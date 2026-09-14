"""GpuGeek OpenAI-compatible Chat Completions provider.

GpuGeek is a gateway rather than a vendor: it fronts models from several
providers behind one OpenAI-shaped `/v1/chat/completions` endpoint, and its
model ids carry the upstream vendor as a prefix (`Vendor2/Claude-4.5-Sonnet`,
`Vendor3/qwen3-max`, `Volcengine/Doubao-Seed-1.6`).

Two consequences shape this file:

* There is no default model. A gateway's catalogue is its own and changes
  without notice, so guessing produces a 404 that reads like a bug in the
  agent. `GPUGEEK_MODEL` or an explicit `--model` is required.
* Thinking level cannot be expressed portably. The upstream models disagree on
  how to ask for it (`enable_thinking`, `reasoning_effort`, a `-thinking`
  suffix in the id), so nothing is sent unless `GPUGEEK_EXTRA_BODY` says to.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from agent.providers._shared import (
    create_openai_compatible_client,
    env_keys_from_values,
    extract_http_status,
    random_env_key,
    should_retry_transient_http_exception,
)
from agent.providers._shared import (
    env_float as _env_float,
)
from agent.providers._shared import (
    env_int as _env_int,
)
from agent.providers.chat_completions import (
    OpenAICompatibleChatCompletionsMixin,
)

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - dotenv is optional

    def load_dotenv(*args: Any, **kwargs: Any) -> None:  # type: ignore
        return None


DEFAULT_GPUGEEK_BASE_URL = "https://api.gpugeek.com/v1"
DEFAULT_GPUGEEK_CONTEXT_TOKENS = 200_000
DEFAULT_GPUGEEK_MAX_TOKENS = 32_000
DEFAULT_GPUGEEK_OUTPUT_SAFETY_TOKENS = 1_024

# Prefixes the gateway uses to namespace its upstreams. Used to tell a gpugeek
# model id apart from an OpenRouter one, which is otherwise the same shape.
GPUGEEK_MODEL_PREFIXES = ("vendor2/", "vendor3/", "volcengine/")

DEFAULT_GPUGEEK_FALLBACK_MODELS = (
    "Vendor2/Claude-4.8-opus",
    "Vendor2/Claude-4.7-opus",
    "Vendor2/Claude-4.6-opus",
    "Vendor2/Claude-4.5-Sonnet",
)
"""Models to move down to when the one in use has no capacity, strongest first.

An authoring run is an hour of dependent turns, so a model that is overloaded for
longer than the retry budget does not fail one request — it throws away the whole run
and everything it had built. Retrying harder does not help when the upstream has no
capacity at all; asking a different one does. Ordered by capability, because the point
is to finish the run on the best model that will actually answer.
"""

CAPACITY_ERROR_FRAGMENTS = (
    "overloaded",
    "no available capacity",
    "capacity",
    "model not found",
    "does not exist",
)
"""Failures that another model would not share. Not timeouts or connection resets:
those are the transport, and the retry loop already rebuilds it."""

UPSTREAM_ROUTING_ERROR_FRAGMENTS = (
    "is not allowed for this account",
    "access to anthropic models is not allowed",
    "no available channel",
)
"""A 400 that describes the upstream account rather than our request.

The gateway fans one model out over several upstream accounts, and not all of them can serve
every model. When it routes a turn to one that cannot, the answer comes back as a 400 naming a
`ValidationException` from the far side -- so the retry loop, which correctly treats a 400 as our
own bad request, gave up, and the model fallback, which keys on capacity wording, did not
recognise it either. One turn's routing killed the whole run: a sweep at four workers lost three
of its first eight cases on turn 1, each on the same account that had just completed a run on the
same model without trouble.

Retried rather than downgraded, because the model is available -- this attempt simply did not
reach it, and the next one is routed afresh. Downgrading the run for it would answer a routing
accident by silently authoring the rest of the asset on a weaker model. A permission problem that
is real still fails, once the retry budget is spent.
"""

logger = logging.getLogger(__name__)


def _is_upstream_routing_error(exc: BaseException) -> bool:
    """Whether this turn reached an upstream that cannot serve the model we asked for."""
    message = str(exc).lower()
    return any(fragment in message for fragment in UPSTREAM_ROUTING_ERROR_FRAGMENTS)


def _load_cwd_dotenv_override() -> None:
    dotenv_path = Path.cwd() / ".env"
    if dotenv_path.exists():
        # Do not clobber already-exported credentials from the caller shell/script.
        load_dotenv(dotenv_path=dotenv_path, override=False)


def gpugeek_api_keys_from_env(env: dict[str, str] | None = None) -> list[str]:
    return env_keys_from_values(
        primary_name="GPUGEEK_API_KEY",
        pool_name="GPUGEEK_API_KEYS",
        env=env,
    )


def gpugeek_api_key_from_env(env: dict[str, str] | None = None) -> str | None:
    _load_cwd_dotenv_override()
    return random_env_key(
        primary_name="GPUGEEK_API_KEY",
        pool_name="GPUGEEK_API_KEYS",
        env=env,
    )


def is_gpugeek_model_id(model_id: str | None) -> bool:
    """Whether a model id looks like it belongs to the GpuGeek catalogue.

    Articraft infers a provider from the model id and reads any bare `/` as
    OpenRouter. GpuGeek ids also contain a `/`, so they have to be recognised by
    their vendor prefix before that rule runs.
    """
    normalized = (model_id or "").strip().lower()
    return normalized.startswith(GPUGEEK_MODEL_PREFIXES)


class GpuGeekLLM(OpenAICompatibleChatCompletionsMixin):
    """GpuGeek Chat Completions client for tool-calling workflows."""

    provider_name = "gpugeek"
    provider_label = "GpuGeek"
    supports_image_content = True
    logger = logger

    def __init__(
        self,
        model_id: str | None = None,
        *,
        thinking_level: str = "high",
        dry_run: bool = False,
    ):
        _load_cwd_dotenv_override()
        resolved_model = model_id or os.environ.get("GPUGEEK_MODEL") or ""
        if not resolved_model.strip():
            raise ValueError(
                "GpuGeek has no default model; its catalogue is its own and changes. "
                "Pass --model or set GPUGEEK_MODEL. Run `amx llm doctor --provider "
                "gpugeek` to list what the gateway currently serves."
            )
        self.model_id = resolved_model
        self.base_url = (os.environ.get("GPUGEEK_BASE_URL") or DEFAULT_GPUGEEK_BASE_URL).rstrip("/")
        self.thinking_level = thinking_level
        self.max_tokens = _env_int("GPUGEEK_MAX_TOKENS", DEFAULT_GPUGEEK_MAX_TOKENS)
        self.context_tokens = _env_int("GPUGEEK_CONTEXT_TOKENS", DEFAULT_GPUGEEK_CONTEXT_TOKENS)
        self.output_safety_tokens = _env_int(
            "GPUGEEK_OUTPUT_SAFETY_TOKENS",
            DEFAULT_GPUGEEK_OUTPUT_SAFETY_TOKENS,
        )
        self.request_timeout_seconds = _env_float("GPUGEEK_REQUEST_TIMEOUT_SECONDS", 900.0)
        self.max_attempts = max(1, int(_env_float("GPUGEEK_MAX_ATTEMPTS", 12)))
        self.retry_base_seconds = _env_float("GPUGEEK_RETRY_BASE_SECONDS", 1.0)
        self.retry_max_seconds = _env_float("GPUGEEK_RETRY_MAX_SECONDS", 60.0)
        self.extra_body = _gpugeek_extra_body()
        self.fallback_models = _gpugeek_fallback_models(self.model_id)

        if dry_run:
            self._client = None
            self._client_is_async = False
            return

        api_key = gpugeek_api_key_from_env()
        if not api_key:
            raise ValueError(
                "GpuGeek credentials not found. Set GPUGEEK_API_KEY or GPUGEEK_API_KEYS."
            )

        self._api_key = api_key
        self._client, self._client_is_async = create_openai_compatible_client(
            provider_label="GpuGeek",
            api_key=api_key,
            base_url=self.base_url,
            timeout_seconds=self.request_timeout_seconds,
        )

    def _chat_extra_body(self) -> dict[str, Any] | None:
        return dict(self.extra_body) if self.extra_body else None

    def _should_retry_exception(self, exc: BaseException) -> bool:
        return should_retry_transient_http_exception(exc) or _is_upstream_routing_error(exc)

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """One turn, moving to a weaker model if this one has no capacity.

        The switch is permanent for the rest of the run. Going back would mean waiting
        out the same outage again at some later turn, having already spent the retry
        budget discovering it. Every turn's trace records the model that answered it, so
        which model authored which part of the asset stays auditable.
        """
        while True:
            try:
                return await super().generate_with_tools(system_prompt, messages, tools)
            except Exception as error:
                nxt = self._next_model_after_capacity_failure(error)
                if nxt is None:
                    raise
                logger.warning(
                    "gpugeek: %s has no capacity (%s); continuing this run on %s",
                    self.model_id,
                    type(error).__name__,
                    nxt,
                )
                self.model_id = nxt

    def _next_model_after_capacity_failure(self, exc: BaseException) -> str | None:
        message = str(exc).lower()
        status = extract_http_status(exc)
        capacity = any(fragment in message for fragment in CAPACITY_ERROR_FRAGMENTS) or status in {
            429,
            503,
        }
        if not capacity:
            return None
        remaining = [
            candidate
            for candidate in self.fallback_models
            if candidate.lower() != self.model_id.lower()
        ]
        if not remaining:
            return None
        self.fallback_models = remaining[1:]
        return remaining[0]

    async def _recover_transient_transport(self, exc: BaseException) -> None:
        """Replace a broken HTTP connection pool before the next retry."""
        if extract_http_status(exc) is not None:
            return
        old_client = self._client
        close = getattr(old_client, "close", None)
        if close is not None:
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass
        self._client, self._client_is_async = create_openai_compatible_client(
            provider_label="GpuGeek",
            api_key=self._api_key,
            base_url=self.base_url,
            timeout_seconds=self.request_timeout_seconds,
        )


def _gpugeek_fallback_models(selected: str) -> list[str]:
    """The chain to walk down from `selected`, set by `GPUGEEK_MODEL_FALLBACKS`.

    Anything at or above the selected model is dropped: a run asked for 4.6 because 4.8
    was not wanted, and quietly promoting it on the first overload would answer a
    question nobody asked.
    """
    raw = os.environ.get("GPUGEEK_MODEL_FALLBACKS", "").strip()
    chain = (
        [item.strip() for item in raw.split(",") if item.strip()]
        if raw
        else list(DEFAULT_GPUGEEK_FALLBACK_MODELS)
    )
    lowered = [item.lower() for item in chain]
    if selected.lower() in lowered:
        chain = chain[lowered.index(selected.lower()) + 1 :]
    return chain


def _gpugeek_extra_body() -> dict[str, Any]:
    """Escape hatch for per-upstream knobs the gateway happens to forward.

    Sent verbatim. Nothing is inferred from `thinking_level`, because the models
    behind this gateway do not agree on how reasoning is requested and an
    unrecognised field is a 400 on some of them.
    """
    raw = os.environ.get("GPUGEEK_EXTRA_BODY", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("GPUGEEK_EXTRA_BODY is not valid JSON; ignoring it")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("GPUGEEK_EXTRA_BODY must be a JSON object; ignoring it")
        return {}
    return parsed
