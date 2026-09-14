"""A minimal structured-output LLM client.

Everything the agents in this project ask a model for is a filled-in pydantic schema: a
part's parameters, a repair patch, a grounding verdict. Nothing asks for code or XML. So
this wraps exactly one call shape — "return an instance of this model" — and validates the
result before handing it back.

Providers are a table rather than a branch because the interesting one here is neither
Anthropic nor OpenAI but an OpenAI-compatible gateway (`gpugeek`), and a gateway is not
the same thing as the API it imitates: it may not implement `response_format`, its model
ids are its own, and it is the component most likely to be swapped. So the OpenAI-shaped
path negotiates down through three ways of asking for JSON — tool call, then
`json_object`, then plain instruction with fenced-block extraction — instead of assuming
the strictest one works. `probe` exists for the same reason: when a gateway is
misconfigured the failure arrives as an unrelated schema error three layers up, and being
able to ask "can you reach it, and what does it serve" first is worth a command of its own.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

Schema = TypeVar("Schema", bound=BaseModel)

DEFAULT_GPUGEEK_MODEL = "Vendor2/Claude-4.8-opus"
"""The gateway model this project is set up against, for authoring and for grading.

A gateway's catalogue is its own and changes, so `amx llm doctor --provider gpugeek` is
still the way to see what it serves. Naming a default anyway is the lesser evil: without
one, a missing `GPUGEEK_MODEL` turned into a sweep where every case failed to generate.
"""


@dataclass(frozen=True)
class Provider:
    """How to reach one API, and what to call by default."""

    kind: str
    """`anthropic` or `openai`; the wire format, not the vendor."""

    key_env: str
    base_url_env: str
    default_base_url: str = ""
    default_model: str = ""


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        kind="anthropic",
        key_env="ANTHROPIC_API_KEY",
        base_url_env="ANTHROPIC_BASE_URL",
        default_model="claude-sonnet-4-5",
    ),
    "openai": Provider(
        kind="openai",
        key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        default_model="gpt-4.1",
    ),
    "gpugeek": Provider(
        kind="openai",
        key_env="GPUGEEK_API_KEY",
        base_url_env="GPUGEEK_BASE_URL",
        default_model=DEFAULT_GPUGEEK_MODEL,
    ),
}

DEFAULT_MODELS = {name: p.default_model for name, p in PROVIDERS.items() if p.default_model}


class LlmUnavailable(RuntimeError):
    """No credentials for the selected provider."""


class StructuredOutputError(RuntimeError):
    """The model answered, but not with a valid instance of the requested schema."""


@dataclass
class Call:
    """One request/response pair, kept so runs can be audited without a receipt protocol."""

    purpose: str
    model: str
    schema: str
    attempts: int
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "model": self.model,
            "schema": self.schema,
            "attempts": self.attempts,
            "error": self.error,
        }


@dataclass
class LlmClient:
    provider: str = field(default_factory=lambda: os.environ.get("AMX_LLM_PROVIDER", "anthropic"))
    model: str = ""
    max_tokens: int = 8192
    attempts: int = 2
    temperature: float = 0.0
    calls: list[Call] = field(default_factory=list)
    trace_dir: Path | None = None
    """Where to write each exchange, prompt and all.

    Off by default because most callers do not want the files, but a repair loop does: the
    only way to tell "the model proposed something bad" from "we asked it badly" after the
    fact is to still have the prompt. Set this and every call lands as a numbered pair of
    files under it.
    """

    def __post_init__(self) -> None:
        self.provider = self.provider.lower()
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"unknown provider {self.provider!r}; expected one of {sorted(PROVIDERS)}"
            )
        self.spec = PROVIDERS[self.provider]
        self.model = self.model or os.environ.get("AMX_LLM_MODEL") or self.spec.default_model
        if not self.model:
            raise ValueError(
                f"{self.provider} has no default model; set AMX_LLM_MODEL or pass model=. "
                f"Run `amx llm doctor --provider {self.provider}` to list what it serves."
            )

    @property
    def api_key(self) -> str:
        return os.environ.get(self.spec.key_env, "")

    @property
    def base_url(self) -> str:
        return os.environ.get(self.spec.base_url_env) or self.spec.default_base_url

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def structured(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        schema: type[Schema],
        images: list[bytes] | None = None,
    ) -> Schema:
        """Ask for one instance of `schema`. Retries once on a schema violation."""
        if not self.available:
            raise LlmUnavailable(
                f"{self.provider} has no API key; set {self.spec.key_env} (see .env.example)"
            )
        record = Call(purpose=purpose, model=self.model, schema=schema.__name__, attempts=0)
        self.calls.append(record)

        message = user
        last_error = ""
        for attempt in range(1, self.attempts + 1):
            record.attempts = attempt
            raw = self._request(system, message, schema, images if attempt == 1 else None)
            try:
                parsed = schema.model_validate(raw)
            except ValidationError as error:
                self._trace(record, attempt, system, message, raw, str(error))
                last_error = str(error)
                message = (
                    f"{user}\n\nYour previous answer did not validate against the schema:\n"
                    f"{last_error}\n\nReturn a corrected object."
                )
            else:
                self._trace(record, attempt, system, message, raw, None)
                return parsed
        record.error = last_error
        raise StructuredOutputError(
            f"{self.model} could not produce a valid {schema.__name__}: {last_error}"
        )

    def _trace(
        self,
        record: Call,
        attempt: int,
        system: str,
        user: str,
        raw: Any,
        error: str | None,
    ) -> None:
        """Write one exchange. Never raises — an unwritable trace is not worth a failed run."""
        if self.trace_dir is None:
            return
        try:
            directory = Path(self.trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            index = len(self.calls)
            stem = f"{index:03d}-{record.purpose.replace('.', '-')}-a{attempt}"
            (directory / f"{stem}-prompt.txt").write_text(
                f"model: {self.model}\nschema: {record.schema}\n\n"
                f"--- system ---\n{system}\n\n--- user ---\n{user}\n"
            )
            (directory / f"{stem}-response.json").write_text(
                json.dumps(
                    {"error": error, "raw": raw}, indent=2, ensure_ascii=False, default=str
                )
                + "\n"
            )
        except OSError:
            pass

    def _request(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        images: list[bytes] | None,
    ) -> dict[str, Any]:
        if self.spec.kind == "anthropic":
            return self._anthropic(system, user, schema, images)
        return self._openai(system, user, schema, images)

    def _anthropic(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        images: list[bytes] | None,
    ) -> dict[str, Any]:
        import anthropic

        client = anthropic.Anthropic(
            api_key=self.api_key, **({"base_url": self.base_url} if self.base_url else {})
        )
        tool = {
            "name": "submit",
            "description": f"Return one {schema.__name__}.",
            "input_schema": schema.model_json_schema(),
        }
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for blob in images or ():
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _b64(blob),
                    },
                }
            )
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit"},
            messages=[{"role": "user", "content": content}],
        )
        for block in response.content:
            if block.type == "tool_use":
                return dict(block.input)
        raise StructuredOutputError(f"{self.model} returned no tool call")

    def _openai(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        images: list[bytes] | None,
    ) -> dict[str, Any]:
        """Ask an OpenAI-shaped endpoint for JSON, in descending order of strictness.

        A gateway that rejects `tools` or `response_format` answers with a 400 naming the
        parameter, which is recoverable by dropping it; anything else is a real failure and
        is re-raised rather than retried into silence.
        """
        import openai

        client = openai.OpenAI(
            api_key=self.api_key, **({"base_url": self.base_url} if self.base_url else {})
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for blob in images or ():
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_b64(blob)}"},
                }
            )
        json_schema = schema.model_json_schema()
        base: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        instruction = (
            f"{system}\n\nReply with a single JSON object matching this schema, and nothing "
            f"else:\n{json.dumps(json_schema, ensure_ascii=False)}"
        )
        plain = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ]
        strategies: list[tuple[str, dict[str, Any]]] = [
            (
                "tool",
                {
                    **base,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "submit",
                                "description": f"Return one {schema.__name__}.",
                                "parameters": json_schema,
                            },
                        }
                    ],
                    "tool_choice": {"type": "function", "function": {"name": "submit"}},
                },
            ),
            ("json_object", {**base, "messages": plain, "response_format": {"type": "json_object"}}),
            ("text", {**base, "messages": plain}),
        ]

        unsupported: list[str] = []
        for name, kwargs in strategies:
            try:
                response = client.chat.completions.create(**kwargs)
            except openai.BadRequestError as error:
                unsupported.append(f"{name}: {error}")
                continue
            message = response.choices[0].message
            for call in getattr(message, "tool_calls", None) or ():
                return _loads(call.function.arguments, self.model)
            if message.content:
                return _loads(message.content, self.model)
            unsupported.append(f"{name}: empty response")
        raise StructuredOutputError(
            f"{self.model} at {self.base_url or 'the default endpoint'} accepted none of the "
            "JSON request forms: " + "; ".join(unsupported)
        )


@dataclass
class Probe:
    """What `amx llm doctor` found, in the order the checks would fail."""

    provider: str
    base_url: str
    key_present: bool
    reachable: bool
    detail: str = ""
    models: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"provider   {self.provider}",
            f"endpoint   {self.base_url or '(vendor default)'}",
            f"api key    {'set' if self.key_present else 'MISSING'}",
            f"reachable  {'yes' if self.reachable else 'NO'}",
        ]
        if self.detail:
            lines.append(f"detail     {self.detail}")
        if self.models:
            shown = self.models[:40]
            lines.append(f"models     {len(self.models)} served")
            lines.extend(f"  {m}" for m in shown)
            if len(self.models) > len(shown):
                lines.append(f"  ... and {len(self.models) - len(shown)} more")
        return "\n".join(lines)


def probe(provider: str, *, timeout: float = 20.0) -> Probe:
    """Ask an endpoint what it serves, over stdlib HTTP.

    Deliberately not through the vendor SDK: the question being answered is whether the
    network and the credential work at all, and an SDK turns a captive portal or a DNS
    result into a retry loop and then a generic timeout.
    """
    name = provider.lower()
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; expected one of {sorted(PROVIDERS)}")
    spec = PROVIDERS[name]
    key = os.environ.get(spec.key_env, "")
    base = os.environ.get(spec.base_url_env) or spec.default_base_url
    if not base:
        if spec.kind == "anthropic":
            base = "https://api.anthropic.com/v1"
        elif name == "openai":
            base = "https://api.openai.com/v1"
        else:
            return Probe(
                provider=name,
                base_url="",
                key_present=bool(key),
                reachable=False,
                detail=f"set {spec.base_url_env} to your own OpenAI-compatible endpoint; this repo ships none",
            )
    result = Probe(provider=name, base_url=base, key_present=bool(key), reachable=False)

    url = base.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if spec.kind == "anthropic":
        headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
    else:
        headers["Authorization"] = f"Bearer {key}"
    try:
        with urllib.request.urlopen(  # noqa: S310 — the URL is configuration, not input
            urllib.request.Request(url, headers=headers), timeout=timeout
        ) as response:
            body = response.read().decode("utf-8", "replace")
        result.reachable = True
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")[:400]
        # An HTTP status means the endpoint is there; 401 is a credential problem, not a
        # network one, and saying so saves an hour of looking at the wrong thing.
        result.reachable = True
        result.detail = f"HTTP {error.code}: {body.strip()}"
        return result
    except Exception as error:  # noqa: BLE001 — every transport failure reads the same here
        result.detail = f"{type(error).__name__}: {error}"
        if "portal" in str(error).lower() or isinstance(error, TimeoutError):
            result.detail += "  (a captive portal or firewall would look like this)"
        return result

    if body.lstrip().startswith("<"):
        result.reachable = False
        result.detail = (
            "the endpoint returned HTML, not JSON — almost always a captive portal or proxy "
            f"intercepting the request: {body.strip()[:200]}"
        )
        return result
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        result.detail = f"unparseable response: {error}"
        return result
    entries = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(entries, list):
        result.models = sorted(
            str(e.get("id", e)) if isinstance(e, dict) else str(e) for e in entries
        )
    else:
        result.detail = f"unexpected shape: {body[:200]}"
    return result


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _loads(text: str, model: str) -> dict[str, Any]:
    """Parse JSON that may arrive fenced or with prose around it."""
    text = text.strip()
    for candidate in (text, *(m.group(1).strip() for m in _FENCE.finditer(text))):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, dict):
                return value
    raise StructuredOutputError(f"{model} returned non-JSON: {text[:300]}")


def _b64(blob: bytes) -> str:
    import base64

    return base64.b64encode(blob).decode("ascii")
