"""Guards on the handful of edits made to the vendored Articraft tree.

`vendor/articraft` is a copy of someone else's repository, and the day it gets refreshed
from upstream every edit listed in `vendor/PROVENANCE.md` disappears. Most of them fail
loudly when that happens — no gpugeek provider means no runs at all — but two do not:

* `_maybe_inject_tool_attachments` reverting to absent means the grounded agent simply
  stops sending renders, and every visual check quietly becomes text-only.
* `agent_cls` reverting to unthreaded means grounded runs silently fall back to the stock
  agent, which finishes as soon as the code compiles.

Both would look like a quality regression in the model rather than a missing seam. These
tests make the refresh fail instead.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from amx.paths import activate_articraft


@pytest.fixture(scope="module")
def articraft():
    return activate_articraft()


def test_the_provider_enum_knows_about_gpugeek(articraft):
    from articraft.values import ProviderName, infer_provider_from_model_id

    assert ProviderName.GPUGEEK.value == "gpugeek"
    # The prefix rule has to come before the generic `/` rule, or every gpugeek model id
    # is read as OpenRouter and the run authenticates against the wrong gateway.
    assert infer_provider_from_model_id("Vendor2/Claude-4.5-Sonnet") is ProviderName.GPUGEEK
    assert infer_provider_from_model_id("anthropic/claude-3") is ProviderName.OPENROUTER


def test_the_factory_can_build_a_gpugeek_client(articraft):
    from agent.providers.factory import ProviderConfig, ProviderConstructors, create_provider_client

    assert hasattr(ProviderConstructors(), "gpugeek")
    client = create_provider_client(
        ProviderConfig(provider="gpugeek", model_id="Vendor2/Claude-4.5-Sonnet"), dry_run=True
    )
    assert client.model_id == "Vendor2/Claude-4.5-Sonnet"


def test_gpugeek_defaults_tolerate_multi_minute_gateway_outages(articraft, monkeypatch):
    from agent.providers.gpugeek import GpuGeekLLM

    for name in (
        "GPUGEEK_MAX_ATTEMPTS",
        "GPUGEEK_RETRY_BASE_SECONDS",
        "GPUGEEK_RETRY_MAX_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    client = GpuGeekLLM(model_id="Vendor2/Claude-4.8-opus", dry_run=True)
    assert client.max_attempts == 12
    assert client.retry_base_seconds == 1.0
    assert client.retry_max_seconds == 60.0


def test_an_overloaded_model_hands_the_run_to_a_weaker_one(articraft, monkeypatch):
    """Retrying harder does not create capacity; asking a different model does.

    An authoring run is an hour of dependent turns, so an upstream that is out of
    capacity for longer than the retry budget does not cost one request — it discards
    the whole run and everything it had built.
    """
    from agent.providers.gpugeek import GpuGeekLLM

    monkeypatch.delenv("GPUGEEK_MODEL_FALLBACKS", raising=False)
    client = GpuGeekLLM(model_id="Vendor2/Claude-4.8-opus", dry_run=True)
    overloaded = RuntimeError("Error code: 400 - the model is overloaded, try again later")

    walked = []
    while (nxt := client._next_model_after_capacity_failure(overloaded)) is not None:
        client.model_id = nxt
        walked.append(nxt)

    assert walked == [
        "Vendor2/Claude-4.7-opus",
        "Vendor2/Claude-4.6-opus",
        "Vendor2/Claude-4.5-Sonnet",
    ]


def test_a_run_is_never_promoted_to_a_model_it_declined(articraft, monkeypatch):
    monkeypatch.delenv("GPUGEEK_MODEL_FALLBACKS", raising=False)
    from agent.providers.gpugeek import GpuGeekLLM

    client = GpuGeekLLM(model_id="Vendor2/Claude-4.6-opus", dry_run=True)

    assert client.fallback_models == ["Vendor2/Claude-4.5-Sonnet"]


def test_a_malformed_request_is_not_blamed_on_capacity(articraft, monkeypatch):
    """Switching model would hide a bug in the request and lose the stronger model."""
    monkeypatch.delenv("GPUGEEK_MODEL_FALLBACKS", raising=False)
    from agent.providers.gpugeek import GpuGeekLLM

    client = GpuGeekLLM(model_id="Vendor2/Claude-4.8-opus", dry_run=True)

    assert client._next_model_after_capacity_failure(ValueError("invalid tool schema")) is None


def _routing_error() -> RuntimeError:
    """The 400 the gateway returns when it routes a turn to an account without the model."""
    return RuntimeError(
        "Error code: 400 - {'code': 400, 'message': 'InvokeModel: operation error Bedrock "
        "Runtime: InvokeModel, https response error StatusCode: 400, ValidationException: "
        "Access to Anthropic models is not allowed for this account.'}"
    )


def test_a_turn_routed_to_an_account_without_the_model_is_tried_again(articraft, monkeypatch):
    """The model is there; this attempt did not reach it, and the next one is routed afresh.

    The gateway spreads one model over several upstream accounts and not all of them can serve
    it. A 400 is our own bad request as a rule, so the retry loop gave up on the first one --
    and a sweep at four workers lost three of its first eight cases on turn 1, each on the same
    account that had just finished a run on the same model.
    """
    monkeypatch.delenv("GPUGEEK_MODEL_FALLBACKS", raising=False)
    from agent.providers.gpugeek import GpuGeekLLM

    client = GpuGeekLLM(model_id="Vendor2/Claude-5-Opus", dry_run=True)

    assert client._should_retry_exception(_routing_error())


def test_a_turn_with_no_channel_for_the_model_is_tried_again(articraft, monkeypatch):
    """The gateway sometimes answers 400 'no available channel' while the model is still listed.

    Twenty-one cases of one sweep died on turn 1 to this wording. It is the same class of
    accident as an account without the model: retry the same id, do not author the rest of
    the asset on a weaker one.
    """
    monkeypatch.delenv("GPUGEEK_MODEL_FALLBACKS", raising=False)
    from agent.providers.gpugeek import GpuGeekLLM

    client = GpuGeekLLM(model_id="Vendor2/Claude-5-Opus", dry_run=True)
    error = RuntimeError(
        "Error code: 400 - {'code': 400, 'message': 'no available channel for model: Claude-5-opus'}"
    )

    assert client._should_retry_exception(error)
    assert client._next_model_after_capacity_failure(error) is None
    assert client.model_id == "Vendor2/Claude-5-Opus"


def test_a_routing_accident_does_not_downgrade_the_rest_of_the_run(articraft, monkeypatch):
    """Answering it with a weaker model would author the rest of the asset on one, unasked."""
    monkeypatch.delenv("GPUGEEK_MODEL_FALLBACKS", raising=False)
    from agent.providers.gpugeek import GpuGeekLLM

    client = GpuGeekLLM(model_id="Vendor2/Claude-5-Opus", dry_run=True)

    assert client._next_model_after_capacity_failure(_routing_error()) is None
    assert client.model_id == "Vendor2/Claude-5-Opus"


def test_retry_jitter_never_collapses_to_an_immediate_retry(articraft):
    from agent.providers._shared import async_retry

    attempts = 0
    delays: list[float] = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary")
        return "ok"

    async def record_sleep(delay: float):
        delays.append(delay)

    result = asyncio.run(
        async_retry(
            operation,
            max_attempts=3,
            should_retry=lambda _error: True,
            base_delay=2.0,
            max_delay=60.0,
            logger=__import__("logging").getLogger(__name__),
            context="test",
            sleep_fn=record_sleep,
            rng=lambda: 0.0,
        )
    )
    assert result == "ok"
    assert delays == [1.0, 2.0]


def test_gpugeek_gets_the_prompt_written_for_its_tool_surface(articraft):
    from sdk._profiles import get_sdk_profile

    profile = get_sdk_profile("sdk")
    assert profile.prompt_name_for_provider("gpugeek") == profile.openrouter_prompt_name


def test_the_registry_still_takes_a_grounding_flag(articraft):
    from agent.tools import build_tool_registry

    plain = build_tool_registry("gpugeek").get_all_tool_names()
    grounded = build_tool_registry("gpugeek", grounding=True).get_all_tool_names()
    assert "check_physical_grounding" not in plain
    assert {
        "check_physical_grounding",
        "check_protocol_grounding",
        "check_visual_grounding",
    } <= set(grounded)


def test_the_turn_loop_still_calls_the_attachment_hook(articraft):
    """Without this call the grounded agent's renders never reach the model."""
    from agent.harness import ArticraftAgent

    assert hasattr(ArticraftAgent, "_maybe_inject_tool_attachments")
    assert "_maybe_inject_tool_attachments" in inspect.getsource(ArticraftAgent.run)


def test_agent_cls_reaches_the_place_that_builds_the_agent(articraft):
    """Without this thread, a grounded run silently falls back to the stock agent."""
    from agent import single_run

    for function in (single_run.run_from_input, single_run.execute_single_run):
        assert "agent_cls" in inspect.signature(function).parameters
    assert "agent_cls=agent_cls" in inspect.getsource(single_run.run_from_input)
    assert "async with agent_cls(" in inspect.getsource(single_run.execute_single_run)


def test_the_runner_lets_a_caller_choose_the_agent_class(articraft):
    """The seam above is useless if the wrapper on top of it overrides the choice.

    It did, for a while: `runner._execute_single_run` passed `agent_cls=ArticraftAgent`
    unconditionally, so every grounded run died with a duplicate-argument TypeError before
    the agent was ever built. The fix is a `setdefault`, and this is the test that says so.
    """
    from agent import runner

    source = inspect.getsource(runner._execute_single_run)
    assert 'setdefault("agent_cls"' in source
    assert "agent_cls=ArticraftAgent," not in source


def test_the_trace_records_turn_boundaries(articraft):
    """`amx trace` cannot split a transcript into turns without these two events."""
    from agent.traces import TraceWriter

    for name in ("write_llm_request", "write_llm_result"):
        assert hasattr(TraceWriter, name)
    assert "write_llm_request" in inspect.getsource(_harness_run(articraft))


def _harness_run(articraft):
    from agent.harness import ArticraftAgent

    return ArticraftAgent.run


def test_the_system_prompt_name_selects_a_provider_variant(articraft):
    """A path defeats the provider substitution; a bare filename is what enables it."""
    from pathlib import Path

    from agent.prompts.loader import resolve_system_prompt_path

    from amx.asset.generate import SYSTEM_PROMPT

    assert "/" not in SYSTEM_PROMPT
    resolved = resolve_system_prompt_path(
        SYSTEM_PROMPT, provider="gpugeek", sdk_package="sdk", repo_root=Path(articraft)
    )
    assert resolved.name == "designer_system_prompt_openrouter.txt"
    assert resolved.is_file()


def test_gpugeek_is_allowed_to_send_images(articraft):
    from agent.tools import SUPPORTED_IMAGE_MIME_TYPES_BY_PROVIDER

    assert "image/png" in SUPPORTED_IMAGE_MIME_TYPES_BY_PROVIDER["gpugeek"]
