"""The finish gate, which is the whole point of the grounded agent.

Stock Articraft stops when the code compiles. These tests pin down the added behaviour:
the agent will not accept a finish attempt while a required check has never been run,
while one has gone stale behind an edit, or while one is failing — and it does accept it
once all three of those are false.

No LLM is involved. The agent object is constructed and its gate is driven directly,
because what is being tested is the gate's logic and not a model's willingness to obey it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from amx.grounding.spec import (
    ComponentTarget,
    DimensionTarget,
    GroundingSpec,
    StabilityTarget,
)
from amx.paths import PROJECT_ROOT, activate_articraft
from amx.report import Finding, RepairTarget, Report, Severity

SAMPLE_TUBE = PROJECT_ROOT / "examples" / "wetlab_transfer" / "assets" / "sample_tube" / "model.py"

pytest.importorskip("cadquery", reason="the Articraft compiler needs the geometry backend")


@pytest.fixture(scope="module")
def articraft():
    activate_articraft()


@pytest.fixture
def spec() -> GroundingSpec:
    return GroundingSpec(
        asset_id="sample_tube",
        asset_class="microcentrifuge tube",
        summary="1.5 mL microcentrifuge tube with a hinged flat cap",
        dimensions=[
            DimensionTarget(
                id="DIM-OD", name="barrel_diameter", value_m=0.0108, kind="diameter_outer"
            )
        ],
        components=[ComponentTarget(id="CMP-1", name="barrel")],
        stability=StabilityTarget(),
    )


def _agent(tmp_path: Path, spec: GroundingSpec | None, **kwargs):
    from agent.grounded_harness import GroundedArticraftAgent

    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model.py"
    model.write_text(SAMPLE_TUBE.read_text())
    return GroundedArticraftAgent(
        file_path=str(model),
        provider="gpugeek",
        model_id="Vendor2/Claude-4.5-Sonnet",
        display_enabled=False,
        grounding_spec=spec,
        grounding_dir=tmp_path / "grounding",
        **kwargs,
    )


def _tool_call(name: str) -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_grounding_tools_are_offered_only_when_there_is_a_spec(articraft, tmp_path):
    with_spec = _agent(tmp_path / "a", GroundingSpec(asset_id="x", dimensions=[
        DimensionTarget(id="D", name="d", value_m=0.01, kind="extent_z")
    ]))
    without = _agent(tmp_path / "b", None)
    assert "check_physical_grounding" in with_spec.tool_registry.get_all_tool_names()
    assert "check_physical_grounding" not in without.tool_registry.get_all_tool_names()


def test_required_checks_follow_what_the_spec_actually_states(articraft, tmp_path, spec):
    agent = _agent(tmp_path, spec)
    required = agent.required_checks()
    assert "check_physical_grounding" in required
    assert "check_protocol_grounding" in required
    # No visual features were stated, so a visual check has nothing to hold it to.
    assert "check_visual_grounding" not in required


def test_a_check_measures_the_real_model_and_answers_in_grounding_signals(
    articraft, tmp_path, spec
):
    agent = _agent(tmp_path, spec)
    result, message = asyncio.run(agent._execute_tool(_tool_call("check_physical_grounding")))

    assert result.is_success(), result.error
    assert result.output.startswith("<grounding_signals>")
    assert "status=success" in result.output
    assert message["role"] == "tool" and message["name"] == "check_physical_grounding"
    assert json.loads(message["content"])["result"] == result.output


def test_a_wrong_dimension_comes_back_as_a_failure_with_both_numbers(articraft, tmp_path):
    wrong = GroundingSpec(
        asset_id="sample_tube",
        dimensions=[
            DimensionTarget(
                id="DIM-OD", name="barrel_diameter", value_m=0.030, kind="diameter_outer"
            )
        ],
    )
    agent = _agent(tmp_path, wrong)
    result, _ = asyncio.run(agent._execute_tool(_tool_call("check_physical_grounding")))
    assert "status=failure" in result.output
    assert "10.79 mm" in result.output and "30.00 mm" in result.output


def test_results_are_written_beside_the_run(articraft, tmp_path, spec):
    agent = _agent(tmp_path, spec)
    asyncio.run(agent._execute_tool(_tool_call("check_physical_grounding")))
    written = sorted((tmp_path / "grounding" / "checks").glob("*.json"))
    assert written, "the check should leave its report on disk"
    assert json.loads(written[0].read_text())["kind"] == "grounding-physical"


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def _finish(agent, conversation, *, edited: bool = True):
    """Attempt to finish, having edited the geometry first unless told otherwise.

    `edited=True` is the default because it is what a run does: the model is refused, it
    changes something, it submits again. `edited=False` is the model pressing the button
    twice over identical code, which the breaker must not count as two attempts.
    """
    if edited:
        agent.grounding_state.mark_mutated()
    return asyncio.run(
        agent._handle_finish_attempt(
            conversation, message="done", turn_count=1, tool_call_count=1, usage={}
        )
    )


def _pretend_compiled(agent) -> None:
    """Mark the code fresh without running the compiler.

    The gate's compile arm is Articraft's and already tested there; what these tests are
    about is what happens after it is satisfied.
    """
    agent._latest_code_is_fresh = lambda: True  # type: ignore[method-assign]


def test_an_unmeasured_asset_is_measured_before_it_is_allowed_to_finish(
    articraft, tmp_path, spec
):
    """The gate does not ask the model to go and check; it checks.

    Refusing the turn and saying "now call the tools" is correct but costs a turn per
    check, and the budget ran out before the last revision was ever measured. What must
    not happen is a submission of an unmeasured revision, and running the checks here
    rules that out more firmly than asking does.
    """
    agent = _agent(tmp_path, spec)
    _pretend_compiled(agent)
    required = agent.required_checks()
    assert agent.grounding_state.stale_or_missing(required) == required

    result = _finish(agent, [])

    assert agent.grounding_state.stale_or_missing(required) == []
    assert agent.grounding_state.reports["check_physical_grounding"].kind == "grounding-physical"
    assert result is not None and result.success, "this tube does match its spec"


def test_a_failing_measurement_taken_at_the_gate_refuses_the_finish(articraft, tmp_path):
    wrong = GroundingSpec(
        asset_id="sample_tube",
        dimensions=[
            DimensionTarget(
                id="DIM-OD", name="barrel_diameter", value_m=0.030, kind="diameter_outer"
            )
        ],
    )
    agent = _agent(tmp_path, wrong)
    _pretend_compiled(agent)
    conversation: list[dict] = []

    assert _finish(agent, conversation) is None
    reminder = conversation[-1]["content"]
    assert "<grounding_required>" in reminder
    assert "should be 30.00 mm" in reminder


def test_a_passing_asset_can_finish(articraft, tmp_path, spec):
    agent = _agent(tmp_path, spec)
    _pretend_compiled(agent)
    for tool in agent.required_checks():
        agent.grounding_state.record(tool, Report(kind=tool, subject="sample_tube"))

    assert not agent.grounding_state.stale_or_missing(agent.required_checks())
    assert not agent.grounding_state.failing(agent.required_checks())


def test_a_failing_check_blocks_the_finish_and_repeats_why(articraft, tmp_path, spec):
    agent = _agent(tmp_path, spec)
    _pretend_compiled(agent)
    for tool in agent.required_checks():
        agent.grounding_state.record(tool, Report(kind=tool, subject="sample_tube"))
    agent.grounding_state.record(
        "check_physical_grounding",
        Report(
            kind="grounding-physical",
            subject="sample_tube",
            findings=[
                Finding(
                    code="G-DIM",
                    severity=Severity.FAILURE,
                    summary="barrel_diameter measures 10.79 mm but should be 30.00 mm",
                    repair_target=RepairTarget.ASSET,
                )
            ],
        ),
    )
    conversation: list[dict] = []
    # No edit: these reports are the freshly recorded state of the code being submitted, and
    # bumping the revision would make them stale and send the gate off to re-measure.
    assert _finish(agent, conversation, edited=False) is None
    assert "should be 30.00 mm" in conversation[-1]["content"]


def test_an_edit_makes_an_earlier_pass_stale(articraft, tmp_path, spec):
    agent = _agent(tmp_path, spec)
    _pretend_compiled(agent)
    for tool in agent.required_checks():
        agent.grounding_state.record(tool, Report(kind=tool, subject="sample_tube"))
    assert not agent.grounding_state.stale_or_missing(agent.required_checks())

    agent._mark_code_mutated("replace")
    assert agent.grounding_state.stale_or_missing(agent.required_checks())

    # Finishing re-measures rather than trusting the earlier pass: the placeholder
    # reports above are replaced by real ones taken against the edited code.
    _finish(agent, [])
    assert not agent.grounding_state.stale_or_missing(agent.required_checks())
    assert agent.grounding_state.reports["check_physical_grounding"].kind == "grounding-physical"


# --------------------------------------------------------------------------- #
# the circuit breaker
# --------------------------------------------------------------------------- #


def _unsatisfiable_spec() -> GroundingSpec:
    """A target no edit to this tube can reach: it is 10.79 mm and asked to be 30 mm.

    Which is the shape of the real thing. SYF-001's specification asked for a 0.22 µm
    membrane pore rating, read as a bounding-box extent, and the model was refused 51
    times over a number that was never a length.
    """
    return GroundingSpec(
        asset_id="sample_tube",
        dimensions=[
            DimensionTarget(
                id="DIM-OD", name="barrel_diameter", value_m=0.030, kind="diameter_outer"
            )
        ],
    )


def test_the_same_refusal_three_times_stops_blocking(articraft, tmp_path):
    """Fourteen of sixty cases spent their whole budget on a failure they could not fix."""
    agent = _agent(tmp_path, _unsatisfiable_spec())
    _pretend_compiled(agent)
    conversation: list[dict] = []

    assert _finish(agent, conversation) is None
    assert _finish(agent, conversation) is None

    result = _finish(agent, conversation)
    assert result is not None and result.success


def test_letting_the_run_finish_does_not_move_the_score(articraft, tmp_path):
    """The failure stays a failure; only the gate stops spending turns on it.

    Downgrading it to a warning would be the obvious implementation and the wrong one:
    `judge._finding_credit` awards a warning a flat 0.5 and grades a failure on how far off
    it actually was, so softening the severity would hand back half the item's weight for a
    dimension that is 64% out.
    """
    from amx.bench.judge import _finding_credit

    agent = _agent(tmp_path, _unsatisfiable_spec())
    _pretend_compiled(agent)
    for _ in range(3):
        _finish(agent, [])

    report = agent.grounding_state.reports["check_physical_grounding"]
    stuck = report.failures
    assert [f.code for f in stuck] == ["G-DIM-HARD"]
    assert stuck[0].severity is Severity.FAILURE
    assert stuck[0].detail["unsatisfiable"] is True
    assert _finding_credit(stuck[0]) == 0.0

    # The gate no longer counts it, which is the whole difference.
    assert agent.grounding_state.failing(agent.required_checks()) == []


def test_the_refusals_before_the_third_still_say_what_is_wrong(articraft, tmp_path):
    """The breaker must not make the first two refusals any less useful.

    Those two are the run's real chance to fix the thing, so they carry the measurement
    exactly as before. Nothing is said about the breaker: a model told which failure is
    about to stop blocking it has been handed a reason to wait rather than fix.
    """
    agent = _agent(tmp_path, _unsatisfiable_spec())
    _pretend_compiled(agent)
    conversation: list[dict] = []

    _finish(agent, conversation)
    _finish(agent, conversation)

    refusals = [m for m in conversation if "<grounding_required>" in str(m.get("content"))]
    assert len(refusals) == 2
    assert all("should be 30.00 mm" in m["content"] for m in refusals)


def test_the_ruling_reaches_the_evidence_on_disk(articraft, tmp_path):
    agent = _agent(tmp_path, _unsatisfiable_spec())
    _pretend_compiled(agent)
    for _ in range(3):
        _finish(agent, [])

    written = sorted((tmp_path / "grounding" / "checks").glob("*-physical.json"))
    findings = json.loads(written[-1].read_text())["findings"]
    assert [f["severity"] for f in findings] == ["failure"]
    assert findings[0]["detail"]["unsatisfiable"] is True


def _attempt(state, reports: list[Report]) -> list[str]:
    """A finish attempt that follows a real edit.

    The breaker only counts a repeat when the geometry moved in between, so a test that
    submits twice without touching anything is exercising that guard rather than the count.
    """
    state.mark_mutated()
    return state.note_refusal(reports)


def _refusal(code: str, subject: str = "sample_tube/x") -> list[Report]:
    return [
        Report(
            kind="grounding-physical",
            subject="sample_tube",
            findings=[
                Finding(code=code, severity=Severity.FAILURE, summary=f"{code} failed", subject=subject)
            ],
        )
    ]


def _mixed_refusal(codes: list[str]) -> list[Report]:
    return [
        Report(
            kind="grounding-physical",
            subject="sample_tube",
            findings=[
                Finding(
                    code=code,
                    severity=Severity.FAILURE,
                    summary=f"{code} failed",
                    subject="sample_tube/x",
                )
                for code in codes
            ],
        )
    ]


def test_a_run_that_fails_differently_every_time_is_still_stuck(articraft, tmp_path, spec):
    """The identical-set count restarts whenever the set changes, and cannot see this.

    Restarting is deliberate: fixing one thing and breaking another is progress and should keep
    its budget. But a run where *which* subset fails keeps moving never reaches the threshold and
    never stops. PCR-001 was refused 35 times that way, alternating an unmeasurable bore with a
    wrong one against a well count that changed on every edit, spent 61% of its turns pressing
    finish, and submitted nothing at all -- a zero where a scored asset was available.

    What is persistently in the way gets declared; what drifted past once does not.
    """
    state = _agent(tmp_path, spec).grounding_state

    declared: list[str] = []
    for index in range(8):
        # The bore is in the way every time. The other failure is different on each attempt, so
        # the set never repeats and the identical-set counter restarts every time.
        declared = _attempt(state, _mixed_refusal(["G-BORE", f"G-DRIFT-{index}"]))

    assert declared == ["G-BORE:sample_tube/x"]
    assert "G-BORE:sample_tube/x" in state.unsatisfiable
    assert not any(key.startswith("G-DRIFT") for key in state.unsatisfiable)


def test_a_run_making_progress_keeps_its_budget(articraft, tmp_path, spec):
    """Refusals that stop coming are not what is blocking a run, however many there were."""
    state = _agent(tmp_path, spec).grounding_state

    for index in range(7):
        assert _attempt(state, _mixed_refusal([f"G-MOVING-{index}"])) == []

    assert state.unsatisfiable == set()


def test_pressing_finish_again_over_untouched_code_is_not_another_attempt(
    articraft, tmp_path, spec
):
    """Three attempts means three tries, not three submissions of one try.

    The count is what decides a requirement is unreachable and stops enforcing it, so what
    it counts had better be work. A model that resubmits identical geometry has not tried
    again, and letting that reach the threshold would let any run opt out of any check by
    pressing the button three times.
    """
    state = _agent(tmp_path, spec).grounding_state

    # The first submission is a real attempt and counts as one.
    assert state.note_refusal(_refusal("G-STUCK")) == []
    # Four more over untouched code. Without the guard these alone would reach the
    # threshold and the check would stop being enforced for the rest of the run.
    for _ in range(4):
        assert state.note_refusal(_refusal("G-STUCK")) == []
    assert state.unsatisfiable == set()

    # Two attempts with an edit behind each, which brings the genuine count to three.
    assert _attempt(state, _refusal("G-STUCK")) == []
    assert _attempt(state, _refusal("G-STUCK")) == ["G-STUCK:sample_tube/x"]


def test_a_run_that_is_getting_somewhere_keeps_its_budget(articraft, tmp_path, spec):
    """Only an unchanging set of failures is evidence of being stuck.

    A model that fixes one thing and breaks another is working, and counting those
    refusals together would cut it off after three edits.
    """
    state = _agent(tmp_path, spec).grounding_state

    for code in ("G-ONE", "G-TWO", "G-THREE", "G-FOUR"):
        assert _attempt(state, _refusal(code)) == []
    assert state.unsatisfiable == set()


def test_the_count_restarts_when_the_failures_change(articraft, tmp_path, spec):
    state = _agent(tmp_path, spec).grounding_state

    assert _attempt(state, _refusal("G-ONE")) == []
    assert _attempt(state, _refusal("G-ONE")) == []
    assert _attempt(state, _refusal("G-TWO")) == []
    assert _attempt(state, _refusal("G-TWO")) == []
    # Two of each, so neither has been seen the three consecutive times it takes.
    assert state.unsatisfiable == set()

    assert _attempt(state, _refusal("G-TWO")) == ["G-TWO:sample_tube/x"]


def test_a_fixable_failure_beside_a_stuck_one_still_blocks(articraft, tmp_path, spec):
    """Ruling one thing unreachable must not wave the rest of the report through."""
    agent = _agent(tmp_path, spec)
    state = agent.grounding_state
    for _ in range(3):
        _attempt(state, _refusal("G-STUCK"))
    assert state.unsatisfiable == {"G-STUCK:sample_tube/x"}

    mixed = Report(
        kind="grounding-physical",
        subject="sample_tube",
        findings=[
            Finding(
                code="G-STUCK",
                severity=Severity.FAILURE,
                summary="stuck",
                subject="sample_tube/x",
            ),
            Finding(
                code="G-FIXABLE",
                severity=Severity.FAILURE,
                summary="fixable",
                subject="sample_tube/y",
            ),
        ],
    )
    state.record("check_physical_grounding", mixed)
    stored = state.reports["check_physical_grounding"]

    assert [f.code for f in state.blocking_failures(stored)] == ["G-FIXABLE"]
    assert state.failing(["check_physical_grounding"]) == [stored]

    conversation: list[dict] = []
    agent._append_grounding_required_reminder(conversation, [], [stored])
    assert "fixable" in conversation[-1]["content"]
    assert "stuck" not in conversation[-1]["content"]


def test_a_failure_is_only_declared_unreachable_once(articraft, tmp_path, spec):
    """The reminder should name it when it is decided, and not nag afterwards."""
    state = _agent(tmp_path, spec).grounding_state

    for _ in range(3):
        declared = _attempt(state, _refusal("G-ONE"))
    assert declared == ["G-ONE:sample_tube/x"]
    assert _attempt(state, _refusal("G-ONE")) == []


def test_the_measured_value_moving_does_not_reset_the_count(articraft, tmp_path, spec):
    """Keying on the summary would read a stuck run as a changing one.

    Every edit nudges the measurement, so the sentence differs each time even when the
    failure is identical. Code and subject are what make two refusals the same refusal.
    """
    state = _agent(tmp_path, spec).grounding_state

    def drifting(measured: float) -> list[Report]:
        return [
            Report(
                kind="grounding-physical",
                subject="sample_tube",
                findings=[
                    Finding(
                        code="G-DIM-HARD",
                        severity=Severity.FAILURE,
                        subject="sample_tube/barrel_diameter",
                        summary=f"barrel_diameter measures {measured:.2f} mm but should be 30.00 mm",
                    )
                ],
            )
        ]

    assert _attempt(state, drifting(10.79)) == []
    assert _attempt(state, drifting(11.02)) == []
    assert _attempt(state, drifting(10.94)) == ["G-DIM-HARD:sample_tube/barrel_diameter"]


def test_the_visual_check_is_required_whenever_the_task_states_an_appearance(
    articraft, tmp_path
):
    """Renders are how a floating or detached part is caught; numbers never see it."""
    from amx.grounding.spec import VisualTarget

    stated = GroundingSpec(
        asset_id="sample_tube",
        components=[ComponentTarget(id="CMP-1", name="barrel")],
        visual=VisualTarget(features=["a conical barrel below a cylindrical neck"]),
    )
    agent = _agent(tmp_path, stated)
    assert "check_visual_grounding" in agent.required_checks()


def test_without_a_spec_the_gate_is_articraft_s_own(articraft, tmp_path):
    agent = _agent(tmp_path, None)
    assert not agent.grounding_enabled
    assert agent.required_checks() == []


def test_the_prompt_gains_a_grounding_section_only_when_the_tools_are_there(
    articraft, tmp_path, spec
):
    grounded = _agent(tmp_path / "a", spec)
    plain = _agent(tmp_path / "b", None)
    assert "check_physical_grounding" in grounded.system_prompt
    assert "A cavity has to be a real subtraction from the solid." in grounded.system_prompt
    assert "check_physical_grounding" not in plain.system_prompt
