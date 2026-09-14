"""What the plan infers from its own actions, and what the loop keeps between rounds.

No physics here. Both modules are bookkeeping, and both have a failure mode where the
bookkeeping is quietly wrong and everything downstream believes it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from amx.loop.judge import Judgement
from amx.loop.loop import LoopResult, Round
from amx.loop.repair import (
    MAX_WAYPOINT_DELTA_M,
    PatchRejected,
    RepairPatch,
    WaypointPatch,
    apply_patch,
)
from amx.report import Finding, Report, Severity
from amx.sim.plan import Actuate, Grip, Hold, Move, OperationPlan, PoseTarget
from amx.sim.scene import Bench, RobotRef, Workcell

TOOL_DOWN = (3.14159, 0.0, 0.0)


def _move(step: str, z: float) -> Move:
    return Move(step_id=step, duration_s=0.5, target=PoseTarget(pos=(0.3, 0.1, z), euler=TOOL_DOWN))


def test_grip_phases_read_the_direction_not_the_number():
    """A grasp is a narrowing and a release is a widening, at whatever absolute width.

    The same plan shape has to work for a 10 mm tube and a 30 mm conical. Anything keyed to
    a threshold gets one of the two wrong, and gets it wrong silently: the judge is handed
    a policy whose retention and release phases point at the wrong actions.
    """
    for grasp, release in ((0.009, 0.030), (0.028, 0.060)):
        plan = OperationPlan(
            plan_id="p", protocol_id="p", manipulated_asset="x",
            actions=[
                _move("step.approach", 0.3),
                Grip(step_id="step.approach", duration_s=0.2, width_m=release),
                Grip(step_id="step.grasp", duration_s=0.2, width_m=grasp),
                _move("step.lift", 0.4),
                Grip(step_id="step.release", duration_s=0.2, width_m=release),
                Hold(step_id="step.settle", duration_s=0.2),
            ],
        )
        assert plan.grip_step_ids() == ["step.grasp", "step.lift"]
        assert plan.release_step_ids() == ["step.release"]


def test_a_plan_that_opens_first_is_not_holding_during_the_open():
    """Opening the hand before approaching must not count as taking hold of something."""
    plan = OperationPlan(
        plan_id="p", protocol_id="p", manipulated_asset="x",
        actions=[
            Grip(step_id="step.open", duration_s=0.2, width_m=0.030),
            _move("step.descend", 0.2),
            Grip(step_id="step.grasp", duration_s=0.2, width_m=0.009),
        ],
    )
    assert "step.open" not in plan.grip_step_ids()
    assert "step.grasp" in plan.grip_step_ids()


def test_a_plan_that_grasps_immediately_is_still_understood():
    """A gripper parks open, so a first `Grip` that narrows is a grasp even unannounced.

    This is the case that used to produce a policy with no retention and no release phases,
    and therefore a judge with nothing to check about the grasp.
    """
    plan = OperationPlan(
        plan_id="p", protocol_id="p", manipulated_asset="x",
        actions=[
            Grip(step_id="step.grasp", duration_s=0.2, width_m=0.009),
            _move("step.lift", 0.4),
            Grip(step_id="step.release", duration_s=0.2, width_m=0.030),
        ],
    )
    assert plan.grip_step_ids() == ["step.grasp", "step.lift"]
    assert plan.release_step_ids() == ["step.release"]


def test_grip_width_is_bounded_by_something_physical():
    """A metre-wide grasp is a typo, and the schema is where a typo should stop."""
    with pytest.raises(ValidationError):
        Grip(step_id="s", duration_s=0.2, width_m=1.5)
    with pytest.raises(ValidationError):
        Grip(step_id="s", duration_s=0.2, width_m=-0.01)


def test_plan_round_trips_through_json(tmp_path):
    """The loop writes a plan every round and reads it back, so this cannot drift."""
    plan = OperationPlan(
        plan_id="p", protocol_id="p", manipulated_asset="x",
        actions=[
            _move("step.a", 0.3),
            Grip(step_id="step.b", duration_s=0.2, width_m=0.01),
            Actuate(step_id="step.c", duration_s=0.2, actuator="rack/lock", value=1.0),
            Hold(step_id="step.d", duration_s=0.1),
        ],
    )
    path = plan.write(tmp_path / "plan.json")
    again = OperationPlan.read(path)
    assert again.model_dump(mode="json") == plan.model_dump(mode="json")
    # The action union has to survive the trip as its concrete types, not as dicts.
    assert isinstance(again.actions[1], Grip)
    assert isinstance(again.actions[2], Actuate)


def _cell(robot_dir):
    return Workcell(
        workcell_id="patched",
        robot=RobotRef(model=robot_dir.name),
        bench=Bench(size_xy=(1.0, 0.8)),
        assets=[],
        fixtures=[],
    )


def test_waypoint_patch_moves_only_what_it_names(robot_dir):
    """A patch is applied deterministically, so a repair round changes one thing knowably."""
    plan = OperationPlan(
        plan_id="p", protocol_id="p", manipulated_asset="x",
        actions=[_move("step.a", 0.30), _move("step.b", 0.40)],
    )
    _, patched, _ = apply_patch(
        workcell=_cell(robot_dir),
        plan=plan,
        parts={},
        patch=RepairPatch(
            diagnosis="lift higher",
            waypoints=[WaypointPatch(action_id="a01.move", position_delta_m=(0.0, 0.0, 0.02))],
        ),
    )
    assert patched.actions[0].target.pos == pytest.approx((0.3, 0.1, 0.30))
    assert patched.actions[1].target.pos == pytest.approx((0.3, 0.1, 0.42))
    # The original is untouched, so a round can always fall back to what it started from.
    assert plan.actions[1].target.pos == pytest.approx((0.3, 0.1, 0.40))


def test_a_waypoint_patch_cannot_teleport_the_arm(robot_dir):
    """Deltas are capped per round, so a bad proposal degrades rather than destroys.

    The loop hands a model the judge's findings and applies what comes back. Without a cap,
    one confused proposal moves a waypoint half a metre and every subsequent round is
    reasoning about a cell that no longer resembles the one that was measured.

    Clamped rather than rejected, deliberately: an over-large proposal is usually right
    about the direction and wrong about the magnitude, so taking the step it can is more
    useful than discarding the round.
    """
    plan = OperationPlan(
        plan_id="p", protocol_id="p", manipulated_asset="x", actions=[_move("step.a", 0.30)]
    )
    _, patched, _ = apply_patch(
        workcell=_cell(robot_dir),
        plan=plan,
        parts={},
        patch=RepairPatch(
            waypoints=[WaypointPatch(action_id="a00.move", position_delta_m=(0.0, 0.0, 0.5))]
        ),
    )
    moved = patched.actions[0].target.pos[2] - 0.30
    assert 0.0 < moved <= MAX_WAYPOINT_DELTA_M + 1e-9


def test_a_patch_naming_a_missing_action_is_rejected(robot_dir):
    """Silently ignoring an unknown action id would make a round look applied when it was not."""
    plan = OperationPlan(
        plan_id="p", protocol_id="p", manipulated_asset="x", actions=[_move("step.a", 0.3)]
    )
    with pytest.raises(PatchRejected):
        apply_patch(
            workcell=_cell(robot_dir),
            plan=plan,
            parts={},
            patch=RepairPatch(
                waypoints=[WaypointPatch(action_id="a99.move", position_delta_m=(0.0, 0.0, 0.01))]
            ),
        )


def test_an_empty_patch_is_not_silently_a_change(robot_dir):
    """`propose_repair` returning nothing has to be distinguishable from a repair."""
    plan = OperationPlan(
        plan_id="p", protocol_id="p", manipulated_asset="x", actions=[_move("step.a", 0.3)]
    )
    patch = RepairPatch(diagnosis="nothing to do")
    assert patch.empty
    _, after, _ = apply_patch(
        workcell=_cell(robot_dir), plan=plan, parts={}, patch=patch
    )
    assert after.model_dump() == plan.model_dump()


def _round(index: int, *, failures: int, warnings: int = 0, verdict: str = "FAIL",
           dfm_failures: int = 0) -> Round:
    """A round carrying only what `score` reads, so the ordering can be tested on its own."""

    def findings(count: int, severity: Severity) -> list[Finding]:
        return [
            Finding(code=f"X{i}", severity=severity, summary="", subject="s")
            for i in range(count)
        ]

    report = Report(
        kind="episode",
        subject="cell",
        findings=findings(failures, Severity.FAILURE) + findings(warnings, Severity.WARNING),
    )
    dfm = Report(kind="dfm", subject="cell", findings=findings(dfm_failures, Severity.FAILURE))
    return Round(
        index=index,
        directory=Path(f"round-{index:02d}"),
        design=None,  # type: ignore[arg-type]
        dfm=dfm,
        judgement=Judgement(case_dir=Path("."), verdict=verdict, reason="", report=report),
    )


def test_the_best_round_is_the_one_that_passed_not_the_last_one():
    """A loop that improves and then regresses must still deliver the round that worked."""
    result = LoopResult(run_dir=Path("run"))
    result.rounds = [
        _round(0, failures=3),
        _round(1, failures=0, verdict="PASS"),
        _round(2, failures=5),
    ]
    assert result.passed
    assert result.best is not None and result.best.index == 1


def test_an_unmakeable_round_loses_to_a_makeable_one_that_failed_as_often():
    """DFM failures count against a round, or the loop would deliver a part nobody can print."""
    result = LoopResult(run_dir=Path("run"))
    result.rounds = [_round(0, failures=1, dfm_failures=2), _round(1, failures=1)]
    assert result.best is not None and result.best.index == 1


def test_an_inconclusive_round_ranks_behind_a_conclusive_failure():
    """Zero failures because nothing could be judged is not better than a known failure."""
    result = LoopResult(run_dir=Path("run"))
    result.rounds = [_round(0, failures=0, verdict="INCONCLUSIVE"), _round(1, failures=4)]
    assert result.best is not None and result.best.index == 1


def test_ties_are_broken_towards_the_earlier_round():
    """Two equally good rounds should resolve to the first, so a run is reproducible."""
    result = LoopResult(run_dir=Path("run"))
    result.rounds = [_round(0, failures=2, warnings=1), _round(1, failures=2, warnings=1)]
    assert result.best is not None and result.best.index == 0
