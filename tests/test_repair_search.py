"""The rule-derived repairs, and the frontier that decides which design to grow.

Both are tested without a simulator. The heuristics are a pure function of a finding and
a plan, and the frontier is a pure function of a list of scored rounds, so driving them
directly is both faster and a sharper test than watching a whole loop and inferring what
it must have decided.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from amx.loop.heuristics import CLEARANCE_STEP_M, heuristic_patch
from amx.loop.judge import Judgement
from amx.loop.loop import LoopConfig, Round, _next_to_expand, _only_asset_failures
from amx.report import Finding, RepairTarget, Report, Severity
from amx.sim.plan import Move, OperationPlan, PoseTarget


def _plan() -> OperationPlan:
    return OperationPlan(
        plan_id="p",
        actions=[
            Move(action_id="approach", step_id="step.pick", duration_s=2.0,
                 target=PoseTarget(pos=(0.3, 0.0, 0.25))),
            Move(action_id="lower", step_id="step.pick", duration_s=1.5,
                 target=PoseTarget(pos=(0.3, 0.0, 0.10))),
        ],
    )


def _finding(code, target, *, subject="beaker", metrics=None, detail=None) -> Finding:
    return Finding(
        code=code,
        severity=Severity.FAILURE,
        subject=subject,
        summary="",
        repair_target=target,
        metrics=metrics or {},
        detail=detail or {},
    )


def test_a_collision_is_lifted_by_a_multiple_of_how_deep_it_was():
    """The size of the correction should track the size of the problem, not a constant."""
    patch = heuristic_patch(
        [
            _finding(
                "P1_CONTACT_PENETRATION_EXCEEDED",
                RepairTarget.TRAJECTORY,
                metrics={"penetration_m": 0.006},
                detail={"peak_action_id": "lower"},
            )
        ],
        plan=_plan(),
        part_ids=set(),
    )
    (waypoint,) = patch.waypoints
    assert waypoint.action_id == "lower"
    assert waypoint.position_delta_m[2] == pytest.approx(0.012)
    assert waypoint.duration_scale > 1.0


def test_the_lift_is_capped_however_deep_the_collision_was():
    patch = heuristic_patch(
        [
            _finding(
                "P2_FORBIDDEN_CONTACT",
                RepairTarget.TRAJECTORY,
                metrics={"penetration_m": 5.0},
                detail={"peak_action_id": "lower"},
            )
        ],
        plan=_plan(),
        part_ids=set(),
    )
    assert patch.waypoints[0].position_delta_m[2] == pytest.approx(0.05)


def test_two_findings_on_one_waypoint_become_one_combined_correction():
    """Two reports of one mistake should not produce two competing patches."""
    patch = heuristic_patch(
        [
            _finding("P1_GEOMETRIC_PENETRATION", RepairTarget.TRAJECTORY,
                     metrics={"penetration_m": 0.002}, detail={"peak_action_id": "lower"}),
            _finding("C1_MINIMUM_CLEARANCE_VIOLATED", RepairTarget.TRAJECTORY,
                     metrics={"clearance_m": 0.010}, detail={"peak_action_id": "lower"}),
        ],
        plan=_plan(),
        part_ids=set(),
    )
    assert len(patch.waypoints) == 1
    assert patch.waypoints[0].position_delta_m[2] == pytest.approx(0.020)


def test_a_body_not_reaching_the_socket_floor_is_lowered_not_lifted():
    patch = heuristic_patch(
        [
            _finding("R5_LABWARE_NOT_TOUCHING_SOCKET_FLOOR", RepairTarget.TRAJECTORY,
                     metrics={"gap_m": 0.004}, detail={"peak_action_id": "lower"}),
        ],
        plan=_plan(),
        part_ids=set(),
    )
    assert patch.waypoints[0].position_delta_m[2] < 0


def test_a_tight_socket_is_opened_and_a_loose_one_is_closed():
    tight = heuristic_patch(
        [_finding("R2_RECEIVER_SIDE_CLEARANCE_LOST", RepairTarget.FIXTURE, subject="rack/socket")],
        plan=_plan(),
        part_ids={"rack"},
    )
    loose = heuristic_patch(
        [_finding("R2_RECEIVER_SOCKET_TOO_LOOSE", RepairTarget.FIXTURE, subject="rack/socket")],
        plan=_plan(),
        part_ids={"rack"},
    )
    assert tight.parameters[0].params["clearance_m"] == pytest.approx(CLEARANCE_STEP_M)
    assert loose.parameters[0].params["clearance_m"] == pytest.approx(-CLEARANCE_STEP_M)


def test_a_finding_with_no_rule_produces_nothing_rather_than_a_guess():
    """An empty patch is the signal to ask the model, so it must not be faked."""
    patch = heuristic_patch(
        [_finding("K3_NON_FINITE_STATE", RepairTarget.TRAJECTORY)],
        plan=_plan(),
        part_ids=set(),
    )
    assert patch.empty


def _scored(index: int, failures: int, *, expanded: bool = False, judged: bool = True) -> Round:
    report = Report(
        kind="episode",
        subject="cell",
        findings=[
            Finding(code=f"X{i}", severity=Severity.FAILURE, summary="", subject="s")
            for i in range(failures)
        ],
    )
    return Round(
        index=index,
        directory=Path(f"round-{index:02d}"),
        design=None,  # type: ignore[arg-type]
        dfm=Report(kind="dfm", subject="cell"),
        judgement=Judgement(case_dir=Path("."), verdict="FAIL", reason="", report=report)
        if judged
        else None,
        expanded=expanded,
    )


def test_the_frontier_grows_the_best_design_not_the_newest_one():
    """This is the backtracking: a round that made things worse must not be built on."""
    rounds = [_scored(0, 2, expanded=True), _scored(1, 5)]
    chosen = _next_to_expand(rounds, LoopConfig())
    assert chosen is not None and chosen.index == 1

    rounds.append(_scored(2, 1))
    rounds[1].expanded = True
    assert _next_to_expand(rounds, LoopConfig()).index == 2

    # Round 2 turned out to be a dead end; the frontier should not have forgotten round 0's
    # other options just because a worse design came after it.
    rounds[2].expanded = True
    rounds.append(_scored(3, 9))
    assert _next_to_expand(rounds, LoopConfig()) is None


def test_the_beam_bounds_how_much_of_the_history_stays_eligible():
    rounds = [_scored(i, i, expanded=(i == 0)) for i in range(6)]
    chosen = _next_to_expand(rounds, LoopConfig(beam=2))
    assert chosen is not None and chosen.index == 1
    rounds[1].expanded = True
    assert _next_to_expand(rounds, LoopConfig(beam=2)) is None


def test_backtracking_off_keeps_the_loop_a_chain():
    rounds = [_scored(0, 1, expanded=True), _scored(1, 8)]
    chosen = _next_to_expand(rounds, LoopConfig(backtrack=False))
    assert chosen is not None and chosen.index == 1


def test_failures_that_are_all_the_assets_fault_are_recognised_as_such():
    """Nothing in a patch can fix a massless body, so the loop must stop patching."""
    asset_only = _scored(0, 0)
    asset_only.judgement.report.findings = [
        _finding("S1_MOVABLE_BODY_MASSLESS", RepairTarget.ASSET),
        _finding("S2_MOVABLE_BODY_VISUAL_ONLY", RepairTarget.ASSET),
    ]
    assert _only_asset_failures(asset_only)

    mixed = _scored(1, 0)
    mixed.judgement.report.findings = [
        _finding("S1_MOVABLE_BODY_MASSLESS", RepairTarget.ASSET),
        _finding("P2_FORBIDDEN_CONTACT", RepairTarget.TRAJECTORY),
    ]
    assert not _only_asset_failures(mixed)
