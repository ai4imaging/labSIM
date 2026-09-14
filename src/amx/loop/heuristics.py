"""Repairs derived from the finding alone, without asking anything.

Two reasons this exists, and neither is "the API might be down".

The first is that a search needs candidates that differ. Sampling one model twice for the
same failure produces two paraphrases of one idea far more often than two ideas, and a
frontier fed on paraphrases is a repair loop wearing a costume. A rule-derived patch is
reliably a *different* attempt, because it comes from somewhere else.

The second is that a good fraction of these findings do not need a model. `sim_judge`
reports which body collided, during which action, and by how much. "Lift the waypoint by
the depth of the interpenetration" is not an insight — it is arithmetic on numbers already
in the finding, and it is right often enough to be worth trying before spending a turn.

Where the arithmetic runs out, this returns nothing and the model gets asked.
"""

from __future__ import annotations

from amx.loop.repair import (
    MAX_LAYOUT_DELTA_M,
    MAX_WAYPOINT_DELTA_M,
    LayoutPatch,
    ParameterPatch,
    RepairPatch,
    WaypointPatch,
)
from amx.report import Finding, RepairTarget
from amx.sim.plan import Move, OperationPlan

DEFAULT_LIFT_M = 0.008
"""Fallback vertical nudge when a finding says something collided but not how deeply."""

CLEARANCE_STEP_M = 0.0005
"""How much to open or close a socket per round. Half a millimetre is the smallest change
worth making to a printed fit and about the largest that is safe to make blind."""

SLOW_DOWN = 1.5
LENGTHEN_SETTLE = 1.6


def heuristic_patch(
    failures: list[Finding],
    *,
    plan: OperationPlan,
    part_ids: set[str],
) -> RepairPatch:
    """The obvious repair for each failure, where there is one.

    Returns an empty patch when nothing here applies, which the caller should read as
    "ask the model" rather than as "nothing is wrong".
    """
    waypoints: dict[str, WaypointPatch] = {}
    layout: dict[str, LayoutPatch] = {}
    parameters: dict[str, ParameterPatch] = {}
    reasons: list[str] = []

    for finding in failures:
        family = finding.code.split("_", 1)[0]
        if finding.repair_target is RepairTarget.TRAJECTORY:
            _trajectory_rule(finding, family, plan, waypoints, reasons)
        elif finding.repair_target is RepairTarget.LAYOUT:
            _layout_rule(finding, layout, reasons)
        elif finding.repair_target is RepairTarget.FIXTURE:
            _fixture_rule(finding, part_ids, parameters, reasons)

    return RepairPatch(
        diagnosis=(
            "Rule-derived repair, from the findings' own measurements: "
            + "; ".join(reasons)
        )
        if reasons
        else "",
        waypoints=list(waypoints.values()),
        layout=list(layout.values()),
        parameters=list(parameters.values()),
    )


def _trajectory_rule(
    finding: Finding,
    family: str,
    plan: OperationPlan,
    waypoints: dict[str, WaypointPatch],
    reasons: list[str],
) -> None:
    action_id = _action_of(finding, plan)
    if action_id is None:
        return

    if family in {"P1", "P2", "P3", "C1"}:
        # Something was hit or came too close. Lift the offending move by the depth it
        # was short by, so the size of the correction matches the size of the problem.
        lift = min(MAX_WAYPOINT_DELTA_M, max(DEFAULT_LIFT_M, _depth_of(finding) * 2.0))
        _merge(
            waypoints,
            WaypointPatch(
                action_id=action_id,
                position_delta_m=(0.0, 0.0, lift),
                duration_scale=SLOW_DOWN,
                reason=f"{finding.code}: lift clear by {lift * 1000:.1f} mm and slow the approach",
            ),
        )
        reasons.append(f"lifted {action_id} by {lift * 1000:.1f} mm")
        return

    if family in {"K1", "K4", "P4"}:
        # Moving too fast: a jump, a force spike, or a body that passed through something
        # between two steps. All three are the same fix.
        _merge(
            waypoints,
            WaypointPatch(
                action_id=action_id,
                duration_scale=SLOW_DOWN,
                reason=f"{finding.code}: slow the motion so the solver can resolve it",
            ),
        )
        reasons.append(f"slowed {action_id}")
        return

    if finding.code in {"R3_LABWARE_STILL_MOVING", "R3_LABWARE_TILTED", "R3_LABWARE_NOT_SEATED"}:
        _merge(
            waypoints,
            WaypointPatch(
                action_id=action_id,
                duration_scale=LENGTHEN_SETTLE,
                reason=f"{finding.code}: give the release longer to settle",
            ),
        )
        reasons.append(f"lengthened {action_id}")
        return

    if finding.code in {
        "R5_LABWARE_NOT_TOUCHING_SOCKET_FLOOR",
        "R4_LABWARE_INSERTION_DEPTH_OUT_OF_RANGE",
    }:
        # Not all the way in. Lower it by the gap that was measured.
        gap = _metric(finding, ("gap_m", "depth_m", "distance_m")) or DEFAULT_LIFT_M
        drop = -min(MAX_WAYPOINT_DELTA_M, abs(gap) * 1.2)
        _merge(
            waypoints,
            WaypointPatch(
                action_id=action_id,
                position_delta_m=(0.0, 0.0, drop),
                reason=f"{finding.code}: lower by {abs(drop) * 1000:.1f} mm to reach the floor",
            ),
        )
        reasons.append(f"lowered {action_id} by {abs(drop) * 1000:.1f} mm")


def _layout_rule(
    finding: Finding, layout: dict[str, LayoutPatch], reasons: list[str]
) -> None:
    target = finding.subject.split("/")[0]
    if not target:
        return
    if finding.code in {"S4_INITIAL_PENETRATION", "S6_RESTING_GEOMETRY_INTERPENETRATION"}:
        lift = min(MAX_LAYOUT_DELTA_M, max(0.002, _depth_of(finding) * 1.5))
        layout.setdefault(
            target,
            LayoutPatch(
                target_id=target,
                position_delta_m=(0.0, 0.0, lift),
                reason=f"{finding.code}: raise clear of what it is intersecting",
            ),
        )
        reasons.append(f"raised {target} by {lift * 1000:.1f} mm")
    elif finding.code == "F1_UNSUPPORTED_FLOATING_BODY":
        gap = _metric(finding, ("gap_m", "height_m", "distance_m")) or 0.003
        layout.setdefault(
            target,
            LayoutPatch(
                target_id=target,
                position_delta_m=(0.0, 0.0, -min(MAX_LAYOUT_DELTA_M, abs(gap))),
                reason=f"{finding.code}: drop it onto what should be supporting it",
            ),
        )
        reasons.append(f"dropped {target} onto its support")


def _fixture_rule(
    finding: Finding,
    part_ids: set[str],
    parameters: dict[str, ParameterPatch],
    reasons: list[str],
) -> None:
    part_id = next((item for item in part_ids if item in finding.subject), None)
    if part_id is None:
        return
    # A socket is either too tight or too loose, and `clearance_m` is the one parameter
    # every socketed template spells the same way.
    looser = finding.code != "R2_RECEIVER_SOCKET_TOO_LOOSE"
    step = CLEARANCE_STEP_M if looser else -CLEARANCE_STEP_M
    existing = parameters.get(part_id)
    current = existing.params.get("clearance_m", 0.0) if existing else 0.0
    parameters[part_id] = ParameterPatch(
        part_id=part_id,
        params={"clearance_m": current + step},
        reason=f"{finding.code}: {'open' if looser else 'close'} the fit by "
        f"{abs(step) * 1000:.1f} mm",
    )
    reasons.append(f"{'opened' if looser else 'closed'} {part_id}")


def _merge(waypoints: dict[str, WaypointPatch], patch: WaypointPatch) -> None:
    """Combine two rules that landed on the same action, rather than letting one win.

    Two findings on one waypoint usually mean one mistake seen twice, and applying the
    larger correction of the two is closer to right than applying whichever was reported
    first.
    """
    existing = waypoints.get(patch.action_id)
    if existing is None:
        waypoints[patch.action_id] = patch
        return
    combined = tuple(
        a if abs(a) >= abs(b) else b
        for a, b in zip(existing.position_delta_m, patch.position_delta_m, strict=True)
    )
    waypoints[patch.action_id] = WaypointPatch(
        action_id=patch.action_id,
        position_delta_m=combined,  # type: ignore[arg-type]
        width_m=patch.width_m if patch.width_m is not None else existing.width_m,
        value=patch.value if patch.value is not None else existing.value,
        duration_scale=max(existing.duration_scale, patch.duration_scale),
        reason=f"{existing.reason}; {patch.reason}",
    )


def _action_of(finding: Finding, plan: OperationPlan) -> str | None:
    """Which action a finding happened during.

    `sim_judge` records the action at the peak of the event, which is what should be
    moved. Failing that, the last `Move` is a defensible guess: releases and insertions —
    the failures without a recorded peak — happen at the end.
    """
    detail = finding.detail
    for key in ("peak_action_id", "action_id", "peak_step_id"):
        value = detail.get(key)
        if isinstance(value, str) and any(a.action_id == value for a in plan.actions):
            return value
    moves = [action.action_id for action in plan.actions if isinstance(action, Move)]
    return moves[-1] if moves else None


def _depth_of(finding: Finding) -> float:
    return abs(
        _metric(finding, ("penetration_m", "depth_m", "overlap_m", "clearance_m", "distance_m"))
        or 0.0
    )


def _metric(finding: Finding, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in finding.metrics:
            return float(finding.metrics[key])
        # sim_judge is not always consistent about units in the key name.
        if f"{key}m" in finding.metrics:
            return float(finding.metrics[f"{key}m"])
    for key in keys:
        base = key.removesuffix("_m")
        millimetres = finding.metrics.get(f"{base}_mm")
        if millimetres is not None:
            return float(millimetres) / 1000.0
    return None
