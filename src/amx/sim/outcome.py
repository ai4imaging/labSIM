"""Did the task actually happen?

`sim_judge` is thorough about how a run went — what struck what, what interpenetrated, what
was still moving — but its seating criteria hang off a declared receiver interface, and a
plan that names no socket geometry gets none. The first end-to-end run of this pipeline
exercised exactly that gap: the fingers closed on air 25 mm from the tube, the tube never
moved, and the verdict was PASS, because nothing in the policy had asked whether the tube
arrived.

A verdict that cannot fail is not feedback, and the whole third part of this pipeline is
built on the assumption that a failure says something. So the objective is checked here as
well, directly and unconditionally, against the last recorded state: the `Destination` says
where the part belongs, and this says whether it is there. It costs one forward evaluation
and it is the one check that must never be skippable.

These findings sit alongside the judge's in the same report, and route to the same repairs.
"""

from __future__ import annotations

import numpy as np

from amx.report import Finding, RepairTarget, Severity
from amx.sim.plan import Destination, OperationPlan
from amx.sim.run import EpisodeResult

TILT_REFERENCE = np.array([0.0, 0.0, 1.0])


def check_destination(plan: OperationPlan, result: EpisodeResult) -> list[Finding]:
    """Compare where the manipulated part ended up with where the plan said it should be.

    Returns an empty list when the plan declares no destination — a plan is allowed to be
    an exploratory poke — but never returns "nothing to say" for one that does.
    """
    destination = plan.destination
    if destination is None or not plan.manipulated_asset:
        return []

    subject = plan.manipulated_asset
    if result.diverged_at_step is not None:
        # Saying anything about where the part ended up would be a fabrication: MuJoCo puts
        # the state back to the beginning when it gives up, so the final pose of a diverged
        # run is the initial pose, and measuring it reports that nothing ever happened.
        return [
            Finding(
                code="O-DIVERGED",
                severity=Severity.FAILURE,
                subject=subject,
                summary=(
                    f"the simulation diverged at step {result.diverged_at_step}, so whether "
                    f"{subject} reached {destination.site} is unknown. Something was driven "
                    "into something else hard enough to break the solver; the judge's "
                    "penetration findings for the steps before that point say where."
                ),
                repair_target=RepairTarget.TRAJECTORY,
                metrics={"diverged_at_step": float(result.diverged_at_step)},
            )
        ]
    start = result.subject_start
    final = result.subject_final
    if start is None or final is None:
        return [
            Finding(
                code="O-SUBJECT-MISSING",
                severity=Severity.FAILURE,
                subject=subject,
                summary=(
                    f"the plan names {subject!r} as the part being moved and gives it a "
                    "destination, but the episode recorded no pose for it, so whether the "
                    "task succeeded cannot be established"
                ),
                repair_target=RepairTarget.LAYOUT,
            )
        ]

    findings = [
        _travelled(destination, subject, start.position, final.position),
        _over_target(destination, subject, final.position, result.destination_xy),
        _seated(destination, subject, final.position),
        _upright(destination, subject, final.quaternion),
    ]
    return [f for f in findings if f is not None]


def _travelled(
    destination: Destination, subject: str, start: np.ndarray, final: np.ndarray
) -> Finding | None:
    """The one check that catches a plan which did nothing at all."""
    moved = float(np.linalg.norm(final - start))
    required = destination.minimum_transfer_distance_m
    metrics = {"travelled_m": moved}
    thresholds = {"minimum_transfer_distance_m": required}
    if moved >= required:
        return Finding(
            code="O-TRANSFER",
            severity=Severity.INFO,
            subject=subject,
            summary=f"{subject} moved {moved * 1000:.1f} mm during the episode.",
            metrics=metrics,
        )
    return Finding(
        code="O-TRANSFER",
        severity=Severity.FAILURE,
        subject=subject,
        summary=(
            f"{subject} moved {moved * 1000:.1f} mm, short of the {required * 1000:.0f} mm "
            "the plan expects a transfer to cover. It was never picked up: check that the "
            "grasp width is under the part's own width and that the tool centre reaches it."
        ),
        repair_target=RepairTarget.TRAJECTORY,
        metrics=metrics,
        thresholds=thresholds,
    )


def _over_target(
    destination: Destination,
    subject: str,
    final: np.ndarray,
    target_xy: tuple[float, float] | None,
) -> Finding | None:
    if target_xy is None:
        return None
    error = float(np.linalg.norm(final[:2] - np.asarray(target_xy)))
    metrics = {"xy_error_m": error}
    if error <= destination.tolerance_xy_m:
        return Finding(
            code="O-PLACEMENT-XY",
            severity=Severity.INFO,
            subject=subject,
            summary=f"{subject} came to rest {error * 1000:.1f} mm from {destination.site}.",
            metrics=metrics,
        )
    return Finding(
        code="O-PLACEMENT-XY",
        severity=Severity.FAILURE,
        subject=subject,
        summary=(
            f"{subject} came to rest {error * 1000:.1f} mm from {destination.site}, outside "
            f"the {destination.tolerance_xy_m * 1000:.0f} mm the plan allows. Either the "
            "release waypoint is off, or the part was dropped short of it."
        ),
        repair_target=RepairTarget.TRAJECTORY,
        metrics=metrics,
        thresholds={"tolerance_xy_m": destination.tolerance_xy_m},
    )


def _seated(destination: Destination, subject: str, final: np.ndarray) -> Finding | None:
    low, high = destination.seated_z_range_m
    z = float(final[2])
    metrics = {"centre_z_m": z}
    thresholds = {"seated_z_min_m": low, "seated_z_max_m": high}
    if low <= z <= high:
        return Finding(
            code="O-SEATED",
            severity=Severity.INFO,
            subject=subject,
            summary=f"{subject}'s centre settled at z = {z * 1000:.1f} mm, within the seated band.",
            metrics=metrics,
        )
    deviation = max(low - z, z - high)
    direction = "above" if z > high else "below"
    return Finding(
        code="O-SEATED",
        severity=Severity.FAILURE,
        subject=subject,
        summary=(
            f"{subject}'s centre settled at z = {z * 1000:.1f} mm, {deviation * 1000:.1f} mm "
            f"{direction} the band the plan calls seated. It is perched on or beside the "
            "socket rather than in it: the insertion waypoint needs to go deeper, or the "
            "socket is too small to admit it."
        ),
        repair_target=RepairTarget.TRAJECTORY,
        metrics={**metrics, "deviation_m": deviation},
        thresholds=thresholds,
    )


def _upright(destination: Destination, subject: str, quaternion: np.ndarray) -> Finding | None:
    axis = _body_z(quaternion)
    tilt = float(np.arccos(np.clip(float(axis @ TILT_REFERENCE), -1.0, 1.0)))
    metrics = {"tilt_rad": tilt}
    if tilt <= destination.maximum_tilt_rad:
        return Finding(
            code="O-UPRIGHT",
            severity=Severity.INFO,
            subject=subject,
            summary=f"{subject} finished {np.degrees(tilt):.1f}° off vertical.",
            metrics=metrics,
        )
    return Finding(
        code="O-UPRIGHT",
        severity=Severity.FAILURE,
        subject=subject,
        summary=(
            f"{subject} finished {np.degrees(tilt):.1f}° off vertical, past the "
            f"{np.degrees(destination.maximum_tilt_rad):.0f}° the plan allows. Something with "
            "liquid in it has been laid over, or it caught the rim on the way in."
        ),
        repair_target=RepairTarget.TRAJECTORY,
        metrics=metrics,
        thresholds={"maximum_tilt_rad": destination.maximum_tilt_rad},
    )


def _body_z(quaternion: np.ndarray) -> np.ndarray:
    """The part's own up-axis in world coordinates, from a `w x y z` quaternion."""
    w, x, y, z = (float(v) for v in quaternion)
    return np.array(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )
