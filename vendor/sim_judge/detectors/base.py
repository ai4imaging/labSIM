"""The contract shared by all detectors.

Every detector answers a single question: "on this step, is there an instance of the
problem I am responsible for?" It does not need to know that the other detectors exist,
and it does not merge intervals itself — collapsing per-step hits into event spans is
the job of :class:`~sim_judge.report.aggregate.EventAggregator`.

Splitting things this way buys two things: each piece of detection logic can be read and
tested on its own, and adding a new class of check only requires writing one ``feed``
method, without touching the scheduling or reporting code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from collections.abc import Iterable

import mujoco

from sim_judge.defaults import GenericThresholds
from sim_judge.loader.policy import Policy
from sim_judge.replay import Frame, Replayer
from sim_judge.report.finding import Observation, Severity, Subject, Tier
from sim_judge.world.geometry import UpAxis
from sim_judge.world.naming import GeomInfo, NameResolver
from sim_judge.world.timeline import Timeline


@dataclass(frozen=True, slots=True)
class DetectorContext:
    """The read-only environment shared by all detectors."""

    model: mujoco.MjModel
    resolver: NameResolver
    policy: Policy
    timeline: Timeline
    thresholds: GenericThresholds
    replayer: Replayer
    """Used by detectors that need on-demand geom distance queries (``mj_geomDistance``).

    The query depends on the current configuration held in ``MjData``, so it may only be
    called while ``feed`` is processing the current frame.
    """

    up_axis: UpAxis | None
    """The up direction, defined as the direction opposite gravity. ``None`` means the
    model declares no gravity, so there is no meaningful vertical direction to speak of."""

    timestep_s: float
    """The simulation timestep of the recording. Used to convert thresholds expressed in
    seconds into a number of steps."""

    def steps_for(self, seconds: float) -> int:
        """Convert a duration into a number of steps, never fewer than 1.

        Thresholds are declared in seconds and converted here so that the same criteria
        cover the same amount of physical time on a recording with a 1 ms timestep as on
        one with a 5 ms timestep, rather than the same number of solver steps.
        """
        if self.timestep_s <= 0.0:
            return 1
        return max(1, round(seconds / self.timestep_s))


@runtime_checkable
class Detector(Protocol):
    """The detector protocol."""

    code_prefix: str

    @property
    def enabled(self) -> bool:
        """Whether this detector has anything to judge in the current scene.

        Returns ``False`` when the required policy declarations are missing, or when the
        scene lacks the precondition for the judgement (for instance, "floating" is
        meaningless if no gravity is declared). Such detectors are dropped at assembly
        time rather than running empty on every frame.
        """

    def feed(self, frame: Frame) -> Iterable[Observation]:
        """Process one frame and emit every hit found on that step."""

    def finalize(self) -> Iterable[Observation]:
        """Emit additional findings after the trajectory has been fully traversed. Used
        for checks that can only conclude once the whole replay has been seen."""

    def reset_continuity(self) -> None:
        """Tell the detector that the next frame is not temporally contiguous with the
        previous one.

        Rescan mode only traverses a handful of non-adjacent windows, which may be tens
        of seconds apart. Detectors that difference consecutive frames (position jumps,
        tunneling) must discard their history here, otherwise they would read normal
        motion across a window boundary as a configuration jump. Detectors that keep no
        history need not implement this.
        """


class BaseDetector:
    """Provides the constructor plus a few conveniences for building :class:`Observation`."""

    code_prefix = "?"

    def __init__(self, context: DetectorContext) -> None:
        self.ctx = context
        self.model = context.model
        self.resolver = context.resolver
        self.policy = context.policy
        self.limits = context.thresholds

    @property
    def enabled(self) -> bool:
        return True

    def feed(self, frame: Frame) -> Iterable[Observation]:  # pragma: no cover - implemented by subclasses
        return ()

    def finalize(self) -> Iterable[Observation]:
        return ()

    def reset_continuity(self) -> None:
        return None


def geom_pair_subject(geom_a: GeomInfo, geom_b: GeomInfo) -> Subject:
    return Subject(
        kind="geom_pair",
        a_name=geom_a.name,
        a_entity=geom_a.entity_id or None,
        a_body=geom_a.body_name,
        b_name=geom_b.name,
        b_entity=geom_b.entity_id or None,
        b_body=geom_b.body_name,
    )


def body_subject(resolver: NameResolver, body_id: int) -> Subject:
    return Subject(
        kind="body",
        a_name=resolver.body_name(body_id),
        a_entity=resolver.body_entity(body_id) or None,
        a_body=resolver.body_name(body_id),
    )


def joint_subject(model: mujoco.MjModel, joint_id: int) -> Subject:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"<joint:{joint_id}>"
    body_id = int(model.jnt_bodyid[joint_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"<body:{body_id}>"
    return Subject(kind="joint", a_name=name, a_body=body_name)


def declared(
    code: str,
    step: int,
    subject: Subject,
    metrics: dict[str, float],
    peak_metric: str,
    thresholds: dict[str, float],
    rule_id: str | None,
    *,
    min_duration_steps: int = 1,
    merge_gap_steps: int | None = None,
    detail: dict | None = None,
) -> Observation:
    """Build a hit for a violation of an explicit declaration. Hits of this kind force the
    overall verdict to FAIL."""
    return Observation(
        code=code,
        severity=Severity.FAILURE,
        tier=Tier.DECLARED,
        step_index=step,
        subject=subject,
        metrics=metrics,
        peak_metric=peak_metric,
        thresholds=thresholds,
        rule_id=rule_id,
        min_duration_steps=min_duration_steps,
        merge_gap_steps=merge_gap_steps,
        detail=detail or {},
    )


def generic(
    code: str,
    step: int,
    subject: Subject,
    metrics: dict[str, float],
    peak_metric: str,
    thresholds: dict[str, float],
    *,
    severity: Severity = Severity.WARNING,
    min_duration_steps: int = 1,
    merge_gap_steps: int | None = None,
    detail: dict | None = None,
) -> Observation:
    """Build a hit from a generic geometric/physical criterion.

    These default to warning level only: the policy has said nothing about these objects,
    and the judge does not get to define failure on the task author's behalf. Only cases
    that cannot be numerically valid at all (NaN, infinity) are explicitly promoted to
    FAILURE by the caller.
    """
    return Observation(
        code=code,
        severity=severity,
        tier=Tier.GENERIC,
        step_index=step,
        subject=subject,
        metrics=metrics,
        peak_metric=peak_metric,
        thresholds=thresholds,
        rule_id=None,
        min_duration_steps=min_duration_steps,
        merge_gap_steps=merge_gap_steps,
        detail=detail or {},
    )
