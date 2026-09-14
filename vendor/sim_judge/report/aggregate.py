"""Merge step-by-step hits into event intervals.

A single real physical anomaly gets detected over and over on hundreds or thousands of
consecutive steps. Emitting all of them would drown the report, so hits are grouped by
"same diagnostic code + same subject", hits adjacent in time are merged into one
:class:`Finding`, and the most severe step in that interval is recorded.

Two merge parameters:

``gap_steps``
    The gap allowed between adjacent hits. Taken from the policy's
    ``runtime_feedback.event_gap_steps`` -- contact flickers frame by frame near the
    threshold, and a gap of one or two steps should not split one event in two.

``min_duration_steps``
    Supplied by the detector along with the hit. Floating has to last 0.2 s to count,
    whereas a forbidden contact is conclusive after a single step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from collections.abc import Iterable, Mapping

from sim_judge.report.finding import Finding, Observation, Severity, Subject, Tier
from sim_judge.world.timeline import Timeline


@dataclass
class _OpenEvent:
    """An event interval still being accumulated."""

    code: str
    severity: Severity
    tier: Tier
    subject: Subject
    rule_id: str | None
    min_duration_steps: int
    merge_gap_steps: int

    first_step: int
    last_step: int
    peak_step: int
    peak_value: float
    peak_metric: str
    peak_metrics: Mapping[str, float]
    thresholds: Mapping[str, float]
    detail: Mapping[str, Any]
    observation_count: int = 1
    step_ids: dict[str, None] = field(default_factory=dict)
    action_ids: dict[str, None] = field(default_factory=dict)

    def absorb(self, observation: Observation) -> None:
        self.last_step = observation.step_index
        self.observation_count += 1
        if observation.severity.rank > self.severity.rank:
            self.severity = observation.severity
        if observation.peak_value > self.peak_value:
            self.peak_step = observation.step_index
            self.peak_value = observation.peak_value
            self.peak_metrics = observation.metrics
            self.thresholds = observation.thresholds
            self.detail = observation.detail


class EventAggregator:
    """Consumes :class:`Observation` as a stream and produces :class:`Finding`."""

    def __init__(self, timeline: Timeline, gap_steps: int, summarizer: Summarizer) -> None:
        self._timeline = timeline
        self._gap = max(gap_steps, 0)
        self._summarize = summarizer
        self._open: dict[tuple[Any, ...], _OpenEvent] = {}
        self._closed: list[Finding] = []

    def add(self, observation: Observation) -> None:
        key = observation.group_key
        event = self._open.get(key)
        if event is not None and observation.step_index - event.last_step <= event.merge_gap_steps + 1:
            event.absorb(observation)
            self._record_location(event, observation.step_index)
            return
        if event is not None:
            self._close(event)
        self._open[key] = self._start(observation)

    def extend(self, observations: Iterable[Observation]) -> None:
        for observation in observations:
            self.add(observation)

    def finish(self) -> list[Finding]:
        for event in self._open.values():
            self._close(event)
        self._open.clear()
        self._closed.sort(key=Finding.sort_key)
        return self._closed

    # -- internals ---------------------------------------------------------

    def _start(self, observation: Observation) -> _OpenEvent:
        event = _OpenEvent(
            code=observation.code,
            severity=observation.severity,
            tier=observation.tier,
            subject=observation.subject,
            rule_id=observation.rule_id,
            min_duration_steps=observation.min_duration_steps,
            merge_gap_steps=max(self._gap, observation.merge_gap_steps or 0),
            first_step=observation.step_index,
            last_step=observation.step_index,
            peak_step=observation.step_index,
            peak_value=observation.peak_value,
            peak_metric=observation.peak_metric,
            peak_metrics=observation.metrics,
            thresholds=observation.thresholds,
            detail=observation.detail,
        )
        self._record_location(event, observation.step_index)
        return event

    def _record_location(self, event: _OpenEvent, step_index: int) -> None:
        location = self._timeline.locate(step_index)
        event.step_ids[location.step_id] = None
        event.action_ids[location.action_id] = None

    def _close(self, event: _OpenEvent) -> None:
        duration = event.last_step - event.first_step + 1
        if duration < event.min_duration_steps:
            return  # too short, treated as numerical jitter

        first = self._timeline.locate(event.first_step)
        last = self._timeline.locate(event.last_step)
        peak = self._timeline.locate(event.peak_step)

        finding = Finding(
            code=event.code,
            severity=event.severity,
            tier=event.tier,
            subject=event.subject,
            first_step=event.first_step,
            last_step=event.last_step,
            time_range_s=(first.time_s, last.time_s),
            step_ids=tuple(event.step_ids),
            action_ids=tuple(event.action_ids),
            peak_step=event.peak_step,
            peak_time_s=peak.time_s,
            peak_step_id=peak.step_id,
            peak_action_id=peak.action_id,
            peak_metric=event.peak_metric,
            metrics=dict(event.peak_metrics),
            thresholds=dict(event.thresholds),
            observation_count=event.observation_count,
            summary="",
            rule_id=event.rule_id,
            detail=dict(event.detail),
        )
        self._closed.append(replace(finding, summary=self._summarize(finding)))


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


class Summarizer:
    """Turns a :class:`Finding` into a single sentence.

    Summary templates are dispatched by diagnostic code; when no template is found it
    falls back to a generic "code + peak metric" sentence, so a new detector that forgot
    to write a template still gets a non-empty line in the report.
    """

    def __call__(self, finding: Finding) -> str:
        builder = _TEMPLATES.get(finding.code, _generic_summary)
        window = _window_phrase(finding)
        try:
            return f"{builder(finding)} ({window})"
        except (KeyError, ValueError):
            return f"{_generic_summary(finding)} ({window})"


def _window_phrase(finding: Finding) -> str:
    action = finding.peak_action_id or "unknown action"
    step = finding.peak_step_id or "unknown step"
    if finding.duration_steps > 1:
        return (
            f"{step} · {action}, steps {finding.first_step}-{finding.last_step}"
            f", peak at step {finding.peak_step} / {finding.peak_time_s:.3f} s"
        )
    return f"{step} · {action}, step {finding.peak_step} / {finding.peak_time_s:.3f} s"


def _mm(value: float) -> str:
    return f"{value * 1000:.3f} mm"


def _pair(finding: Finding) -> str:
    return finding.subject.describe()


def _generic_summary(finding: Finding) -> str:
    value = finding.metrics.get(finding.peak_metric, float("nan"))
    return f"{finding.code}: {_pair(finding)}, {finding.peak_metric}={value:.6g}"


def _p1_declared(f: Finding) -> str:
    return (
        f"{_pair(f)} penetrates {_mm(f.metrics['penetration_m'])}, "
        f"exceeding the {_mm(f.thresholds['maximum_penetration_m'])} the rule allows"
    )


def _p1_generic(f: Finding) -> str:
    ratio = f.metrics.get("penetration_ratio", 0.0)
    thickness = f.thresholds.get("thinner_geom_half_extent_m", 0.0)
    if ratio >= 2.0:
        verdict = "it has passed clean through that geom"
    elif ratio >= 1.0:
        verdict = "it is past the half-thickness of that geom"
    else:
        verdict = "the push-in is on the deep side"
    return (
        f"{_pair(f)} penetrates {_mm(f.metrics['penetration_m'])}, "
        f"{ratio:.2f}x the half-thickness of the thinner side ({_mm(thickness)}), {verdict}; "
        "the policy does not declare this contact pair, so this is a generic criterion warning"
    )


def _p5_solver_blind(f: Finding) -> str:
    """P5 needs to say two things at once: how deep the overlap is, and why nothing in the
    simulation ever pushed back on it."""
    ratio = f.metrics.get("penetration_ratio", 0.0)
    thickness = f.thresholds.get("thicker_geom_half_extent_m", 0.0)
    visual = [
        side
        for side, key in (("first", "visual_only_a"), ("second", "visual_only_b"))
        if f.detail.get(key)
    ]
    if len(visual) == 2:
        cause = "both geoms are visual-only, so the pair can never collide"
    elif visual:
        cause = f"the {visual[0]} geom is visual-only, so the pair can never collide"
    else:
        cause = "the collision masks of the two geoms exclude each other, so the pair can never collide"
    return (
        f"{_pair(f)} share the same volume, {_mm(f.metrics['penetration_m'])} deep, "
        f"{ratio:.2f}x the half-thickness of the thicker side ({_mm(thickness)}); "
        f"{cause}, and the solver reports no contact anywhere between the two bodies, "
        "so no force ever resists it"
    )


def _s6_resting(f: Finding) -> str:
    """S6 is about the asset, not the run, so the phrasing must not sound like an event."""
    ratio = f.metrics.get("penetration_ratio", 0.0)
    return (
        f"{_pair(f)} already overlap by {_mm(f.metrics['penetration_m'])} in the initial "
        f"state, {ratio:.2f}x the half-thickness of the thicker side; the scene ships this "
        "way, so it is measured as a baseline and only growth beyond it is judged as an event"
    )


def _p6_unreported(f: Finding) -> str:
    ratio = f.metrics.get("penetration_ratio", 0.0)
    thickness = f.thresholds.get("thicker_geom_half_extent_m", 0.0)
    return (
        f"{_pair(f)} overlap by {_mm(f.metrics['penetration_m'])}, "
        f"{ratio:.2f}x the half-thickness of the thicker side ({_mm(thickness)}), "
        "yet the solver produced no contact for a pair its collision masks do allow"
    )


def _p2(f: Finding) -> str:
    message = f.detail.get("message") or "a contact the rules forbid occurred"
    return f"{message} Actual contact {_pair(f)}, penetration {_mm(f.metrics['penetration_m'])}"


def _p3_device(f: Finding) -> str:
    return (
        f"{_pair(f)} is a robot <-> device contact outside the allowlist, "
        f"penetration {_mm(f.metrics['penetration_m'])}"
    )


def _p3_environment(f: Finding) -> str:
    return (
        f"{_pair(f)} is a robot <-> environment contact outside the allowlist, "
        f"penetration {_mm(f.metrics['penetration_m'])}"
    )


def _p3_self(f: Finding) -> str:
    return (
        f"{_pair(f)} the robot and the tool it carries press into each other by "
        f"{_mm(f.metrics['penetration_m'])}, "
        f"exceeding the {_mm(f.thresholds['maximum_penetration_m'])} allowed"
    )


def _p4(f: Finding) -> str:
    return (
        f"{_pair(f)} moved {_mm(f.metrics['displacement_m'])} in a single step, "
        f"more than its own smallest extent {_mm(f.thresholds['body_min_extent_m'])}, "
        "so it may have tunnelled through a thin wall between two frames"
    )


def _f1(f: Finding) -> str:
    return (
        f"{_pair(f)} sits at height {f.metrics['height_m']:.4f} m with no contact at all "
        f"and almost no motion (speed {f.metrics['speed_m_s']:.5f} m/s, vertical acceleration "
        f"{f.metrics['vertical_accel_m_s2']:.3f} m/s²), i.e. it is floating unphysically"
    )


def _f2(f: Finding) -> str:
    partners = ", ".join(f.detail.get("contact_partners", [])) or "unknown objects"
    return (
        f"{_pair(f)} only touches {partners}, and following the contact chain downwards "
        "reaches no grounded body, so the support chain is broken"
    )


def _f3(f: Finding) -> str:
    return (
        f"{_pair(f)} weighs {f.metrics['weight_n']:.4f} N but the net vertical contact force "
        f"is only {f.metrics['net_vertical_contact_force_n']:.4f} N, yet it is not falling "
        f"(vertical acceleration {f.metrics['vertical_accel_m_s2']:.3f} m/s²), "
        "so the support is unphysical"
    )


def _f4_gap(f: Finding) -> str:
    return (
        f"{_pair(f)} has a {_mm(f.metrics['gap_m'])} gap to its support surface "
        f"{f.detail.get('support_geom', '?')}, exceeding the {_mm(f.thresholds['maximum_gap_m'])} "
        "allowed, so the base is off its support"
    )


def _f4_penetration(f: Finding) -> str:
    return (
        f"{_pair(f)} sinks {_mm(f.metrics['penetration_m'])} into its support surface "
        f"{f.detail.get('support_geom', '?')}, "
        f"exceeding the {_mm(f.thresholds['maximum_penetration_m'])} allowed"
    )


def _c1(f: Finding) -> str:
    message = f.detail.get("message") or "the safety clearance between the two is too small"
    return (
        f"{message} The closest pair is {_pair(f)}, "
        f"gap {_mm(f.metrics['clearance_m'])}, "
        f"below the required {_mm(f.thresholds['minimum_clearance_m'])}"
    )


def _k1(f: Finding) -> str:
    return (
        f"{_pair(f)} moved {_mm(f.metrics['displacement_m'])} while the current velocity only "
        f"accounts for {_mm(f.metrics['explainable_m'])}, so the configuration jumped"
    )


def _k2(f: Finding) -> str:
    return (
        f"joint {_pair(f)} position {f.metrics['position']:.6f} is outside the permitted range "
        f"[{f.thresholds['lower']:.6f}, {f.thresholds['upper']:.6f}], by {f.metrics['violation']:.6f}"
    )


def _k3_linear(f: Finding) -> str:
    return (
        f"{_pair(f)} reached a linear speed of {f.metrics['speed_m_s']:.3f} m/s, "
        "the solution has gone numerically unstable"
    )


def _k3_angular(f: Finding) -> str:
    return (
        f"{_pair(f)} reached an angular speed of {f.metrics['speed_rad_s']:.3f} rad/s, "
        "the solution has gone numerically unstable"
    )


def _k3_nan(f: Finding) -> str:
    return "the state vector contains non-finite values, the simulation has diverged"


def _k4_normal(f: Finding) -> str:
    return (
        f"{_pair(f)} normal contact force {f.metrics['normal_force_n']:.3f} N, "
        f"above the rule limit of {f.thresholds['maximum_normal_force_n']:.3f} N"
    )


def _k4_tangential(f: Finding) -> str:
    return (
        f"{_pair(f)} tangential contact force {f.metrics['tangential_force_n']:.3f} N, "
        f"above the rule limit of {f.thresholds['maximum_tangential_force_n']:.3f} N"
    )


def _k5(f: Finding) -> str:
    return (
        f"the MuJoCo engine raised warning {f.detail.get('warning', f.subject.a_name)}, "
        "the physics solution cannot be trusted"
    )


def _r1(f: Finding) -> str:
    limit = next(iter(f.thresholds.values()), 0.0)
    return (
        f"at rest, {_pair(f)} sits {_mm(f.metrics['radial_error_m'])} off the socket centre, "
        f"exceeding the {_mm(limit)} limit "
        f"(side clearance is only {_mm(f.metrics['side_clearance_m'])})"
    )


def _r1_transient(f: Finding) -> str:
    return (
        f"while seating after release, {_pair(f)} drifted up to "
        f"{_mm(f.metrics['radial_error_m'])} off the socket centre, exceeding the "
        f"{_mm(f.thresholds['maximum_interface_alignment_error_m'])} interface alignment "
        f"requirement ({f.observation_count} steps hit); this metric is judged at rest, "
        "so drift during the motion is only a hint"
    )


def _r2(f: Finding) -> str:
    return (
        f"at rest, {_pair(f)} has {_mm(f.metrics['side_clearance_m'])} side clearance, "
        f"below the required {_mm(f.thresholds['minimum_side_clearance_m'])}, "
        "so the tube wall is squeezing the socket wall"
    )


def _r2_transient(f: Finding) -> str:
    return (
        f"while seating after release, {_pair(f)} side clearance dropped as low as "
        f"{_mm(f.metrics['side_clearance_m'])}, below the required "
        f"{_mm(f.thresholds['minimum_side_clearance_m'])}, so the tube wall pushed against the "
        f"socket wall ({f.observation_count} steps hit); this metric is judged at rest, "
        "so squeezing during the motion is only a hint"
    )


def _r3_tilt(f: Finding) -> str:
    return (
        f"{_pair(f)} final tilt {f.metrics['tilt_rad']:.4f} rad, "
        f"exceeding the {f.thresholds['maximum_tilt_rad']:.4f} rad allowed"
    )


def _r3_seat(f: Finding) -> str:
    return (
        f"{_pair(f)} final centre height {f.metrics['center_z_m']:.4f} m is outside the seated "
        f"range [{f.thresholds['seated_center_z_min_m']:.4f}, "
        f"{f.thresholds['seated_center_z_max_m']:.4f}], off by {_mm(f.metrics['deviation_m'])}"
    )


def _r3_moving(f: Finding) -> str:
    return (
        f"{_pair(f)} is still moving at {f.metrics['speed_m_s']:.5f} m/s in the final state, "
        f"above the {f.thresholds['maximum_final_speed_m_s']:.5f} m/s allowed, "
        "so it has not settled"
    )


def _r2_loose(f: Finding) -> str:
    return (
        f"{_pair(f)} final side clearance {_mm(f.metrics['side_clearance_m'])}, "
        f"above the {_mm(f.thresholds['maximum_side_clearance_m'])} allowed, so the socket is "
        "looser than the labware and it will still rattle once dropped in"
    )


def _r4_depth(f: Finding) -> str:
    depth = f.metrics["insertion_depth_m"]
    low = f.thresholds["minimum_insertion_depth_m"]
    high = f.thresholds["maximum_insertion_depth_m"]
    how = (
        "too shallow, it may only be resting on the socket mouth"
        if depth < low
        else "too deep, it is already past the socket floor"
    )
    return (
        f"{_pair(f)} final insertion depth {_mm(depth)} is outside the required "
        f"[{_mm(low)}, {_mm(high)}], {how}"
    )


def _r5_floor(f: Finding) -> str:
    return (
        f"{_pair(f)} does not touch the socket floor "
        f"{f.detail.get('socket_floor_geom', '')} in the final state, "
        "so it never really bottoms out"
    )


def _s1_mass(f: Finding) -> str:
    return (
        f"movable body {_pair(f)} has a mass of only {f.metrics['mass_kg']:.3e} kg, "
        "which is no effective mass"
    )


def _s1_inertia(f: Finding) -> str:
    return (
        f"movable body {_pair(f)} has a smallest principal inertia of only "
        f"{f.metrics['minimum_principal_inertia_kg_m2']:.3e} kg·m², "
        "which is no effective inertia"
    )


def _s2_entity(f: Finding) -> str:
    return (
        f"entity {_pair(f)} is declared physical, but none of its "
        f"{int(f.metrics['body_count'])} bodies carries any collision geometry"
    )


def _s2_body(f: Finding) -> str:
    return (
        f"movable body {_pair(f)} has visual geometry only and no collision geometry, "
        "so it will pass straight through other objects"
    )


def _s3(f: Finding) -> str:
    return (
        f"{_pair(f)} visual and collision geometry centres are "
        f"{_mm(f.metrics['center_offset_m'])} apart, "
        f"exceeding the {_mm(f.thresholds['maximum_center_offset_m'])} allowed"
    )


def _s4(f: Finding) -> str:
    return (
        f"in the initial configuration {_pair(f)} already overlaps "
        f"{_mm(f.metrics['penetration_m'])}, which is a placement or modelling defect"
    )


def _s5_gap(f: Finding) -> str:
    return (
        f"tool mount faces {_pair(f)} have a {_mm(f.metrics['gap_m'])} gap, "
        f"exceeding the {_mm(f.thresholds['maximum_gap_m'])} allowed, so they are not seated flush"
    )


def _s5_penetration(f: Finding) -> str:
    return (
        f"tool mount faces {_pair(f)} press into each other by {_mm(f.metrics['penetration_m'])}, "
        f"exceeding the {_mm(f.thresholds['maximum_penetration_m'])} allowed"
    )


_TEMPLATES = {
    "P1_CONTACT_PENETRATION_EXCEEDED": _p1_declared,
    "P1_GEOMETRIC_PENETRATION": _p1_generic,
    "P2_FORBIDDEN_CONTACT": _p2,
    "P3_UNLISTED_ROBOT_DEVICE_CONTACT": _p3_device,
    "P3_UNLISTED_ROBOT_ENVIRONMENT_CONTACT": _p3_environment,
    "P3_UNLISTED_ROBOT_SELF_CONTACT": _p3_self,
    "P4_POSSIBLE_TUNNELING": _p4,
    "P5_SOLVER_BLIND_INTERPENETRATION": _p5_solver_blind,
    "P6_UNREPORTED_INTERPENETRATION": _p6_unreported,
    "S6_RESTING_GEOMETRY_INTERPENETRATION": _s6_resting,
    "F1_UNSUPPORTED_FLOATING_BODY": _f1,
    "F2_SUPPORT_CHAIN_BROKEN": _f2,
    "F3_GRAVITY_INCONSISTENT_SUPPORT": _f3,
    "F4_SUPPORT_GAP_EXCEEDED": _f4_gap,
    "F4_SUPPORT_PENETRATION_EXCEEDED": _f4_penetration,
    "C1_MINIMUM_CLEARANCE_VIOLATED": _c1,
    "K1_POSITION_JUMP": _k1,
    "K2_JOINT_LIMIT_VIOLATION": _k2,
    "K3_LINEAR_SPEED_EXPLOSION": _k3_linear,
    "K3_ANGULAR_SPEED_EXPLOSION": _k3_angular,
    "K3_NON_FINITE_STATE": _k3_nan,
    "K4_CONTACT_NORMAL_FORCE_EXCEEDED": _k4_normal,
    "K4_CONTACT_TANGENTIAL_FORCE_EXCEEDED": _k4_tangential,
    "K5_ENGINE_WARNING": _k5,
    "R1_RECEIVER_ALIGNMENT_ERROR": _r1,
    "R1_RECEIVER_ALIGNMENT_TRANSIENT": _r1_transient,
    "R2_RECEIVER_SIDE_CLEARANCE_LOST": _r2,
    "R2_RECEIVER_SIDE_CLEARANCE_TRANSIENT": _r2_transient,
    "R2_RECEIVER_SOCKET_TOO_LOOSE": _r2_loose,
    "R3_LABWARE_TILTED": _r3_tilt,
    "R3_LABWARE_NOT_SEATED": _r3_seat,
    "R3_LABWARE_STILL_MOVING": _r3_moving,
    "R4_LABWARE_INSERTION_DEPTH_OUT_OF_RANGE": _r4_depth,
    "R5_LABWARE_NOT_TOUCHING_SOCKET_FLOOR": _r5_floor,
    "S1_MOVABLE_BODY_MASSLESS": _s1_mass,
    "S1_MOVABLE_BODY_INERTIALESS": _s1_inertia,
    "S2_ENTITY_WITHOUT_COLLISION_GEOMETRY": _s2_entity,
    "S2_MOVABLE_BODY_VISUAL_ONLY": _s2_body,
    "S3_VISUAL_COLLISION_OFFSET": _s3,
    "S4_INITIAL_PENETRATION": _s4,
    "S5_TOOL_MOUNT_GAP": _s5_gap,
    "S5_TOOL_MOUNT_PENETRATION": _s5_penetration,
}
