"""Parses ``bound-operation.json`` into a structured judging policy.

Design principle: **the judge hard-codes no task thresholds of its own**. Questions like how
deep counts as penetration, how large a gap counts as floating, or which two parts must never
touch are always answered by the policy object parsed here. Only where the policy stays silent
do we fall back to the generic geometric criteria in :mod:`sim_judge.defaults`, and in that case
the conclusion is downgraded to an advisory note.

Geom and body names are matched uniformly with :func:`fnmatch.fnmatchcase`, because the policy
makes heavy use of wildcard spellings such as ``robot:tool:jaw_0__*``. Paired rules are always
matched as unordered pairs.

**Parsing is tolerant of malformed entries.** Different tasks vary a lot in how completely the
fields of their ``bound-operation.json`` are filled in, and one badly written rule should not
block the entire judgement. An entry that is missing a required field is skipped and recorded in
:attr:`Policy.issues`, so the report can honestly state that the rule never took effect instead
of the parser aborting with an exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import partial
from typing import Any, TypeVar
from collections.abc import Callable, Iterable, Mapping, Sequence

ANY_PHASE = "*"

T = TypeVar("T")

# The exceptions that dict / sequence / numeric conversions raise when an entry is malformed.
_MALFORMED = (KeyError, TypeError, ValueError, IndexError)


def _matches(name: str, pattern: str) -> bool:
    return fnmatchcase(name, pattern)


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(name, p) for p in patterns)


def _phase_matches(patterns: Sequence[str], labels: Sequence[str]) -> bool:
    """Whether the phase restriction matches the current frame.

    ``phases`` in the policy may be written either as action names (``release``) or as protocol
    step ids (``step.p03.load``) — both namespaces show up across different tasks. If we honoured
    only one of them, rules written in the other spelling would match nothing at all, and would do
    so **with no warning whatsoever**: no error, and no finding either. So we match against both
    labels together.
    """
    if not patterns:
        return True
    return any(fnmatchcase(label, p) for label in labels if label for p in patterns)


def _pair_matches(a: str, b: str, pattern_a: str, pattern_b: str) -> bool:
    """Unordered pair matching: both (a,b) and (b,a) count as a match."""
    return (_matches(a, pattern_a) and _matches(b, pattern_b)) or (
        _matches(b, pattern_a) and _matches(a, pattern_b)
    )


# --------------------------------------------------------------------------- #
# Contact rules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ContactRule:
    """A contact constraint on one pair of geoms during a particular phase."""

    rule_id: str
    disposition: str
    """One of ``required`` / ``allowed`` / ``forbidden``."""

    geom_a: str
    geom_b: str
    phases: tuple[str, ...]
    maximum_penetration_m: float | None
    maximum_normal_force_n: float | None
    maximum_tangential_force_n: float | None
    message: str
    repair_target: str | None

    def applies_to(self, geom_a: str, geom_b: str, phase: Sequence[str]) -> bool:
        return self.covers_phase(phase) and _pair_matches(geom_a, geom_b, self.geom_a, self.geom_b)

    def covers_phase(self, phase: Sequence[str]) -> bool:
        return _phase_matches(self.phases, phase)

    @property
    def specificity(self) -> int:
        """Fewer wildcards means more specific. Used to pick the closest-fitting rule when one geom pair matches several."""
        return -(self.geom_a.count("*") + self.geom_b.count("*") + sum(p == ANY_PHASE for p in self.phases))


@dataclass(frozen=True, slots=True)
class ClearanceRule:
    """The minimum clearance that must be maintained between two groups of bodies."""

    rule_id: str
    body_a: str
    body_b: str
    minimum_clearance_m: float
    geom_groups: tuple[int, ...]
    excluded_body_pairs: tuple[tuple[str, str], ...]
    phases: tuple[str, ...]
    message: str
    repair_target: str | None

    def covers_bodies(self, body_a: str, body_b: str) -> bool:
        """Whether this body pair falls under this rule, ignoring phase."""
        if not _pair_matches(body_a, body_b, self.body_a, self.body_b):
            return False
        return not any(_pair_matches(body_a, body_b, pa, pb) for pa, pb in self.excluded_body_pairs)

    def covers_phase(self, phase: Sequence[str]) -> bool:
        return _phase_matches(self.phases, phase)

    def applies_to(self, body_a: str, body_b: str, phase: Sequence[str]) -> bool:
        return self.covers_phase(phase) and self.covers_bodies(body_a, body_b)


# --------------------------------------------------------------------------- #
# Scene semantics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SupportedBody:
    """A body declared to rest on a support surface."""

    body: str
    component_id: str
    support_geom: str | None
    support_point_local_m: tuple[float, float, float]
    maximum_gap_m: float
    maximum_penetration_m: float


@dataclass(frozen=True, slots=True)
class PhysicalEntity:
    """Physical completeness requirements for a group of bodies."""

    entity_id: str
    body_patterns: tuple[str, ...]
    require_contact_geometry: bool
    require_mass_inertia_if_movable: bool

    def covers(self, body_name: str) -> bool:
        return _matches_any(body_name, self.body_patterns)


@dataclass(frozen=True, slots=True)
class VisualCollisionPair:
    """How closely a visual geom and its collision geom must coincide."""

    component_id: str
    visual_geom: str
    collision_geom: str
    maximum_center_offset_m: float


@dataclass(frozen=True, slots=True)
class ReceiverSpec:
    """Seating criteria for the receiver socket (the sample well in the centrifuge rotor).

    This is the core quantitative contract that separates "the tube went in" from "the tube is
    wedged at an angle in the mouth of the well".
    """

    interface_id: str
    labware_body: str
    labware_geom: str
    socket_site: str
    socket_floor_geom: str
    socket_wall_geoms: tuple[str, ...]
    inner_radius_m: float
    minimum_side_clearance_m: float
    maximum_side_clearance_m: float
    minimum_insertion_depth_m: float
    maximum_insertion_depth_m: float
    maximum_axis_tilt_rad: float
    maximum_interface_alignment_error_m: float
    require_floor_contact: bool

    labware_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Direction of the labware's own "upright" axis in its body-local frame. The tilt angle is the angle between it and the scene vertical.

    The overwhelming majority of labware is modelled with local +z as that axis, hence the
    default; models with a different axis can override it in the policy.
    """


@dataclass(frozen=True, slots=True)
class LabwareSpec:
    """Final-state criteria for the labware being transported."""

    body: str
    contact_geom: str
    free_joint: str
    maximum_tilt_rad: float
    maximum_final_speed_m_s: float
    receiver_center_xy_m: tuple[float, float]
    maximum_receiver_xy_error_m: float
    seated_center_z_range_m: tuple[float, float]
    minimum_transfer_distance_m: float


@dataclass(frozen=True, slots=True)
class MountGeomPair:
    """A mating surface between the tool and the flange."""

    pair_id: str
    mount_geom: str
    tool_geom: str
    maximum_gap_m: float
    maximum_penetration_m: float


@dataclass(frozen=True, slots=True)
class ToolMountIntegrity:
    """Constraints on the integrity of the tool mounting."""

    mount_body: str
    tool_root_body: str
    tool_geom_prefix: str
    maximum_self_penetration_m: float
    mount_geom_pairs: tuple[MountGeomPair, ...]
    clearance_geom_pairs: tuple[MountGeomPair, ...]
    allowed_self_collision_geom_pairs: tuple[tuple[str, str], ...]


# --------------------------------------------------------------------------- #
# Geom role classification
# --------------------------------------------------------------------------- #


class GeomRole:
    """The role a geom plays when judging unlisted contacts."""

    ROBOT = "robot"
    TOOL = "tool"
    DEVICE = "device"
    ENVIRONMENT = "environment"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class UnlistedContactPolicy:
    """How to dispose of contacts that fall outside the whitelist."""

    forbid_robot_device: bool
    forbid_robot_environment: bool
    forbid_robot_tool: bool
    maximum_unlisted_tool_penetration_m: float
    robot_geom_prefix: str
    tool_body_prefix: str
    extra_tool_body_prefixes: tuple[str, ...]
    device_geom_prefix: str
    environment_geom_patterns: tuple[str, ...]
    allowed_robot_device_body_pairs: tuple[tuple[str, str], ...]

    def classify(self, geom_name: str) -> str:
        """Determine a geom's role from its name. The tool prefix must be tested before the robot prefix."""
        if geom_name.startswith(self.tool_body_prefix) or any(
            geom_name.startswith(p) for p in self.extra_tool_body_prefixes
        ):
            return GeomRole.TOOL
        if geom_name.startswith(self.robot_geom_prefix):
            return GeomRole.ROBOT
        if geom_name.startswith(self.device_geom_prefix):
            return GeomRole.DEVICE
        if _matches_any(geom_name, self.environment_geom_patterns):
            return GeomRole.ENVIRONMENT
        return GeomRole.OTHER

    def is_allowed_body_pair(self, body_a: str, body_b: str) -> bool:
        return any(
            _pair_matches(body_a, body_b, pa, pb)
            for pa, pb in self.allowed_robot_device_body_pairs
        )


# --------------------------------------------------------------------------- #
# Top-level policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Policy:
    """A structured view of ``bound-operation.json``."""

    task_id: str
    schema_version: str
    enabled: bool

    contact_rules: tuple[ContactRule, ...]
    clearance_rules: tuple[ClearanceRule, ...]
    unlisted: UnlistedContactPolicy
    event_gap_steps: int
    clearance_sample_stride_steps: int
    fail_on_engine_warning: bool

    supported_bodies: tuple[SupportedBody, ...]
    physical_entities: tuple[PhysicalEntity, ...]
    visual_collision_pairs: tuple[VisualCollisionPair, ...]
    visual_exemption_patterns: tuple[str, ...]
    minimum_movable_body_mass_kg: float
    minimum_movable_body_inertia_kg_m2: float
    bench_top_geom: str | None

    receiver: ReceiverSpec | None
    labware: LabwareSpec | None
    tool_mount: ToolMountIntegrity | None

    joint_limit_tolerance: float
    release_phase_names: tuple[str, ...]
    retention_phase_names: tuple[str, ...]
    final_phase_name: str | None

    issues: tuple[str, ...] = ()
    """Notes on malformed entries that parsing skipped. The report lists them verbatim, so that a rule which was written but never took effect does not get buried."""

    # -- Query interface ---------------------------------------------------

    def contact_rule_for(self, geom_a: str, geom_b: str, phase: Sequence[str]) -> ContactRule | None:
        """Return the most specific rule governing this geom pair in this phase, or ``None`` if no rule governs it.

        ``forbidden`` takes precedence over ``required``/``allowed``: a prohibition is a hard
        boundary and must not be overridden by a looser wildcard rule.
        """
        candidates = [r for r in self.contact_rules if r.applies_to(geom_a, geom_b, phase)]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda r: (r.disposition == "forbidden", r.specificity),
        )

    def is_visual_exempt(self, geom_name: str) -> bool:
        return _matches_any(geom_name, self.visual_exemption_patterns)

    def unenforced_contracts(self) -> tuple[str, ...]:
        """List the contracts the policy declares but for which this judge has no detector.

        Every line the task author wrote in ``bound-operation.json`` was written deliberately.
        Whatever the judge cannot actually check has to be said out loud — otherwise the author
        assumes that writing a clause means it is being judged, while a PASS in the report in fact
        never covered those clauses. Once a detector is added, just delete the matching entry here.
        """
        pending: list[str] = []
        if self.retention_phase_names:
            pending.append(
                f"retention_phase_names declares {len(self.retention_phase_names)} retention phases, "
                "but the judge has no detector for whether the labware stays gripped throughout the grasp."
            )
        if self.labware and self.labware.minimum_transfer_distance_m > 0.0:
            pending.append(
                f"labware.minimum_transfer_distance_m={self.labware.minimum_transfer_distance_m} m "
                "declares a minimum transfer distance, but the judge has no detector for it."
            )
        if self.tool_mount and self.tool_mount.maximum_self_penetration_m < float("inf"):
            pending.append(
                "tool_mount_integrity.maximum_self_penetration_m declares an upper bound on tool self-penetration, "
                "but the judge only checks that the mounting faces are seated (S5); it performs no tool self-collision check."
            )
        return tuple(pending)

    def entity_for_body(self, body_name: str) -> PhysicalEntity | None:
        matches = [e for e in self.physical_entities if e.covers(body_name)]
        if not matches:
            return None
        # A longer pattern is more specific: "labware" should win over "*".
        return max(matches, key=lambda e: max(len(p) for p in e.body_patterns))


class _Issues:
    """Collects the entries that were skipped during parsing."""

    def __init__(self) -> None:
        self._messages: list[str] = []

    def each(
        self, kind: str, items: Iterable[Any], parse: Callable[[Any], T]
    ) -> tuple[T, ...]:
        """Parse entry by entry, skipping and recording the malformed ones."""
        parsed: list[T] = []
        for index, item in enumerate(items or ()):
            try:
                parsed.append(parse(item))
            except _MALFORMED as error:
                label = _entry_label(item, index)
                self._messages.append(f"{kind} entry {index + 1} ({label}) has missing fields and was skipped: {error}")
        return tuple(parsed)

    def one(self, kind: str, raw: Any, parse: Callable[[Any], T | None]) -> T | None:
        """Parse a single section; if it is malformed, discard the whole section and record it."""
        try:
            return parse(raw)
        except _MALFORMED as error:
            self._messages.append(f"the {kind} section has missing fields, so its criteria will not take effect: {error}")
            return None

    @property
    def messages(self) -> tuple[str, ...]:
        return tuple(self._messages)


def _entry_label(item: Any, index: int) -> str:
    if isinstance(item, Mapping):
        for key in ("rule_id", "component_id", "entity_id", "body", "pair_id"):
            if item.get(key):
                return str(item[key])
    return f"#{index + 1}"


def load_policy(document: Mapping[str, Any]) -> Policy:
    """Build a :class:`Policy` from an already-loaded ``bound-operation.json`` dictionary."""
    feedback = document.get("runtime_feedback") or {}
    semantics = document.get("scene_semantics") or {}
    completeness = semantics.get("physics_completeness") or {}
    workcell = semantics.get("workcell") or {}
    critic = semantics.get("component_attachment_critic") or {}
    device = document.get("device") or {}
    issues = _Issues()

    return Policy(
        task_id=str(document.get("task_id", "")),
        schema_version=str(document.get("schema_version", "")),
        enabled=bool(feedback.get("enabled", True)),
        contact_rules=issues.each("contact_rules", feedback.get("contact_rules"), _parse_contact_rule),
        clearance_rules=issues.each(
            "clearance_rules", feedback.get("clearance_rules"), _parse_clearance_rule
        ),
        unlisted=_parse_unlisted(feedback, device),
        event_gap_steps=int(feedback.get("event_gap_steps", 2)),
        clearance_sample_stride_steps=max(1, int(feedback.get("clearance_sample_stride_steps", 1))),
        fail_on_engine_warning=bool(feedback.get("fail_on_engine_warning", True)),
        supported_bodies=issues.each(
            "supported_bodies", workcell.get("supported_bodies"), _parse_supported
        ),
        physical_entities=issues.each(
            "physical_entities", completeness.get("physical_entities"), _parse_physical_entity
        ),
        visual_collision_pairs=_parse_visual_pairs(critic, issues),
        visual_exemption_patterns=tuple(
            pattern
            for exemption in completeness.get("visual_exemptions", [])
            for pattern in exemption.get("geom_patterns", [])
        ),
        minimum_movable_body_mass_kg=float(completeness.get("minimum_movable_body_mass_kg", 0.0)),
        minimum_movable_body_inertia_kg_m2=float(
            completeness.get("minimum_movable_body_inertia_kg_m2", 0.0)
        ),
        bench_top_geom=workcell.get("bench_top_geom"),
        receiver=issues.one("scene_semantics.receiver", semantics.get("receiver"), _parse_receiver),
        labware=issues.one("labware", _labware_section(document, semantics), _parse_labware),
        tool_mount=issues.one(
            "tool_mount_integrity", document.get("tool_mount_integrity"), _parse_tool_mount
        ),
        joint_limit_tolerance=float(document.get("joint_limit_tolerance", 0.0)),
        release_phase_names=tuple(document.get("release_phase_names", [])),
        retention_phase_names=tuple(document.get("retention_phase_names", [])),
        final_phase_name=document.get("final_phase_name"),
        issues=issues.messages,
    )


def _labware_section(
    document: Mapping[str, Any], semantics: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """The labware declaration drifts between schema versions, so look in both places.

    Honouring only one location would make the entire final-state criteria silently vanish for
    documents written the other way — the hardest class of failure to notice.
    """
    return document.get("labware") or semantics.get("labware")


# --------------------------------------------------------------------------- #
# Per-section parsing
# --------------------------------------------------------------------------- #


def _parse_contact_rule(raw: Mapping[str, Any]) -> ContactRule:
    return ContactRule(
        rule_id=str(raw.get("rule_id", "<anonymous>")),
        disposition=str(raw.get("disposition", "allowed")),
        geom_a=str(raw["geom_a"]),
        geom_b=str(raw["geom_b"]),
        phases=tuple(raw.get("phases") or (ANY_PHASE,)),
        maximum_penetration_m=_optional_float(raw.get("maximum_penetration_m")),
        maximum_normal_force_n=_optional_float(raw.get("maximum_normal_force_n")),
        maximum_tangential_force_n=_optional_float(raw.get("maximum_tangential_force_n")),
        # Prefer an English message when the policy supplies one, otherwise fall back to the author's Chinese text.
        message=str(raw.get("message_en") or raw.get("message_zh", "")),
        repair_target=raw.get("repair_target"),
    )


def _parse_clearance_rule(raw: Mapping[str, Any]) -> ClearanceRule:
    return ClearanceRule(
        rule_id=str(raw.get("rule_id", "<anonymous>")),
        body_a=str(raw["body_a"]),
        body_b=str(raw["body_b"]),
        minimum_clearance_m=float(raw.get("minimum_clearance_m", 0.0)),
        geom_groups=tuple(int(g) for g in raw.get("geom_groups") or ()),
        excluded_body_pairs=tuple(
            (str(p[0]), str(p[1])) for p in raw.get("excluded_body_pairs") or ()
        ),
        phases=tuple(raw.get("phases") or (ANY_PHASE,)),
        # Prefer an English message when the policy supplies one, otherwise fall back to the author's Chinese text.
        message=str(raw.get("message_en") or raw.get("message_zh", "")),
        repair_target=raw.get("repair_target"),
    )


def _parse_unlisted(feedback: Mapping[str, Any], device: Mapping[str, Any]) -> UnlistedContactPolicy:
    return UnlistedContactPolicy(
        forbid_robot_device=bool(feedback.get("forbid_unlisted_robot_device_contacts", False)),
        forbid_robot_environment=bool(
            feedback.get("forbid_unlisted_robot_environment_contacts", False)
        ),
        forbid_robot_tool=bool(feedback.get("forbid_unlisted_robot_tool_contacts", False)),
        maximum_unlisted_tool_penetration_m=float(
            feedback.get("maximum_unlisted_robot_tool_penetration_m", 0.0)
        ),
        robot_geom_prefix=str(feedback.get("robot_geom_prefix", "robot:")),
        tool_body_prefix=str(feedback.get("tool_body_prefix", "robot:tool:")),
        extra_tool_body_prefixes=tuple(feedback.get("tool_body_prefixes") or ()),
        device_geom_prefix=str(feedback.get("device_geom_prefix", "")),
        environment_geom_patterns=tuple(feedback.get("environment_geom_patterns") or ()),
        allowed_robot_device_body_pairs=tuple(
            (str(p[0]), str(p[1])) for p in device.get("allowed_robot_device_body_pairs") or ()
        ),
    )


def _parse_supported(raw: Mapping[str, Any]) -> SupportedBody:
    return SupportedBody(
        body=str(raw["body"]),
        component_id=str(raw.get("component_id", "")),
        support_geom=raw.get("support_geom"),
        support_point_local_m=_vector3(raw.get("support_point_local_m"), (0.0, 0.0, 0.0)),
        maximum_gap_m=float(raw.get("maximum_gap_m", 0.0)),
        maximum_penetration_m=float(raw.get("maximum_penetration_m", 0.0)),
    )


def _parse_physical_entity(raw: Mapping[str, Any]) -> PhysicalEntity:
    return PhysicalEntity(
        entity_id=str(raw.get("entity_id", "")),
        body_patterns=tuple(raw.get("body_patterns") or ()),
        require_contact_geometry=bool(raw.get("require_contact_geometry", False)),
        require_mass_inertia_if_movable=bool(raw.get("require_mass_inertia_if_movable", False)),
    )


def _parse_visual_pairs(critic: Mapping[str, Any], issues: _Issues) -> tuple[VisualCollisionPair, ...]:
    if not critic.get("enabled", False):
        return ()
    pairs: list[VisualCollisionPair] = []
    for component in critic.get("components", []) or ():
        component_id = str(component.get("component_id", ""))
        pairs.extend(
            issues.each(
                f"visual_collision_pairs[{component_id}]",
                component.get("visual_collision_pairs"),
                partial(_parse_visual_pair, component_id),
            )
        )
    return tuple(pairs)


def _parse_visual_pair(component_id: str, raw: Mapping[str, Any]) -> VisualCollisionPair:
    return VisualCollisionPair(
        component_id=component_id,
        visual_geom=str(raw["visual_geom"]),
        collision_geom=str(raw["collision_geom"]),
        maximum_center_offset_m=float(raw.get("maximum_center_offset_m", 0.0)),
    )


def _parse_receiver(raw: Mapping[str, Any] | None) -> ReceiverSpec | None:
    if not raw:
        return None
    return ReceiverSpec(
        interface_id=str(raw.get("interface_id", "")),
        labware_body=str(raw["labware_body"]),
        labware_geom=str(raw["labware_geom"]),
        socket_site=str(raw.get("socket_site", "")),
        socket_floor_geom=str(raw["socket_floor_geom"]),
        socket_wall_geoms=tuple(raw.get("socket_wall_geoms") or ()),
        inner_radius_m=float(raw["inner_radius_m"]),
        minimum_side_clearance_m=float(raw.get("minimum_side_clearance_m", 0.0)),
        maximum_side_clearance_m=float(raw.get("maximum_side_clearance_m", float("inf"))),
        minimum_insertion_depth_m=float(raw.get("minimum_insertion_depth_m", 0.0)),
        maximum_insertion_depth_m=float(raw.get("maximum_insertion_depth_m", float("inf"))),
        maximum_axis_tilt_rad=float(raw.get("maximum_axis_tilt_rad", float("inf"))),
        maximum_interface_alignment_error_m=float(
            raw.get("maximum_interface_alignment_error_m", float("inf"))
        ),
        require_floor_contact=bool(raw.get("require_floor_contact", False)),
        labware_axis_local=_vector3(raw.get("labware_axis_local"), (0.0, 0.0, 1.0)),
    )


def _vector3(
    value: Any, fallback: tuple[float, float, float]
) -> tuple[float, float, float]:
    if value is None:
        return fallback
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_labware(raw: Mapping[str, Any] | None) -> LabwareSpec | None:
    if not raw:
        return None
    center = raw.get("receiver_center_xy_m") or (0.0, 0.0)
    z_range = raw.get("seated_center_z_range_m") or (float("-inf"), float("inf"))
    return LabwareSpec(
        body=str(raw["body"]),
        contact_geom=str(raw["contact_geom"]),
        free_joint=str(raw.get("free_joint", "")),
        maximum_tilt_rad=float(raw.get("maximum_tilt_rad", float("inf"))),
        maximum_final_speed_m_s=float(raw.get("maximum_final_speed_m_s", float("inf"))),
        receiver_center_xy_m=(float(center[0]), float(center[1])),
        maximum_receiver_xy_error_m=float(raw.get("maximum_receiver_xy_error_m", float("inf"))),
        seated_center_z_range_m=(float(z_range[0]), float(z_range[1])),
        minimum_transfer_distance_m=float(raw.get("minimum_transfer_distance_m", 0.0)),
    )


def _parse_tool_mount(raw: Mapping[str, Any] | None) -> ToolMountIntegrity | None:
    if not raw:
        return None

    def pairs(key: str) -> tuple[MountGeomPair, ...]:
        return tuple(
            MountGeomPair(
                pair_id=str(p.get("pair_id", "")),
                mount_geom=str(p.get("mount_geom") or p.get("geom_a", "")),
                tool_geom=str(p.get("tool_geom") or p.get("geom_b", "")),
                maximum_gap_m=float(p.get("maximum_gap_m", float("inf"))),
                maximum_penetration_m=float(p.get("maximum_penetration_m", float("inf"))),
            )
            for p in raw.get(key) or ()
        )

    return ToolMountIntegrity(
        mount_body=str(raw.get("mount_body", "")),
        tool_root_body=str(raw.get("tool_root_body", "")),
        tool_geom_prefix=str(raw.get("tool_geom_prefix", "")),
        maximum_self_penetration_m=float(raw.get("maximum_self_penetration_m", float("inf"))),
        mount_geom_pairs=pairs("mount_geom_pairs"),
        clearance_geom_pairs=pairs("clearance_geom_pairs"),
        allowed_self_collision_geom_pairs=tuple(
            (str(p[0]), str(p[1])) for p in raw.get("allowed_self_collision_geom_pairs") or ()
        ),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
