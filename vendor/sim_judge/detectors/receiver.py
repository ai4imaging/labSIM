"""Class R: whether the labware is actually seated inside the receiver socket.

This class answers the question "did it go in?", and complements the class P penetration
checks: a tube can perfectly well avoid any penetration while sitting crooked at the mouth
of the socket without reaching the bottom, and conversely it can be seated perfectly
straight while pressing into the thin socket floor.

The criteria come from the ``scene_semantics.receiver`` and ``labware`` declarations, and
are split into two tiers by **when they are evaluated**:

**Final-state judgements** (declared-rule violations, which decide the verdict)
    ``maximum_receiver_xy_error_m``, ``seated_center_z_range_m``,
    ``maximum_final_speed_m_s``. These quantities describe what things should look like
    once everything has come to rest, so they are a final-state contract. The instant the
    gripper lets go, the tube is inevitably still wobbling, and applying these thresholds
    across the whole run would fail a perfectly normal seating process.

**Transient warnings** (generic criteria, which do not veto the verdict)
    Between release and settling, the tube really can knock against the socket wall for a
    moment. That is a physical fact worth knowing, but the policy never declared that
    contact is forbidden during the process, so it is reported as a warning only.

``R1`` radial alignment error
``R2`` side clearance
``R3`` final-state tilt / seating height / residual speed
``R4`` insertion depth
``R5`` whether it reached the socket floor
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import mujoco
import numpy as np

from sim_judge.detectors.base import BaseDetector, DetectorContext, body_subject, declared, generic
from sim_judge.replay import Frame, free_joint_of_body
from sim_judge.report.finding import Observation, Subject
from sim_judge.world.geometry import extents, horizontal_offset, quaternion_to_axis


@dataclass(frozen=True, slots=True)
class _Anchor:
    """The model indices and the nominal socket position used by the judgement, resolved
    once at construction time."""

    labware_body: int
    labware_geom: int
    labware_radius_m: float
    """Half-width of the labware in the horizontal plane, derived generically from the geom
    type rather than assuming it is an upright cylinder."""

    labware_half_height_m: float
    """Vertical distance from the labware's geom origin to its bottom face."""

    socket_center: np.ndarray
    """World coordinates of the socket center. Only its components in the horizontal plane
    take part in the alignment judgement."""

    inner_radius_m: float

    socket_site: int
    """The socket's interface reference site, or ``-1`` if the model has none. Insertion
    depth is measured with its height as the zero point."""

    socket_floor_geom: int
    """The socket floor geom, or ``-1`` if undeclared. Used to decide whether the labware
    really did reach the bottom."""


@dataclass(frozen=True, slots=True)
class _Seating:
    """Every measurement of the labware relative to the receiver socket on a given step."""

    step_index: int
    radial_error_m: float
    side_clearance_m: float
    tilt_rad: float
    center_z_m: float
    speed_m_s: float

    insertion_depth_m: float | None
    """How far the labware's bottom face has sunk below the interface reference plane.
    ``None`` means the model has no usable reference site."""

    floor_touched: bool
    """Whether the labware is currently pressing on the socket floor geom."""


class ReceiverSeatingDetector(BaseDetector):
    """R1 / R2 / R3."""

    code_prefix = "R"

    def __init__(self, context: DetectorContext) -> None:
        super().__init__(context)
        self.spec = self.policy.receiver
        self.labware = self.policy.labware
        self._up = self.ctx.up_axis
        self._anchor = self._resolve_anchor()
        self._post_release = self._actions_from_release()
        self._final_action = self.policy.final_phase_name
        self._declared_final: _Seating | None = None
        self._last_seen: _Seating | None = None

    @property
    def enabled(self) -> bool:
        return self._anchor is not None and bool(self._post_release)

    def feed(self, frame: Frame) -> Iterable[Observation]:
        if not self.enabled or frame.action_id not in self._post_release:
            return ()

        seating = self._measure(frame)
        self._last_seen = seating
        if self._final_action in frame.phase_labels:
            self._declared_final = seating
        return self._transient_observations(seating)

    def finalize(self) -> Iterable[Observation]:
        """Final-state judgement. It lives here because it needs the last frame, and which
        frame that is only becomes known once the traversal is over.

        The action declared by the policy as the closing phase is preferred; when the
        policy declares none, or the declared name does not match anything in this
        recording, we fall back to the last frame we saw — that is what "final state"
        means anyway, and a single name mismatch should not silence the entire class of
        checks.
        """
        seating = self._declared_final or self._last_seen
        spec, labware = self.spec, self.labware
        if seating is None or spec is None or self._anchor is None:
            return
        # The labware section is optional: it only supplies a few extra final-state
        # contracts, and its absence does not affect the interface-level criteria.
        xy_limit = labware.maximum_receiver_xy_error_m if labware else float("inf")
        subject = self._subject()
        common = {
            "radial_error_m": seating.radial_error_m,
            "side_clearance_m": seating.side_clearance_m,
            "tilt_rad": seating.tilt_rad,
            "center_z_m": seating.center_z_m,
            "speed_m_s": seating.speed_m_s,
        }

        if seating.radial_error_m > xy_limit:
            yield declared(
                code="R1_RECEIVER_ALIGNMENT_ERROR",
                step=seating.step_index,
                subject=subject,
                metrics={**common, "excess_m": seating.radial_error_m - xy_limit},
                peak_metric="radial_error_m",
                thresholds={"maximum_receiver_xy_error_m": xy_limit},
                rule_id="labware.maximum_receiver_xy_error_m",
                detail={"interface_id": spec.interface_id, "evaluated_at": "final"},
            )
        elif seating.radial_error_m > spec.maximum_interface_alignment_error_m:
            # The interface-level threshold is tighter than the labware-level one: not
            # far enough off to count as misplaced, but already outside what the
            # interface requires for alignment.
            yield declared(
                code="R1_RECEIVER_ALIGNMENT_ERROR",
                step=seating.step_index,
                subject=subject,
                metrics={
                    **common,
                    "excess_m": seating.radial_error_m - spec.maximum_interface_alignment_error_m,
                },
                peak_metric="radial_error_m",
                thresholds={
                    "maximum_interface_alignment_error_m": spec.maximum_interface_alignment_error_m
                },
                rule_id="receiver.maximum_interface_alignment_error_m",
                detail={"interface_id": spec.interface_id, "evaluated_at": "final"},
            )

        if seating.side_clearance_m < spec.minimum_side_clearance_m:
            yield declared(
                code="R2_RECEIVER_SIDE_CLEARANCE_LOST",
                step=seating.step_index,
                subject=subject,
                metrics={
                    **common,
                    "shortfall_m": spec.minimum_side_clearance_m - seating.side_clearance_m,
                },
                peak_metric="shortfall_m",
                thresholds={"minimum_side_clearance_m": spec.minimum_side_clearance_m},
                rule_id="receiver.minimum_side_clearance_m",
                detail={"interface_id": spec.interface_id, "evaluated_at": "final"},
            )

        if seating.side_clearance_m > spec.maximum_side_clearance_m:
            # Excessive clearance means the socket is far too loose for the labware: it
            # goes in, but it will rattle around inside.
            yield declared(
                code="R2_RECEIVER_SOCKET_TOO_LOOSE",
                step=seating.step_index,
                subject=subject,
                metrics={
                    **common,
                    "excess_m": seating.side_clearance_m - spec.maximum_side_clearance_m,
                },
                peak_metric="excess_m",
                thresholds={"maximum_side_clearance_m": spec.maximum_side_clearance_m},
                rule_id="receiver.maximum_side_clearance_m",
                detail={"interface_id": spec.interface_id, "evaluated_at": "final"},
            )

        limit = min(
            spec.maximum_axis_tilt_rad,
            labware.maximum_tilt_rad if labware else float("inf"),
        )
        if seating.tilt_rad > limit:
            yield declared(
                code="R3_LABWARE_TILTED",
                step=seating.step_index,
                subject=subject,
                metrics=common,
                peak_metric="tilt_rad",
                thresholds={"maximum_tilt_rad": limit},
                rule_id="receiver.maximum_axis_tilt_rad",
            )

        low, high = labware.seated_center_z_range_m if labware else (float("-inf"), float("inf"))
        if not (low <= seating.center_z_m <= high):
            yield declared(
                code="R3_LABWARE_NOT_SEATED",
                step=seating.step_index,
                subject=subject,
                metrics={
                    **common,
                    "deviation_m": max(low - seating.center_z_m, seating.center_z_m - high),
                },
                peak_metric="deviation_m",
                thresholds={"seated_center_z_min_m": low, "seated_center_z_max_m": high},
                rule_id="labware.seated_center_z_range_m",
            )

        if labware and seating.speed_m_s > labware.maximum_final_speed_m_s:
            yield declared(
                code="R3_LABWARE_STILL_MOVING",
                step=seating.step_index,
                subject=subject,
                metrics=common,
                peak_metric="speed_m_s",
                thresholds={"maximum_final_speed_m_s": labware.maximum_final_speed_m_s},
                rule_id="labware.maximum_final_speed_m_s",
            )

        # Insertion depth and floor contact are the direct evidence of whether it actually
        # went in: the labware may well be neither tilted nor wobbling and still be merely
        # perched on the mouth of the socket. The height-range criterion looks at absolute
        # height and breaks as soon as the socket depth changes; these two do not.
        depth, low, high = seating.insertion_depth_m, spec.minimum_insertion_depth_m, spec.maximum_insertion_depth_m
        if depth is not None and not (low <= depth <= high):
            yield declared(
                code="R4_LABWARE_INSERTION_DEPTH_OUT_OF_RANGE",
                step=seating.step_index,
                subject=subject,
                metrics={
                    **common,
                    "insertion_depth_m": depth,
                    "deviation_m": max(low - depth, depth - high),
                },
                peak_metric="deviation_m",
                thresholds={"minimum_insertion_depth_m": low, "maximum_insertion_depth_m": high},
                rule_id="receiver.insertion_depth",
                detail={"interface_id": spec.interface_id, "evaluated_at": "final"},
            )

        if spec.require_floor_contact and not seating.floor_touched:
            yield declared(
                code="R5_LABWARE_NOT_TOUCHING_SOCKET_FLOOR",
                step=seating.step_index,
                subject=subject,
                metrics={**common, "insertion_depth_m": depth or 0.0},
                peak_metric="radial_error_m",
                thresholds={},
                rule_id="receiver.require_floor_contact",
                detail={
                    "interface_id": spec.interface_id,
                    "socket_floor_geom": spec.socket_floor_geom,
                    "evaluated_at": "final",
                },
            )

    # -- transient -----------------------------------------------------------

    def _transient_observations(self, seating: _Seating) -> list[Observation]:
        spec = self.spec
        assert spec is not None
        subject = self._subject()
        metrics = {
            "radial_error_m": seating.radial_error_m,
            "side_clearance_m": seating.side_clearance_m,
        }
        # While seating, the labware bounces back and forth inside the socket and these
        # quantities cross the threshold repeatedly. A large merge gap collapses the whole
        # seating process into one span instead of emitting a dozen findings each a few
        # milliseconds long.
        gap = self.ctx.steps_for(self.limits.oscillation_merge_gap_s)
        results: list[Observation] = []

        if seating.radial_error_m > spec.maximum_interface_alignment_error_m:
            results.append(
                generic(
                    code="R1_RECEIVER_ALIGNMENT_TRANSIENT",
                    step=seating.step_index,
                    subject=subject,
                    metrics=metrics,
                    peak_metric="radial_error_m",
                    thresholds={
                        "maximum_interface_alignment_error_m": spec.maximum_interface_alignment_error_m
                    },
                    merge_gap_steps=gap,
                    detail={"interface_id": spec.interface_id},
                )
            )
        if seating.side_clearance_m < spec.minimum_side_clearance_m:
            results.append(
                generic(
                    code="R2_RECEIVER_SIDE_CLEARANCE_TRANSIENT",
                    step=seating.step_index,
                    subject=subject,
                    metrics={
                        **metrics,
                        "shortfall_m": spec.minimum_side_clearance_m - seating.side_clearance_m,
                    },
                    peak_metric="shortfall_m",
                    thresholds={"minimum_side_clearance_m": spec.minimum_side_clearance_m},
                    merge_gap_steps=gap,
                    detail={"interface_id": spec.interface_id},
                )
            )
        return results

    # -- measurement ---------------------------------------------------------

    def _measure(self, frame: Frame) -> _Seating:
        anchor, up = self._anchor, self._up
        assert anchor is not None and up is not None
        body = anchor.labware_body

        center = np.asarray(frame.xpos[body], dtype=np.float64)
        # The alignment error is the offset within the horizontal plane, where horizontal
        # is defined by the gravity direction rather than by taking the x and y components.
        radial_error = horizontal_offset(up, center - anchor.socket_center)
        # Side clearance = socket inner radius - alignment error - labware half-width. A
        # negative value means the labware wall is already pressing on the socket wall.
        side_clearance = anchor.inner_radius_m - radial_error - anchor.labware_radius_m

        # Insertion depth is measured from the interface site as the zero point, positive
        # downwards: how far the labware's bottom face is below the reference plane.
        bottom = up.height(frame.geom_xpos[anchor.labware_geom]) - anchor.labware_half_height_m
        insertion_depth = (
            None
            if anchor.socket_site < 0
            else up.height(frame.site_xpos[anchor.socket_site]) - bottom
        )

        addresses = free_joint_of_body(self.model, body)
        speed = 0.0
        if addresses is not None:
            _, dof = addresses
            speed = float(np.linalg.norm(frame.qvel[dof : dof + 3]))

        axis = quaternion_to_axis(frame.xquat[body], self.spec.labware_axis_local)
        cosine = abs(float(axis @ up.direction))
        return _Seating(
            step_index=frame.step_index,
            radial_error_m=radial_error,
            side_clearance_m=side_clearance,
            tilt_rad=float(np.arccos(np.clip(cosine, -1.0, 1.0))),
            center_z_m=up.height(center),
            speed_m_s=speed,
            insertion_depth_m=insertion_depth,
            floor_touched=self._touches_floor(frame, anchor),
        )

    @staticmethod
    def _touches_floor(frame: Frame, anchor: _Anchor) -> bool:
        if anchor.socket_floor_geom < 0:
            return False
        pair = {anchor.labware_geom, anchor.socket_floor_geom}
        return any(pair == {c.geom_a.id, c.geom_b.id} for c in frame.contacts)

    # -- construction --------------------------------------------------------

    def _subject(self) -> Subject:
        anchor = self._anchor
        assert anchor is not None and self.spec is not None
        labware = body_subject(self.resolver, anchor.labware_body)
        return Subject(
            kind="body",
            a_name=labware.a_name,
            a_entity=labware.a_entity,
            a_body=labware.a_body,
            b_name=self.spec.interface_id,
        )

    def _resolve_anchor(self) -> _Anchor | None:
        spec, up = self.spec, self._up
        if spec is None or up is None:
            return None
        body_id = self.resolver.body_id(spec.labware_body)
        geom = self.resolver.geom_by_name(spec.labware_geom)
        if body_id is None or geom is None:
            return None
        site_id = self._site_id(spec.socket_site)
        center = self._socket_center(site_id)
        if center is None:
            return None

        # The labware's half-width and half-height are derived generically from the geom
        # type: simply taking size[0] as the radius and size[1] as the half-height only
        # holds for upright cylinders and capsules, and would be wrong for box, sphere or
        # mesh labware.
        shape = extents(self.model, self.ctx.replayer.data, geom.id, up)
        floor = self.resolver.geom_by_name(spec.socket_floor_geom) if spec.socket_floor_geom else None
        return _Anchor(
            labware_body=body_id,
            labware_geom=geom.id,
            labware_radius_m=shape.horizontal_m,
            labware_half_height_m=shape.vertical_m,
            socket_center=center,
            inner_radius_m=spec.inner_radius_m,
            socket_site=site_id,
            socket_floor_geom=floor.id if floor else -1,
        )

    def _site_id(self, name: str | None) -> int:
        if not name:
            return -1
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name))

    def _socket_center(self, site_id: int) -> np.ndarray | None:
        """World coordinates of the receiver socket center.

        The interface site is the model's own definition of where the socket is, and it
        holds under any gravity direction, so it is preferred. With no site we fall back to
        ``labware.receiver_center_xy_m`` — that field is declared in world XY, and is
        therefore only meaningful in scenes where the vertical direction happens to be
        world +Z.

        The site must be read from ``data.site_xpos`` (world frame) and not
        ``model.site_pos`` (the local frame of its parent body): the latter only happens to
        agree when the site hangs directly off worldbody, and if it is attached to a child
        body such as a rotor it will not line up with the labware's world coordinates,
        yielding a plausible-looking but completely wrong alignment error.
        """
        if site_id >= 0:
            return np.asarray(self.ctx.replayer.data.site_xpos[site_id], dtype=np.float64).copy()
        if self.labware is not None:
            x, y = self.labware.receiver_center_xy_m
            return np.array([x, y, 0.0])
        return None

    def _actions_from_release(self) -> frozenset[str]:
        """The release action itself plus every action that follows it.

        Derived from the action order on the timeline rather than hard-coding phase names,
        so that this keeps working on a different task. ``release_phase_names`` may hold
        either action names or protocol step ids, and both are accepted.
        """
        release = set(self.policy.release_phase_names)
        if not release:
            return frozenset()
        spans = self.ctx.timeline.spans
        first = next(
            (i for i, s in enumerate(spans) if s.action_id in release or s.step_id in release),
            None,
        )
        return frozenset() if first is None else frozenset(s.action_id for s in spans[first:])
