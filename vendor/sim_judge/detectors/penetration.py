"""Class P: penetration.

The judgement has two tiers, following the same layering principle as
:mod:`sim_judge.defaults`:

``P1`` penetration over the limit
    First check whether ``bound-operation.json`` declares a ``maximum_penetration_m`` for
    this geom pair; if it does, judge by the declaration, and any violation is a FAILURE.
    With no declaration we fall back to a generic geometric criterion: non-dimensionalize
    the penetration depth by dividing it by the smallest half-extent of the thinner of the
    two geoms, and warn when the ratio exceeds the threshold. A relative quantity rather
    than an absolute one, because 0.5 mm of intrusion is irrelevant on a 20 mm thick
    chassis but means a full breach of a 2 mm thin socket floor.

``P2`` forbidden contact
    For pairs matching a rule with ``disposition: forbidden``, the mere existence of
    contact is a FAILURE, regardless of depth.

``P3`` unlisted contact
    When the policy enables the ``forbid_unlisted_*`` switches, any robot-to-device,
    robot-to-environment or robot-to-self contact outside the allowlist.

``P4`` tunneling
    A free body whose single-step displacement exceeds its own smallest extent, meaning it
    could have passed through a thin wall between two frames without any contact being
    detected.

Penetration on step 0 is handled by S4 in :mod:`sim_judge.detectors.static_scene` —
overlapping already in the initial configuration is an asset layout defect, which is a
different thing from being pressed in during the run, and the two should not be mixed
into the same finding.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sim_judge.detectors.base import (
    BaseDetector,
    DetectorContext,
    body_subject,
    declared,
    generic,
    geom_pair_subject,
)
from sim_judge.loader.policy import GeomRole
from sim_judge.replay import ContactSample, Frame, free_body_ids
from sim_judge.report.finding import Observation


class PenetrationDetector(BaseDetector):
    """P1 / P2 / P3: judge penetration and illegal contact one contact point at a time."""

    code_prefix = "P"

    def feed(self, frame: Frame) -> Iterable[Observation]:
        if frame.step_index == 0:
            return ()  # the initial configuration is S4's responsibility
        results: list[Observation] = []
        for contact in frame.contacts:
            results.extend(self._judge_contact(frame, contact))
        return results

    def _judge_contact(self, frame: Frame, contact: ContactSample) -> Iterable[Observation]:
        geom_a, geom_b = contact.geom_a, contact.geom_b
        rule = self.policy.contact_rule_for(geom_a.name, geom_b.name, frame.phase_labels)

        if rule is not None and rule.disposition == "forbidden":
            yield declared(
                code="P2_FORBIDDEN_CONTACT",
                step=frame.step_index,
                subject=geom_pair_subject(geom_a, geom_b),
                metrics={
                    "penetration_m": contact.depth_m,
                    "normal_force_n": contact.normal_force_n,
                },
                peak_metric="penetration_m",
                thresholds={},
                rule_id=rule.rule_id,
                detail={"message": rule.message, "repair_target": rule.repair_target},
            )
            return

        if rule is not None and rule.maximum_penetration_m is not None:
            limit = rule.maximum_penetration_m
            if contact.depth_m > limit:
                yield declared(
                    code="P1_CONTACT_PENETRATION_EXCEEDED",
                    step=frame.step_index,
                    subject=geom_pair_subject(geom_a, geom_b),
                    metrics={
                        "penetration_m": contact.depth_m,
                        "normal_force_n": contact.normal_force_n,
                        "excess_m": contact.depth_m - limit,
                    },
                    peak_metric="penetration_m",
                    thresholds={"maximum_penetration_m": limit},
                    rule_id=rule.rule_id,
                    detail={"message": rule.message, "repair_target": rule.repair_target},
                )
            return

        if rule is None:
            unlisted = self._judge_unlisted(frame, contact)
            if unlisted is not None:
                yield unlisted
                return

        yield from self._judge_generic_depth(frame, contact)

    # -- P3 ----------------------------------------------------------------

    def _judge_unlisted(self, frame: Frame, contact: ContactSample) -> Observation | None:
        """Decide whether this contact is one of the unlisted contacts that the policy
        explicitly forbids."""
        policy = self.policy.unlisted
        role_a = policy.classify(contact.geom_a.name)
        role_b = policy.classify(contact.geom_b.name)
        roles = {role_a, role_b}
        robotic = {GeomRole.ROBOT, GeomRole.TOOL}

        if not (roles & robotic):
            return None  # contacts not involving the robot are outside these switches

        subject = geom_pair_subject(contact.geom_a, contact.geom_b)
        base_metrics = {
            "penetration_m": contact.depth_m,
            "normal_force_n": contact.normal_force_n,
        }

        if (
            GeomRole.DEVICE in roles
            and policy.forbid_robot_device
            and not policy.is_allowed_body_pair(contact.geom_a.body_name, contact.geom_b.body_name)
        ):
            return declared(
                code="P3_UNLISTED_ROBOT_DEVICE_CONTACT",
                step=frame.step_index,
                subject=subject,
                metrics=base_metrics,
                peak_metric="penetration_m",
                thresholds={},
                rule_id="forbid_unlisted_robot_device_contacts",
                detail={"role_a": role_a, "role_b": role_b},
            )

        if GeomRole.ENVIRONMENT in roles and policy.forbid_robot_environment:
            return declared(
                code="P3_UNLISTED_ROBOT_ENVIRONMENT_CONTACT",
                step=frame.step_index,
                subject=subject,
                metrics=base_metrics,
                peak_metric="penetration_m",
                thresholds={},
                rule_id="forbid_unlisted_robot_environment_contacts",
                detail={"role_a": role_a, "role_b": role_b},
            )

        # Contact between the robot body and the tool mounted on it. Adjacent links always
        # overlap a little, so this class allows the tolerance the policy declares as
        # "maximum_unlisted_robot_tool_penetration_m" (the attribute below drops the
        # "robot_" that the JSON field carries).
        if roles <= robotic and policy.forbid_robot_tool:
            limit = policy.maximum_unlisted_tool_penetration_m
            if contact.depth_m > limit:
                return declared(
                    code="P3_UNLISTED_ROBOT_SELF_CONTACT",
                    step=frame.step_index,
                    subject=subject,
                    metrics={**base_metrics, "excess_m": contact.depth_m - limit},
                    peak_metric="penetration_m",
                    thresholds={"maximum_penetration_m": limit},
                    rule_id="forbid_unlisted_robot_tool_contacts",
                    detail={"role_a": role_a, "role_b": role_b},
                )
        return None

    # -- P1 generic tier -----------------------------------------------------

    def _judge_generic_depth(self, frame: Frame, contact: ContactSample) -> Iterable[Observation]:
        scale = min(contact.geom_a.min_half_extent_m, contact.geom_b.min_half_extent_m)
        if not np.isfinite(scale) or scale <= 0.0:
            return  # semi-infinite bodies (ground planes, height fields) have no thickness

        limit = max(self.limits.penetration_absolute_floor_m, self.limits.penetration_ratio_warning * scale)
        if contact.depth_m <= limit:
            return

        ratio = contact.depth_m / scale
        thinner = contact.geom_a if contact.geom_a.min_half_extent_m <= contact.geom_b.min_half_extent_m else contact.geom_b
        yield generic(
            code="P1_GEOMETRIC_PENETRATION",
            step=frame.step_index,
            subject=geom_pair_subject(contact.geom_a, contact.geom_b),
            metrics={
                "penetration_m": contact.depth_m,
                "penetration_ratio": ratio,
                "normal_force_n": contact.normal_force_n,
            },
            peak_metric="penetration_m",
            thresholds={
                "penetration_m": limit,
                "thinner_geom_half_extent_m": scale,
                "ratio_warning": self.limits.penetration_ratio_warning,
                "ratio_severe": self.limits.penetration_ratio_severe,
            },
            detail={
                "thinner_geom": thinner.name,
                # 1x the half-thickness = past the mid-plane, 2x = all the way out the
                # other side.
                "past_half_thickness": ratio >= self.limits.penetration_ratio_severe,
                "passed_through": ratio >= 2.0 * self.limits.penetration_ratio_severe,
            },
        )


class TunnelingDetector(BaseDetector):
    """P4: a free body's single-step displacement is large enough that it may have passed
    through a thin wall between two frames.

    Contact detection only happens at discrete instants; the motion between two frames is
    invisible. If the distance covered in one step exceeds the body's own smallest extent,
    it could have crossed an entire thin wall without leaving any contact record behind.
    """

    code_prefix = "P"

    def __init__(self, context: DetectorContext) -> None:
        super().__init__(context)
        self._bodies = [b for b in free_body_ids(self.model) if float(self.model.body_mass[b]) > 0.0]
        self._extent = {b: self._body_extent(b) for b in self._bodies}
        self._previous: dict[int, np.ndarray] = {}
        self._previous_step: int | None = None

    def feed(self, frame: Frame) -> Iterable[Observation]:
        results: list[Observation] = []
        gap = 1 if self._previous_step is None else frame.step_index - self._previous_step

        for body_id in self._bodies:
            position = frame.xpos[body_id].copy()
            previous = self._previous.get(body_id)
            self._previous[body_id] = position
            if previous is None:
                continue

            travelled = float(np.linalg.norm(position - previous))
            extent = self._extent[body_id]
            # Scale the criterion by the number of steps actually spanned, so that
            # sampling with --stride does not produce false positives.
            limit = self.limits.tunneling_displacement_ratio * extent * max(gap, 1)
            if extent <= 0.0 or travelled <= limit:
                continue

            results.append(
                generic(
                    code="P4_POSSIBLE_TUNNELING",
                    step=frame.step_index,
                    subject=body_subject(self.resolver, body_id),
                    metrics={"displacement_m": travelled, "step_gap": float(gap)},
                    peak_metric="displacement_m",
                    thresholds={"displacement_m": limit, "body_min_extent_m": extent},
                )
            )

        self._previous_step = frame.step_index
        return results

    def reset_continuity(self) -> None:
        self._previous.clear()
        self._previous_step = None

    def _body_extent(self, body_id: int) -> float:
        extents = [
            g.min_half_extent_m
            for g in self.resolver.geoms_of_body(body_id)
            if g.collidable and np.isfinite(g.min_half_extent_m)
        ]
        return min(extents) if extents else 0.0


class ContactForceDetector(BaseDetector):
    """K4: contact force exceeds the upper limit declared by a rule.

    Only the task author can say what the force limit should be — the same 30 N is a
    normal press on a button but crushes a thin wall. So there is no generic fallback
    here; only explicit declarations are judged.
    """

    code_prefix = "K"

    def feed(self, frame: Frame) -> Iterable[Observation]:
        for contact in frame.contacts:
            rule = self.policy.contact_rule_for(contact.geom_a.name, contact.geom_b.name, frame.phase_labels)
            if rule is None:
                continue
            subject = geom_pair_subject(contact.geom_a, contact.geom_b)

            if rule.maximum_normal_force_n is not None and contact.normal_force_n > rule.maximum_normal_force_n:
                yield declared(
                    code="K4_CONTACT_NORMAL_FORCE_EXCEEDED",
                    step=frame.step_index,
                    subject=subject,
                    metrics={
                        "normal_force_n": contact.normal_force_n,
                        "penetration_m": contact.depth_m,
                    },
                    peak_metric="normal_force_n",
                    thresholds={"maximum_normal_force_n": rule.maximum_normal_force_n},
                    rule_id=rule.rule_id,
                    detail={"message": rule.message},
                )

            if (
                rule.maximum_tangential_force_n is not None
                and contact.tangential_force_n > rule.maximum_tangential_force_n
            ):
                yield declared(
                    code="K4_CONTACT_TANGENTIAL_FORCE_EXCEEDED",
                    step=frame.step_index,
                    subject=subject,
                    metrics={"tangential_force_n": contact.tangential_force_n},
                    peak_metric="tangential_force_n",
                    thresholds={"maximum_tangential_force_n": rule.maximum_tangential_force_n},
                    rule_id=rule.rule_id,
                    detail={"message": rule.message},
                )
