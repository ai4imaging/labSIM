"""Class F: floating and support anomalies.

Floating has to be told apart from two perfectly normal situations:

* **Free fall** — having no contact is correct, but the object should be accelerating
  downwards at g.
* **Held up by a joint** — a robot arm link being clear of every contact surface is the
  normal case; it is supported by its joints rather than by contact.

So floating is only judged for bodies that are **driven by a free joint and have mass**,
and the criterion is being at rest rather than being contact-free: an object that touches
nothing and yet is not accelerating downwards must be held up by some non-physical force.

``F1`` unsupported free body: zero contacts and nearly at rest for several steps in a row.
``F2`` broken support chain: contacts exist, but following the contact graph downwards
never reaches a grounded body.
``F3`` gravity consistency: almost no vertical acceleration, yet the net contact force
falls far short of balancing gravity.

The vertical direction always comes from :class:`~sim_judge.world.geometry.UpAxis`, which
is defined as the opposite of the model's declared gravity rather than assuming world +Z
is up. If the model declares no gravity (a zero-g scene, say), floating loses its meaning
altogether and this entire class of criteria is switched off.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

from sim_judge.detectors.base import (
    BaseDetector,
    DetectorContext,
    body_subject,
    declared,
    generic,
)
from sim_judge.loader.policy import SupportedBody
from sim_judge.replay import Frame, free_body_ids, free_joint_of_body, grounded_body_ids
from sim_judge.report.finding import Observation
from sim_judge.world.geometry import body_point_world, support_height


class FloatingDetector(BaseDetector):
    """F1 / F2 / F3."""

    code_prefix = "F"

    def __init__(self, context: DetectorContext) -> None:
        super().__init__(context)
        self._grounded = grounded_body_ids(self.model)
        self._up = self.ctx.up_axis
        self._bodies = [
            body_id
            for body_id in free_body_ids(self.model)
            if float(self.model.body_mass[body_id]) > 0.0
        ]
        self._dof = {b: free_joint_of_body(self.model, b) for b in self._bodies}
        self._min_steps = self.ctx.steps_for(self.limits.floating_min_duration_s)

    @property
    def enabled(self) -> bool:
        return bool(self._bodies) and self._up is not None

    def feed(self, frame: Frame) -> Iterable[Observation]:
        if not self.enabled:
            return ()

        supported = self._bodies_reaching_ground(frame)
        results: list[Observation] = []
        for body_id in self._bodies:
            results.extend(self._judge_body(frame, body_id, supported))
        return results

    # -- a single free body --------------------------------------------------

    def _judge_body(self, frame: Frame, body_id: int, supported: set[int]) -> Iterable[Observation]:
        addresses = self._dof[body_id]
        up = self._up
        if addresses is None or up is None:
            return
        _, dof_start = addresses

        velocity = frame.qvel[dof_start : dof_start + 3]
        acceleration = frame.qacc[dof_start : dof_start + 3]
        speed = float(np.linalg.norm(velocity))
        # The vertical component is positive upwards, so in free fall it is about -g.
        vertical_accel = up.component(acceleration)
        mass = float(self.model.body_mass[body_id])
        weight = mass * up.gravity_m_s2

        contacts = frame.contacts_of_body(body_id)
        subject = body_subject(self.resolver, body_id)

        # Criterion for free fall: vertical acceleration close to gravitational
        # acceleration.
        falling = vertical_accel <= -up.gravity_m_s2 * (1.0 - self.limits.gravity_consistency_accel_ratio)

        if not contacts:
            if speed < self.limits.floating_speed_epsilon_m_s and not falling:
                yield generic(
                    code="F1_UNSUPPORTED_FLOATING_BODY",
                    step=frame.step_index,
                    subject=subject,
                    metrics={
                        "speed_m_s": speed,
                        "vertical_accel_m_s2": vertical_accel,
                        "height_m": up.height(frame.xpos[body_id]),
                    },
                    peak_metric="speed_m_s",
                    thresholds={
                        "speed_m_s": self.limits.floating_speed_epsilon_m_s,
                        "gravity_m_s2": up.gravity_m_s2,
                    },
                    min_duration_steps=self._min_steps,
                )
            return

        if body_id not in supported:
            partners = sorted({self.resolver.body_name(c.other_body(body_id)) for c in contacts})
            yield generic(
                code="F2_SUPPORT_CHAIN_BROKEN",
                step=frame.step_index,
                subject=subject,
                metrics={"contact_count": float(len(contacts)), "speed_m_s": speed},
                peak_metric="contact_count",
                thresholds={},
                min_duration_steps=self._min_steps,
                detail={"contact_partners": partners},
            )
            return

        # F3: the body has contacts and is grounded, yet those contacts do not carry its
        # own weight while it also refuses to fall — something non-physical is holding it up.
        net_vertical = sum(up.component(c.force_on_body(body_id)) for c in contacts)
        if not falling and net_vertical < weight * (1.0 - self.limits.gravity_consistency_force_tolerance):
            yield generic(
                code="F3_GRAVITY_INCONSISTENT_SUPPORT",
                step=frame.step_index,
                subject=subject,
                metrics={
                    "net_vertical_contact_force_n": net_vertical,
                    "weight_n": weight,
                    "vertical_accel_m_s2": vertical_accel,
                },
                peak_metric="weight_n",
                thresholds={
                    "minimum_support_force_n": weight * (1.0 - self.limits.gravity_consistency_force_tolerance)
                },
                min_duration_steps=self._min_steps,
            )

    # -- contact graph -------------------------------------------------------

    def _bodies_reaching_ground(self, frame: Frame) -> set[int]:
        """Flood outwards from the grounded bodies along contact relations to obtain the
        set of bodies that are supported on this step.

        Grounded bodies are the ones kinematically welded to the world — their position is
        determined by joints rather than by contact. A free body only counts as supported
        if it rests on them, directly or indirectly.
        """
        neighbours: dict[int, list[int]] = {}
        for contact in frame.contacts:
            first, second = contact.body_pair
            if first == second:
                continue
            neighbours.setdefault(first, []).append(second)
            neighbours.setdefault(second, []).append(first)

        reached = set(self._grounded)
        frontier = [b for b in self._grounded if b in neighbours]
        while frontier:
            current = frontier.pop()
            for neighbour in neighbours.get(current, ()):
                if neighbour not in reached:
                    reached.add(neighbour)
                    frontier.append(neighbour)
        return reached


class SupportedBodyDetector(BaseDetector):
    """F4: a base is either lifted off its support surface or sunk into it.

    ``scene_semantics.workcell.supported_bodies`` declares which bodies must sit on a
    support surface, along with the permitted gap and penetration. The criterion is the
    **vertical distance from the support point to the top of the support surface**:
    positive means lifted off, negative means sunk in.

    We deliberately avoid the generic convex distance between geoms here. An equipment
    base is often a single solid box tens of centimeters tall, and when it touches the
    bench the convex distance algorithm reports the penetration depth along the shallowest
    separation direction, producing an absurd figure of several hundred millimeters. The
    policy supplies ``support_point_local_m`` precisely to say which point should be
    measured, and measuring along it is what was actually meant.

    These quantities barely change over time (the bodies involved are all welded to the
    world), so sampling at a fairly large stride is enough.
    """

    code_prefix = "F"

    def __init__(self, context: DetectorContext) -> None:
        super().__init__(context)
        self._stride = self.ctx.steps_for(self.limits.support_sample_interval_s)
        self._up = self.ctx.up_axis
        self._targets = self._resolve_targets()
        self._next_sample = 0

    @property
    def enabled(self) -> bool:
        return bool(self._targets) and self._up is not None

    def feed(self, frame: Frame) -> Iterable[Observation]:
        # Decide based on how far we are from the last sample rather than by taking the
        # step index modulo the stride: the replay itself may be running with --stride,
        # and when the two strides are coprime the modulo would almost never sample.
        up = self._up
        if up is None or frame.step_index < self._next_sample:
            return ()
        self._next_sample = frame.step_index + self._stride

        model, data = self.model, self.ctx.replayer.data
        for target in self._targets:
            spec = target.spec
            support_point = body_point_world(model, data, target.body_id, spec.support_point_local_m)
            surface_z = support_height(model, data, target.support_geom_id, up)
            gap = up.height(support_point) - surface_z
            subject = body_subject(self.resolver, target.body_id)

            if gap > spec.maximum_gap_m:
                code, metric, value, limit = "F4_SUPPORT_GAP_EXCEEDED", "gap_m", gap, spec.maximum_gap_m
            elif -gap > spec.maximum_penetration_m:
                code, metric, value, limit = (
                    "F4_SUPPORT_PENETRATION_EXCEEDED",
                    "penetration_m",
                    -gap,
                    spec.maximum_penetration_m,
                )
            else:
                continue

            yield declared(
                code=code,
                step=frame.step_index,
                subject=subject,
                metrics={metric: value, "excess_m": value - limit},
                peak_metric=metric,
                thresholds={f"maximum_{metric}": limit},
                rule_id=f"supported_body:{spec.component_id}",
                # The sampling stride is far larger than the event merge gap; without
                # telling the aggregator, one continuous state would be split into many
                # single-step events.
                merge_gap_steps=self._stride,
                detail={"support_geom": target.support_geom_name},
            )

    def _resolve_targets(self) -> list[_SupportTarget]:
        """Resolve the policy declarations into targets that can be queried directly.

        ``support_geom`` may be empty, in which case we fall back to the bench top — the
        policy treats "sitting on the bench" as the default convention. Bodies or geoms
        that the declaration references but that cannot be resolved are simply skipped:
        that is a mismatch between policy and scene, which the static class S checks are
        responsible for reporting, and it should not turn into noise here.
        """
        fallback = self.policy.bench_top_geom
        targets: list[_SupportTarget] = []
        for spec in self.policy.supported_bodies:
            body_id = self.resolver.body_id(spec.body)
            support_name = spec.support_geom or fallback
            support = self.resolver.geom_by_name(support_name) if support_name else None
            if body_id is not None and support is not None:
                targets.append(_SupportTarget(spec, body_id, support.id, support.name))
        return targets


@dataclass(frozen=True, slots=True)
class _SupportTarget:
    spec: SupportedBody
    body_id: int
    support_geom_id: int
    support_geom_name: str
