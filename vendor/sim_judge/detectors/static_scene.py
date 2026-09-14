"""Class S: static scene sanity.

This class looks only at the initial frame and at the model itself; what it checks is
**asset quality** rather than the run. Running it first is genuinely valuable: a movable
body with no mass, or an entity with no collision geometry, robs every subsequent dynamics
finding of its meaning, and reporting it up front avoids misreading an asset defect as a
trajectory problem.

``S1`` movable body missing mass or inertia
``S2`` declared as a physical entity yet has no collision geometry
``S3`` center offset between visual and collision geoms over the limit
``S4`` self-penetration already present in the initial configuration
``S5`` poor mating of the tool mount face (gap or intrusion beyond the
``tool_mount_integrity`` declaration)
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
from sim_judge.replay import Frame
from sim_judge.report.finding import Observation, Subject


class StaticSceneDetector(BaseDetector):
    """S1..S5. Only frame 0 is processed; every later frame is skipped outright."""

    code_prefix = "S"

    _SEARCH_RANGE_M = 0.05

    def __init__(self, context: DetectorContext) -> None:
        super().__init__(context)
        self._done = False

    def feed(self, frame: Frame) -> Iterable[Observation]:
        if self._done or frame.step_index != 0:
            return ()
        self._done = True
        return [
            *self._check_mass_and_inertia(frame),
            *self._check_contact_geometry(frame),
            *self._check_visual_collision_parity(frame),
            *self._check_initial_penetration(frame),
            *self._check_tool_mount(frame),
        ]

    # -- S1 ----------------------------------------------------------------

    def _check_mass_and_inertia(self, frame: Frame) -> Iterable[Observation]:
        minimum_mass = self.policy.minimum_movable_body_mass_kg
        minimum_inertia = self.policy.minimum_movable_body_inertia_kg_m2

        for body_id in range(1, self.model.nbody):
            if int(self.model.body_dofnum[body_id]) == 0:
                continue  # a welded body does not need mass
            name = self.resolver.body_name(body_id)
            entity = self.policy.entity_for_body(name)
            if entity is None or not entity.require_mass_inertia_if_movable:
                continue

            mass = float(self.model.body_mass[body_id])
            inertia = float(np.min(self.model.body_inertia[body_id]))
            subject = body_subject(self.resolver, body_id)

            if mass < minimum_mass:
                yield declared(
                    code="S1_MOVABLE_BODY_MASSLESS",
                    step=frame.step_index,
                    subject=subject,
                    metrics={"mass_kg": mass},
                    peak_metric="mass_kg",
                    thresholds={"minimum_movable_body_mass_kg": minimum_mass},
                    rule_id=f"physics_completeness:{entity.entity_id}",
                )
            if inertia < minimum_inertia:
                yield declared(
                    code="S1_MOVABLE_BODY_INERTIALESS",
                    step=frame.step_index,
                    subject=subject,
                    metrics={"minimum_principal_inertia_kg_m2": inertia},
                    peak_metric="minimum_principal_inertia_kg_m2",
                    thresholds={"minimum_movable_body_inertia_kg_m2": minimum_inertia},
                    rule_id=f"physics_completeness:{entity.entity_id}",
                )

    # -- S2 ----------------------------------------------------------------

    def _check_contact_geometry(self, frame: Frame) -> Iterable[Observation]:
        """Two granularities: an entire entity without a single collision geom is a hard
        defect, whereas an individual movable body without one is merely a risk.

        The latter is only reported as a warning — movable parts such as knobs and
        indicator lights may genuinely have been authored as visual-only on purpose.
        """
        for entity in self.policy.physical_entities:
            if not entity.require_contact_geometry:
                continue
            bodies = [
                body_id
                for body_id in range(self.model.nbody)
                if entity.covers(self.resolver.body_name(body_id))
            ]
            if not bodies:
                continue

            if not any(g.collidable for b in bodies for g in self.resolver.geoms_of_body(b)):
                yield declared(
                    code="S2_ENTITY_WITHOUT_COLLISION_GEOMETRY",
                    step=frame.step_index,
                    subject=Subject(kind="scene", a_name=entity.entity_id, a_entity=entity.entity_id),
                    metrics={"body_count": float(len(bodies))},
                    peak_metric="body_count",
                    thresholds={},
                    rule_id=f"physics_completeness:{entity.entity_id}",
                    detail={"body_patterns": list(entity.body_patterns)},
                )
                continue

            for body_id in bodies:
                if int(self.model.body_dofnum[body_id]) == 0:
                    continue
                geoms = self.resolver.geoms_of_body(body_id)
                visible = [g for g in geoms if not self.policy.is_visual_exempt(g.name)]
                if visible and not any(g.collidable for g in visible):
                    yield generic(
                        code="S2_MOVABLE_BODY_VISUAL_ONLY",
                        step=frame.step_index,
                        subject=body_subject(self.resolver, body_id),
                        metrics={"geom_count": float(len(visible))},
                        peak_metric="geom_count",
                        thresholds={},
                        detail={"entity_id": entity.entity_id},
                    )

    # -- S3 ----------------------------------------------------------------

    def _check_visual_collision_parity(self, frame: Frame) -> Iterable[Observation]:
        positions = self.ctx.replayer.data.geom_xpos
        for pair in self.policy.visual_collision_pairs:
            visual = self.resolver.geom_by_name(pair.visual_geom)
            collision = self.resolver.geom_by_name(pair.collision_geom)
            if visual is None or collision is None:
                continue
            offset = float(np.linalg.norm(positions[visual.id] - positions[collision.id]))
            if offset <= pair.maximum_center_offset_m:
                continue
            yield declared(
                code="S3_VISUAL_COLLISION_OFFSET",
                step=frame.step_index,
                subject=geom_pair_subject(visual, collision),
                metrics={"center_offset_m": offset},
                peak_metric="center_offset_m",
                thresholds={"maximum_center_offset_m": pair.maximum_center_offset_m},
                rule_id=f"component_attachment_critic:{pair.component_id}",
            )

    # -- S4 ----------------------------------------------------------------

    def _check_initial_penetration(self, frame: Frame) -> Iterable[Observation]:
        """Overlap already present in the initial configuration.

        This is a different kind of problem from being pressed in during the run (P1): the
        former is a placement or modelling mistake, the latter a trajectory or contact
        stiffness issue. The class P detectors therefore skip step 0, so the two are never
        reported twice.
        """
        for contact in frame.contacts:
            scale = min(contact.geom_a.min_half_extent_m, contact.geom_b.min_half_extent_m)
            rule = self.policy.contact_rule_for(contact.geom_a.name, contact.geom_b.name, frame.phase_labels)
            declared_limit = rule.maximum_penetration_m if rule else None

            if declared_limit is not None:
                if contact.depth_m <= declared_limit:
                    continue
                yield declared(
                    code="S4_INITIAL_PENETRATION",
                    step=frame.step_index,
                    subject=geom_pair_subject(contact.geom_a, contact.geom_b),
                    metrics={"penetration_m": contact.depth_m},
                    peak_metric="penetration_m",
                    thresholds={"maximum_penetration_m": declared_limit},
                    rule_id=rule.rule_id if rule else None,
                )
                continue

            if not np.isfinite(scale) or scale <= 0.0:
                continue
            limit = max(
                self.limits.penetration_absolute_floor_m,
                self.limits.penetration_ratio_warning * scale,
            )
            if contact.depth_m <= limit:
                continue
            yield generic(
                code="S4_INITIAL_PENETRATION",
                step=frame.step_index,
                subject=geom_pair_subject(contact.geom_a, contact.geom_b),
                metrics={"penetration_m": contact.depth_m, "penetration_ratio": contact.depth_m / scale},
                peak_metric="penetration_m",
                thresholds={"penetration_m": limit, "thinner_geom_half_extent_m": scale},
            )

    # -- S5 ----------------------------------------------------------------

    def _check_tool_mount(self, frame: Frame) -> Iterable[Observation]:
        mount = self.policy.tool_mount
        if mount is None:
            return

        checks = [(p, "mount") for p in mount.mount_geom_pairs]
        checks += [(p, "clearance") for p in mount.clearance_geom_pairs]

        for pair, kind in checks:
            first = self.resolver.geom_by_name(pair.mount_geom)
            second = self.resolver.geom_by_name(pair.tool_geom)
            if first is None or second is None:
                continue
            distance = self.ctx.replayer.geom_distance(first.id, second.id, self._SEARCH_RANGE_M)
            subject = geom_pair_subject(first, second)

            # A mating face may neither lift away nor bite into its counterpart; a
            # clearance face only has to avoid biting in.
            if kind == "mount" and distance > pair.maximum_gap_m:
                yield declared(
                    code="S5_TOOL_MOUNT_GAP",
                    step=frame.step_index,
                    subject=subject,
                    metrics={"gap_m": distance, "excess_m": distance - pair.maximum_gap_m},
                    peak_metric="gap_m",
                    thresholds={"maximum_gap_m": pair.maximum_gap_m},
                    rule_id=f"tool_mount_integrity:{pair.pair_id}",
                )
            elif -distance > pair.maximum_penetration_m:
                yield declared(
                    code="S5_TOOL_MOUNT_PENETRATION",
                    step=frame.step_index,
                    subject=subject,
                    metrics={
                        "penetration_m": -distance,
                        "excess_m": -distance - pair.maximum_penetration_m,
                    },
                    peak_metric="penetration_m",
                    thresholds={"maximum_penetration_m": pair.maximum_penetration_m},
                    rule_id=f"tool_mount_integrity:{pair.pair_id}",
                )
