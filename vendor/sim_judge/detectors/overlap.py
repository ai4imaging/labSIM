"""Class P, geometric tier: interpenetration that the contact solver never reports.

Every other penetration check in this package reads ``data.contact``, and that list only
ever holds pairs the *collision model* is willing to generate contacts for. Two geoms
filtered apart by contype/conaffinity, or a pair where one side is a purely visual geom,
produce no contact at all: the objects sweep through each other with zero force and the
recording looks perfectly clean. On the reference centrifuge scene that blind spot hides
a 20 mm intrusion of the robot wrist into the machine lid at a step where MuJoCo reports
no robot-to-device contact whatsoever.

A judge that only asks the solver "did anything touch?" cannot see this class of defect by
construction, so this detector ignores the solver and measures geometry instead.

Pipeline, cheapest stage first:

broad phase, bodies
    Bounding-sphere rejection between whole bodies. Two dozen bodies means a few hundred
    tests, which throws away the overwhelming majority of geom pairs for free.

broad phase, geoms
    Bounding-sphere rejection from ``model.geom_rbound``, vectorised over the geom pairs
    of whichever body pairs survived.

narrow phase, stage 1
    ``mj_geomDistance``. Cheap and conservative in the right direction, but for mesh geoms
    MuJoCo works on the *convex hull*. A concave shell therefore over-claims badly: a tube
    resting legitimately inside the centrifuge chamber measures 48 mm deep inside the
    housing mesh, because the hull fills the very cavity the tube is sitting in.

narrow phase, stage 2
    For any candidate involving a mesh, re-test the witness segment against the true
    triangle soup with a ray-parity containment test. Measured on the reference scene this
    rejects every hull artifact (0 of 9 sample points genuinely inside the mesh) while
    keeping the real intrusion (8 of 9), so the two populations separate on their own
    rather than by a tuned cutoff.

Three structural rules keep the output honest:

*Bodies already in contact are left to P1.* If the solver is resolving any contact between
the two bodies, whatever their geoms overlap is ordinary soft-contact compression: a
gripper pad sinking into the tube it grips, a pusher into the button it presses, a tube
settling onto a socket floor. Measured on the reference scene, dropping those pairs takes
the confirmed overlaps from 76 down to 15 and leaves the four deepest -- all four of them
links of the same arm buried in the same lid -- at the top of the list, with the residue an
order of magnitude shallower. The division of labour is exact: P1 judges depth wherever a
contact exists, P5 judges the case P1 structurally cannot see.


*Linked bodies are exempt.* A pair is only eligible when the two geoms sit on different
bodies that are not parent and child. Linked bodies share material around their joint by
design -- a shoulder capsule always reaches into the base it rotates on. That single rule
turns out to subsume all twelve ``exclude`` directives the reference model declares for
itself, so nothing scene-specific has to be configured. Exclusions are otherwise *not*
honoured: a hand-written exclusion is precisely the kind of thing that hides a defect.

*Overlap present at rest is subtracted, not reported as an event.* Whatever the very first
recorded state already overlaps is a property of the asset, not something the run did --
on the reference scene that covers the lid latch, which is modelled as a tab embedded in
its catch. Those pairs are reported once as a static defect, and for the rest of the run
the criterion measures how much *deeper* the overlap got, so a pair that starts 3 mm in and
ends up 30 mm in is still caught.
"""

from __future__ import annotations

from collections.abc import Iterable

import mujoco
import numpy as np

from sim_judge.detectors.base import (
    BaseDetector,
    DetectorContext,
    generic,
    geom_pair_subject,
)
from sim_judge.replay import Frame
from sim_judge.report.finding import Observation, Severity

#: How many points along the witness segment get tested for containment in the true mesh.
#: The segment spans the two deepest points ``mj_geomDistance`` reports, so sampling it end
#: to end reveals whether the overlap lies in solid material or in a cavity the hull merely
#: spans.
_WITNESS_SAMPLES = 9

#: Ray direction for the parity test, in the mesh's local frame. An oblique, irrational-ish
#: direction avoids the degenerate cases where a ray runs exactly along an edge or straight
#: through a vertex of an axis-aligned mesh.
_RAY = np.array([0.5773502691896258, 0.5773502691896258, 0.5773502691896258])


class UnmodelledOverlapDetector(BaseDetector):
    """P5 / P6 / S6: geoms sharing the same volume with the solver saying nothing.

    ``P5`` covers pairs the collision model can never resolve, where the physics is
    fiction whatever the recording shows. ``P6`` covers pairs the model *could* have
    resolved but did not on this step, which points at a solver or margin problem rather
    than at the scene description. ``S6`` reports the overlap the scene ships with.
    """

    code_prefix = "P"

    def __init__(self, context: DetectorContext) -> None:
        super().__init__(context)
        self._replayer = context.replayer
        self._sample_interval = context.steps_for(self.limits.overlap_sample_interval_s)
        self._merge_gap = max(
            context.steps_for(self.limits.overlap_merge_gap_s), 2 * self._sample_interval
        )

        self._bodies, self._body_radius, self._geom_pairs = _eligible_pairs(self.model)
        self._triangles: dict[int, np.ndarray] = {}
        self._fromto = np.zeros(6, dtype=np.float64)

        #: Depth each pair already overlaps in the initial recorded state, subtracted from
        #: later measurements so that asset-level overlap is not re-reported every step.
        self._resting_depth: dict[tuple[int, int], float] = {}
        self._resting_known = False

    @property
    def enabled(self) -> bool:
        return len(self._geom_pairs) > 0

    def feed(self, frame: Frame) -> Iterable[Observation]:
        if frame.step_index == 0:
            return self._capture_resting_state(frame)
        if frame.step_index % self._sample_interval:
            return ()
        return self._judge_frame(frame)

    # -- the resting configuration -------------------------------------------

    def _capture_resting_state(self, frame: Frame) -> list[Observation]:
        """Record what already overlaps before the run starts, and report it once."""
        findings: list[Observation] = []
        for geom_a, geom_b, depth in self._overlaps(frame):
            self._resting_depth[(geom_a, geom_b)] = depth
            measurement = self._measure(frame, geom_a, geom_b, depth, resting_depth=0.0)
            if measurement is not None:
                findings.append(measurement.as_static_defect())
        self._resting_known = True
        return findings

    # -- per-frame judgement --------------------------------------------------

    def _judge_frame(self, frame: Frame) -> Iterable[Observation]:
        for geom_a, geom_b, depth in self._overlaps(frame):
            resting = self._resting_depth.get((geom_a, geom_b), 0.0)
            measurement = self._measure(frame, geom_a, geom_b, depth, resting_depth=resting)
            if measurement is not None:
                yield measurement.as_event()

    def _overlaps(self, frame: Frame) -> Iterable[tuple[int, int, float]]:
        """Every eligible geom pair currently overlapping, hull-level, cheapest test first."""
        model, data = self.model, self._replayer.data
        centres = frame.xpos[self._bodies]
        in_contact = _bodies_in_contact(frame)

        for (body_a, body_b), (left, right) in self._geom_pairs.items():
            if (int(self._bodies[body_a]), int(self._bodies[body_b])) in in_contact:
                # These two are already touching, so the solver is resolving them and any
                # overlap between their geoms is ordinary soft-contact compression -- the
                # gripper pad sinking into the tube it grips, the pusher into the button it
                # presses. Judging that depth is P1's job, and it does it from the contact
                # itself. What is left for this detector is the case P1 structurally cannot
                # see: two bodies sharing volume with nothing anywhere between them.
                continue

            separation = centres[body_a] - centres[body_b]
            reach = self._body_radius[body_a] + self._body_radius[body_b]
            if separation @ separation > reach * reach:
                continue

            delta = frame.geom_xpos[left] - frame.geom_xpos[right]
            span = model.geom_rbound[left] + model.geom_rbound[right]
            close = np.einsum("ij,ij->i", delta, delta) < span * span
            if not close.any():
                continue

            for geom_a, geom_b in zip(left[close], right[close], strict=True):
                distance = mujoco.mj_geomDistance(
                    model, data, int(geom_a), int(geom_b), 0.0, self._fromto
                )
                if distance < 0.0:
                    yield int(geom_a), int(geom_b), -float(distance)

    def _measure(
        self, frame: Frame, geom_a: int, geom_b: int, depth: float, *, resting_depth: float
    ) -> _Overlap | None:
        """Turn a hull-level overlap into a finding, or reject it."""
        info_a = self.resolver.geom(geom_a)
        info_b = self.resolver.geom(geom_b)

        # The *thicker* of the two sets the scale here; see the calibration note on
        # unmodelled_overlap_ratio_warning for why this is inverted relative to P1.
        scale = max(info_a.min_half_extent_m, info_b.min_half_extent_m)
        if not np.isfinite(scale) or scale <= 0.0:
            return None  # semi-infinite geoms have no thickness to measure against

        # What the run is responsible for is the depth beyond what the scene shipped with.
        excess = depth - resting_depth
        limit = max(
            self.limits.penetration_absolute_floor_m,
            self.limits.unmodelled_overlap_ratio_warning * scale,
        )
        if excess <= limit:
            return None

        inside_fraction = self._fraction_inside_true_mesh(geom_a, geom_b)
        if inside_fraction < self.limits.overlap_witness_inside_fraction:
            return None  # a convex-hull artifact: the overlap sits in a cavity

        return _Overlap(
            detector=self,
            frame=frame,
            info_a=info_a,
            info_b=info_b,
            depth_m=depth,
            excess_m=excess,
            resting_depth_m=resting_depth,
            scale_m=scale,
            limit_m=limit,
            inside_fraction=inside_fraction,
            solver_blind=not _solver_can_resolve(self.model, geom_a, geom_b),
        )

    # -- exact mesh containment ------------------------------------------------

    def _fraction_inside_true_mesh(self, geom_a: int, geom_b: int) -> float:
        """How much of the witness segment lies in solid material rather than a cavity.

        ``mj_geomDistance`` reduces a mesh to its convex hull, which reports an object
        resting inside a concave shell as deeply embedded in it. Only mesh geoms can lie
        this way; a pair of exact primitives is already trustworthy and returns 1.0
        without doing any work.
        """
        meshes = [
            geom
            for geom in (geom_a, geom_b)
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        ]
        if not meshes:
            return 1.0

        start, end = self._fromto[:3], self._fromto[3:]
        samples = start + np.linspace(0.0, 1.0, _WITNESS_SAMPLES)[:, None] * (end - start)

        # The overlap has to survive every mesh involved: a point sitting in the cavity of
        # either shell is not inside solid material.
        return min(float(self._points_inside_mesh(geom, samples).mean()) for geom in meshes)

    def _points_inside_mesh(self, geom: int, points_world: np.ndarray) -> np.ndarray:
        """Ray-parity containment test against the geom's true triangle soup.

        Shoots one ray per point and counts surface crossings; an odd count means the point
        is enclosed. Because it works on the raw triangles it can tell solid material from
        an open cavity, which is the whole reason this stage exists.
        """
        triangles = self._mesh_triangles(geom)
        data = self._replayer.data
        rotation = data.geom_xmat[geom].reshape(3, 3)
        local = (points_world - data.geom_xpos[geom]) @ rotation

        corner = triangles[:, 0]
        edge_1 = triangles[:, 1] - corner
        edge_2 = triangles[:, 2] - corner

        # Moller-Trumbore with a fixed ray direction, so every term that does not depend on
        # the ray origin is computed once for the whole batch of points.
        pivot = np.cross(_RAY, edge_2)
        determinant = np.einsum("fi,fi->f", edge_1, pivot)
        usable = np.abs(determinant) > 1e-12
        inverse = np.zeros_like(determinant)
        inverse[usable] = 1.0 / determinant[usable]

        offset = local[:, None, :] - corner[None, :, :]
        u = np.einsum("pfi,fi->pf", offset, pivot) * inverse
        cross = np.cross(offset, edge_1[None, :, :])
        v = (cross @ _RAY) * inverse
        t = np.einsum("pfi,fi->pf", cross, edge_2) * inverse

        hit = usable & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-9)
        return hit.sum(axis=1) % 2 == 1

    def _mesh_triangles(self, geom: int) -> np.ndarray:
        """``(faces, 3, 3)`` triangle vertices in the geom's local frame, cached per geom."""
        cached = self._triangles.get(geom)
        if cached is not None:
            return cached

        model = self.model
        mesh = int(model.geom_dataid[geom])
        vertex_start = int(model.mesh_vertadr[mesh])
        vertex_count = int(model.mesh_vertnum[mesh])
        face_start = int(model.mesh_faceadr[mesh])
        face_count = int(model.mesh_facenum[mesh])

        vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
        vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(model.mesh_face[face_start : face_start + face_count]).reshape(-1, 3)

        triangles = vertices[faces]
        self._triangles[geom] = triangles
        return triangles


class _Overlap:
    """One confirmed overlap, ready to be phrased either as a static defect or an event.

    Kept as a small object rather than a tuple because the resting-state pass and the
    per-frame pass build the identical measurement and only differ in how they label it.
    """

    __slots__ = (
        "_detector",
        "_frame",
        "_info_a",
        "_info_b",
        "depth_m",
        "excess_m",
        "resting_depth_m",
        "scale_m",
        "limit_m",
        "inside_fraction",
        "solver_blind",
    )

    def __init__(
        self,
        *,
        detector: UnmodelledOverlapDetector,
        frame: Frame,
        info_a,
        info_b,
        depth_m: float,
        excess_m: float,
        resting_depth_m: float,
        scale_m: float,
        limit_m: float,
        inside_fraction: float,
        solver_blind: bool,
    ) -> None:
        self._detector = detector
        self._frame = frame
        self._info_a = info_a
        self._info_b = info_b
        self.depth_m = depth_m
        self.excess_m = excess_m
        self.resting_depth_m = resting_depth_m
        self.scale_m = scale_m
        self.limit_m = limit_m
        self.inside_fraction = inside_fraction
        self.solver_blind = solver_blind

    @property
    def _ratio(self) -> float:
        return self.depth_m / self.scale_m

    @property
    def _physical(self) -> bool:
        """Whether any of the physics is actually corrupted by this overlap.

        Two purely decorative geoms passing through each other is a rendering defect: no
        force, mass or constraint is involved on either side. Worth saying, not worth
        failing a run over.
        """
        return self._info_a.collidable or self._info_b.collidable

    def _common(self) -> dict:
        limits = self._detector.limits
        return {
            "step": self._frame.step_index,
            "subject": geom_pair_subject(self._info_a, self._info_b),
            "metrics": {
                "penetration_m": self.depth_m,
                "penetration_ratio": self._ratio,
                "penetration_excess_m": self.excess_m,
                "witness_inside_fraction": self.inside_fraction,
            },
            "peak_metric": "penetration_m",
            "merge_gap_steps": self._detector._merge_gap,
            "thresholds": {
                "penetration_m": self.limit_m,
                "thicker_geom_half_extent_m": self.scale_m,
                "ratio_warning": limits.unmodelled_overlap_ratio_warning,
                "ratio_severe": limits.unmodelled_overlap_ratio_severe,
            },
            "detail": {
                "scale_geom": (
                    self._info_a.name
                    if self._info_a.min_half_extent_m >= self._info_b.min_half_extent_m
                    else self._info_b.name
                ),
                "solver_blind": self.solver_blind,
                "visual_only_a": not self._info_a.collidable,
                "visual_only_b": not self._info_b.collidable,
                "resting_overlap_m": self.resting_depth_m,
                "affects_physics": self._physical,
            },
        }

    def as_static_defect(self) -> Observation:
        """The scene already ships with this overlap, so it is an asset problem."""
        return generic(
            code="S6_RESTING_GEOMETRY_INTERPENETRATION",
            severity=Severity.WARNING,
            **self._common(),
        )

    def as_event(self) -> Observation:
        """The run drove these two geoms into each other.

        Once the intrusion reaches a sizeable fraction of the thicker geom's half-thickness
        it is no longer a grazing mismatch between visual and collision geometry but one
        object well inside another's material, and with no contact anywhere between the two
        bodies nothing in the simulation will ever push it back out -- so it is a failure,
        provided real physics is involved.
        """
        severe = self._ratio >= self._detector.limits.unmodelled_overlap_ratio_severe
        return generic(
            code=(
                "P5_SOLVER_BLIND_INTERPENETRATION"
                if self.solver_blind
                else "P6_UNREPORTED_INTERPENETRATION"
            ),
            severity=Severity.FAILURE if (severe and self._physical) else Severity.WARNING,
            **self._common(),
        )


def _eligible_pairs(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    """Enumerate the geom pairs worth measuring, grouped by the body pair they belong to.

    Grouping is what makes the per-frame cost bearable: one sphere test per body pair
    discards all of that pair's geom combinations at once, and only the handful of body
    pairs that are actually near each other ever reach the geom-level test.

    Returns the body ids in play, their bounding radii, and the geom index arrays keyed by
    position within that body list.
    """
    bodies = sorted({int(model.geom_bodyid[g]) for g in range(model.ngeom)})
    index_of = {body: position for position, body in enumerate(bodies)}

    geoms_of: dict[int, list[int]] = {body: [] for body in bodies}
    for geom in range(model.ngeom):
        geoms_of[int(model.geom_bodyid[geom])].append(geom)

    # A body's reach is the furthest any of its geoms extends from the body origin.
    radius = np.array(
        [
            max(
                float(np.linalg.norm(model.geom_pos[g])) + float(model.geom_rbound[g])
                for g in geoms_of[body]
            )
            for body in bodies
        ],
        dtype=np.float64,
    )

    parent = model.body_parentid
    pairs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for position_a, body_a in enumerate(bodies):
        for body_b in bodies[position_a + 1 :]:
            if parent[body_a] == body_b or parent[body_b] == body_a:
                continue  # linked bodies share material at their joint by construction
            left = np.repeat(geoms_of[body_a], len(geoms_of[body_b]))
            right = np.tile(geoms_of[body_b], len(geoms_of[body_a]))
            pairs[(position_a, index_of[body_b])] = (
                left.astype(np.int32),
                right.astype(np.int32),
            )

    return np.asarray(bodies, dtype=np.int32), radius, pairs


def _bodies_in_contact(frame: Frame) -> set[tuple[int, int]]:
    """Body pairs the solver is currently resolving a contact for, as ordered id pairs."""
    return {
        (min(contact.geom_a.body_id, contact.geom_b.body_id),
         max(contact.geom_a.body_id, contact.geom_b.body_id))
        for contact in frame.contacts
    }


def _solver_can_resolve(model: mujoco.MjModel, geom_a: int, geom_b: int) -> bool:
    """Whether MuJoCo's collision filter would ever let this pair produce a contact.

    Mirrors the engine's own rule: the pair passes when either geom's contype shares a bit
    with the other's conaffinity. Purely visual geoms have both masks at zero and can never
    pass, which is exactly the case this detector exists to catch.
    """
    contype_a, affinity_a = int(model.geom_contype[geom_a]), int(model.geom_conaffinity[geom_a])
    contype_b, affinity_b = int(model.geom_contype[geom_b]), int(model.geom_conaffinity[geom_b])
    return bool((contype_a & affinity_b) or (contype_b & affinity_a))
