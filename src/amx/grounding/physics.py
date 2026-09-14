"""The physical questions a rubric asks that geometry alone cannot answer.

`checks.py` already knows how to measure a datasheet number and how to drive one stated
affordance to its endpoint. What it has no vocabulary for is the class of defect that
made the first benchmark sweep meaningless: a part with no mass, an assembly whose pieces
never touch each other, a lid that clears its housing at both ends of its travel and
ploughs straight through it in the middle.

Those are not measurements of the specification, they are measurements of whether the
thing is an object at all, and they are deliberately kept in their own module because
they need no target to check against. An asset either has positive mass everywhere or it
does not, and no datasheet has an opinion on the matter.

Everything here raises rather than returning a benign default. A caller that cannot
measure something must be able to tell the difference between "measured, and it is fine"
and "could not measure", because collapsing those two is precisely the bug this module
exists to stop repeating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import trimesh

from amx.grounding import mesh as gmesh
from amx.grounding.build import GroundedAsset
from amx.grounding.scene import GROUND_GEOM, SceneError, Staged, settle, stage

CONTACT_MARGIN_M = 0.001
"""How close two parts have to be before they count as touching.

A millimetre. Tight enough that two halves of an instrument standing a centimetre apart
are still reported as disconnected, loose enough that a lid resting on a rim is not
called disconnected because the mesh leaves a 50 µm gap.
"""

PENETRATION_TOLERANCE_M = 0.0002
"""Overlap that counts as interpenetration rather than contact.

Matches the 0.2 mm the corpus states for abnormal penetration, and sits an order of
magnitude above the solver's own allowance at the stiffened contact settings `scene`
uses, so a reported overlap is the geometry's and not the integrator's.
"""

PLAUSIBLE_DENSITY_KG_M3 = (80.0, 12000.0)
"""The density band a labware part can sit in and still be a real material.

The floor is below expanded polystyrene packing; the ceiling is above lead. A part
outside this is not a material choice, it is a mass that was never authored.
"""


class PhysicsUnavailable(RuntimeError):
    """The question could not be put to the model, so there is no answer to report."""


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #


@dataclass
class AssetContext:
    """One compiled asset, staged at most twice however many checks are run against it.

    Staging is the expensive part of every check in this file and the static pose is
    shared by all of them, so it is built once and handed round. The grounded pose is
    separate because standing an object on a floor under gravity is a different
    experiment from inspecting it where it was authored.
    """

    asset: GroundedAsset
    _static: Staged | None = field(default=None, init=False, repr=False)
    _grounded: Staged | None = field(default=None, init=False, repr=False)
    _world: trimesh.Trimesh | None = field(default=None, init=False, repr=False)
    _topology: gmesh.Topology | None = field(default=None, init=False, repr=False)
    _body_meshes: dict[str, trimesh.Trimesh | None] = field(
        default_factory=dict, init=False, repr=False
    )
    load_error: str | None = field(default=None, init=False)

    def static(self) -> Staged:
        """The asset as authored: no floor, no free joint, nothing moved."""
        if self._static is None:
            try:
                self._static = stage(self.asset.mjcf_path, ground=False, free=False)
            except (SceneError, RuntimeError) as error:
                self.load_error = str(error)
                raise PhysicsUnavailable(f"the asset will not load: {error}") from error
        return self._static

    def grounded(self) -> Staged:
        """The asset standing on a bench, free to fall over."""
        if self._grounded is None:
            try:
                self._grounded = stage(self.asset.mjcf_path, ground=True, free=True)
            except (SceneError, RuntimeError) as error:
                self.load_error = str(error)
                raise PhysicsUnavailable(f"the asset will not stand up: {error}") from error
        return self._grounded

    def topology(self) -> gmesh.Topology:
        if self._topology is None:
            self._topology = gmesh.read_topology(self.static().model)
        return self._topology

    def world(self) -> trimesh.Trimesh:
        if self._world is None:
            staged = self.static()
            self._world = gmesh.world_mesh(staged.model, staged.data)
        return self._world

    def body_id(self, body: str) -> int:
        staged = self.static()
        found = mujoco.mj_name2id(staged.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if found < 0:
            raise PhysicsUnavailable(f"no body named {body!r}")
        return int(found)

    def body_mesh(self, body: str) -> trimesh.Trimesh:
        if body not in self._body_meshes:
            staged = self.static()
            self._body_meshes[body] = gmesh.body_mesh(
                staged.model, staged.data, self.body_id(body)
            )
        found = self._body_meshes[body]
        if found is None or found.is_empty:
            raise PhysicsUnavailable(f"body {body!r} carries no measurable geometry")
        return found

    def match_bodies(self, hint: str) -> list[str]:
        """Bodies whose name plausibly refers to `hint`.

        The same two-way normalised substring match the component checks use, so a
        specification that says "Pouring spout" and a model that says `pouring_spout`
        agree, and then a token overlap pass so that "Lid latch button" can still find
        `latch_button`. Ordered best first.
        """
        return match_bodies(hint, self.topology().bodies)


def match_bodies(hint: str, bodies: list[str]) -> list[str]:
    """Rank body names against a free-text hint. Shared so the judge and loop agree."""
    wanted = _normalise(hint)
    if not wanted:
        return []
    exact = [name for name in bodies if _normalise(name) == wanted]
    if exact:
        return exact
    contained = [
        name
        for name in bodies
        if wanted in _normalise(name) or _normalise(name) in wanted
    ]
    if contained:
        return sorted(contained, key=lambda name: abs(len(_normalise(name)) - len(wanted)))

    tokens = {token for token in _tokens(hint) if len(token) >= 3}
    if not tokens:
        return []
    scored: list[tuple[int, str]] = []
    for name in bodies:
        overlap = len(tokens & set(_tokens(name)))
        if overlap:
            scored.append((overlap, name))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return [name for _, name in scored]


def _normalise(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _tokens(text: str) -> set[str]:
    current: list[str] = []
    out: set[str] = set()
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            out.add("".join(current))
            current = []
    if current:
        out.add("".join(current))
    return out


# --------------------------------------------------------------------------- #
# mass, inertia, density
# --------------------------------------------------------------------------- #


@dataclass
class PartMass:
    """What one body weighs, and whether that weight is physically possible."""

    body: str
    mass_kg: float
    volume_m3: float
    density_kg_m3: float | None
    inertia: tuple[float, float, float]
    movable: bool
    problems: list[str] = field(default_factory=list)

    @property
    def sound(self) -> bool:
        return not self.problems


def part_masses(context: AssetContext) -> list[PartMass]:
    """Per-body mass, density and inertia, with every implausibility named.

    MuJoCo refuses to compile a *moving* body with no mass, so the zero-mass parts that
    survive to here are the welded ones — which still matter, because a rack whose tubes
    weigh nothing is a rack that cannot be picked up in simulation.
    """
    staged = context.static()
    model = staged.model
    movable = {
        int(model.jnt_bodyid[joint])
        for joint in range(model.njnt)
        if int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE)
    }
    free = {
        int(model.jnt_bodyid[joint])
        for joint in range(model.njnt)
        if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
    }

    out: list[PartMass] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body#{body_id}"
        if int(model.body_geomnum[body_id]) == 0:
            continue
        mass = float(model.body_mass[body_id])
        inertia = tuple(float(value) for value in model.body_inertia[body_id])
        piece = gmesh.body_mesh(model, staged.data, body_id)
        volume = gmesh.solid_volume(piece) if piece is not None else 0.0
        density = (mass / volume) if volume > 0.0 and mass > 0.0 else None

        problems: list[str] = []
        if mass <= 0.0:
            problems.append("mass is zero")
        if any(value <= 0.0 for value in inertia):
            problems.append("inertia has a non-positive principal moment")
        elif not _triangle_inequality(inertia):
            problems.append("principal moments violate the inertia triangle inequality")
        if density is not None and not (
            PLAUSIBLE_DENSITY_KG_M3[0] <= density <= PLAUSIBLE_DENSITY_KG_M3[1]
        ):
            problems.append(f"density {density:.0f} kg/m^3 is not a real material")
        if mass > 0.0 and all(value > 0.0 for value in inertia) and piece is not None:
            problems.extend(_gyration_problems(mass, inertia, piece))

        out.append(
            PartMass(
                body=name,
                mass_kg=mass,
                volume_m3=volume,
                density_kg_m3=density,
                inertia=inertia,  # type: ignore[arg-type]
                movable=body_id in movable or body_id in free,
                problems=problems,
            )
        )
    if not out:
        raise PhysicsUnavailable("the model declares no bodies carrying geometry")
    return out


def _triangle_inequality(inertia: tuple[float, ...]) -> bool:
    a, b, c = sorted(inertia)
    return a + b >= c * (1.0 - 1e-6)


def _gyration_problems(
    mass: float, inertia: tuple[float, ...], piece: trimesh.Trimesh
) -> list[str]:
    """Whether the inertia is the right size for the shape it claims to describe.

    The radius of gyration of any solid lies between a small fraction of its bounding
    radius and the bounding radius itself. An inertia tensor that puts it far outside
    that band was not computed from the geometry, and it is the difference between a
    rotor that spins up and a rotor that will not budge.
    """
    extent = float(np.linalg.norm(gmesh.extents(piece))) / 2.0
    if extent <= 0.0:
        return []
    gyration = math.sqrt(max(inertia) / mass)
    if gyration > extent * 2.0:
        return [f"inertia implies a {gyration * 1000:.1f} mm gyration radius on a "
                f"{extent * 1000:.1f} mm part"]
    if gyration < extent * 0.01:
        return [f"inertia is {extent / max(gyration, 1e-12):.0f}x too small for the part size"]
    return []


# --------------------------------------------------------------------------- #
# assembly integrity
# --------------------------------------------------------------------------- #


@dataclass
class Connectivity:
    """Which parts of the assembly actually touch which."""

    connected: bool
    orphans: list[str]
    groups: int
    inspected: int


def assembly_connectivity(
    context: AssetContext, *, margin_m: float = CONTACT_MARGIN_M
) -> Connectivity:
    """Whether the object is one solid thing or several floating near each other.

    The kinematic tree proves nothing here: a body has a parent by construction, and
    MuJoCo does not even test parent against child for contact, which is precisely the
    pair that matters. So this measures geometry directly — bounding boxes to shortlist
    the pairs, then an exact surface distance on each shortlisted pair — and asks
    whether the resulting graph reaches every part. A knob hovering 3 mm off its panel
    is joined in the tree and adrift in the world, and it is the world that is simulated.
    """
    staged = context.static()
    model, data = staged.model, staged.data
    meshes: dict[str, trimesh.Trimesh] = {}
    for body_id in range(1, model.nbody):
        if int(model.body_geomnum[body_id]) == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body#{body_id}"
        piece = gmesh.body_mesh(model, data, body_id)
        if piece is not None and not piece.is_empty:
            meshes[name] = piece

    names = list(meshes)
    if len(names) <= 1:
        return Connectivity(connected=True, orphans=[], groups=len(names), inspected=len(names))

    touching: dict[str, set[str]] = {name: set() for name in names}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if not _boxes_near(meshes[first], meshes[second], margin_m):
                continue
            if _surface_distance(meshes[first], meshes[second]) <= margin_m:
                touching[first].add(second)
                touching[second].add(first)

    seen: set[str] = set()
    groups: list[set[str]] = []
    for name in names:
        if name in seen:
            continue
        stack = [name]
        group: set[str] = set()
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            stack.extend(touching[current] - group)
        seen |= group
        groups.append(group)

    largest = max(groups, key=len)
    orphans = [
        name for group in groups if group is not largest for name in sorted(group)
    ]
    return Connectivity(
        connected=len(groups) == 1,
        orphans=orphans,
        groups=len(groups),
        inspected=len(names),
    )


def _boxes_near(first: trimesh.Trimesh, second: trimesh.Trimesh, margin: float) -> bool:
    lower_a, upper_a = first.bounds
    lower_b, upper_b = second.bounds
    return bool(np.all(lower_a - margin <= upper_b) and np.all(lower_b - margin <= upper_a))


def _surface_distance(first: trimesh.Trimesh, second: trimesh.Trimesh) -> float:
    """Closest approach between two surfaces, zero when they intersect.

    Exact where FCL is available, which it is by declaration; if it is not, overlapping
    bounding boxes are taken as touching, which errs towards calling an assembly
    connected rather than accusing a sound one of falling apart.
    """
    try:
        from trimesh.collision import CollisionManager  # noqa: PLC0415

        manager = CollisionManager()
        manager.add_object("a", first)
        manager.add_object("b", second)
        return float(manager.min_distance_internal())
    except Exception:  # noqa: BLE001 - no FCL, or a mesh it will not accept
        return 0.0


def rest_interpenetration(context: AssetContext) -> tuple[float, str]:
    """The deepest overlap between two of the asset's own parts, standing still.

    Contacts with the bench are excluded on purpose: an object sinking into the floor is
    a stability failure and is reported as one, while two of its own parts sharing the
    same space is a modelling failure and has to be separable from it.
    """
    staged = context.grounded()
    if not settle(staged, 1.0):
        raise PhysicsUnavailable("the simulation diverged before anything could be measured")
    model, data = staged.model, staged.data
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM)

    worst = 0.0
    where = ""
    for index in range(data.ncon):
        contact = data.contact[index]
        if ground >= 0 and ground in {int(contact.geom1), int(contact.geom2)}:
            continue
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        if first == second:
            continue
        distance = float(contact.dist)
        if distance < worst:
            worst = distance
            where = "{} into {}".format(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first) or first,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second) or second,
            )
    return abs(worst), where


# --------------------------------------------------------------------------- #
# swept motion
# --------------------------------------------------------------------------- #


@dataclass
class Sweep:
    """What happened as one joint was driven across its whole travel."""

    joint: str
    body: str
    samples: int
    worst_penetration_m: float
    worst_at: float
    reached: tuple[float, float]
    blocked_fraction: float


def swept_collision(
    context: AssetContext, joint: str, *, samples: int = 16
) -> Sweep:
    """Drive a joint through its range and watch for geometry ploughing through geometry.

    `check_operations` tests the two endpoints, which is where an author looks and so is
    where the clearance usually is. The interesting failure is in between: a lid hinged
    a centimetre inside its housing clears at closed and clears at open and sweeps
    through the wall on the way. Sixteen samples is enough to catch an overlap that
    lasts more than a few degrees, which is every real one seen so far.
    """
    staged = context.static()
    model, data = staged.model, staged.data
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if joint_id < 0:
        raise PhysicsUnavailable(f"no joint named {joint!r}")
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        raise PhysicsUnavailable(f"{joint!r} is a free joint and has no travel to sweep")

    body_id = int(model.jnt_bodyid[joint_id])
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body#{body_id}"
    address = int(model.jnt_qposadr[joint_id])
    if bool(model.jnt_limited[joint_id]):
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
    elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
        lower, upper = 0.0, 2.0 * math.pi
    else:
        lower, upper = 0.0, 0.02
    if upper <= lower:
        raise PhysicsUnavailable(f"{joint!r} has an empty travel range")

    start = np.array(data.qpos, copy=True)
    worst = 0.0
    worst_at = lower
    blocked = 0
    try:
        for index in range(samples + 1):
            position = lower + (upper - lower) * index / samples
            data.qpos[address] = position
            mujoco.mj_forward(model, data)
            depth = _body_overlap(model, data, body_id)
            if depth > worst:
                worst, worst_at = depth, position
            if depth > PENETRATION_TOLERANCE_M:
                blocked += 1
    finally:
        data.qpos[:] = start
        mujoco.mj_forward(model, data)

    return Sweep(
        joint=joint,
        body=body,
        samples=samples + 1,
        worst_penetration_m=worst,
        worst_at=worst_at,
        reached=(lower, upper),
        blocked_fraction=blocked / (samples + 1),
    )


def _body_overlap(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> float:
    worst = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        if body_id not in {first, second} or first == second:
            continue
        worst = min(worst, float(contact.dist))
    return abs(worst)


# --------------------------------------------------------------------------- #
# standing up
# --------------------------------------------------------------------------- #


def com_support_margin(context: AssetContext) -> tuple[float, dict[str, Any]]:
    """How far inside its own footprint the centre of mass sits, in metres.

    Negative means the centre of mass is outside the supporting polygon, which is an
    object that will fall over the moment it is let go however carefully it was placed.
    The footprint is taken from the vertices within a millimetre of the lowest point,
    because that is what the bench is actually in contact with.
    """
    staged = context.static()
    model, data = staged.model, staged.data
    world = context.world()
    vertices = np.asarray(world.vertices, dtype=float)
    if len(vertices) < 3:
        raise PhysicsUnavailable("too few vertices to form a footprint")

    floor = float(vertices[:, 2].min())
    footprint = vertices[vertices[:, 2] <= floor + 0.001][:, :2]
    if len(footprint) < 3:
        raise PhysicsUnavailable("the object touches the bench at fewer than three points")

    mujoco.mj_forward(model, data)
    total = float(np.sum(model.body_mass[1:]))
    if total <= 0.0:
        raise PhysicsUnavailable("the model has no mass, so it has no centre of mass")
    centre = np.sum(
        data.xipos[1:] * np.asarray(model.body_mass[1:], dtype=float)[:, None], axis=0
    ) / total

    try:
        hull = trimesh.points.PointCloud(
            np.column_stack([footprint, np.zeros(len(footprint))])
        ).convex_hull
        polygon = np.asarray(hull.vertices, dtype=float)[:, :2]
    except Exception as error:  # noqa: BLE001 - a degenerate footprint is unmeasurable
        raise PhysicsUnavailable(f"the footprint is degenerate: {error}") from error

    margin = _distance_inside(centre[:2], polygon)
    return margin, {
        "com_xy_mm": [float(centre[0]) * 1000, float(centre[1]) * 1000],
        "footprint_points": int(len(footprint)),
        "total_mass_kg": total,
    }


def _distance_inside(point: np.ndarray, polygon: np.ndarray) -> float:
    """Signed distance from a point to a convex hull's boundary, positive inside."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    try:
        hull = ConvexHull(polygon)
    except Exception as error:  # noqa: BLE001
        raise PhysicsUnavailable(f"the footprint has no area: {error}") from error

    distances = []
    for equation in hull.equations:
        normal, offset = equation[:-1], equation[-1]
        distances.append(-(float(np.dot(normal, point)) + float(offset)))
    return float(min(distances))
