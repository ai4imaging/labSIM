"""Turning a compiled MuJoCo model back into geometry you can put a caliper on.

MuJoCo is a physics engine, and what it exposes readily is bounding boxes and contact
normals. A datasheet asks a different kind of question — what is the outside diameter,
how thick is the wall, how much does it hold — and none of those survive a bounding box.
A beaker's bounding box is wider than the beaker because of its spout, and a bounding
box cannot see a cavity at all.

So the model is flattened back into one world-space triangle soup and measured with
cross-sections, which is what a person with a caliper would do:

* `profile_at` slices perpendicular to the part's axis and reads the outer ring, the
  inner ring and the gap between them.
* `cavity_volume` integrates the area of the inner rings from the bottom up to the
  lowest height at which the section stops being a ring. That height is the overflow
  edge, whether it is the rim or a pouring spout, and it falls out of the geometry
  rather than having to be declared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import trimesh

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

_PRIMITIVE_SECTIONS = 64
"""Facets used when tessellating an analytic geom. A 250 mL beaker modelled as a
cylinder is 70 mm across; 64 facets put the chord error near 30 µm, which is two
orders of magnitude under the tolerance any of these datasheets state."""


class GeometryUnavailable(RuntimeError):
    """The model carries no triangles that can be measured."""


@dataclass
class Profile:
    """One cross-section, reduced to the numbers a datasheet quotes."""

    height: float
    outer_diameter: float
    inner_diameter: float | None
    wall_thickness: float | None
    ring_count: int
    outer_area: float
    inner_area: float
    feature_count: int = 1
    """How many alike features this section cuts through.

    1 for a beaker, 96 for a PCR rack. The numbers above describe one member of that
    population, so a rack's `inner_diameter` is one well's bore rather than an average
    over 96 wells and two hinge posts.
    """
    feature_centres: tuple[tuple[float, float], ...] = ()
    """Where each member of the population sits in the section plane.

    Kept so a stated centre-to-centre pitch can be measured. Nothing else in the corpus
    could confirm a 9 mm well pitch, and 9 mm is most of what makes a rack a rack.
    """
    inner_span: float | None = None
    """The narrowest clear width across the opening, whatever shape it is.

    `inner_diameter` is a median radius doubled, which is the right reading for a round bore
    and the wrong one for anything else: a deep-well plate states an 8.2 mm "square internal
    mouth width", and the diameter of a square is not a number anyone quotes. This is the
    smaller side of the opening's bounding box, so it is the side of a square, the narrow way
    across a slot, and the diameter of a circle.
    """

    @property
    def is_enclosed(self) -> bool:
        """Whether anything at this height is walled in on every side."""
        return self.ring_count > 0


@dataclass
class CavityMeasurement:
    """A cavity volume and the evidence that the number has converged."""

    volume_m3: float
    overflow_height: float
    floor_height: float
    inner_diameter: float
    centre_xy: tuple[float, float]
    convergence_rel: float
    samples: int
    method: str = "cross-section integration to the lowest overflow edge"

    @property
    def volume_ml(self) -> float:
        return self.volume_m3 * 1e6

    @property
    def converged(self) -> bool:
        return self.convergence_rel <= 0.01


@dataclass
class Topology:
    """The body/joint tree, named, for checking a component list against."""

    bodies: list[str] = field(default_factory=list)
    joints: dict[str, str] = field(default_factory=dict)
    """Joint name -> the body it drives."""
    parents: dict[str, str] = field(default_factory=dict)
    sites: list[str] = field(default_factory=list)
    geoms_per_body: dict[str, int] = field(default_factory=dict)
    mesh_geoms: int = 0
    primitive_geoms: int = 0


def world_mesh(model: mujoco.MjModel, data: mujoco.MjData) -> trimesh.Trimesh:
    """Every visible geom, tessellated and placed, as one mesh in world coordinates.

    Planes are skipped: they are infinite, and a ground plane in the asset's own file
    would otherwise swallow every measurement taken from the bounding box.
    """
    pieces: list[trimesh.Trimesh] = []
    for geom in range(model.ngeom):
        piece = _geom_mesh(model, geom)
        if piece is None or piece.is_empty:
            continue
        transform = np.eye(4)
        transform[:3, :3] = data.geom_xmat[geom].reshape(3, 3)
        transform[:3, 3] = data.geom_xpos[geom]
        piece.apply_transform(transform)
        pieces.append(piece)
    if not pieces:
        raise GeometryUnavailable("the compiled model has no measurable geometry")
    return trimesh.util.concatenate(pieces)


def body_mesh(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> trimesh.Trimesh | None:
    """Just the geoms hanging off one body, placed in world coordinates.

    A datasheet's "clear inner diameter" belongs to the tube, not to the rack the tube
    sits in, and measuring it on the whole world mesh reads the rack. Restricting the
    triangle soup to one body is what makes a local feature measurable at all.
    """
    pieces: list[trimesh.Trimesh] = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) != body_id:
            continue
        piece = _geom_mesh(model, geom)
        if piece is None or piece.is_empty:
            continue
        transform = np.eye(4)
        transform[:3, :3] = data.geom_xmat[geom].reshape(3, 3)
        transform[:3, 3] = data.geom_xpos[geom]
        piece.apply_transform(transform)
        pieces.append(piece)
    if not pieces:
        return None
    return trimesh.util.concatenate(pieces)


def solid_volume(mesh: trimesh.Trimesh) -> float:
    """The volume of material in a part, in cubic metres.

    Watertight meshes report their own volume. The rest fall back to the convex hull,
    which overstates a hollow part and so understates its density — the conservative
    direction when the number is being used to ask whether a density is physical.
    """
    try:
        if mesh.is_watertight:
            volume = abs(float(mesh.volume))
            if volume > 0.0:
                return volume
    except Exception:  # noqa: BLE001 - a degenerate mesh has no volume to report
        pass
    try:
        return abs(float(mesh.convex_hull.volume))
    except Exception:  # noqa: BLE001
        return 0.0


def _geom_mesh(model: mujoco.MjModel, geom: int) -> trimesh.Trimesh | None:
    kind = int(model.geom_type[geom])
    size = model.geom_size[geom]
    types = mujoco.mjtGeom
    if kind == types.mjGEOM_PLANE:
        return None
    if kind == types.mjGEOM_MESH:
        return _mesh_asset(model, int(model.geom_dataid[geom]))
    if kind == types.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * size[:3])
    if kind == types.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=3, radius=float(size[0]))
    if kind == types.mjGEOM_ELLIPSOID:
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        sphere.apply_scale(size[:3])
        return sphere
    if kind == types.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(
            radius=float(size[0]), height=2.0 * float(size[1]), sections=_PRIMITIVE_SECTIONS
        )
    if kind == types.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(
            radius=float(size[0]), height=2.0 * float(size[1]), count=[16, _PRIMITIVE_SECTIONS]
        )
    return None


def _mesh_asset(model: mujoco.MjModel, mesh_id: int) -> trimesh.Trimesh | None:
    if mesh_id < 0:
        return None
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    if vert_count <= 0 or face_count <= 0:
        return None
    vertices = np.array(model.mesh_vert[vert_start : vert_start + vert_count], dtype=float)
    faces = np.array(model.mesh_face[face_start : face_start + face_count], dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def axis_bounds(mesh: trimesh.Trimesh, axis: str = "z") -> tuple[float, float]:
    index = AXIS_INDEX[axis]
    lower, upper = mesh.bounds
    return float(lower[index]), float(upper[index])


def extents(mesh: trimesh.Trimesh) -> np.ndarray:
    lower, upper = mesh.bounds
    return np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)


def top_surface_range(
    mesh: trimesh.Trimesh, axis: str = "z", *, samples: int = 24
) -> tuple[float, float] | None:
    """The lowest and highest the object's upper surface gets, measured from its base.

    An object with a flat top has one height and these are equal. Plenty of them do not: a
    stepped rack has a high row and a low row, a sloped instrument has a tall back and a short
    front, and a specification quotes both, from the same datum, as two numbers. Read as one
    height each they are two claims about a single extent, so at most one could ever be true of
    whatever was built and the other failed regardless -- which is why they were declined instead.

    Found by dropping a ray down the axis onto the object from above at each point of a grid over
    its footprint, and keeping the highest surface each ray hits. The maximum of those is the
    overall height; the minimum is the lowest part of the top. Rays that miss entirely are what
    a footprint being non-rectangular looks like, and are discarded rather than read as zero.
    """
    index = AXIS_INDEX[axis]
    plane = [i for i in range(3) if i != index]
    lower, upper = mesh.bounds
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        return None

    # Inset the grid so rays do not grave the vertical outer walls, where a hit is a
    # coin toss between the wall and nothing.
    spans = [(upper[i] - lower[i]) for i in plane]
    if min(spans) <= 0.0:
        return None
    axes = [
        np.linspace(lower[i] + span * 0.02, upper[i] - span * 0.02, samples)
        for i, span in zip(plane, spans, strict=True)
    ]
    grid_a, grid_b = np.meshgrid(*axes)
    origins = np.zeros((grid_a.size, 3))
    origins[:, plane[0]] = grid_a.ravel()
    origins[:, plane[1]] = grid_b.ravel()
    origins[:, index] = upper[index] + max(spans)

    directions = np.zeros_like(origins)
    directions[:, index] = -1.0

    try:
        points = mesh.ray.intersects_location(origins, directions, multiple_hits=False)[0]
    except Exception:  # noqa: BLE001 - no ray engine, or a mesh it cannot index
        return None
    if len(points) == 0:
        return None

    heights = points[:, index] - lower[index]
    heights = heights[np.isfinite(heights)]
    if len(heights) == 0:
        return None
    return float(heights.min()), float(heights.max())


def profile_at(
    mesh: trimesh.Trimesh, height: float, axis: str = "z", *, repeated: bool = True
) -> Profile | None:
    """Slice across the part's axis and reduce the section to diameters.

    Diameters are taken as the median radius about the section's own centroid, doubled,
    rather than from the section's bounding box. A pouring spout, a handle or a moulded
    graduation panel all widen the bounding box without changing the diameter anyone
    would quote, and the median ignores them because they are a minority of the
    perimeter.
    """
    index = AXIS_INDEX[axis]
    normal = np.zeros(3)
    normal[index] = 1.0
    origin = mesh.bounds.mean(axis=0).copy()
    origin[index] = height

    try:
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            return None
        planar, _ = section.to_2D()
        polygons = _section_polygons(planar)
    except Exception:  # noqa: BLE001 - a degenerate slice is a missing measurement
        return None
    if not polygons:
        return None

    from shapely.geometry import Polygon  # noqa: PLC0415

    population = _dominant_population(polygons) if repeated else [max(polygons, key=_area)]
    representative = population[len(population) // 2]
    outer_ring = np.asarray(representative.exterior.coords, dtype=float)
    outer_diameter = 2.0 * _median_radius(outer_ring)
    outer_area = float(Polygon(representative.exterior).area)
    centres = tuple(
        (float(polygon.centroid.x), float(polygon.centroid.y)) for polygon in population
    )

    interiors = list(representative.interiors)
    if not interiors:
        return Profile(
            height=height,
            outer_diameter=outer_diameter,
            inner_diameter=None,
            wall_thickness=None,
            ring_count=0,
            outer_area=outer_area,
            inner_area=0.0,
            feature_count=len(population),
            feature_centres=centres,
        )

    inner_area = float(sum(Polygon(ring).area for ring in interiors))

    # A repeated feature reaches a section in one of two ways, and only one of them was read.
    # PCR-001's wells stand up as separate tubes, so the slice returns 96 outlines and the
    # population above finds them. A rack or a plate drilled through a solid slab returns one
    # outline with a hole in it per position, so the population was one and the openings went
    # uncounted -- no pitch, no count, on exactly the objects whose pitch is the point of them.
    feature_count = len(population)
    if feature_count == 1 and len(interiors) > 1:
        openings = _dominant_population([Polygon(ring) for ring in interiors])
        feature_count = len(openings)
        centres = tuple(
            (float(opening.centroid.x), float(opening.centroid.y)) for opening in openings
        )
        representative_opening = openings[len(openings) // 2]
        inner_ring = np.asarray(representative_opening.exterior.coords, dtype=float)
    else:
        widest = max(interiors, key=lambda ring: Polygon(ring).area)
        inner_ring = np.asarray(widest.coords, dtype=float)

    inner_diameter = 2.0 * _median_radius(inner_ring)
    lower = inner_ring[:, :2].min(axis=0)
    upper = inner_ring[:, :2].max(axis=0)
    return Profile(
        height=height,
        outer_diameter=outer_diameter,
        inner_diameter=inner_diameter,
        wall_thickness=max(0.0, (outer_diameter - inner_diameter) / 2.0),
        ring_count=len(interiors),
        outer_area=outer_area,
        inner_area=inner_area,
        feature_count=feature_count,
        feature_centres=centres,
        inner_span=float(min(upper - lower)),
    )


def repeated_feature_count(
    mesh: trimesh.Trimesh, *, axis: str = "z", samples: int = 19
) -> tuple[int, float]:
    """How many alike features the object repeats, and where that was read.

    Taken from the section that finds the most of them rather than from one fixed height,
    because repeated features do not all begin and end together: PCR-001's wells run from
    the base plate to the top face and its hinge posts overlap only part of that, and a
    slice taken where the lid ribs cross would count ribs.

    Returns `(0, 0.0)` when no section yields anything countable.
    """
    low, high = axis_bounds(mesh, axis)
    if high <= low:
        return 0, 0.0
    best, at = 0, 0.0
    for step in range(1, samples + 1):
        fraction = step / (samples + 1)
        profile = profile_at(mesh, low + (high - low) * fraction, axis=axis)
        if profile is not None and profile.feature_count > best:
            best, at = profile.feature_count, fraction
    return best, at


def feature_pitch(profile: Profile) -> float | None:
    """Centre-to-centre spacing of a section's repeated features, or None if it has none.

    The median nearest-neighbour distance, which is what a catalogue means by pitch on a
    grid: every well on a 9 mm plate has a neighbour 9 mm away, whether it sits in the
    middle with four of them or in a corner with two. A mean over all pairs would report
    the size of the plate instead.
    """
    if len(profile.feature_centres) < 2:
        return None
    centres = np.asarray(profile.feature_centres, dtype=float)
    # Squared distances between every pair; the diagonal is self and must not win.
    deltas = centres[:, None, :] - centres[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(np.median(nearest))


def feature_pitch_axes(profile: Profile) -> tuple[float, float] | None:
    """The closer and the wider of a grid's two centre-to-centre spacings.

    `feature_pitch` answers with the nearest neighbour, which on a square grid is the whole
    story and on a rectangular one is half of it. A tube rack is quoted with a clear gap of 7 mm
    along a row and 13 mm between rows: one grid, two spacings, and no way to check either from
    a single number.

    Read per axis, because a rack's rows are laid out along the model's axes: the feature centres
    take a handful of distinct positions along each one, and the spacing is the median step
    between neighbouring positions. Which axis is which is not read -- an author may lay the rows
    along x or along y and neither is wrong -- so they come back ordered by size, the same way a
    footprint's two sides do.
    """
    if len(profile.feature_centres) < 2:
        return None
    centres = np.asarray(profile.feature_centres, dtype=float)
    spacings = []
    for axis in (0, 1):
        step = _median_step(centres[:, axis])
        if step is not None:
            spacings.append(step)
    if not spacings:
        return None
    if len(spacings) == 1:
        # Features in a single line have one spacing, and no second one to report.
        return spacings[0], spacings[0]
    return min(spacings), max(spacings)


def _median_step(values: np.ndarray) -> float | None:
    """The typical gap between the distinct positions these coordinates cluster at.

    Positions within a tolerance of each other are one row: a section's centroids do not land on
    exactly equal coordinates, and treating each as its own row would report a spacing of nearly
    zero.
    """
    ordered = np.sort(values)
    span = float(ordered[-1] - ordered[0])
    if span <= 0.0:
        return None
    tolerance = span * 0.02
    positions = [float(ordered[0])]
    for value in ordered[1:]:
        if value - positions[-1] > tolerance:
            positions.append(float(value))
    if len(positions) < 2:
        return None
    return float(np.median(np.diff(positions)))


def _area(polygon: Any) -> float:
    return float(polygon.area)


def _dominant_population(polygons: list[Any]) -> list[Any]:
    """The group of alike shapes this section is mostly made of, smallest area first.

    Taking the single largest polygon was wrong on any object built from repeated
    features. PCR-001's rack cuts 98 shapes at mid-height: 96 wells, each a 5.2 mm ring
    around a 4 mm bore, and two solid 5.2 mm hinge posts the author added for decoration.
    A solid disc has more area than a ring of the same diameter, so the largest polygon
    was always a hinge post — which has no interior, so the bore came back unmeasurable
    and a rack whose wells are 23% too narrow to hold a tube was never contradicted.

    Grouping by area and taking the biggest group asks the right question: what is this
    section mostly? Ninety-six wells outvote two posts. A beaker sections into one wall
    and perhaps a handle, every group has one member, and the tie falls to the larger
    area — the wall, which is what it always was.
    """
    ranked = sorted(polygons, key=lambda polygon: polygon.area)
    groups: list[list[Any]] = []
    for polygon in ranked:
        # Same size to within 5% is the same kind of feature. Tessellation of a curved
        # wall varies a little between one instance and the next, and identical wells
        # must not land in 96 groups of one.
        if groups and polygon.area <= groups[-1][-1].area * 1.05:
            groups[-1].append(polygon)
        else:
            groups.append([polygon])
    return max(groups, key=lambda group: (len(group), group[-1].area))


def _section_polygons(planar: Any) -> list[Any]:
    """The material region of a cross-section, as shapely polygons.

    trimesh's own `polygons_full` does this and is used first because it is faster and
    right whenever it works. It stops working on exactly the meshes this project produces:
    a generated asset is a union of primitives that were never merged, so a wall and the
    graduation panel glued to it cross the section plane as two overlapping rings, and
    building a `Polygon(shell, holes)` out of rings that overlap raises rather than
    returning a worse answer.

    The fallback computes the same thing the robust way. Rings at even containment depth
    are material and rings at odd depth are holes — the standard even-odd rule — and
    unioning each group before subtracting lets rings overlap freely, which is the whole
    problem. Every ring is passed through `buffer(0)` first, which is shapely's idiom for
    repairing a self-touching ring into a valid polygon.

    Without this a beaker with graduations reports its cavity as ending at the bottom of
    the graduation panel, because the sections just above it come back empty and the
    overflow scan believes it has found the rim.
    """
    from shapely.geometry import MultiPolygon, Polygon  # noqa: PLC0415
    from shapely.ops import unary_union  # noqa: PLC0415

    try:
        polygons = list(planar.polygons_full)
        if polygons:
            return polygons
    except Exception:  # noqa: BLE001 - fall through to the robust path
        pass

    rings: list[Any] = []
    for entity in planar.entities:
        # Closed loops only. An unmerged mesh sections into a mix of closed loops — the
        # real boundaries — and open arcs, which are where two coincident surfaces met the
        # plane. An open arc has no enclosed area, and closing it by joining its endpoints
        # invents a shape that is not in the geometry; at best it is a sliver and at worst
        # it swallows the bore and the section reads as solid.
        if not entity.closed:
            continue
        points = entity.discrete(planar.vertices)
        if len(points) < 4:
            continue
        candidate = Polygon(points).buffer(0)
        if candidate.is_empty or candidate.area <= 0:
            continue
        rings.append(candidate)
    if not rings:
        return []

    solids, holes = [], []
    for index, ring in enumerate(rings):
        depth = sum(
            1
            for other_index, other in enumerate(rings)
            if other_index != index and other.area > ring.area and other.contains(ring)
        )
        (holes if depth % 2 else solids).append(ring)
    if not solids:
        return []

    region = unary_union(solids)
    if holes:
        region = region.difference(unary_union(holes))
    if region.is_empty:
        return []
    return list(region.geoms) if isinstance(region, MultiPolygon) else [region]


def _median_radius(ring: np.ndarray) -> float:
    """Median distance from a closed ring to its own centroid."""
    points = ring[:, :2] if ring.shape[1] > 2 else ring
    centre = points.mean(axis=0)
    return float(np.median(np.linalg.norm(points - centre, axis=1)))


def representative_profile(
    mesh: trimesh.Trimesh, *, axis: str = "z", fraction: float = 0.5
) -> Profile | None:
    """The section at a stated height up the part, retried if that height is degenerate.

    A slice can land exactly on a floor, a lip or a seam, where it degenerates into
    something with no interior. Nudging is not cheating: the datasheet says "mid-height
    wall", not "the wall at exactly 47.5 mm".
    """
    low, high = axis_bounds(mesh, axis)
    span = high - low
    if span <= 0.0:
        return None
    for offset in (0.0, 0.03, -0.03, 0.07, -0.07, 0.12):
        where = fraction + offset
        if not 0.02 < where < 0.98:
            continue
        profile = profile_at(mesh, low + span * where, axis=axis)
        if profile is not None and profile.outer_diameter > 0.0:
            return profile
    return None


def outer_diameter(mesh: trimesh.Trimesh, *, axis: str = "z") -> float | None:
    """The diameter of the body, taken as the median of several heights.

    One section can land on a base flare or a rolled rim. The median over the middle of
    the part is the number a catalogue means by "outside diameter".
    """
    diameters = [
        profile.outer_diameter
        for fraction in (0.3, 0.4, 0.5, 0.6, 0.7)
        if (profile := representative_profile(mesh, axis=axis, fraction=fraction)) is not None
    ]
    return float(np.median(diameters)) if diameters else None


def cavity_volume(
    mesh: trimesh.Trimesh, *, axis: str = "z", samples: int = 240
) -> CavityMeasurement | None:
    """Integrate the enclosed cross-sectional area up to the lowest overflow edge.

    The overflow edge is not declared anywhere; it is found. Scanning upwards, a height
    is part of the cavity as long as its section still has an interior ring — that is
    what "walled in on every side" means. The first height at which the ring opens is
    where the contents would run out, and it lands on a pouring spout exactly as
    readily as on a plain rim.

    Runs twice at different resolutions and reports how far the two answers are apart,
    because an integral quoted without a convergence figure is a guess.
    """
    coarse = _integrate_cavity(mesh, axis=axis, samples=samples)
    fine = _integrate_cavity(mesh, axis=axis, samples=samples * 2)
    if fine is None or coarse is None:
        return None

    convergence = 0.0
    if fine.volume_m3 > 0.0:
        convergence = abs(fine.volume_m3 - coarse.volume_m3) / fine.volume_m3
    fine.convergence_rel = float(convergence)
    return fine


def _integrate_cavity(
    mesh: trimesh.Trimesh, *, axis: str, samples: int
) -> CavityMeasurement | None:
    low, high = axis_bounds(mesh, axis)
    span = high - low
    if span <= 0.0 or samples < 8:
        return None

    step = span / samples
    # Sample at cell centres. A slice coplanar with the floor or the rim is degenerate
    # and reports nothing useful.
    heights = [low + (index + 0.5) * step for index in range(samples)]
    # `repeated=False` keeps this integral about the object's own interior. The population
    # rule is right for reading one well's bore and wrong here: a bottle with four moulded
    # grip ribs sections into four solid shapes and one walled one, and the ribs would
    # outvote the wall — leaving a bottle with a perfectly good interior reported as solid.
    profiles = [profile_at(mesh, height, axis=axis, repeated=False) for height in heights]
    enclosed = [
        profile is not None and profile.is_enclosed for profile in profiles
    ]

    # Where the cavity ends. A real overflow edge — a rim, a pouring spout notch — opens
    # the wall and it stays open all the way to the top of the object. A decorative band,
    # a moulded panel, or two surfaces that happen to be coincident open it for a
    # millimetre or two and then it closes again, and those are not places liquid runs
    # out. So the cavity ends at the start of the *final* run of open slices, and every
    # gap before that is bridged.
    #
    # Reading it this way rather than stopping at the first gap is worth 250 of a
    # beaker's 296 mL, and the same code answers the agent's in-loop question, so getting
    # it wrong sent the agent chasing a cavity defect that was never there.
    last_closed = max((i for i, ok in enumerate(enclosed) if ok), default=None)
    if last_closed is None:
        return None
    first_closed = next(i for i, ok in enumerate(enclosed) if ok)

    areas: list[float] = []
    diameters: list[float] = []
    floor = heights[first_closed]
    overflow = heights[last_closed]
    carried = 0.0
    for index in range(first_closed, last_closed + 1):
        profile = profiles[index]
        if profile is not None and profile.is_enclosed:
            carried = profile.inner_area
            if profile.inner_diameter:
                diameters.append(profile.inner_diameter)
        areas.append(carried)
    centre = mesh.bounds.mean(axis=0)
    horizontal = [i for i in range(3) if i != AXIS_INDEX[axis]]
    return CavityMeasurement(
        volume_m3=float(sum(areas) * step),
        overflow_height=float(overflow),
        floor_height=float(floor),
        inner_diameter=float(np.median(diameters)) if diameters else 0.0,
        centre_xy=(float(centre[horizontal[0]]), float(centre[horizontal[1]])),
        convergence_rel=0.0,
        samples=samples,
    )


def read_topology(model: mujoco.MjModel) -> Topology:
    """Names and parentage, so a required-component list can be checked against them."""
    topology = Topology()
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body#{body}"
        if body == 0:
            continue
        topology.bodies.append(name)
        parent = int(model.body_parentid[body])
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) or "world"
        topology.parents[name] = parent_name
        topology.geoms_per_body[name] = int(model.body_geomnum[body])

    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or f"joint#{joint}"
        body = int(model.jnt_bodyid[joint])
        topology.joints[name] = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body#{body}"
        )

    for site in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if name:
            topology.sites.append(name)

    for geom in range(model.ngeom):
        if int(model.geom_type[geom]) == mujoco.mjtGeom.mjGEOM_MESH:
            topology.mesh_geoms += 1
        else:
            topology.primitive_geoms += 1
    return topology
