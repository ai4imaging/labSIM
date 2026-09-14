"""Can this part actually be made?

Every check here is deterministic and geometric. No model is consulted, because whether a
1.1 mm wall survives an FDM nozzle is not a matter of opinion, and a part that passes these
checks is one you can send to a printer or a shop without looking at it first.

The checks are the ones that catch real failures in lab fixtures, in the order they bite:

    watertight / single body   a mesh that is not a solid cannot be sliced at all
    wall thickness             the most common failure, and invisible in a render
    overhang                   droops or needs support inside a pocket where it cannot be removed
    feature size               ribs and pins below the process's resolution simply do not appear
    build volume               a part nobody can fit on the machine
    fit clearance              a hole cut to nominal will not accept its shaft
    mass                       an adapter heavy enough to matter to the arm's payload

Wall thickness is measured by shooting a ray inward from each of a few thousand surface
samples and taking the distance to the next surface it hits. That is the standard ray-based
thickness estimate: it is exact for slabs and conservative for curved sections, and unlike a
medial-axis method it needs nothing beyond ray intersection.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from amx.codesign.materials import Material
from amx.codesign.parts import PartGeometry
from amx.report import Finding, RepairTarget, Report, Severity

THICKNESS_SAMPLES = 4000
THICKNESS_EPSILON = 1e-5
THICKNESS_OPPOSITION = -math.cos(math.pi / 4.0)
"""How nearly the far surface must face back at the near one for the pair to count as a wall.

Without this, every hole whose axis is not perpendicular to the face it breaks through
reports a wall of nearly zero: right at the lip, the inward ray immediately grazes into the
hole and returns the millimetre or so to its wall. That edge is a corner, not a wall, and
the two surfaces meeting at it are nowhere near parallel — so requiring the far face to lie
within 45 deg of antiparallel discards exactly those readings and keeps the real ones.
"""
OVERHANG_AREA_TOLERANCE = 0.02
"""Fraction of total surface area allowed to be unsupported before it is reported.

A few stray facets on a curved underside are a meshing artefact, not a manufacturing
problem; a genuine unsupported face is a large contiguous patch.
"""

MAX_TOOL_MASS_KG = 1.5
"""Above this, an arm-mounted part starts to matter to a UR5e's 5 kg payload and its
dynamics, not just to whether it fits.
"""


def check_part(
    geometry: PartGeometry, *, mounted_on_arm: bool = False, build_direction: str = "z"
) -> Report:
    """Run every manufacturability check against one built part."""
    report = Report(kind="dfm", subject=f"{geometry.part_id} ({geometry.template})")
    target = RepairTarget.TOOL if mounted_on_arm else RepairTarget.FIXTURE
    mesh = geometry.mesh
    spec = geometry.material

    report.findings.extend(_solidity(geometry, target))
    if not mesh.is_watertight:
        report.notes.append(
            "the mesh is not a closed solid, so wall thickness and overhang were not measured"
        )
        return report

    report.findings.append(_wall_thickness(geometry, spec, target))
    report.findings.append(_overhang(geometry, spec, target, build_direction))
    report.findings.append(_feature_size(geometry, spec, target))
    report.findings.append(_build_volume(geometry, spec, target))
    report.findings.extend(_clearances(geometry, spec, target))
    if mounted_on_arm:
        report.findings.append(_tool_mass(geometry))
    report.notes.append(
        f"{spec.label}: {geometry.mesh.volume * 1e6:.2f} cm3, {geometry.mass_kg * 1000:.1f} g"
    )
    if spec.notes:
        report.notes.append(spec.notes)
    return report


def _solidity(geometry: PartGeometry, target: RepairTarget) -> list[Finding]:
    mesh = geometry.mesh
    findings: list[Finding] = []

    if not mesh.is_watertight:
        findings.append(
            Finding(
                code="D-SOLID-OPEN",
                severity=Severity.FAILURE,
                subject=geometry.part_id,
                summary=(
                    "the mesh is not watertight, so no slicer will accept it. The parameters "
                    "have produced features that meet exactly on a tangent plane instead of "
                    "overlapping; nudge the dimensions that control that interface."
                ),
                repair_target=target,
                metrics={"euler_number": float(mesh.euler_number)},
            )
        )
    if mesh.body_count > 1:
        findings.append(
            Finding(
                code="D-SOLID-SPLIT",
                severity=Severity.FAILURE,
                subject=geometry.part_id,
                summary=(
                    f"the part is {mesh.body_count} disconnected pieces rather than one. At "
                    "these dimensions its features no longer touch: shorten a reach, thicken a "
                    "web, or reduce a tilt."
                ),
                repair_target=target,
                metrics={"bodies": float(mesh.body_count)},
            )
        )
    if mesh.volume <= 0.0:
        findings.append(
            Finding(
                code="D-SOLID-INVERTED",
                severity=Severity.FAILURE,
                subject=geometry.part_id,
                summary="the mesh encloses no positive volume; its faces are wound inside out.",
                repair_target=target,
                metrics={"volume_m3": float(mesh.volume)},
            )
        )
    if not findings:
        findings.append(
            Finding(
                code="D-SOLID",
                severity=Severity.INFO,
                subject=geometry.part_id,
                summary=f"one closed solid, {mesh.volume * 1e6:.2f} cm3.",
                metrics={"volume_m3": float(mesh.volume)},
            )
        )
    return findings


def _wall_thickness(
    geometry: PartGeometry, spec: Material, target: RepairTarget
) -> Finding:
    thickness = measure_wall_thickness(geometry.mesh)
    if thickness is None:
        return Finding(
            code="D-WALL",
            severity=Severity.WARNING,
            subject=geometry.part_id,
            summary="no inward ray hit another surface, so wall thickness could not be measured.",
        )
    if thickness >= spec.minimum_wall_m:
        return Finding(
            code="D-WALL",
            severity=Severity.INFO,
            subject=geometry.part_id,
            summary=(
                f"thinnest section is {thickness * 1000:.2f} mm, above the "
                f"{spec.minimum_wall_m * 1000:.2f} mm minimum for {spec.label}."
            ),
            metrics={"minimum_wall_m": thickness},
            thresholds={"required_wall_m": spec.minimum_wall_m},
        )
    return Finding(
        code="D-WALL",
        severity=Severity.FAILURE,
        subject=geometry.part_id,
        summary=(
            f"thinnest section is {thickness * 1000:.2f} mm, below the "
            f"{spec.minimum_wall_m * 1000:.2f} mm {spec.label} can produce. It will come off "
            "the machine translucent and split under the first load. Increase the wall or web "
            "thickness, or move to a process with a finer minimum."
        ),
        repair_target=target,
        metrics={"minimum_wall_m": thickness},
        thresholds={"required_wall_m": spec.minimum_wall_m},
    )


def measure_wall_thickness(
    mesh: trimesh.Trimesh, samples: int = THICKNESS_SAMPLES
) -> float | None:
    """Ray-based minimum wall thickness.

    Samples the surface, steps just inside along the inward normal, and measures the
    distance to the next surface along that ray, keeping only the rays that land on a
    surface facing back at the one they left (see `THICKNESS_OPPOSITION`). Sampling is
    seeded so a given mesh always yields the same number: a manufacturability check that
    fluctuates between runs would make the repair loop chase noise.
    """
    points, face_ids = trimesh.sample.sample_surface_even(mesh, samples, seed=0)
    if len(points) == 0:
        return None
    normals = mesh.face_normals[face_ids]
    origins = points - normals * THICKNESS_EPSILON

    locations, ray_ids, hit_faces = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=-normals, multiple_hits=False
    )
    if len(ray_ids) == 0:
        return None
    distances = np.linalg.norm(locations - origins[ray_ids], axis=1)
    opposition = np.einsum("ij,ij->i", mesh.face_normals[hit_faces], normals[ray_ids])
    usable = distances[
        (distances > THICKNESS_EPSILON * 10.0) & (opposition <= THICKNESS_OPPOSITION)
    ]
    if usable.size == 0:
        return None
    return float(usable.min())


def _overhang(
    geometry: PartGeometry, spec: Material, target: RepairTarget, build_direction: str
) -> Finding:
    axis = {"x": 0, "y": 1, "z": 2}[build_direction]
    mesh = geometry.mesh
    # A downward-facing face is self-supporting while it makes at least
    # (90 deg - maximum_overhang) with the build plate. Expressed on the normal, that is a
    # limit on how much of the normal points straight down.
    limit = math.cos(spec.maximum_overhang_rad)
    downward = -mesh.face_normals[:, axis]
    unsupported = downward > limit
    area = float(mesh.area_faces[unsupported].sum())
    fraction = area / float(mesh.area) if mesh.area > 0 else 0.0

    if spec.maximum_overhang_rad >= math.pi / 2.0 - 1e-6:
        return Finding(
            code="D-OVERHANG",
            severity=Severity.INFO,
            subject=geometry.part_id,
            summary=f"{spec.label} is self-supporting, so overhangs are unconstrained.",
        )
    if fraction <= OVERHANG_AREA_TOLERANCE:
        return Finding(
            code="D-OVERHANG",
            severity=Severity.INFO,
            subject=geometry.part_id,
            summary=(
                f"{fraction * 100:.1f}% of the surface is unsupported in the {build_direction} "
                "build direction, which needs no support material."
            ),
            metrics={"unsupported_area_fraction": fraction},
        )
    return Finding(
        code="D-OVERHANG",
        severity=Severity.WARNING,
        subject=geometry.part_id,
        summary=(
            f"{fraction * 100:.1f}% of the surface ({area * 1e4:.1f} cm2) overhangs by more than "
            f"{math.degrees(spec.maximum_overhang_rad):.0f} deg in the {build_direction} build "
            "direction. It will print, but with support material that has to be removed — check "
            "that none of it is inside a pocket you cannot reach, or build along another axis."
        ),
        repair_target=target,
        metrics={"unsupported_area_fraction": fraction, "unsupported_area_m2": area},
        thresholds={"tolerated_fraction": OVERHANG_AREA_TOLERANCE},
    )


def _feature_size(
    geometry: PartGeometry, spec: Material, target: RepairTarget
) -> Finding:
    """The smallest dimension the part declares, judged against the process resolution.

    Read off the collision primitives rather than the mesh, because they are the part's own
    statement of what its features are, and a mesh's smallest triangle says nothing.
    """
    if not geometry.collision:
        return Finding(
            code="D-FEATURE",
            severity=Severity.INFO,
            subject=geometry.part_id,
            summary="the part declares no primitives, so feature size was not checked.",
        )
    smallest = min(p.minimum_extent() for p in geometry.collision)
    if smallest >= spec.minimum_feature_m:
        return Finding(
            code="D-FEATURE",
            severity=Severity.INFO,
            subject=geometry.part_id,
            summary=(
                f"smallest declared feature is {smallest * 1000:.2f} mm, above the "
                f"{spec.minimum_feature_m * 1000:.2f} mm resolution of {spec.label}."
            ),
            metrics={"minimum_feature_m": smallest},
            thresholds={"required_feature_m": spec.minimum_feature_m},
        )
    return Finding(
        code="D-FEATURE",
        severity=Severity.FAILURE,
        subject=geometry.part_id,
        summary=(
            f"smallest declared feature is {smallest * 1000:.2f} mm, below the "
            f"{spec.minimum_feature_m * 1000:.2f} mm {spec.label} can resolve. It will not "
            "appear on the finished part, so whatever it was for will not work."
        ),
        repair_target=target,
        metrics={"minimum_feature_m": smallest},
        thresholds={"required_feature_m": spec.minimum_feature_m},
    )


def _build_volume(
    geometry: PartGeometry, spec: Material, target: RepairTarget
) -> Finding:
    extents = geometry.extents_m
    if spec.fits_build_volume(extents):
        return Finding(
            code="D-ENVELOPE",
            severity=Severity.INFO,
            subject=geometry.part_id,
            summary=(
                f"{extents[0] * 1000:.0f} x {extents[1] * 1000:.0f} x {extents[2] * 1000:.0f} mm "
                f"fits the {spec.label} build volume."
            ),
        )
    return Finding(
        code="D-ENVELOPE",
        severity=Severity.FAILURE,
        subject=geometry.part_id,
        summary=(
            f"{extents[0] * 1000:.0f} x {extents[1] * 1000:.0f} x {extents[2] * 1000:.0f} mm does "
            f"not fit the {spec.label} build volume of "
            f"{' x '.join(f'{v * 1000:.0f}' for v in spec.build_volume_m)} mm, in any "
            "orientation. Reduce the largest dimension or split the part."
        ),
        repair_target=target,
        metrics={"extent_x_m": extents[0], "extent_y_m": extents[1], "extent_z_m": extents[2]},
    )


def _clearances(
    geometry: PartGeometry, spec: Material, target: RepairTarget
) -> list[Finding]:
    """Each fit the part claims, against what the process actually holds."""
    findings: list[Finding] = []
    for label, value in geometry.stated_clearances.items():
        if value >= spec.hole_allowance_m:
            findings.append(
                Finding(
                    code="D-FIT",
                    severity=Severity.INFO,
                    subject=f"{geometry.part_id}/{label}",
                    summary=(
                        f"{label} clearance is {value * 1000:.2f} mm, at or above the "
                        f"{spec.hole_allowance_m * 1000:.2f} mm {spec.label} needs."
                    ),
                    metrics={"clearance_m": value},
                    thresholds={"required_clearance_m": spec.hole_allowance_m},
                )
            )
            continue
        findings.append(
            Finding(
                code="D-FIT",
                severity=Severity.FAILURE,
                subject=f"{geometry.part_id}/{label}",
                summary=(
                    f"{label} clearance is {value * 1000:.2f} mm, under the "
                    f"{spec.hole_allowance_m * 1000:.2f} mm {spec.label} shrinks a feature by. "
                    "The mating part will not go in without being forced, and forcing it is how "
                    "a fixture ends up holding its contents at an angle."
                ),
                repair_target=target,
                metrics={"clearance_m": value},
                thresholds={"required_clearance_m": spec.hole_allowance_m},
            )
        )
    return findings


def _tool_mass(geometry: PartGeometry) -> Finding:
    mass = geometry.mass_kg
    if mass <= MAX_TOOL_MASS_KG:
        return Finding(
            code="D-TOOL-MASS",
            severity=Severity.INFO,
            subject=geometry.part_id,
            summary=f"{mass * 1000:.0f} g on the flange, well inside the arm's payload.",
            metrics={"mass_kg": mass},
        )
    return Finding(
        code="D-TOOL-MASS",
        severity=Severity.FAILURE,
        subject=geometry.part_id,
        summary=(
            f"{mass:.2f} kg hanging off the flange exceeds the {MAX_TOOL_MASS_KG:.1f} kg this "
            "pipeline allows for tooling. Hollow it out, shorten it, or make it in a lighter "
            "material."
        ),
        repair_target=RepairTarget.TOOL,
        metrics={"mass_kg": mass},
        thresholds={"maximum_mass_kg": MAX_TOOL_MASS_KG},
    )
