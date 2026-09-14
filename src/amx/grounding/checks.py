"""The checks themselves: measure the asset, compare against the spec, report findings.

Grouped by what they cost and what they can be trusted to say:

* `check_dimensions` is a caliper. Deterministic, fast, and the one an author can act
  on most directly.
* `check_topology` reads the body tree and integrates the cavity. Deterministic, and
  slower only because sectioning a mesh two hundred times is not free.
* `check_protocol` runs physics: does it stand up, can a probe get in and stay in, does
  it survive being tipped and put back.
* `check_visual` renders it and asks a model. The least reliable of the four, so it is
  advisory unless a caller says otherwise.

Every finding carries `repair_target=ASSET`, because at this point in the pipeline the
asset is the only thing there is to repair.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field

from amx.grounding import mesh as gmesh
from amx.grounding.build import GroundedAsset
from amx.grounding.physics import match_bodies
from amx.grounding.scene import (
    GROUND_GEOM,
    STUDIO_SURFACE_MATERIAL,
    SceneError,
    settle,
    stage,
    tilt_angle,
    worst_penetration,
)
from amx.grounding.spec import DimensionTarget, GroundingSpec, OperationTarget
from amx.llm import LlmClient, LlmUnavailable
from amx.report import Finding, RepairTarget, Report, Severity, interval_finding

VIEW_ANGLES: dict[str, tuple[float, float]] = {
    "front": (90.0, -10.0),
    "back": (-90.0, -10.0),
    "left": (0.0, -10.0),
    "right": (180.0, -10.0),
    "side": (0.0, -10.0),
    "top": (90.0, -89.0),
    "bottom": (90.0, 89.0),
    "iso": (45.0, -25.0),
    "front_left_45deg": (45.0, -25.0),
}


# --------------------------------------------------------------------------- #
# dimensions
# --------------------------------------------------------------------------- #


def check_dimensions(asset: GroundedAsset, spec: GroundingSpec) -> Report:
    """Measure each stated dimension on the compiled geometry."""
    report = Report(kind="grounding-dimensions", subject=asset.asset_id)
    if not spec.dimensions and spec.total_mass_kg is None:
        report.notes.append("no dimensional targets were stated, so nothing was measured")
        return report

    try:
        staged = stage(asset.mjcf_path, ground=False, free=False)
    except SceneError as error:
        report.findings.append(_unmeasurable(asset.asset_id, f"the asset will not load: {error}"))
        return report

    try:
        world = gmesh.world_mesh(staged.model, staged.data)
    except gmesh.GeometryUnavailable as error:
        report.findings.append(_unmeasurable(asset.asset_id, str(error)))
        return report

    parts = _part_meshes(staged, {target.part for target in spec.dimensions if target.part})
    for target in spec.dimensions:
        if not target.part:
            mesh = _without_part(staged, target.part_excluded) if target.part_excluded else world
            report.findings.append(_dimension_finding(asset.asset_id, mesh or world, target))
            continue
        mesh = parts.get(target.part)
        if mesh is None:
            report.findings.append(_unnamed_part(asset.asset_id, target, staged))
            continue
        report.findings.append(_dimension_finding(asset.asset_id, mesh, target))

    if spec.total_mass_kg is not None:
        total = float(staged.model.body_mass.sum())
        finding = interval_finding(
            code="G-MASS",
            subject=asset.asset_id,
            measured=total,
            interval=spec.total_mass_kg,
            unit="kg",
            repair_target=RepairTarget.ASSET,
            what="total mass",
        )
        if finding:
            report.findings.append(finding)
    return report


def _part_meshes(staged, wanted: set[str]) -> dict[str, trimesh.Trimesh]:
    """Resolve each named part to just its own geometry, in world coordinates.

    Measuring a part on the assembled mesh reads whatever else happens to be widest, which
    is how one `diameter_outer` came to be shared by a 57 mm bowl, a 10 mm stem and a 48 mm
    support disc. `gmesh.body_mesh` has been able to do this all along; the dimension check
    simply never asked it to.
    """
    if not wanted:
        return {}
    topology = gmesh.read_topology(staged.model)
    meshes: dict[str, trimesh.Trimesh] = {}
    for part in wanted:
        body = match_body(part, topology.bodies)
        if body is None:
            continue
        body_id = mujoco.mj_name2id(staged.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id < 0:
            continue
        mesh = gmesh.body_mesh(staged.model, staged.data, body_id)
        if mesh is not None and not mesh.is_empty:
            meshes[part] = mesh
    return meshes


def _without_part(staged, excluded: str) -> trimesh.Trimesh | None:
    """The assembly with one part left out, or None if that leaves nothing to measure.

    None also covers the part not being there to leave out, which is how a location that reads
    "without treating the surface" ends up measured as the whole object: the caller falls back.
    """
    topology = gmesh.read_topology(staged.model)
    unwanted = match_body(excluded, topology.bodies)
    if unwanted is None:
        return None
    keep = []
    for name in topology.bodies:
        if name == unwanted:
            continue
        body_id = mujoco.mj_name2id(staged.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        mesh = gmesh.body_mesh(staged.model, staged.data, body_id)
        if mesh is not None and not mesh.is_empty:
            keep.append(mesh)
    if not keep:
        return None
    return trimesh.util.concatenate(keep)


def _unnamed_part(asset_id: str, target: DimensionTarget, staged) -> Finding:
    """A part-scoped target whose part is not in the tree.

    A warning rather than a failure: the measurement was not taken, so there is no evidence
    the dimension is wrong, and treating an unrecognised name as a wrong size would create
    exactly the unfixable refusal the part field exists to remove. The missing part itself
    is `check_topology`'s to report, and it does, against the stated component list.
    """
    bodies = gmesh.read_topology(staged.model).bodies
    return Finding(
        code="G-DIM-UNSCOPED",
        severity=Severity.WARNING,
        subject=f"{asset_id}/{target.name}",
        summary=(
            f"{target.name} is stated for part {target.part!r}, which has no matching body, "
            f"so it was not measured. Bodies present: {', '.join(bodies) or '(none)'}."
        ),
        repair_target=RepairTarget.ASSET,
        detail={"part": target.part, "kind": target.kind},
    )


def measure_dimension(world: trimesh.Trimesh, target: DimensionTarget) -> float | None:
    """Turn geometry into the one number this target is about, or None if it cannot be read."""
    extents = gmesh.extents(world)
    kind = target.kind
    if kind == "extent_x":
        return float(extents[0])
    if kind == "extent_y":
        return float(extents[1])
    if kind == "extent_z":
        return float(extents[2])
    if kind == "extent_z_min":
        # The lowest the top gets, for the shorter of the two heights a stepped or sloped
        # object is quoted with. Equal to `extent_z` when the top is flat.
        span = gmesh.top_surface_range(world)
        return float(span[0]) if span else None
    if kind == "interior_extent_z":
        # How deep the enclosed space is, which is not how tall the object is.
        cavity = gmesh.cavity_volume(world)
        if cavity is None:
            return None
        depth = cavity.overflow_height - cavity.floor_height
        return float(depth) if depth > 0.0 else None
    if kind == "extent_max":
        # "Overall length" on a hand tool is the longest dimension of it, whichever way
        # the author happened to lay the part out.
        return float(max(extents))
    if kind == "footprint_max":
        return float(max(extents[0], extents[1]))
    if kind == "footprint_min":
        # The shorter of the two bench-plane extents. Which one a datasheet calls the
        # width depends on how the author laid the part out, and the asset is not wrong
        # for having put the long side along y.
        return float(min(extents[0], extents[1]))
    if kind == "diameter_outer":
        return gmesh.outer_diameter(world, axis=target.axis)

    fraction = target.height_fraction() or 0.5
    profile = gmesh.representative_profile(world, axis=target.axis, fraction=fraction)
    if profile is None:
        return None
    if kind == "diameter_inner":
        return profile.inner_diameter
    if kind == "width_inner":
        return profile.inner_span
    if kind == "wall_thickness":
        return profile.wall_thickness
    if kind == "pitch":
        return gmesh.feature_pitch(profile)
    if kind in {"feature_gap_min", "feature_gap_max"}:
        # The material between neighbours: their spacing less the opening itself.
        spacings = gmesh.feature_pitch_axes(profile)
        if spacings is None or profile.inner_span is None:
            return None
        spacing = spacings[0] if kind == "feature_gap_min" else spacings[1]
        gap = spacing - profile.inner_span
        return float(gap) if gap > 0.0 else None
    return None


def _dimension_finding(
    asset_id: str, world: trimesh.Trimesh, target: DimensionTarget
) -> Finding:
    measured = measure_dimension(world, target)
    subject = f"{asset_id}/{target.name}"
    if measured is None or measured <= 0.0:
        return Finding(
            code="G-DIM-UNMEASURABLE",
            severity=Severity.WARNING,
            subject=subject,
            summary=(
                f"{target.name} ({target.kind}) could not be measured on this geometry. "
                f"For an inner diameter or a wall thickness that usually means the part "
                f"has no enclosed interior: a cross-section through it is solid, not a ring."
            ),
            repair_target=RepairTarget.ASSET,
            detail={"measurement_location": target.measurement_location},
        )

    error = abs(measured - target.value_m) / abs(target.value_m)
    metrics = {
        "measured_m": measured,
        "target_m": target.value_m,
        "relative_error": error,
    }
    thresholds = {"tolerance_rel": target.tolerance_rel, "hard_fail_rel": target.hard_fail_rel}

    if error <= target.tolerance_rel:
        return Finding(
            code="G-DIM",
            severity=Severity.INFO,
            subject=subject,
            summary=(
                f"{target.name} measures {measured * 1000:.2f} mm against a target of "
                f"{target.value_m * 1000:.2f} mm ({error * 100:.1f}% error)."
            ),
            metrics=metrics,
            thresholds=thresholds,
        )

    code = "G-DIM-HARD" if error > target.hard_fail_rel else "G-DIM"
    return Finding(
        code=code,
        severity=Severity.FAILURE,
        subject=subject,
        summary=(
            f"{target.name} measures {measured * 1000:.2f} mm but should be "
            f"{target.value_m * 1000:.2f} mm — {error * 100:.1f}% off, against a "
            f"{target.tolerance_rel * 100:.0f}% tolerance. Measured as {target.kind}"
            + (f" on part {target.part!r}" if target.part else "")
            + (f" at {target.measurement_location}." if target.measurement_location else ".")
        ),
        repair_target=RepairTarget.ASSET,
        metrics=metrics,
        thresholds=thresholds,
    )


# --------------------------------------------------------------------------- #
# topology and cavity
# --------------------------------------------------------------------------- #


def check_topology(asset: GroundedAsset, spec: GroundingSpec) -> Report:
    """Check the required components exist and the cavity holds what it should."""
    report = Report(kind="grounding-topology", subject=asset.asset_id)
    try:
        staged = stage(asset.mjcf_path, ground=False, free=False)
    except SceneError as error:
        report.findings.append(_unmeasurable(asset.asset_id, f"the asset will not load: {error}"))
        return report

    topology = gmesh.read_topology(staged.model)
    report.notes.append(
        f"model tree: {len(topology.bodies)} bodies, {len(topology.joints)} joints, "
        f"{topology.mesh_geoms} mesh geoms, {topology.primitive_geoms} primitive geoms"
    )

    cavity_required = any(c.kind == "cavity" for c in spec.components) or spec.cavity is not None
    for component in spec.components:
        finding = _component_finding(asset.asset_id, component, topology)
        if finding is not None:
            report.findings.append(finding)

    if not cavity_required:
        return report

    try:
        world = gmesh.world_mesh(staged.model, staged.data)
    except gmesh.GeometryUnavailable as error:
        report.findings.append(_unmeasurable(asset.asset_id, str(error)))
        return report

    report.findings.extend(_repeated_cavity_findings(asset.asset_id, world, spec))

    measurement = gmesh.cavity_volume(world)
    if measurement is None:
        report.findings.append(
            Finding(
                code="G-CAVITY-MISSING",
                severity=Severity.FAILURE,
                subject=asset.asset_id,
                summary=(
                    "no enclosed cavity was found: no horizontal cross-section of this "
                    "object comes out as a ring, so it is a solid shape rather than a "
                    "vessel. The interior has to be subtracted from the body."
                ),
                repair_target=RepairTarget.ASSET,
            )
        )
        return report

    metrics = {
        "volume_ml": measurement.volume_ml,
        "overflow_height_m": measurement.overflow_height,
        "floor_height_m": measurement.floor_height,
        "convergence_rel": measurement.convergence_rel,
    }
    if spec.cavity is None:
        report.findings.append(
            Finding(
                code="G-CAVITY",
                severity=Severity.INFO,
                subject=asset.asset_id,
                summary=(
                    f"the largest enclosed cavity holds {measurement.volume_ml:.1f} mL to its "
                    f"overflow edge at {measurement.overflow_height * 1000:.1f} mm. No capacity "
                    f"is stated for this asset, so the number is reported and not graded"
                    + (
                        f" — and it is not the {repeated[0].name!r} this asset is required to "
                        f"repeat {repeated[0].quantity} times, which is counted separately."
                        if (repeated := [c for c in spec.components
                                         if c.kind == "cavity" and c.quantity > 1])
                        else "."
                    )
                ),
                metrics=metrics,
            )
        )
        return report

    if not measurement.converged:
        report.findings.append(
            Finding(
                code="G-CAVITY-CONVERGENCE",
                severity=Severity.WARNING,
                subject=asset.asset_id,
                summary=(
                    f"the cavity volume has not converged: two resolutions disagree by "
                    f"{measurement.convergence_rel * 100:.2f}%. The value below is the finer one."
                ),
                metrics=metrics,
                thresholds={"convergence_rel": spec.cavity.convergence_tolerance_rel},
            )
        )

    wanted = spec.cavity.minimum_volume_ml
    if measurement.volume_ml >= wanted:
        report.findings.append(
            Finding(
                code="G-CAVITY-VOLUME",
                severity=Severity.INFO,
                subject=asset.asset_id,
                summary=(
                    f"the cavity holds {measurement.volume_ml:.1f} mL to its overflow edge, "
                    f"against a required {wanted:.0f} mL."
                ),
                metrics=metrics,
                thresholds={"minimum_ml": wanted},
            )
        )
    else:
        report.findings.append(
            Finding(
                code="G-CAVITY-VOLUME",
                severity=Severity.FAILURE,
                subject=asset.asset_id,
                summary=(
                    f"the cavity holds only {measurement.volume_ml:.1f} mL, against a required "
                    f"{wanted:.0f} mL. Usable volume is measured to the lowest overflow edge, "
                    f"which this object puts at {measurement.overflow_height * 1000:.1f} mm — "
                    f"a low spout or a notch in the rim caps the volume however tall the wall is."
                ),
                repair_target=RepairTarget.ASSET,
                metrics=metrics,
                thresholds={"minimum_ml": wanted},
            )
        )
    return report


def _repeated_cavity_findings(
    asset_id: str, world: trimesh.Trimesh, spec: GroundingSpec
) -> list[Finding]:
    """Count the cavities a specification asks for in quantity.

    A cavity component produced no finding at all before this, because a cavity is not a
    body and the volume integration that stood in for it reports one number for the whole
    object. PCR-001 asks for 96 tube wells; the integration reported the 66 mL tray between
    them, passed, and told the generator nothing about the wells. The generator has to see
    the same count the judge will.
    """
    findings: list[Finding] = []
    for component in spec.components:
        if component.kind != "cavity" or component.quantity <= 1:
            continue
        counted, fraction = gmesh.repeated_feature_count(world)
        subject = f"{asset_id}/{component.id}"
        if counted == component.quantity:
            findings.append(
                Finding(
                    code="G-CAVITY-COUNT",
                    severity=Severity.INFO,
                    subject=subject,
                    summary=f"{counted} × {component.name!r}, as required.",
                    metrics={"counted": float(counted), "required": float(component.quantity)},
                )
            )
            continue
        findings.append(
            Finding(
                code="G-CAVITY-COUNT",
                severity=Severity.FAILURE if component.critical else Severity.WARNING,
                subject=subject,
                summary=(
                    f"{component.name!r} is required {component.quantity} times and a section "
                    f"across the object finds {counted}. Counted at {fraction * 100:.0f}% of the "
                    f"height, as the group of alike shapes the section is mostly made of — so a "
                    f"handful of posts or ribs will not have been mistaken for the openings."
                ),
                repair_target=RepairTarget.ASSET,
                metrics={"counted": float(counted), "required": float(component.quantity)},
            )
        )
    return findings


def _component_finding(
    asset_id: str, component, topology: gmesh.Topology
) -> Finding | None:
    """Match a required component against the body tree by name.

    Matching is on normalised substrings in both directions, because an author writes
    `pouring_spout` for a component the specification calls "Pouring spout" and neither
    spelling is wrong. A cavity is not a body and is checked by volume instead.
    """
    if component.kind in {"cavity", "visual"}:
        return None

    match = match_body(component.name, topology.bodies) or match_body(
        component.id.replace("CMP-", ""), topology.bodies
    )

    if match is None:
        severity = Severity.FAILURE if component.critical else Severity.WARNING
        return Finding(
            code="G-COMPONENT-MISSING",
            severity=severity,
            subject=f"{asset_id}/{component.id}",
            summary=(
                f"required component {component.id} ({component.name!r}) has no matching body. "
                f"Bodies present: {', '.join(topology.bodies) or '(none)'}. Name the part for "
                f"what it is so it can be identified."
            ),
            repair_target=RepairTarget.ASSET,
            detail={"kind": component.kind, "expected_parent": component.parent},
        )

    if component.kind in {"moving", "articulated"} and match not in topology.joints.values():
        return Finding(
            code="G-COMPONENT-STATIC",
            severity=Severity.FAILURE,
            subject=f"{asset_id}/{component.id}",
            summary=(
                f"{component.name!r} is required to move but body {match!r} carries no joint."
            ),
            repair_target=RepairTarget.ASSET,
        )

    return Finding(
        code="G-COMPONENT",
        severity=Severity.INFO,
        subject=f"{asset_id}/{component.id}",
        summary=f"{component.name!r} is present as body {match!r}.",
    )


def _normalise(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def match_body(wanted: str, bodies: list[str]) -> str | None:
    """Find the body a specification means by `wanted`, or None.

    Defers to the ranking the judge uses, which is the whole point. This function did its own
    matching on normalised substrings, and a substring is sensitive to word order where a name
    is not: PCR-001 asks for an "8x12 rack", a model that calls the body `rack_8x12` has named
    it correctly, and "8x12rack" is not a substring of "rack8x12" in either direction. So the
    in-loop check reported the component missing while the judge found it without difficulty,
    and the generator renamed the body four times -- `rack_8x12`, `rack_base`, `rack_8x12_body`,
    `rack_8x12_96_well` -- being refused each time, until it ran out of turns.

    One name cannot be present for one reader and absent for the other.
    """
    ranked = match_bodies(wanted, bodies)
    return ranked[0] if ranked else None


# --------------------------------------------------------------------------- #
# protocol / rigid body
# --------------------------------------------------------------------------- #


def check_protocol(asset: GroundedAsset, spec: GroundingSpec) -> Report:
    """Exercise stated operations, stand it up, insert a probe, and tip it."""
    report = Report(kind="grounding-protocol", subject=asset.asset_id)
    if not spec.operations and spec.stability is None and spec.probe is None and spec.tilt is None:
        report.notes.append("no protocol behaviour was stated, so nothing was simulated")
        return report

    if spec.operations:
        report.extend(check_operations(asset, spec))
    if spec.stability is not None:
        report.findings.append(_stability_finding(asset, spec))
    if spec.probe is not None:
        report.findings.extend(_probe_findings(asset, spec))
    if spec.tilt is not None:
        report.findings.append(_tilt_finding(asset, spec))
    return report


def check_operations(asset: GroundedAsset, spec: GroundingSpec) -> Report:
    """Drive every specified affordance to an endpoint and back in MuJoCo."""
    report = Report(kind="grounding-operations", subject=asset.asset_id)
    if not spec.operations:
        report.notes.append("no physical operations were stated")
        return report
    try:
        staged = stage(asset.mjcf_path, ground=False, free=False, gravity=0.0)
    except SceneError as error:
        report.findings.append(_unmeasurable(asset.asset_id, f"operations could not load: {error}"))
        return report

    import mujoco  # noqa: PLC0415

    body_names = {
        body_id: mujoco.mj_id2name(staged.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        for body_id in range(1, staged.model.nbody)
    }
    anchored = _anchored_bodies(staged.model, spec, body_names)
    for target in spec.operations:
        matches = _operation_bodies(target, body_names)
        if len(matches) < target.count:
            report.findings.append(
                Finding(
                    code="G-OP-PART-MISSING",
                    severity=Severity.FAILURE,
                    subject=f"{asset.asset_id}/{target.id}",
                    summary=(
                        f"{target.name!r} requires {target.count} independently operable Part(s), "
                        f"but only {len(matches)} matching bodies were found. Operable controls "
                        "and mechanisms must not be fused into a panel or housing visual."
                    ),
                    repair_target=RepairTarget.ASSET,
                    detail={"matched_bodies": [body_names[item] for item in matches]},
                )
            )
            continue
        for body_id in matches[: target.count]:
            report.findings.append(
                _exercise_operation(
                    staged, asset.asset_id, target, body_id, body_names[body_id], anchored
                )
            )
    return report


def operable_joints(model, spec: GroundingSpec) -> dict[str, OperationTarget]:
    """Joint name to the stated operation that drives it, for every operation that matched.

    The same body matcher the operation check uses, so a viewer and a grader agree about
    which joints the task actually claims are operable. Joints not in the result belong
    to no stated mechanism: either the author added them, or the matcher cannot see which
    operation they serve, and in both cases driving them illustrates nothing.
    """
    import mujoco  # noqa: PLC0415

    body_names = {
        body_id: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        for body_id in range(1, model.nbody)
    }
    matched: dict[str, OperationTarget] = {}
    for target in spec.operations:
        bodies = _operation_bodies(target, body_names)[: target.count]
        for joint in range(model.njnt):
            if int(model.jnt_bodyid[joint]) not in bodies:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if name:
                matched.setdefault(name, target)
    return matched


def _operation_bodies(target: OperationTarget, body_names: dict[int, str]) -> list[int]:
    wanted = _normalise(target.child_hint)
    direct = [
        body_id
        for body_id, name in body_names.items()
        if wanted and (wanted in _normalise(name) or _normalise(name) in wanted)
    ]
    if direct:
        return direct
    ignored = {"and", "the", "movable", "moving", "matching", "independent", "access"}
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", target.child_hint.lower())
        if len(token) >= 3 and token not in ignored
    }
    scored = []
    for body_id, name in body_names.items():
        body_tokens = set(re.findall(r"[a-z0-9]+", name.lower().replace("_", " ")))
        score = len(tokens & body_tokens)
        if score:
            scored.append((score, body_id))
    scored.sort(reverse=True)
    return [body_id for _, body_id in scored]


def _exercise_operation(
    staged,
    asset_id: str,
    target: OperationTarget,
    body_id: int,
    body: str,
    anchored: dict[int, str],
) -> Finding:
    import mujoco  # noqa: PLC0415

    model, data = staged.model, staged.data
    joints = [joint for joint in range(model.njnt) if int(model.jnt_bodyid[joint]) == body_id]
    if not joints:
        return Finding(
            code="G-OP-FIXED",
            severity=Severity.FAILURE,
            subject=f"{asset_id}/{target.id}/{body}",
            summary=f"{target.name!r} is body {body!r}, but it is fixed and cannot be operated.",
            repair_target=RepairTarget.ASSET,
        )

    type_names = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }
    compatible = [
        joint
        for joint in joints
        if not target.expected_joint_types
        or type_names.get(int(model.jnt_type[joint])) in target.expected_joint_types
    ]
    if not compatible:
        actual = [type_names.get(int(model.jnt_type[joint]), "unknown") for joint in joints]
        return Finding(
            code="G-OP-JOINT-TYPE",
            severity=Severity.FAILURE,
            subject=f"{asset_id}/{target.id}/{body}",
            summary=(
                f"{target.name!r} uses joint type(s) {actual}, but the physical operation "
                f"requires one of {target.expected_joint_types}."
            ),
            repair_target=RepairTarget.ASSET,
            detail={"actual": actual, "expected": target.expected_joint_types},
        )

    joint = _joint_to_drive(model, compatible, target)
    joint_type = type_names[int(model.jnt_type[joint])]
    if target.kind == "press" and joint_type != "slide":
        return Finding(
            code="G-OP-JOINT-TYPE",
            severity=Severity.FAILURE,
            subject=f"{asset_id}/{target.id}/{body}",
            summary=(
                f"{target.name!r} is mounted on a {joint_type} joint, so pressing it turns "
                "it. A button travels along its axis and does not rotate: use a bounded "
                "prismatic joint with an inward-facing axis."
            ),
            repair_target=RepairTarget.ASSET,
            detail={"actual": joint_type, "expected": ["slide"]},
        )
    limited = bool(model.jnt_limited[joint])
    lower, upper = (float(value) for value in model.jnt_range[joint])
    if target.continuous and limited:
        return Finding(
            code="G-OP-RANGE",
            severity=Severity.FAILURE,
            subject=f"{asset_id}/{target.id}/{body}",
            summary=f"{target.name!r} must rotate continuously, but its joint is bounded.",
            repair_target=RepairTarget.ASSET,
        )
    if (
        not target.continuous
        and target.kind != "remove"
        and joint_type in {"hinge", "slide"}
        and not limited
    ):
        return Finding(
            code="G-OP-RANGE",
            severity=Severity.FAILURE,
            subject=f"{asset_id}/{target.id}/{body}",
            summary=(
                f"{target.name!r} has an unlimited {joint_type} joint. Real press, open, slide "
                "and adjustment mechanisms need conservative finite travel limits."
            ),
            repair_target=RepairTarget.ASSET,
        )
    if target.range_min is not None and target.range_max is not None:
        tolerance = max(abs(target.range_max - target.range_min) * 0.05, 1e-5)
        if abs(lower - target.range_min) > tolerance or abs(upper - target.range_max) > tolerance:
            return Finding(
                code="G-OP-RANGE",
                severity=Severity.FAILURE,
                subject=f"{asset_id}/{target.id}/{body}",
                summary=(
                    f"{target.name!r} joint limits [{lower:g}, {upper:g}] do not match the "
                    f"specified [{target.range_min:g}, {target.range_max:g}] SI range."
                ),
                repair_target=RepairTarget.ASSET,
            )
    if target.kind == "press" and (not limited or upper - lower < 0.0002 or upper - lower > 0.02):
        return Finding(
            code="G-OP-PRESS-TRAVEL",
            severity=Severity.FAILURE,
            subject=f"{asset_id}/{target.id}/{body}",
            summary=(
                f"{target.name!r} button travel is {(upper - lower) * 1000:.3g} mm; a physical "
                "button needs a bounded 0.2–20 mm inward stroke."
            ),
            repair_target=RepairTarget.ASSET,
        )

    start_qpos = data.qpos.copy()
    start_pos = data.xpos[body_id].copy()
    start_mat = data.xmat[body_id].copy()
    qpos_address = int(model.jnt_qposadr[joint])
    carried = _subtree(model, body_id) - {body_id}
    dragged = sorted(name for body, name in anchored.items() if body in carried)

    # The whole travel, not only the far end. A lid that clears its housing when shut and
    # when fully open, and ploughs through the rim at 40°, passed an endpoint-only check
    # and is the single most common way an asset looks broken in a render.
    waypoints = _operation_waypoints(target, joint_type, limited, lower, upper,
                                     current=float(start_qpos[qpos_address]))
    worst_penetration_m = 0.0
    worst_penetration_at = 0.0
    finite = True
    translation = 0.0
    rotation = 0.0

    for value in waypoints:
        data.qpos[:] = start_qpos
        if joint_type == "free":
            # Lifting a removable part straight out along the approach axis. A freejoint's
            # qpos is position then quaternion, so this is a pure translation in z.
            data.qpos[qpos_address + 2] += value
        else:
            data.qpos[qpos_address] = value
        mujoco.mj_forward(model, data)

        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qacc).all()):
            finite = False
            break

        penetration = _body_penetration(model, data, body_id)
        if penetration > worst_penetration_m:
            worst_penetration_m, worst_penetration_at = penetration, value
        translation = float(np.linalg.norm(data.xpos[body_id] - start_pos))
        rotation = tilt_angle(start_mat, data.xmat[body_id])

    data.qpos[:] = start_qpos
    mujoco.mj_forward(model, data)
    return_translation = float(np.linalg.norm(data.xpos[body_id] - start_pos))
    return_rotation = tilt_angle(start_mat, data.xmat[body_id])
    moved = translation >= 0.0001 or rotation >= math.radians(0.1)
    returned = return_translation <= 1e-6 and return_rotation <= math.radians(0.01)
    metrics = {
        "translation_mm": translation * 1000.0,
        "rotation_deg": math.degrees(rotation),
        "penetration_mm": worst_penetration_m * 1000.0,
        "penetration_at_q": worst_penetration_at,
        "path_samples": len(waypoints),
        "return_translation_mm": return_translation * 1000.0,
        "return_rotation_deg": math.degrees(return_rotation),
    }

    problems = []
    if not finite:
        problems.append("simulation state became non-finite partway along the travel")
    if not moved:
        problems.append("the body did not measurably move")
    if worst_penetration_m > 0.0002:
        problems.append(
            f"collision penetrates {worst_penetration_m * 1000:.2f} mm at q="
            f"{worst_penetration_at:.4g}, partway along the travel rather than only at rest"
        )
    if target.return_required and not returned:
        problems.append("it did not return to its initial pose")
    reach = _assembly_reach_m(model, data)
    if translation > reach:
        # A 12.5 mm cryovial whose cap "unscrews" by sliding 6.3 m clears every collision
        # test there is, because after the first millimetre there is nothing left to hit.
        problems.append(
            f"it travels {translation * 1000:.0f} mm, past the {reach * 1000:.0f} mm that "
            "encloses the whole object — that is the part leaving, not being operated. A "
            "linear joint's limits are metres; an angle in radians does not belong in them"
        )

    if dragged:
        return Finding(
            code="G-OP-DISTURBANCE",
            severity=Severity.FAILURE,
            subject=f"{asset_id}/{target.id}/{body}",
            summary=(
                f"operating {target.name!r} also moves {', '.join(repr(n) for n in dragged)}, "
                f"which must hold still. Those bodies are children of {body!r} in the "
                "kinematic tree, so this joint carries them. Re-parent: a control hangs off "
                "the housing, the housing does not hang off the control."
            ),
            repair_target=RepairTarget.ASSET,
            metrics=metrics,
            detail={"dragged_bodies": dragged},
        )
    if problems:
        return Finding(
            code="G-OP-EXECUTION",
            severity=Severity.FAILURE,
            subject=f"{asset_id}/{target.id}/{body}",
            summary=f"{target.name!r} is not physically executable: {'; '.join(problems)}.",
            repair_target=RepairTarget.ASSET,
            metrics=metrics,
        )
    return Finding(
        code="G-OP-EXECUTION",
        severity=Severity.INFO,
        subject=f"{asset_id}/{target.id}/{body}",
        summary=(
            f"{target.name!r} ran its whole travel clear and returned: "
            f"{translation * 1000:.2f} mm translation, {math.degrees(rotation):.2f}° rotation "
            f"over {len(waypoints)} sampled positions."
        ),
        metrics=metrics,
    )


def _joint_to_drive(model, compatible: list[int], target: OperationTarget) -> int:
    """Which of a body's joints this operation is actually about.

    Only interesting when a body has several, which is how a floating part is exported:
    three orthogonal slides and a ball, because MuJoCo will not take a freejoint on a
    nested body. Taking the first of those would slide a lid sideways out of its seat.
    Taking the most vertical slide lifts it out, which is what removing something means.
    """
    import mujoco  # noqa: PLC0415

    if len(compatible) == 1 or target.kind != "remove":
        return compatible[0]
    slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
    slides = [joint for joint in compatible if int(model.jnt_type[joint]) == slide]
    if not slides:
        return compatible[0]
    return max(slides, key=lambda joint: abs(float(model.jnt_axis[joint][2])))


def _operation_waypoints(
    target: OperationTarget,
    joint_type: str,
    limited: bool,
    lower: float,
    upper: float,
    *,
    current: float,
    samples: int = 12,
) -> list[float]:
    """Positions to drive this joint through, from where it rests to where it ends up.

    For a freejoint the values are a lift in metres along z rather than a joint
    coordinate, because what is being asked of a removable part is whether it comes out.
    """
    if joint_type == "free":
        return [0.02 * (index + 1) / samples for index in range(samples)]
    if target.continuous:
        end = current + math.pi / 2.0
    elif target.kind == "remove":
        end = upper if limited and abs(upper - current) > 1e-9 else current + 0.02
    else:
        end = upper if abs(upper - current) >= abs(lower - current) else lower
    return [current + (end - current) * (index + 1) / samples for index in range(samples)]


def _subtree(model, root_body: int) -> set[int]:
    """`root_body` and everything hanging off it — everything the joint carries with it.

    Anything in here is allowed to move when the joint moves: a handle on a lid travels
    with the lid, and demanding otherwise would fail a correct assembly.
    """
    subtree = {root_body}
    # A child body always has a higher id than its parent, so one forward pass suffices.
    for body in range(model.nbody):
        walker = body
        while walker > 0:
            if walker in subtree:
                subtree.add(body)
                break
            walker = int(model.body_parentid[walker])
    return subtree


def _anchored_bodies(model, spec: GroundingSpec, body_names: dict[int, str]) -> dict[int, str]:
    """Bodies that must hold still while anything else is operated.

    The object's own root — the housing, the base, the frame — plus every body matching a
    component the specification calls fixed. Nothing operable may carry these, and a
    kinematic tree makes that a structural question rather than a measurement: if the
    housing is a descendant of the knob's joint then turning the knob turns the housing,
    and no amount of getting the axis right will fix it.
    """
    anchored: dict[int, str] = {
        body: name
        for body, name in body_names.items()
        if int(model.body_parentid[body]) == 0
    }
    fixed_names = [
        _normalise(component.name)
        for component in spec.components
        if component.kind == "fixed" and component.name
    ]
    for body, name in body_names.items():
        normalised = _normalise(name)
        if any(
            wanted and (wanted in normalised or normalised in wanted) for wanted in fixed_names
        ):
            anchored[body] = name
    return anchored


def _assembly_reach_m(model, data) -> float:
    """A generous upper bound on how far any part of this object can sensibly travel.

    Four times the radius of the sphere enclosing the whole assembly at rest. Operating a
    mechanism moves it within the object: a lid lifts clear, a drawer runs out its depth,
    a cap comes off the neck. Nothing an object does to itself carries a part further than
    a couple of its own diameters, so a travel past this is the part leaving the scene.
    """
    if model.ngeom == 0:
        return math.inf
    centre = data.geom_xpos[: model.ngeom].mean(axis=0)
    radius = max(
        float(np.linalg.norm(data.geom_xpos[geom] - centre)) + float(model.geom_rbound[geom])
        for geom in range(model.ngeom)
    )
    return 4.0 * radius


def _body_penetration(model, data, body_id: int) -> float:
    worst = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body_id in {body1, body2}:
            worst = min(worst, float(contact.dist))
    return abs(worst)


def _stability_finding(asset: GroundedAsset, spec: GroundingSpec) -> Finding:
    target = spec.stability
    assert target is not None
    try:
        staged = stage(asset.mjcf_path, ground=True, free=True)
    except SceneError as error:
        return _unmeasurable(asset.asset_id, f"could not stage a bench scene: {error}")

    root = staged.root_id
    start_pos = staged.data.xpos[root].copy()
    start_mat = staged.data.xmat[root].copy()

    if not settle(staged, target.duration_s):
        return Finding(
            code="G-STABILITY-DIVERGED",
            severity=Severity.FAILURE,
            subject=asset.asset_id,
            summary=(
                f"the simulation diverged {staged.data.time:.3g} s after the object was placed "
                "on a flat surface. That is almost always a body with no mass or no inertia, "
                "or geometry that starts out intersecting itself."
            ),
            repair_target=RepairTarget.ASSET,
        )

    drift = float(np.linalg.norm(staged.data.xpos[root] - start_pos))
    tilt = tilt_angle(start_mat, staged.data.xmat[root])
    penetration = worst_penetration(staged)
    metrics = {
        "translation_mm": drift * 1000.0,
        "tilt_deg": math.degrees(tilt),
        "penetration_mm": penetration * 1000.0,
    }
    thresholds = {
        "max_translation_mm": target.max_translation_mm,
        "max_tilt_deg": target.max_tilt_deg,
        "max_penetration_mm": target.max_penetration_mm,
    }

    problems = []
    if drift * 1000.0 > target.max_translation_mm:
        problems.append(f"drifted {drift * 1000:.2f} mm")
    if math.degrees(tilt) > target.max_tilt_deg:
        problems.append(f"tipped {math.degrees(tilt):.2f}°")
    if penetration * 1000.0 > target.max_penetration_mm:
        problems.append(f"sank {penetration * 1000:.2f} mm into the surface")

    if not problems:
        return Finding(
            code="G-STABILITY",
            severity=Severity.INFO,
            subject=asset.asset_id,
            summary=(
                f"rests stably: {drift * 1000:.2f} mm drift and {math.degrees(tilt):.2f}° tilt "
                f"over {target.duration_s:.0f} s."
            ),
            metrics=metrics,
            thresholds=thresholds,
        )
    return Finding(
        code="G-STABILITY",
        severity=Severity.FAILURE,
        subject=asset.asset_id,
        summary=(
            f"does not rest stably on a flat surface: it {', '.join(problems)} over "
            f"{target.duration_s:.0f} s. Check that the base is flat, that the centre of mass "
            f"is over the footprint, and that the inertia matches the geometry."
        ),
        repair_target=RepairTarget.ASSET,
        metrics=metrics,
        thresholds=thresholds,
    )


def _probe_findings(asset: GroundedAsset, spec: GroundingSpec) -> list[Finding]:
    """Drop the reference probe through the mouth and see where it ends up."""
    target = spec.probe
    assert target is not None
    radius = target.diameter_mm / 2000.0

    try:
        staged = stage(asset.mjcf_path, ground=False, free=False)
        world = gmesh.world_mesh(staged.model, staged.data)
    except (SceneError, gmesh.GeometryUnavailable) as error:
        return [_unmeasurable(asset.asset_id, f"could not read the cavity: {error}")]

    cavity = gmesh.cavity_volume(world)
    if cavity is None:
        return [
            Finding(
                code="G-PROBE-NO-CAVITY",
                severity=Severity.FAILURE,
                subject=asset.asset_id,
                summary=(
                    "the reference probe has nowhere to go: no enclosed cavity was found, so "
                    "the object cannot accept anything through its mouth."
                ),
                repair_target=RepairTarget.ASSET,
            )
        ]

    findings: list[Finding] = [_probe_clearance_finding(asset, world, cavity, radius, target)]
    if findings[0].severity is Severity.FAILURE:
        # No point dropping something into an opening it cannot fit through.
        return findings

    start = (
        cavity.centre_xy[0],
        cavity.centre_xy[1],
        cavity.overflow_height + radius * 4.0,
    )
    try:
        # Concave collision is essential here: against the convex hull the interior does
        # not exist and the probe would simply rest on the rim, which reads as a pass.
        scene = stage(
            asset.mjcf_path,
            ground=True,
            free=False,
            concave=True,
            probe_radius_m=radius,
            probe_mass_kg=target.mass_g / 1000.0,
            probe_pos=start,
        )
    except SceneError as error:
        findings.append(
            Finding(
                code="G-PROBE-CONTACT-UNAVAILABLE",
                severity=Severity.WARNING,
                subject=asset.asset_id,
                summary=(
                    f"insertion was checked geometrically but not simulated: {error}. MuJoCo "
                    "collides a mesh as its convex hull, so without a convex decomposition a "
                    "dropped probe would rest on the rim whatever the interior looks like."
                ),
            )
        )
        return findings

    probe = scene.probe_id
    if probe is None:
        return [*findings, _unmeasurable(asset.asset_id, "the probe body was not created")]

    if not settle(scene, 3.0):
        findings.append(
            Finding(
                code="G-PROBE-DIVERGED",
                severity=Severity.FAILURE,
                subject=asset.asset_id,
                summary="the simulation diverged while the reference probe was falling in.",
                repair_target=RepairTarget.ASSET,
            )
        )
        return findings

    final_z = float(scene.data.xpos[probe][2])
    floor = cavity.floor_height
    metrics = {
        "probe_final_z_mm": final_z * 1000.0,
        "cavity_floor_mm": floor * 1000.0,
        "cavity_overflow_mm": cavity.overflow_height * 1000.0,
        "released_from_mm": start[2] * 1000.0,
    }
    if final_z < floor - radius:
        findings.append(
            Finding(
                code="G-PROBE-ESCAPE",
                severity=Severity.FAILURE,
                subject=asset.asset_id,
                summary=(
                    f"the reference probe passed through the bottom: it came to rest at "
                    f"{final_z * 1000:.1f} mm, below the cavity floor at {floor * 1000:.1f} mm. "
                    f"The base is either open or too thin to collide against."
                ),
                repair_target=RepairTarget.ASSET,
                metrics=metrics,
            )
        )
    elif final_z > cavity.overflow_height:
        findings.append(
            Finding(
                code="G-PROBE-BLOCKED",
                severity=Severity.FAILURE,
                subject=asset.asset_id,
                summary=(
                    f"the reference probe never got in: it came to rest at "
                    f"{final_z * 1000:.1f} mm, above the overflow edge at "
                    f"{cavity.overflow_height * 1000:.1f} mm, so something is blocking the mouth."
                ),
                repair_target=RepairTarget.ASSET,
                metrics=metrics,
            )
        )
    else:
        findings.append(
            Finding(
                code="G-PROBE",
                severity=Severity.INFO,
                subject=asset.asset_id,
                summary=(
                    f"the reference probe entered through the mouth and came to rest at "
                    f"{final_z * 1000:.1f} mm, inside the cavity and above its floor."
                ),
                metrics=metrics,
            )
        )
    return findings


def _probe_clearance_finding(
    asset: GroundedAsset,
    world: trimesh.Trimesh,
    cavity: gmesh.CavityMeasurement,
    radius: float,
    target,
) -> Finding:
    """Can a disc of the probe's diameter be lowered down the axis without touching?

    This is the geometric half of the insertion test, and it is the half that always
    works: it reads the cross-sections directly and does not depend on how the collision
    geometry happens to be represented. It also localises the obstruction, which a
    dropped ball cannot — knowing the neck closes at 61 mm is actionable in a way that
    "it stopped" is not.
    """
    span = cavity.overflow_height - cavity.floor_height
    if span <= 0.0:
        return Finding(
            code="G-PROBE-CLEARANCE",
            severity=Severity.FAILURE,
            subject=asset.asset_id,
            summary="the cavity has no depth, so nothing can be lowered into it.",
            repair_target=RepairTarget.ASSET,
        )

    narrowest = None
    narrowest_height = cavity.floor_height
    for index in range(24):
        height = cavity.floor_height + span * (index + 0.5) / 24.0
        profile = gmesh.profile_at(world, height)
        if profile is None or profile.inner_diameter is None:
            continue
        if narrowest is None or profile.inner_diameter < narrowest:
            narrowest = profile.inner_diameter
            narrowest_height = height

    if narrowest is None:
        return Finding(
            code="G-PROBE-CLEARANCE",
            severity=Severity.WARNING,
            subject=asset.asset_id,
            summary="the cavity's cross-sections could not be read, so clearance is unknown.",
        )

    metrics = {
        "narrowest_diameter_mm": narrowest * 1000.0,
        "narrowest_at_mm": narrowest_height * 1000.0,
        "probe_diameter_mm": target.diameter_mm,
    }
    if narrowest <= 2.0 * radius:
        return Finding(
            code="G-PROBE-CLEARANCE",
            severity=Severity.FAILURE,
            subject=asset.asset_id,
            summary=(
                f"a {target.diameter_mm:.1f} mm probe cannot be lowered in: the cavity narrows "
                f"to {narrowest * 1000:.1f} mm at {narrowest_height * 1000:.1f} mm above the base."
            ),
            repair_target=RepairTarget.ASSET,
            metrics=metrics,
        )
    return Finding(
        code="G-PROBE-CLEARANCE",
        severity=Severity.INFO,
        subject=asset.asset_id,
        summary=(
            f"a {target.diameter_mm:.1f} mm probe clears the cavity throughout; the tightest "
            f"point is {narrowest * 1000:.1f} mm."
        ),
        metrics=metrics,
    )


def _tilt_finding(asset: GroundedAsset, spec: GroundingSpec) -> Finding:
    """Rotate the object to the stated angle and back, kinematically.

    The rotation is imposed rather than achieved with a gripper: the question here is
    whether the object survives the pose and returns to it, not whether some particular
    arm can hold it. Grasping is part 2's problem.
    """
    target = spec.tilt
    assert target is not None
    try:
        staged = stage(asset.mjcf_path, ground=True, free=True)
    except SceneError as error:
        return _unmeasurable(asset.asset_id, f"could not stage a tilt scene: {error}")

    root = staged.root_id
    if staged.model.nq < 7:
        return Finding(
            code="G-TILT",
            severity=Severity.WARNING,
            subject=asset.asset_id,
            summary="the asset has no free joint to rotate, so the tilt protocol was not run.",
        )

    start_mat = staged.data.xmat[root].copy()
    axis = np.zeros(3)
    axis[{"x": 0, "y": 1, "z": 2}[target.axis]] = 1.0
    angle = math.radians(target.angle_deg)

    reached = _impose_rotation(staged, axis, angle)
    returned = _impose_rotation(staged, axis, 0.0)
    if reached is None or returned is None:
        return Finding(
            code="G-TILT",
            severity=Severity.FAILURE,
            subject=asset.asset_id,
            summary=(
                f"the simulation diverged while the object was rotated {target.angle_deg:.0f}° "
                f"about {target.axis}."
            ),
            repair_target=RepairTarget.ASSET,
        )

    reached_error = abs(math.degrees(reached) - target.angle_deg)
    return_error = math.degrees(tilt_angle(start_mat, staged.data.xmat[root]))
    metrics = {"reached_deg": math.degrees(reached), "return_error_deg": return_error}
    thresholds = {"tolerance_deg": target.tolerance_deg}

    if reached_error <= target.tolerance_deg and return_error <= target.tolerance_deg:
        return Finding(
            code="G-TILT",
            severity=Severity.INFO,
            subject=asset.asset_id,
            summary=(
                f"tilted to {math.degrees(reached):.1f}° about {target.axis} and returned "
                f"upright within {return_error:.2f}°."
            ),
            metrics=metrics,
            thresholds=thresholds,
        )
    return Finding(
        code="G-TILT",
        severity=Severity.FAILURE,
        subject=asset.asset_id,
        summary=(
            f"the tilt protocol did not complete: reached {math.degrees(reached):.1f}° against a "
            f"{target.angle_deg:.0f}° target and came back {return_error:.2f}° off upright."
        ),
        repair_target=RepairTarget.ASSET,
        metrics=metrics,
        thresholds=thresholds,
    )


def _impose_rotation(staged, axis: np.ndarray, angle: float) -> float | None:
    """Set the free joint's orientation directly and let the scene react.

    Returns the achieved angle, or None if the integrator diverged.
    """
    import mujoco  # noqa: PLC0415

    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)
    before = staged.data.xmat[staged.root_id].copy()
    staged.data.qpos[3:7] = quat
    mujoco.mj_forward(staged.model, staged.data)
    achieved = tilt_angle(before, staged.data.xmat[staged.root_id])
    for _ in range(50):
        mujoco.mj_step1(staged.model, staged.data)
        staged.data.qpos[3:7] = quat
        mujoco.mj_forward(staged.model, staged.data)
        if not np.isfinite(staged.data.qacc).all():
            return None
    return achieved


# --------------------------------------------------------------------------- #
# visual
# --------------------------------------------------------------------------- #


class FeatureVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    visible: bool
    observation: str = Field(description="What is actually visible, in one sentence.")


class FeatureVerdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[FeatureVerdict]


def render_views(
    asset: GroundedAsset,
    views: list[str],
    *,
    output_dir: Path | None = None,
    width: int = 768,
    height: int = 768,
) -> list[tuple[str, bytes]]:
    """Render the asset from named directions. Needs a working offscreen GL backend."""
    import io  # noqa: PLC0415

    import mujoco  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    staged = stage(asset.mjcf_path, ground=True, free=False, lit=True)
    model, data = staged.model, staged.data
    # Framed from the asset's own geometry rather than `model.stat`, which now includes the
    # two-metre bench plane: reading the camera distance off that put a 127 mm rack in the
    # middle of an empty floor at a twentieth of its size.
    bounds = gmesh.world_mesh(model, data).bounds
    centre = (np.asarray(bounds[0]) + np.asarray(bounds[1])) / 2.0
    radius = max(float(np.linalg.norm(bounds[1] - bounds[0]) / 2.0), 0.01)
    _give_surfaces_a_finish(model)
    _stand_the_bench_under_it(model, data, float(bounds[0][2]))

    # MuJoCo sizes the offscreen framebuffer from the model, not from the renderer, and
    # the default is 640x480. Asking for anything larger fails unless the model says so
    # first — and a generated asset has no reason to have declared it.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)

    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as error:  # noqa: BLE001 - any GL failure means no renders
        raise RuntimeError(
            f"MuJoCo offscreen rendering failed ({error}); on Linux set MUJOCO_GL=egl or osmesa"
        ) from error

    camera = mujoco.MjvCamera()
    camera.lookat[:] = centre
    # Far enough that a sphere of `radius` about the centre just fits the vertical field of
    # view, plus a tenth for margin. The old fixed 2.4× of a differently-defined radius left
    # the object at about a third of the frame height, so most of every render was empty and
    # the features that were in it were too small to read.
    half_fov = math.radians(float(model.vis.global_.fovy) / 2.0)
    camera.distance = radius / math.tan(half_fov) * 1.1

    out: list[tuple[str, bytes]] = []
    with renderer:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        for view in views:
            azimuth, elevation = VIEW_ANGLES.get(view, VIEW_ANGLES["iso"])
            camera.azimuth, camera.elevation = azimuth, elevation
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                image.save(output_dir / f"{view}.png")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            out.append((view, buffer.getvalue()))
    return out


def _stand_the_bench_under_it(model, data, lowest_z: float) -> None:
    """Put the render's bench at the bottom of the asset rather than at z=0.

    An author is free to build about the origin, and plenty do — a plane fixed at z=0 then
    slices the object in half and the render shows a rack sunk to its waist in the bench.
    Nothing about the asset is wrong in that case, so nothing about it should look wrong.
    """
    import mujoco  # noqa: PLC0415

    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM)
    if ground < 0:
        return
    model.geom_pos[ground][2] = lowest_z
    mujoco.mj_forward(model, data)


def _give_surfaces_a_finish(model) -> None:
    """Make surfaces that were left matte read as the material they are.

    A generated asset sets `rgba` and nothing else, and MuJoCo's default is fully diffuse:
    no highlight, so no cue to curvature, so a moulded rack renders as flat colour and
    looks exactly like a box someone painted. Every plastic and glass object in this corpus
    has a specular highlight in life, and the highlight is most of what makes a render
    legible as a solid.

    Only geoms that made no statement of their own are touched, so an author who did set a
    finish keeps it.
    """
    import mujoco  # noqa: PLC0415

    finish = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, STUDIO_SURFACE_MATERIAL)
    if finish < 0:
        return
    for geom in range(model.ngeom):
        if int(model.geom_matid[geom]) < 0:
            model.geom_matid[geom] = finish


def check_visual(
    asset: GroundedAsset,
    spec: GroundingSpec,
    *,
    client: LlmClient | None = None,
    blocking: bool = False,
    image_dir: Path | None = None,
) -> Report:
    """Render the asset and ask a model whether each required feature is visible."""
    report = Report(kind="grounding-visual", subject=asset.asset_id)
    features = spec.visual.features
    if not features:
        report.notes.append("no visual features were stated, so nothing was checked")
        return report

    try:
        images = render_views(asset, spec.visual.views, output_dir=image_dir)
    except (RuntimeError, SceneError) as error:
        report.notes.append(f"rendering unavailable, visual check skipped: {error}")
        return report

    if client is None:
        try:
            client = LlmClient()
        except (LlmUnavailable, ValueError) as error:
            # The renders still exist and are still worth looking at, so this is a note
            # rather than a failure. The caller attaches them either way.
            report.notes.append(f"visual check skipped, no grader configured: {error}")
            return report

    try:
        answer = client.structured(
            purpose="grounding.visual",
            system=(
                "You inspect renders of a 3D laboratory object and report only what the "
                "images show. Judge each feature independently. If a feature concerns "
                "something the supplied views cannot show, mark it not visible and say so. "
                "Do not re-judge dimensions; you are looking at shape and appearance only."
            ),
            user=(
                f"Object: {spec.summary or spec.asset_class or spec.asset_id}\n\n"
                f"Views supplied, in order: {', '.join(spec.visual.views)}\n\n"
                + (f"Reference: {spec.visual.reference_note}\n\n" if spec.visual.reference_note else "")
                + "Features to judge:\n"
                + "\n".join(f"{i + 1}. {feature}" for i, feature in enumerate(features))
            ),
            schema=FeatureVerdicts,
            images=[blob for _, blob in images],
        )
    except (LlmUnavailable, Exception) as error:  # noqa: BLE001 - a grader outage is a note
        report.notes.append(f"visual check skipped: {type(error).__name__}: {error}")
        return report

    severity = Severity.FAILURE if blocking else Severity.WARNING
    judged = {verdict.feature for verdict in answer.verdicts}
    for verdict in answer.verdicts:
        report.findings.append(
            Finding(
                code="G-VISUAL-CLAIM",
                severity=Severity.INFO if verdict.visible else severity,
                subject=asset.asset_id,
                summary=(
                    f"{'visible' if verdict.visible else 'not visible'}: {verdict.feature} "
                    f"— {verdict.observation}"
                ),
                repair_target=RepairTarget.NONE if verdict.visible else RepairTarget.ASSET,
            )
        )
    for missing in [f for f in features if f not in judged]:
        report.notes.append(f"the grader returned no verdict for: {missing}")
    return report


# --------------------------------------------------------------------------- #
# everything
# --------------------------------------------------------------------------- #


def ground_against_spec(
    asset: GroundedAsset,
    spec: GroundingSpec,
    *,
    client: LlmClient | None = None,
    include_visual: bool = True,
    visual_blocks: bool = False,
    image_dir: Path | None = None,
) -> Report:
    """Run every applicable check and merge the findings into one report."""
    report = Report(kind="grounding", subject=asset.asset_id)
    report.extend(check_dimensions(asset, spec))
    report.extend(check_topology(asset, spec))
    report.extend(check_protocol(asset, spec))
    if include_visual:
        report.extend(
            check_visual(
                asset, spec, client=client, blocking=visual_blocks, image_dir=image_dir
            )
        )
    return report


def _unmeasurable(asset_id: str, detail: str) -> Finding:
    return Finding(
        code="G-UNMEASURABLE",
        severity=Severity.FAILURE,
        subject=asset_id,
        summary=f"the asset could not be measured: {detail}",
        repair_target=RepairTarget.ASSET,
    )
