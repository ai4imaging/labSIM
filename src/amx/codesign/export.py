"""One built part, written out three ways.

The same geometry has to satisfy three readers that want quite different things, and
writing them from one build is what keeps them from drifting apart:

    part.xml    what MuJoCo simulates: primitives for contact, the mesh for looks
    part.stl    what goes to a printer or a shop
    part.json   what a person needs to make and inspect it — parameters, material, fits

The MJCF deliberately does not collide with the mesh. A convex decomposition of a bracket
is slow to load and unstable at the small contact scales a wetlab cell works at, whereas
the primitives each template already declares are both fast and exactly the shape the
designer meant. The mesh is present as a visual so what you see is what gets printed.

Names inside `part.xml` are local. `amx.sim.scene` prefixes them with the part's namespace
when it composes the cell, which is what keeps one part's `base` from colliding with
another's.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from amx.codesign.dfm import check_part
from amx.codesign.parts import Part, PartGeometry, Primitive
from amx.geometry import Pose
from amx.report import Report
from amx.sim import mjcf

VISUAL_RGBA = "0.28 0.42 0.60 1"
COLLISION_GROUP = "3"
"""MuJoCo shows groups 0-2 by default, so collision primitives stay out of the render."""

CONTACT_SOLREF = "0.005 1"
"""A stiffer-than-default contact. Printed fixtures are rigid, and a soft contact lets a
tube visibly sink into its own rack.
"""


def export_part(part: Part, output_dir: Path) -> tuple[PartGeometry, Report]:
    """Build, check and write one part. Returns the geometry and its DFM report."""
    geometry = part.build()
    report = check_part(geometry, mounted_on_arm=_is_tool_side(geometry))
    write_geometry(geometry, output_dir, report=report)
    return geometry, report


def write_geometry(
    geometry: PartGeometry, output_dir: Path, *, report: Report | None = None
) -> Path:
    """Write the three deliverables into `output_dir`. Returns the directory."""
    output_dir = Path(output_dir)
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    stl_name = f"{geometry.part_id}.stl"
    geometry.mesh.export(mesh_dir / stl_name, file_type="stl")

    mjcf.write(_build_mjcf(geometry, stl_name), output_dir / "part.xml")
    (output_dir / "part.json").write_text(
        json.dumps(_datasheet(geometry, report), indent=2, ensure_ascii=False) + "\n"
    )
    if report is not None:
        report.write(output_dir / "dfm.json")
    return output_dir


def _is_tool_side(geometry: PartGeometry) -> bool:
    """Parts that mount on the arm are the ones whose mass matters to it."""
    return geometry.template in {"flange_adapter", "press_finger", "lid_hook"}


def _build_mjcf(geometry: PartGeometry, stl_name: str) -> ET.Element:
    root = ET.Element("mujoco", {"model": geometry.part_id})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "meshes"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "shell", "file": stl_name})
    ET.SubElement(
        asset,
        "material",
        {"name": "surface", "rgba": VISUAL_RGBA, "specular": "0.2", "shininess": "0.3"},
    )

    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", {"name": "base"})
    # State the mass explicitly rather than letting MuJoCo infer it from the primitives:
    # the primitives overlap by design, so their summed volume overstates the real part.
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": " ".join(f"{v:.6g}" for v in geometry.mesh.center_mass),
            "mass": f"{geometry.mass_kg:.6g}",
            "diaginertia": _diagonal_inertia(geometry),
            "quat": "1 0 0 0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "visual",
            "type": "mesh",
            "mesh": "shell",
            "material": "surface",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "mass": "0",
        },
    )
    for index, primitive in enumerate(geometry.collision):
        ET.SubElement(body, "geom", _collision_attrs(geometry, primitive, index))
    for name, pose in geometry.frames.items():
        ET.SubElement(
            body,
            "site",
            {"name": name, "size": "0.003", "group": "4", "rgba": "1 0.6 0 1", **pose.mjcf()},
        )
    return root


def _collision_attrs(
    geometry: PartGeometry, primitive: Primitive, index: int
) -> dict[str, str]:
    attrs: dict[str, str] = {
        "name": primitive.role or f"collision_{index}",
        "type": primitive.shape,
        "size": " ".join(f"{v:.6g}" for v in primitive.size),
        "group": COLLISION_GROUP,
        "condim": "3",
        "friction": f"{geometry.material.friction:.6g} 0.005 0.0001",
        "solref": CONTACT_SOLREF,
        "mass": "0",
        "rgba": "0.8 0.2 0.2 0.3",
    }
    attrs.update(primitive.pose.mjcf())
    return attrs


def _diagonal_inertia(geometry: PartGeometry) -> str:
    """Principal moments of the actual mesh, scaled to the material's density."""
    mesh = geometry.mesh
    scale = geometry.mass_kg / float(mesh.mass) if mesh.mass else 1.0
    moments = [max(float(v) * scale, 1e-9) for v in mesh.principal_inertia_components]
    return " ".join(f"{v:.6g}" for v in moments)


def _datasheet(geometry: PartGeometry, report: Report | None) -> dict[str, object]:
    """What a shop or a colleague needs in order to make and accept this part."""
    material = geometry.material
    extents = geometry.extents_m
    return {
        "part_id": geometry.part_id,
        "template": geometry.template,
        "parameters_m": geometry.parameters,
        "material": {
            "key": material.key,
            "label": material.label,
            "process": material.process,
            "density_kg_m3": material.density_kg_m3,
        },
        "manufacturing": {
            "minimum_wall_m": material.minimum_wall_m,
            "minimum_feature_m": material.minimum_feature_m,
            "hole_allowance_m": material.hole_allowance_m,
            "maximum_overhang_rad": material.maximum_overhang_rad,
            "build_volume_m": list(material.build_volume_m),
        },
        "stated_clearances_m": geometry.stated_clearances,
        "measured": {
            "volume_m3": float(geometry.mesh.volume),
            "mass_kg": geometry.mass_kg,
            "extents_m": list(extents),
            "watertight": bool(geometry.mesh.is_watertight),
            "bodies": int(geometry.mesh.body_count),
        },
        "frames": {name: pose.model_dump() for name, pose in geometry.frames.items()},
        "dfm": None if report is None else {
            "verdict": report.verdict,
            "failures": [f.code for f in report.failures],
            "warnings": [f.code for f in report.warnings],
        },
    }


def frame_pose(geometry: PartGeometry, name: str) -> Pose:
    """One of the part's named interfaces, by name."""
    try:
        return geometry.frames[name]
    except KeyError:
        raise KeyError(
            f"{geometry.part_id} has no frame {name!r}; it has: "
            f"{', '.join(sorted(geometry.frames)) or '(none)'}"
        ) from None
