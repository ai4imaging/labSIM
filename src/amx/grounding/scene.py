"""Putting a bare asset into the smallest world that can answer a question about it.

An exported asset is just the object: no floor, usually no free joint, and nothing to
poke it with. Every physical question — does it stand up, can something be dropped into
it, does it survive being tipped over — needs one or two of those added, and adding
them by string substitution on the XML breaks the moment a model writes its
`<worldbody>` tag with an attribute on it.

So the MJCF is parsed, amended and re-serialised. Meshes are handed to MuJoCo as an
in-memory asset dictionary because a model loaded from a string has no directory to
resolve relative paths against.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

PROBE_BODY = "__probe"
GROUND_GEOM = "__ground"
STUDIO_FLOOR_MATERIAL = "__studio_floor"
STUDIO_SURFACE_MATERIAL = "__studio_surface"


class SceneError(RuntimeError):
    """The asset MJCF could not be turned into a loadable scene."""


@dataclass(frozen=True)
class Staged:
    model: mujoco.MjModel
    data: mujoco.MjData
    root_body: str
    probe_body: str | None = None

    @property
    def root_id(self) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.root_body)

    @property
    def probe_id(self) -> int | None:
        if not self.probe_body:
            return None
        found = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.probe_body)
        return found if found >= 0 else None


def mesh_assets(mjcf_path: Path) -> dict[str, bytes]:
    """Exactly the files the MJCF names, keyed the way MuJoCo will look them up.

    A model loaded from a string has no directory to resolve relative paths against, so
    its assets have to be handed over as bytes. The keys are what MuJoCo asks for:
    `<compiler meshdir>` joined onto each `file` attribute.

    Only referenced files are collected rather than everything in the directory. MuJoCo
    rejects two entries that share a basename, and sweeping up a sibling cache of
    decomposed parts is an easy way to produce exactly that — a failure that surfaces as
    an unloadable model rather than as the bookkeeping mistake it is.
    """
    mjcf_path = Path(mjcf_path)
    base = mjcf_path.parent
    try:
        root = ET.parse(mjcf_path).getroot()
    except ET.ParseError:
        return {}

    compiler = root.find("compiler")
    meshdir = (compiler.get("meshdir") if compiler is not None else None) or ""
    texturedir = (compiler.get("texturedir") if compiler is not None else None) or meshdir

    assets: dict[str, bytes] = {}
    asset_node = root.find("asset")
    if asset_node is None:
        return assets

    for tag, prefix in (("mesh", meshdir), ("hfield", meshdir), ("texture", texturedir)):
        for node in asset_node.findall(tag):
            reference = node.get("file")
            if not reference:
                continue
            key = f"{prefix}/{reference}" if prefix else reference
            path = base / key
            if path.is_file():
                assets[key] = path.read_bytes()
    return assets


CONTACT_SOLREF = "0.004 1"
"""Contact time constant for the surfaces this module adds.

MuJoCo's default of 0.02 s lets a 100 g object settle roughly 0.2 mm into the bench,
which is the same order as the penetration these checks are meant to catch. Stiffening
the added surfaces pushes the solver's own allowance an order of magnitude below the
thresholds so that a reported penetration is the object's fault and not the solver's.
"""


def _add_studio(root: ET.Element, worldbody: ET.Element) -> None:
    """A sky, a bench surface and three lights, so a render reads as a photographed object.

    The renders this replaced were a flat silhouette on black: MuJoCo's default scene has
    one camera-mounted headlight, and a light coming from exactly where the camera is casts
    no visible shading, so a moulded rack and a painted rectangle look identical. Nothing
    here is decoration — a curved surface is only legible as curved because the light
    falling on it changes across it.

    Two of the three lights cast no shadow. One shadow tells you what is resting on what;
    three overlapping ones just look like dirt.
    """
    visual = root.find("visual") or ET.SubElement(root, "visual")
    # 8x multisampling. Edges at this scale are mostly diagonal, and aliasing on them is
    # the single most artificial-looking thing about an otherwise correct render.
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "8"})
    # The default headlight is bright enough to wash out the lights added below.
    ET.SubElement(
        visual,
        "headlight",
        {"ambient": "0.35 0.35 0.38", "diffuse": "0.25 0.25 0.25", "specular": "0.1 0.1 0.1"},
    )
    ET.SubElement(visual, "map", {"shadowclip": "2", "znear": "0.001"})

    asset = root.find("asset") or ET.SubElement(root, "asset")
    # A gradient sky, so a curved highlight has something to reflect other than black.
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.53 0.60 0.68",
            "rgb2": "0.86 0.89 0.92",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "__studio_grid",
            "type": "2d",
            "builtin": "checker",
            # Nearly the same two greys. The bench is there to say which way is down and to
            # catch a shadow, and a high-contrast check competes with the object for
            # attention — which is the opposite of what a reference render is for.
            "rgb1": "0.80 0.80 0.82",
            "rgb2": "0.84 0.84 0.86",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": STUDIO_SURFACE_MATERIAL,
            # Matches MuJoCo's default geom rgba exactly. A geom that set its own colour has
            # a non-default rgba and keeps it; one that did not gets this, which is the grey
            # it was already being drawn in. So the finish is added and no colour is lost.
            "rgba": "0.5 0.5 0.5 1",
            "specular": "0.4",
            "shininess": "0.5",
            "reflectance": "0.08",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": STUDIO_FLOOR_MATERIAL,
            "texture": "__studio_grid",
            # A 20 mm check gives a sense of scale without reading as a pattern.
            "texrepeat": "50 50",
            "texuniform": "true",
            # A matte bench. Any specular on a surface this large puts the key light's
            # reflection right under the object as a blown-out white patch.
            "reflectance": "0.0",
            "specular": "0.05",
            "shininess": "0.1",
        },
    )

    for name, pos, direction, castshadow in (
        ("__key", "0.4 -0.5 0.9", "-0.4 0.5 -0.9", "true"),
        ("__fill", "-0.6 -0.3 0.4", "0.6 0.3 -0.4", "false"),
        ("__rim", "0.1 0.7 0.5", "-0.1 -0.7 -0.5", "false"),
    ):
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": name,
                "pos": pos,
                "dir": direction,
                # Directional. A point light a metre from the object falls off long before
                # the far edge of the bench, which left the floor lit in a patch under the
                # asset and black beyond it — a dark band across the top of every render.
                "directional": "true",
                "castshadow": castshadow,
                "diffuse": "0.55 0.55 0.55" if name == "__key" else "0.25 0.25 0.26",
                "specular": "0.3 0.3 0.3" if name == "__key" else "0.05 0.05 0.05",
            },
        )


def stage(
    mjcf_path: Path,
    *,
    ground: bool = True,
    free: bool = True,
    concave: bool = False,
    drop_height: float = 0.0,
    probe_radius_m: float | None = None,
    probe_mass_kg: float = 1e-4,
    probe_pos: tuple[float, float, float] | None = None,
    timestep: float = 0.001,
    gravity: float = -9.81,
    lit: bool = False,
) -> Staged:
    """Load the asset with a floor, a free joint and optionally a probe above it.

    `concave=True` swaps in a convex decomposition of the collision geometry. Only worth
    the cost when something has to reach the inside of the object; for standing it on a
    bench the hull is both correct and cheaper.

    `lit=True` adds a sky, a bench surface and three lights. It changes nothing a
    measurement reads — a light has no geometry and the bench is a plane, which
    `world_mesh` skips — and it is what makes a render show a moulded object instead of a
    silhouette in a void.
    """
    if concave:
        from amx.grounding.collision import (  # noqa: PLC0415
            DecompositionUnavailable,
            concave_mjcf,
        )

        try:
            mjcf_path = concave_mjcf(mjcf_path)
        except DecompositionUnavailable as error:
            raise SceneError(f"concave collision unavailable: {error}") from error

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise SceneError(f"{mjcf_path} has no <worldbody>")

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{timestep:g}")
    option.set("gravity", f"0 0 {gravity:g}")

    bodies = worldbody.findall("body")
    if not bodies:
        raise SceneError(f"{mjcf_path} declares no bodies")
    asset_body = bodies[0]
    root_name = asset_body.get("name") or "asset"
    asset_body.set("name", root_name)

    if free and asset_body.find("freejoint") is None:
        has_free = any(j.get("type") == "free" for j in asset_body.findall("joint"))
        if not has_free:
            # Prepended so it precedes the geoms, which MJCF requires.
            asset_body.insert(0, ET.Element("freejoint"))

    if drop_height:
        pos = [float(v) for v in (asset_body.get("pos") or "0 0 0").split()]
        pos[2] += drop_height
        asset_body.set("pos", " ".join(f"{v:g}" for v in pos))

    if lit:
        _add_studio(root, worldbody)

    if ground:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": GROUND_GEOM,
                "type": "plane",
                # Wide enough when lit that the bench edge falls outside the frame at every
                # view angle; a horizon cutting across a reference render reads as a defect
                # in the object.
                "size": "20 20 0.05" if lit else "2 2 0.05",
                "pos": "0 0 0",
                "condim": "3",
                "friction": "1 0.005 0.0001",
                "solref": CONTACT_SOLREF,
                **({"material": STUDIO_FLOOR_MATERIAL} if lit else {}),
            },
        )

    probe_name: str | None = None
    if probe_radius_m and probe_radius_m > 0.0:
        probe_name = PROBE_BODY
        where = probe_pos or (0.0, 0.0, 0.2)
        body = ET.SubElement(
            worldbody,
            "body",
            {"name": probe_name, "pos": " ".join(f"{v:g}" for v in where)},
        )
        ET.SubElement(body, "freejoint")
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{probe_name}_geom",
                "type": "sphere",
                "size": f"{probe_radius_m:g}",
                "mass": f"{max(probe_mass_kg, 1e-6):g}",
                "rgba": "0.9 0.2 0.2 1",
                "condim": "3",
                "solref": CONTACT_SOLREF,
            },
        )

    xml = ET.tostring(root, encoding="unicode")
    try:
        model = mujoco.MjModel.from_xml_string(xml, mesh_assets(mjcf_path))
    except Exception as error:  # noqa: BLE001 - MuJoCo raises bare ValueError for XML faults
        raise SceneError(str(error)) from error
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return Staged(model=model, data=data, root_body=root_name, probe_body=probe_name)


def settle(staged: Staged, seconds: float) -> bool:
    """Step for a while. False if the integrator gave up."""
    steps = max(1, int(seconds / staged.model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(staged.model, staged.data)
        if not np.isfinite(staged.data.qacc).all():
            return False
    return True


def worst_penetration(staged: Staged) -> float:
    """Deepest contact interpenetration currently present, in metres.

    MuJoCo reports contact distance as negative when two geoms overlap, so the worst
    penetration is the most negative distance. Contacts with the ground are included:
    an object sinking into the bench is exactly the failure this is looking for.
    """
    worst = 0.0
    for index in range(staged.data.ncon):
        worst = min(worst, float(staged.data.contact[index].dist))
    return abs(worst)


def tilt_angle(before: np.ndarray, after: np.ndarray) -> float:
    """Angle in radians between two 3x3 orientations."""
    relative = after.reshape(3, 3) @ before.reshape(3, 3).T
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))
