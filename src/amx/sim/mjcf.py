"""MJCF surgery: namespacing, mesh flattening and section merging.

Composing a workcell means loading several independently authored MJCF files — a robot, one
or more generated assets, some co-designed parts — that all use short local names like
`base` and `visual` and all reference meshes by their own relative paths. Two things have
to happen before they can share one model: every identifier gets a namespace prefix, and
every mesh gets copied into one flat directory under a name derived from its contents, so
identical meshes collapse and different meshes with the same filename do not.

There is no manifest, no hash ledger and no virtual filesystem here. A composed scene is a
directory with an XML file and a `meshes/` folder, and `mujoco.MjModel.from_xml_path` reads
it directly.
"""

from __future__ import annotations

import hashlib
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Tags whose identifier attribute declares a name that other elements may reference.
DECLARING_TAGS = frozenset(
    {
        "body", "joint", "freejoint", "geom", "site", "camera", "light", "mesh", "material",
        "texture", "hfield", "skin", "default", "fixed", "spatial", "motor", "position",
        "velocity", "general", "adhesion", "cylinder", "muscle", "pair", "exclude", "weld",
        "connect", "distance", "jointpos", "framepos", "framequat", "touch", "force",
        "torque", "rangefinder", "actuatorfrc", "numeric", "text", "tuple", "key", "flex",
    }
)

# Attributes that carry values, never identifiers. Guards against a geom named "box"
# turning `type="box"` into a reference.
LITERAL_ATTRS = frozenset(
    {
        "type", "file", "group", "rgba", "pos", "quat", "axis", "size", "range", "euler",
        "axisangle", "xyaxes", "zaxis", "fromto", "friction", "solref", "solimp", "mass",
        "density", "diaginertia", "fullinertia", "damping", "stiffness", "armature",
        "margin", "gap", "kp", "kv", "ctrlrange", "forcerange", "gear", "coef", "scale",
        "contype", "conaffinity", "condim", "limited", "ctrllimited", "forcelimited",
        "springlength", "solreflimit", "solimplimit", "mode", "fovy", "meshdir", "texturedir",
        "dclass", "value", "data", "user", "priority", "shellinertia", "inertiagrouprange",
    }
)

# Sections merged wholesale when one model is folded into another, in MJCF's own order.
MERGE_SECTIONS = (
    "default",
    "asset",
    "tendon",
    "actuator",
    "contact",
    "equality",
    "sensor",
)

# Sections that describe the whole simulation and so may only be declared by the scene.
GLOBAL_SECTIONS = ("compiler", "option", "size", "statistic", "visual", "keyframe")

MESH_SUFFIXES = frozenset({".stl", ".obj", ".msh", ".ply", ".dae"})


class MjcfError(RuntimeError):
    pass


def load(path: Path) -> ET.Element:
    path = Path(path)
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as error:
        raise MjcfError(f"{path} is not parseable MJCF: {error}") from error


def identifier_attr(tag: str) -> str:
    """`<default>` declares itself with `class`; everything else uses `name`."""
    return "class" if tag == "default" else "name"


def collect_names(*roots: ET.Element) -> set[str]:
    names: set[str] = set()
    for root in roots:
        for element in root.iter():
            if element.tag not in DECLARING_TAGS:
                continue
            value = element.get(identifier_attr(element.tag))
            if value:
                names.add(value)
    return names


def apply_prefix(root: ET.Element, prefix: str, names: set[str] | None = None) -> None:
    """Move every declaration and reference in `root` under `prefix`.

    References are found by value: an attribute is rewritten when its value matches a name
    the same model declares. That avoids maintaining a table of which MJCF attribute
    references which kind of entity, which is long and changes between MuJoCo versions.
    """
    if not prefix:
        return
    names = names if names is not None else collect_names(root)
    for element in root.iter():
        declares = identifier_attr(element.tag) if element.tag in DECLARING_TAGS else None
        for attr, value in list(element.attrib.items()):
            if attr in LITERAL_ATTRS:
                continue
            if attr == declares:
                element.set(attr, prefix + value)
            elif value in names:
                element.set(attr, prefix + value)


def absorb_meshes(root: ET.Element, source_dir: Path, mesh_dir: Path) -> dict[str, str]:
    """Copy every referenced mesh into `mesh_dir` under a content-addressed name.

    Returns the original reference to new filename mapping. Identical bytes reaching the
    same destination is the common case for a scene with several copies of one part, and it
    resolves to a single file.
    """
    mesh_dir.mkdir(parents=True, exist_ok=True)
    search_roots = [source_dir, *(source_dir / d for d in ("meshes", "assets"))]
    mapping: dict[str, str] = {}

    for mesh in root.iter("mesh"):
        reference = mesh.get("file")
        if not reference:
            continue
        resolved = _resolve_mesh(reference, search_roots)
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()[:16]
        flat = f"{digest}{resolved.suffix.lower()}"
        destination = mesh_dir / flat
        if not destination.exists():
            shutil.copyfile(resolved, destination)
        mesh.set("file", flat)
        mapping[reference] = flat
    return mapping


def _resolve_mesh(reference: str, search_roots: list[Path]) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    for base in search_roots:
        found = base / reference
        if found.is_file():
            return found
        # Some exporters write a bare filename while the file sits in a subdirectory.
        matches = sorted(base.rglob(candidate.name)) if base.is_dir() else []
        if matches:
            return matches[0]
    searched = ", ".join(str(p) for p in search_roots)
    raise MjcfError(f"mesh {reference!r} is referenced but was not found under: {searched}")


def strip_globals(root: ET.Element) -> None:
    """Drop simulation-wide declarations so the scene's own settings are the only ones."""
    for tag in GLOBAL_SECTIONS:
        for element in root.findall(tag):
            root.remove(element)


def section(root: ET.Element, tag: str) -> ET.Element:
    found = root.find(tag)
    if found is None:
        found = ET.SubElement(root, tag)
    return found


def merge_sections(destination: ET.Element, source: ET.Element) -> None:
    """Fold `source`'s non-body sections into `destination`."""
    for tag in MERGE_SECTIONS:
        origin = source.find(tag)
        if origin is None or len(origin) == 0:
            continue
        target = section(destination, tag)
        for child in list(origin):
            target.append(child)


def root_bodies(root: ET.Element) -> list[ET.Element]:
    world = root.find("worldbody")
    if world is None:
        raise MjcfError("model has no <worldbody>")
    return [child for child in world if child.tag == "body"]


def worldbody_extras(root: ET.Element) -> list[ET.Element]:
    """Sites, lights and cameras declared directly on the worldbody, not inside a body."""
    world = root.find("worldbody")
    if world is None:
        return []
    return [child for child in world if child.tag != "body"]


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise MjcfError(f"no body named {name!r}")


def order_sections(root: ET.Element) -> None:
    """Reorder children into MJCF's documented order.

    MuJoCo accepts most orderings, but keeping it canonical makes a composed scene.xml
    readable and diffable, which matters when a repair loop writes twenty of them.
    """
    order = [
        "compiler", "option", "size", "visual", "statistic", "default", "custom", "asset",
        "worldbody", "deformable", "contact", "equality", "tendon", "actuator", "sensor",
        "keyframe",
    ]
    rank = {tag: index for index, tag in enumerate(order)}
    children = sorted(root, key=lambda element: rank.get(element.tag, len(order)))
    for child in list(root):
        root.remove(child)
    for child in children:
        root.append(child)


def write(root: ET.Element, path: Path) -> Path:
    order_sections(root)
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    return path
