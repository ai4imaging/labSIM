"""Giving a vessel an interior that physics can actually feel.

MuJoCo collides a mesh as its convex hull. For a beaker that hull is a solid cylinder:
the cavity does not exist as far as contact is concerned, and anything dropped towards
the mouth lands on the rim. Every question of the form "can this be put into that" is
therefore unanswerable on the exported asset as it stands, and would silently answer
"no" for a perfectly good beaker.

The fix is an approximate convex decomposition of the collision geometry. It is not
free — a few seconds per mesh — so the result is cached beside the build it came from,
and it is only requested by the checks that need contact with the interior.

The original mesh geom is kept and demoted to visual, rather than removed. It carries
the body's mass and inertia, and reproducing those across twenty convex fragments would
change the physics for no reason.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

COLLISION_GROUP = "3"
"""MuJoCo's conventional group for collision-only geometry. Keeps the fragments out of
the default render, so a screenshot still shows the object rather than its hull soup."""

_THRESHOLD = 0.05
"""CoACD concavity threshold. Lower means more parts and a tighter fit. 0.05 resolves a
3 mm wall on a 70 mm beaker without producing hundreds of fragments."""

_MAX_PARTS = 64
"""Above this the contact solver costs more than the check is worth, and a decomposition
this fragmented usually means the source mesh is broken rather than merely concave."""


class DecompositionUnavailable(RuntimeError):
    """Concave collision could not be produced, so contact with an interior is not testable."""


def concave_mjcf(mjcf_path: Path, *, cache_dir: Path | None = None) -> Path:
    """Return a variant of the MJCF whose collision geometry is genuinely concave.

    Raises `DecompositionUnavailable` if CoACD is missing or a mesh defeats it, so the
    caller can fall back to a geometric answer and say which one it used.
    """
    mjcf_path = Path(mjcf_path).resolve()
    base = mjcf_path.parent
    # A sibling, never a child. Writing the decomposed parts inside the asset's own
    # directory means the next thing to walk that directory finds two meshes with the
    # same basename, which MuJoCo refuses to load.
    target_dir = Path(cache_dir) if cache_dir else base.parent / f"{base.name}-concave"
    target = target_dir / mjcf_path.name
    if target.is_file():
        return target

    try:
        import coacd  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - coacd is a declared dependency
        raise DecompositionUnavailable("coacd is not installed") from error

    coacd.set_log_level("error")
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    asset_node = root.find("asset")
    worldbody = root.find("worldbody")
    if asset_node is None or worldbody is None:
        raise DecompositionUnavailable("the MJCF has no <asset> or no <worldbody>")

    compiler = root.find("compiler")
    meshdir = (compiler.get("meshdir") if compiler is not None else None) or ""

    staged_dir = target_dir
    shutil.rmtree(staged_dir, ignore_errors=True)
    staged_dir.mkdir(parents=True, exist_ok=True)
    for entry in base.iterdir():
        if entry.name == staged_dir.name:
            continue
        if entry.is_dir():
            shutil.copytree(entry, staged_dir / entry.name, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, staged_dir / entry.name)

    mesh_files = {
        node.get("name"): node.get("file")
        for node in asset_node.findall("mesh")
        if node.get("name") and node.get("file")
    }

    produced: dict[str, list[str]] = {}
    for geom in _collision_mesh_geoms(worldbody):
        mesh_name = geom.get("mesh")
        if mesh_name is None or mesh_name in produced:
            continue
        source = mesh_files.get(mesh_name)
        if source is None:
            continue
        loaded = _load_mesh(staged_dir / meshdir / source)
        if loaded is None:
            continue
        produced[mesh_name] = _write_parts(
            loaded, mesh_name=mesh_name, mesh_root=staged_dir / meshdir
        )

    if not produced:
        raise DecompositionUnavailable("no collidable mesh geometry was found to decompose")

    for name, parts in produced.items():
        for part in parts:
            ET.SubElement(asset_node, "mesh", {"name": part, "file": f"{part}.stl"})

    for parent in worldbody.iter():
        for geom in list(parent.findall("geom")):
            mesh_name = geom.get("mesh")
            if mesh_name not in produced or not _is_collidable(geom):
                continue
            contype = geom.get("contype", "1")
            conaffinity = geom.get("conaffinity", "1")
            condim = geom.get("condim", "3")
            friction = geom.get("friction")
            # Demote the original: it keeps the mass and inertia and stops colliding.
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            for part in produced[mesh_name]:
                attrs = {
                    "name": f"{geom.get('name', mesh_name)}__{part}",
                    "type": "mesh",
                    "mesh": part,
                    "mass": "0",
                    "group": COLLISION_GROUP,
                    "contype": contype,
                    "conaffinity": conaffinity,
                    "condim": condim,
                    "rgba": "0.4 0.6 0.9 0.25",
                }
                if friction:
                    attrs["friction"] = friction
                for inherited in ("pos", "quat", "euler"):
                    if geom.get(inherited):
                        attrs[inherited] = geom.get(inherited)
                parent.append(ET.Element("geom", attrs))

    tree.write(target, encoding="unicode")
    return target


def _collision_mesh_geoms(worldbody: ET.Element) -> list[ET.Element]:
    return [
        geom
        for parent in worldbody.iter()
        for geom in parent.findall("geom")
        if geom.get("type") == "mesh" and _is_collidable(geom)
    ]


def _is_collidable(geom: ET.Element) -> bool:
    return geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0"


def _load_mesh(path: Path) -> trimesh.Trimesh | None:
    if not path.is_file():
        return None
    loaded = trimesh.load(path, force="mesh")
    return loaded if isinstance(loaded, trimesh.Trimesh) and not loaded.is_empty else None


def _write_parts(mesh: trimesh.Trimesh, *, mesh_name: str, mesh_root: Path) -> list[str]:
    import coacd  # noqa: PLC0415

    result = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        threshold=_THRESHOLD,
    )
    if not result:
        raise DecompositionUnavailable(f"CoACD returned no parts for mesh {mesh_name!r}")
    if len(result) > _MAX_PARTS:
        raise DecompositionUnavailable(
            f"mesh {mesh_name!r} decomposed into {len(result)} parts, past the {_MAX_PARTS} "
            "the contact solver is worth running with"
        )

    mesh_root.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for index, (vertices, faces) in enumerate(result):
        part = f"{mesh_name}_cvx{index:03d}"
        trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces)).export(
            mesh_root / f"{part}.stl"
        )
        names.append(part)
    return names
