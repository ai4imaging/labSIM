"""Getting from an Articraft `model.py` to something MuJoCo can load.

The grounding tools run inside the authoring loop, where the model may call them
several times without having changed anything — after reading a file, after a failed
edit, or just to look again. Compiling an Articraft model runs CadQuery and OCC and is
by far the slowest thing in the loop, so the result is cached against a hash of the
source and its mesh assets. The hash covers the assets directory too, because a model
that only references a mesh looks unchanged when the mesh underneath it is not.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from amx.paths import PROJECT_ROOT, activate_articraft

CACHE_ROOT = PROJECT_ROOT / ".cache" / "grounding"


class GroundingBuildError(RuntimeError):
    """The model could not be compiled or exported, so there is nothing to measure."""


@dataclass(frozen=True)
class GroundedAsset:
    """A compiled asset on disk, ready to be measured."""

    asset_id: str
    root: Path
    model_path: Path
    urdf_path: Path
    mjcf_path: Path
    source_sha: str
    compile_warnings: list[str] = field(default_factory=list)
    from_cache: bool = False

    @property
    def mesh_dir(self) -> Path:
        return self.mjcf_path.parent / "meshes"


def source_fingerprint(model_path: Path) -> str:
    """A hash over the model source and every mesh it can reach.

    Sorted by relative path so the digest does not depend on directory order, and the
    relative path is hashed alongside the bytes so that renaming a mesh counts as a
    change.
    """
    model_path = Path(model_path).resolve()
    digest = hashlib.sha256()
    digest.update(model_path.read_bytes())

    assets = model_path.parent / "assets"
    if assets.is_dir():
        for entry in sorted(assets.rglob("*")):
            if entry.is_file():
                digest.update(str(entry.relative_to(assets)).encode())
                digest.update(entry.read_bytes())
    return digest.hexdigest()[:16]


def build_grounded_asset(
    model_path: Path,
    *,
    asset_id: str,
    cache_root: Path | None = None,
    use_cache: bool = True,
    run_checks: bool = True,
) -> GroundedAsset:
    """Compile `model.py` to URDF and export it to MJCF, reusing an earlier build.

    Raises `GroundingBuildError` rather than letting Articraft's own exception types
    escape: a caller measuring an asset should not have to know what CadQuery raises
    when a fillet fails.

    `run_checks=False` is for looking only. It skips the compiler's own quality checks,
    which is the only way to get a rejected draft on screen to see what went wrong with
    it. Nothing that measures or scores an asset may pass it: an asset that reached the
    judge without those checks is an asset whose overlaps were never counted.
    """
    model_path = Path(model_path).resolve()
    fingerprint = source_fingerprint(model_path)
    # The flag is part of the cache key, or a viewer's unchecked build would be handed
    # back to the judge as though it had been checked.
    suffix = "" if run_checks else "-unchecked"
    root = Path(cache_root or CACHE_ROOT) / f"{asset_id}-{fingerprint}{suffix}"
    marker = root / "build.json"

    if use_cache and marker.is_file():
        try:
            recorded = json.loads(marker.read_text())
            cached = GroundedAsset(
                asset_id=asset_id,
                root=root,
                model_path=root / "model.py",
                urdf_path=root / "model.urdf",
                mjcf_path=Path(recorded["mjcf"]),
                source_sha=fingerprint,
                compile_warnings=list(recorded.get("warnings", [])),
                from_cache=True,
            )
            if cached.mjcf_path.is_file():
                return cached
        except (OSError, ValueError, KeyError):
            # A half-written cache entry is not worth diagnosing; rebuild over it.
            shutil.rmtree(root, ignore_errors=True)

    activate_articraft()
    from agent.compiler import compile_urdf_report  # noqa: PLC0415
    from agent.mujoco_export import export_record_to_mjcf  # noqa: PLC0415

    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    staged = root / "model.py"
    shutil.copy2(model_path, staged)
    source_assets = model_path.parent / "assets"
    if source_assets.is_dir():
        shutil.copytree(source_assets, root / "assets", dirs_exist_ok=True)

    try:
        # The same checks the authoring loop's own `compile_model` runs. Skipping them
        # here let an asset that the author's compiler had rejected — parts overlapping
        # in the rest pose, a body with no mass, geometry that cannot collide — arrive at
        # the measurements as though it had passed, and be measured, and be scored.
        report = compile_urdf_report(staged, run_checks=run_checks, target="full")
    except Exception as error:  # noqa: BLE001 - Articraft raises a wide range of build errors
        raise GroundingBuildError(f"compile failed: {error}") from error

    urdf_path = root / "model.urdf"
    urdf_path.write_text(report.urdf_xml)

    try:
        exported = export_record_to_mjcf(
            model_path=staged,
            urdf_path=urdf_path,
            output_dir=root / "mjcf",
            record_id=asset_id,
        )
    except Exception as error:  # noqa: BLE001 - export failures are as varied as compiles
        raise GroundingBuildError(f"MJCF export failed: {error}") from error

    warnings = [str(w) for w in report.warnings]
    marker.write_text(
        json.dumps({"mjcf": str(exported.asset_xml_path), "warnings": warnings}, indent=2) + "\n"
    )
    return GroundedAsset(
        asset_id=asset_id,
        root=root,
        model_path=staged,
        urdf_path=urdf_path,
        mjcf_path=exported.asset_xml_path,
        source_sha=fingerprint,
        compile_warnings=warnings,
    )
