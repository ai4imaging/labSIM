"""Part 1: turn a request into a simulation-ready asset.

Articraft already does the hard part — an LLM writes a Python model against its geometry
SDK, and a compile-repair loop iterates until the object builds. This module does not
reimplement any of that. It supplies the request, points Articraft's record store at this
project, and then walks the tail of the pipeline that Articraft itself stops short of:
compile the record to URDF and export it to MJCF.

`AssetBundle` is the only thing the rest of the pipeline sees, and `bundle_from_model` can
produce one without an LLM, which is what the tests and examples use.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from amx.asset.spec import AssetRequest
from amx.paths import ASSETS_DIR, activate_articraft

if TYPE_CHECKING:
    from amx.grounding.spec import GroundingSpec

SYSTEM_PROMPT = "designer_system_prompt.txt"
"""The bare filename, not a path.

Articraft's loader only substitutes the provider-specific variant — `_openrouter`,
`_anthropic` and so on — when it is handed a name it recognises. Passing a path instead
defeats that, and the base name it then looks for does not exist, so the run dies at
construction. The bare name is what the rest of Articraft passes, and it is what makes
the gpugeek profile actually select a prompt.
"""


class AssetGenerationError(RuntimeError):
    """Articraft could not author or compile an object for this request."""


@dataclass(frozen=True)
class AssetBundle:
    """A compiled asset on disk, ready to be placed into a workcell."""

    asset_id: str
    root: Path
    model_path: Path
    urdf_path: Path
    mjcf_path: Path
    mesh_dir: Path

    @property
    def controller_contract_path(self) -> Path:
        return self.mjcf_path.parent / "controller_contract.json"

    def metadata(self) -> dict[str, Any]:
        path = self.root / "asset.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    @classmethod
    def load(cls, root: Path) -> "AssetBundle":
        root = Path(root).resolve()
        meta = json.loads((root / "asset.json").read_text())
        return cls(
            asset_id=meta["asset_id"],
            root=root,
            model_path=root / "model.py",
            urdf_path=root / "model.urdf",
            mjcf_path=root / "mjcf" / "asset.xml",
            mesh_dir=root / "mjcf" / "meshes",
        )


def bundle_from_model(
    model_path: Path,
    *,
    asset_id: str,
    output_root: Path | None = None,
    run_checks: bool = True,
) -> AssetBundle:
    """Compile an Articraft `model.py` and export it to MJCF.

    This is the whole back half of part 1. `generate_asset` calls it after the authoring
    agent lands a model; call it directly when you already have one.
    """
    activate_articraft()
    from agent.compiler import compile_urdf_report  # noqa: PLC0415
    from agent.mujoco_export import export_record_to_mjcf  # noqa: PLC0415

    model_path = Path(model_path).resolve()
    root = Path(output_root or ASSETS_DIR / asset_id).resolve()
    root.mkdir(parents=True, exist_ok=True)

    staged = root / "model.py"
    if staged != model_path:
        shutil.copy2(model_path, staged)
        # Meshes referenced by the model are relative to it, so they travel with it.
        source_assets = model_path.parent / "assets"
        if source_assets.is_dir():
            shutil.copytree(source_assets, root / "assets", dirs_exist_ok=True)

    try:
        report = compile_urdf_report(staged, run_checks=run_checks, target="full")
    except Exception as error:  # noqa: BLE001 - Articraft raises a wide range of build errors
        raise AssetGenerationError(f"{asset_id}: compile failed: {error}") from error

    urdf_path = root / "model.urdf"
    urdf_path.write_text(report.urdf_xml)

    result = export_record_to_mjcf(
        model_path=staged,
        urdf_path=urdf_path,
        output_dir=root / "mjcf",
        record_id=asset_id,
    )

    bundle = AssetBundle(
        asset_id=asset_id,
        root=root,
        model_path=staged,
        urdf_path=urdf_path,
        mjcf_path=result.asset_xml_path,
        mesh_dir=root / "mjcf" / "meshes",
    )
    _write_metadata(bundle, warnings=list(report.warnings))
    return bundle


def generate_asset(
    request: AssetRequest,
    *,
    provider: str = "anthropic",
    model_id: str | None = None,
    thinking_level: str = "medium",
    output_root: Path | None = None,
    grounding_spec: "GroundingSpec | None" = None,
    grounding_blocks: bool = True,
    visual_blocks: bool = False,
) -> AssetBundle:
    """Author a new asset with Articraft, then compile and export it.

    Articraft's record store is pointed at `assets/<asset_id>/articraft/`, so a run leaves
    its whole provenance — every turn, every compile attempt — beside the asset, and never
    writes into the vendored tree.

    Passing a `grounding_spec` swaps in the grounded agent, which offers the model three
    measurement tools and refuses to let it finish while any of them is failing or has
    gone stale. Without one the run is plain Articraft: it stops as soon as the code
    compiles, whatever the code happens to describe.
    """
    root = Path(output_root or ASSETS_DIR / request.asset_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_root = root / "articraft"
    data_root.mkdir(exist_ok=True)

    articraft_root = activate_articraft()
    from agent.runner import run_from_input  # noqa: PLC0415
    from storage.repo import StorageRepo  # noqa: PLC0415
    from storage.revisions import active_model_path  # noqa: PLC0415

    record_id = f"rec_{request.asset_id}"
    run_kwargs = {
        "prompt_text": request.authoring_prompt(),
        "display_prompt": None,
        "repo_root": articraft_root,
        "image_path": _first_image(request),
        "data_root": data_root,
        "provider": provider,
        "model_id": model_id,
        "thinking_level": thinking_level,
        "max_turns": request.max_turns,
        "system_prompt_path": SYSTEM_PROMPT,
        "display_enabled": False,
        "record_id": record_id,
        "label": request.asset_id,
    }
    # Stock Articraft already constructs its own ArticraftAgent and older releases
    # reject this bioSIM extension hook. Only grounded runs need to override it.
    if grounding_spec is not None:
        run_kwargs["agent_cls"] = _agent_factory(
            grounding_spec,
            grounding_dir=root / "grounding",
            grounding_blocks=grounding_blocks,
            visual_blocks=visual_blocks,
        )
    exit_code = asyncio.run(
        run_from_input(
            request.authoring_prompt(),
            **run_kwargs,
        )
    )
    if exit_code != 0:
        # No asset is written. The authoring loop stopped without ever measuring the code
        # it left on disk, so what is there is a draft: a lid halfway through being
        # re-hinged, or the empty scaffold from a run that never got a reply. Exporting
        # one anyway is what produced a benchmark full of assets that could be rendered
        # and were wrong. The trace, the error and Articraft's own record survive under
        # `data_root`, which is where a failure is diagnosed.
        raise AssetGenerationError(
            f"{request.asset_id}: Articraft authoring exited with code {exit_code}; "
            f"no asset was produced. The authoring record and every turn are under {data_root}"
        )

    repo = StorageRepo(root=articraft_root, data_root=data_root)
    model_path = active_model_path(repo, record_id)
    bundle = bundle_from_model(model_path, asset_id=request.asset_id, output_root=root)
    _write_metadata(
        bundle,
        request=request,
        provenance={"record_id": record_id, "provider": provider, "model_id": model_id},
    )
    return bundle


def _agent_factory(
    spec: "GroundingSpec | None",
    *,
    grounding_dir: Path,
    grounding_blocks: bool,
    visual_blocks: bool,
):
    """The agent class Articraft should construct, with grounding already bound to it.

    Articraft builds the agent itself, deep inside its own run pipeline, so the only way
    to hand a subclass extra configuration is to pre-bind it. A partial does that without
    Articraft needing to know any of these parameters exist.
    """
    from agent.harness import ArticraftAgent  # noqa: PLC0415

    if spec is None or spec.is_empty():
        return ArticraftAgent

    from functools import partial  # noqa: PLC0415

    from agent.grounded_harness import GroundedArticraftAgent  # noqa: PLC0415

    grounding_dir.mkdir(parents=True, exist_ok=True)
    spec.write(grounding_dir / "grounding-spec.json")
    return partial(
        GroundedArticraftAgent,
        grounding_spec=spec,
        grounding_dir=grounding_dir,
        grounding_blocks=grounding_blocks,
        visual_blocks=visual_blocks,
    )


def _first_image(request: AssetRequest) -> Path | None:
    for path in request.reference_files:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return path
    return None


def _write_metadata(
    bundle: AssetBundle,
    *,
    request: AssetRequest | None = None,
    warnings: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    path = bundle.root / "asset.json"
    payload: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
    payload["asset_id"] = bundle.asset_id
    payload["mjcf"] = str(bundle.mjcf_path.relative_to(bundle.root))
    if warnings is not None:
        payload["compile_warnings"] = warnings
    if request is not None:
        payload["request"] = request.model_dump(mode="json")
    if provenance is not None:
        payload["provenance"] = provenance
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
