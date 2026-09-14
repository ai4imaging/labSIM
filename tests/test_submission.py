"""What is allowed to count as a submitted asset, and what a run is called.

A sweep that produced 54 authoring failures still wrote 54 scorecards, because the parts
of the pipeline downstream of the author each went looking for a `model.py` and each
found one: a staging copy, a last-good revision from twenty turns earlier, an export
compiled with the quality checks switched off. None of those were submitted by anybody.
These tests pin the boundary: only what the author finished and stood behind is scored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from amx.bench.run import GenerationSettings, _find_model
from amx.cli import _bench_run_dir
from amx.llm import DEFAULT_GPUGEEK_MODEL, LlmClient

# --------------------------------------------------------------------------- #
# what counts as submitted
# --------------------------------------------------------------------------- #


def test_the_submitted_model_is_the_one_at_the_top_of_the_asset_directory(tmp_path: Path):
    asset = tmp_path / "asset"
    (asset / "mjcf").mkdir(parents=True)
    submitted = asset / "model.py"
    submitted.write_text("SUBMITTED = True\n")

    assert _find_model(asset) == submitted


def test_a_working_draft_left_behind_by_a_failed_run_is_not_a_submission(tmp_path: Path):
    """The newest `model.py` anywhere under `asset/` used to win, whatever it was."""
    asset = tmp_path / "asset"
    draft = asset / "records" / "rev-014" / "model.py"
    draft.parent.mkdir(parents=True)
    draft.write_text("HALF_FINISHED = True\n")
    (asset / "model.last-good.py").write_text("TWENTY_TURNS_AGO = True\n")

    assert _find_model(asset) is None


def _request():
    from amx.asset.spec import AssetRequest

    return AssetRequest(asset_id="mor_001", prompt="a mortar and pestle")


def test_failed_authoring_leaves_no_asset_to_score(tmp_path: Path, monkeypatch):
    """A non-zero exit means nothing was submitted, and nothing is written.

    The old behaviour compiled whatever draft it could find with the quality checks off
    and wrote it to `visualization/`, from where the viewer and the judge both picked it
    up. An asset the author's own compiler had rejected is not a partial result.
    """
    pytest.importorskip("cadquery", reason="the Articraft runner needs the geometry backend")
    from amx.asset.generate import AssetGenerationError, generate_asset
    from amx.paths import activate_articraft

    activate_articraft()
    import agent.runner  # noqa: PLC0415

    async def gave_up(*args, **kwargs):
        return 2

    monkeypatch.setattr(agent.runner, "run_from_input", gave_up)
    root = tmp_path / "asset"

    with pytest.raises(AssetGenerationError, match="no asset was produced"):
        generate_asset(_request(), output_root=root, provider="gpugeek")

    assert not (root.parent / "visualization").exists()
    assert not (root / "model.py").exists()


def test_the_active_revision_is_read_through_the_repository(tmp_path: Path, monkeypatch):
    """`active_model_path` takes a `StorageRepo`, not the layout underneath it.

    Handing it the layout raised `AttributeError: no attribute 'read_json'` after the
    authoring loop had already succeeded, which is the worst place for it: the asset was
    finished, and the sweep recorded a generation failure for it anyway.
    """
    pytest.importorskip("cadquery", reason="the Articraft runner needs the geometry backend")
    from amx.asset import generate as generate_module
    from amx.asset.generate import AssetBundle
    from amx.paths import activate_articraft

    activate_articraft()
    import agent.runner  # noqa: PLC0415
    import storage.revisions  # noqa: PLC0415
    from storage.repo import StorageRepo  # noqa: PLC0415

    async def finished(*args, **kwargs):
        return 0

    seen: dict[str, object] = {}

    def fake_active_model_path(repo, record_id):
        seen["repo"] = repo
        return tmp_path / "model.py"

    root = tmp_path / "asset"
    mjcf = root / "mjcf" / "asset.xml"
    bundle = AssetBundle(
        asset_id="mor_001",
        root=root,
        model_path=tmp_path / "model.py",
        urdf_path=root / "model.urdf",
        mjcf_path=mjcf,
        mesh_dir=mjcf.parent / "meshes",
    )
    monkeypatch.setattr(agent.runner, "run_from_input", finished)
    monkeypatch.setattr(storage.revisions, "active_model_path", fake_active_model_path)
    monkeypatch.setattr(generate_module, "bundle_from_model", lambda *a, **k: bundle)

    generate_module.generate_asset(_request(), output_root=root, provider="gpugeek")

    assert isinstance(seen["repo"], StorageRepo)


def test_the_judge_rebuild_runs_the_compiler_s_own_quality_checks():
    """Reading the default off the signature, because this is the whole guarantee.

    `build_grounded_asset` is what turns a submitted `model.py` into something MuJoCo can
    measure. Built with `run_checks=False`, parts that overlap in the rest pose and
    bodies with no mass reach the measurements as though the compiler had passed them.
    """
    import inspect

    from amx.grounding.build import build_grounded_asset

    assert inspect.signature(build_grounded_asset).parameters["run_checks"].default is True


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #


def test_a_sweep_is_named_for_when_it_ran(tmp_path: Path):
    directory = _bench_run_dir(argparse.Namespace(run_dir=None, run_id=None))

    assert directory.parent.name == "bench"
    stamp = directory.name
    assert stamp.startswith("results_")
    assert len(stamp) == len("results_20260911-134500")


def test_an_explicit_directory_or_id_still_wins(tmp_path: Path):
    assert _bench_run_dir(argparse.Namespace(run_dir=tmp_path, run_id=None)) == tmp_path
    named = _bench_run_dir(argparse.Namespace(run_dir=None, run_id="ablation-no-grounding"))
    assert named.name == "ablation-no-grounding"


def test_the_gateway_has_a_default_model_for_authoring_and_for_grading(monkeypatch):
    """A missing `GPUGEEK_MODEL` used to be a sweep in which every case failed."""
    monkeypatch.delenv("AMX_LLM_MODEL", raising=False)

    assert GenerationSettings().model_id == DEFAULT_GPUGEEK_MODEL
    assert LlmClient(provider="gpugeek").model == DEFAULT_GPUGEEK_MODEL


def test_a_model_id_is_only_filled_in_for_the_gateway_it_belongs_to():
    """Model ids are per-provider; handing a gateway slug to Anthropic is a 404."""
    assert GenerationSettings(provider="anthropic").model_id is None
    assert GenerationSettings(model_id="Vendor2/Something-Else").model_id == (
        "Vendor2/Something-Else"
    )
