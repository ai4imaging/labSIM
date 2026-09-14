"""Best-first search over generated `model.py` candidates.

A single Articraft run is a repair loop of width one: it follows the model's next edit
and cannot go back. This wraps that loop in `amx.search`, so a promising first draft that
then gets worse can be abandoned for a sibling, and two different readings of the same
failure can be tried instead of paraphrased.

Each node is one full Articraft authoring run. That is expensive, which is why the
default generation path does not come through here — the caller has to ask, with
`max_nodes > 1`.
"""

from __future__ import annotations

from pathlib import Path

from amx.asset.generate import AssetBundle, AssetGenerationError, bundle_from_model, generate_asset
from amx.asset.spec import AssetRequest
from amx.grounding.build import build_grounded_asset
from amx.grounding.checks import ground_against_spec
from amx.grounding.spec import GroundingSpec
from amx.report import Finding, RepairTarget, Report, Severity
from amx.search import Node, SearchConfig, search
from amx.skills.learn import learn_from_search
from amx.skills.library import SkillLibrary


def generate_with_search(
    request: AssetRequest,
    *,
    spec: GroundingSpec,
    output_root: Path,
    search_dir: Path,
    provider: str = "gpugeek",
    model_id: str | None = None,
    thinking_level: str = "medium",
    grounding_blocks: bool = True,
    visual_blocks: bool = False,
    config: SearchConfig | None = None,
) -> AssetBundle:
    """Generate several candidates and keep the best one the checks can measure."""
    config = config or SearchConfig()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    search_dir = Path(search_dir)
    library = SkillLibrary()
    library.activate()

    seed = _author(
        request,
        spec=spec,
        output_root=output_root / "search-seed",
        provider=provider,
        model_id=model_id,
        thinking_level=thinking_level,
        grounding_blocks=grounding_blocks,
        visual_blocks=visual_blocks,
        extra_prompt=_briefing(library, spec),
    )
    if not seed.source:
        raise AssetGenerationError(
            f"{request.asset_id}: the seed run produced no model.py; see {output_root / 'search-seed'}"
        )

    def evaluate(source: str, node: Node) -> Report:
        node_dir = search_dir / f"eval-{node.id:04d}"
        (node_dir / "model.py").parent.mkdir(parents=True, exist_ok=True)
        (node_dir / "model.py").write_text(source)
        try:
            asset = build_grounded_asset(node_dir / "model.py", asset_id=f"{request.asset_id}-n{node.id}")
        except Exception as error:  # noqa: BLE001
            report = Report(kind="grounding", subject=request.asset_id)
            report.findings.append(
                Finding(
                    code="G-UNMEASURABLE",
                    severity=Severity.FAILURE,
                    subject=request.asset_id,
                    summary=f"candidate {node.id} did not compile: {error}",
                    repair_target=RepairTarget.ASSET,
                )
            )
            return report
        return ground_against_spec(
            asset,
            spec,
            include_visual=False,
            visual_blocks=visual_blocks,
            image_dir=node_dir / "renders",
        )

    def expand(node: Node, wanted: int) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        failures = "\n".join(f.summary for f in (node.report.failures if node.report else [])[:12])
        for index in range(wanted):
            extra = (
                _briefing(library, spec)
                + "\n\nA previous attempt is below. Do not rewrite it from scratch; "
                "keep what already measures correctly and repair the listed failures.\n\n"
                f"<previous_model>\n{node.source}\n</previous_model>\n"
            )
            if failures:
                extra += f"\n<remaining_failures>\n{failures}\n</remaining_failures>\n"
            if index:
                extra += (
                    "\nA sibling of this candidate is being evaluated separately. Take a "
                    "materially different reading of the failures.\n"
                )
            child = _author(
                request,
                spec=spec,
                output_root=output_root / f"search-n{node.id:04d}-s{index}",
                provider=provider,
                model_id=model_id,
                thinking_level=thinking_level,
                grounding_blocks=grounding_blocks,
                visual_blocks=visual_blocks,
                extra_prompt=extra,
            )
            if child.source:
                out.append((child.source, f"from-{node.id}-sample-{index}"))
        return out

    result = search(
        root_source=seed.source,
        evaluate=evaluate,
        expand=expand,
        run_dir=search_dir,
        config=config,
    )
    (search_dir / "search.txt").write_text(result.to_text() + "\n")
    try:
        learn_from_search(result.tree, library=library, asset_class=spec.asset_class, case_id=spec.asset_id)
    except Exception:  # noqa: BLE001 — a lesson that fails to write is not a failed generation
        pass

    winner = result.solved or result.best
    if winner is None or not winner.source:
        if seed.bundle is not None:
            return seed.bundle
        raise AssetGenerationError(f"{request.asset_id}: search produced no usable candidate")

    (output_root / "model.py").write_text(winner.source)
    if seed.bundle is not None and winner.id == 0:
        return seed.bundle
    return bundle_from_model(output_root / "model.py", asset_id=request.asset_id, output_root=output_root)


class _Authored:
    bundle: AssetBundle | None
    source: str

    def __init__(self, bundle: AssetBundle | None, source: str) -> None:
        self.bundle = bundle
        self.source = source


def _author(
    request: AssetRequest,
    *,
    spec: GroundingSpec,
    output_root: Path,
    provider: str,
    model_id: str | None,
    thinking_level: str,
    grounding_blocks: bool,
    visual_blocks: bool,
    extra_prompt: str = "",
) -> _Authored:
    """One Articraft run. Returns whatever `model.py` it left, even if the run failed."""
    patched = request
    if extra_prompt.strip():
        patched = request.model_copy(update={"prompt": request.prompt + "\n\n" + extra_prompt})
    bundle = None
    try:
        bundle = generate_asset(
            patched,
            provider=provider,
            model_id=model_id,
            thinking_level=thinking_level,
            output_root=output_root,
            grounding_spec=spec,
            grounding_blocks=grounding_blocks,
            visual_blocks=visual_blocks,
        )
        source = bundle.model_path.read_text() if bundle.model_path.is_file() else ""
        return _Authored(bundle, source)
    except AssetGenerationError:
        source = _find_source(output_root)
        return _Authored(None, source)


def _find_source(root: Path) -> str:
    candidates = sorted(Path(root).rglob("model.py"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].read_text() if candidates else ""


def _briefing(library: SkillLibrary, spec: GroundingSpec) -> str:
    codes = [target.id for target in spec.dimensions] + [c.id for c in spec.components]
    lessons = library.relevant(codes=codes, asset_class=spec.asset_class)
    return library.briefing(lessons)
