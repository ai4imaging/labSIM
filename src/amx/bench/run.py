"""Running the benchmark: generate, judge, and sweep.

Generation and judging are separate entry points that meet on disk, which is what makes
this usable. A sweep across 64 cases takes hours and will be interrupted; keeping the
submitted asset as a plain directory means judging can be re-run against it as often as
the judge changes, and a resumed sweep can tell what is already done by looking rather
than by consulting a database it also has to keep correct.

Every run writes into one directory per case:

    runs/<sweep>/<CASE-ID>/
      asset/                 the generated asset, model.py and MJCF and meshes
      agent/turns/turn-NNN/  every ReAct turn: prompt, response, tools, signals, renders
      grounding-spec.json    what the loop held itself to, derived from input.md
      rubric.json            the compiled answer key this was scored against
      checks/                 per-item evidence: reports, measurements, renders
      scorecard.json         the score, the gates, the certification status
      scorecard.txt          the same, readable
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amx.bench.case import BenchCase, discover, load_case
from amx.bench.compiler import load_rubric
from amx.bench.judge import Scorecard, judge_asset
from amx.grounding.build import GroundedAsset, build_grounded_asset
from amx.grounding.spec import GroundingSpec
from amx.llm import DEFAULT_GPUGEEK_MODEL, LlmClient


@dataclass
class GenerationSettings:
    provider: str = "gpugeek"
    model_id: str | None = None
    """Which model authors the asset. Left unset, the gateway's default is filled in below.

    Only for `gpugeek`: model ids are per-provider, and handing a gateway slug to the
    Anthropic client is a 404 rather than a fallback.
    """

    thinking_level: str = "medium"
    max_turns: int = 100
    grounding: bool = True
    """Whether the generator gets the measurement tools and the finish gate.

    Off is the honest baseline: plain Articraft, which stops as soon as the code compiles.
    The difference between the two columns is the thing this project is claiming.
    """

    visual_blocks: bool = False
    concise_prompt: bool = False
    """Show plain Articraft one AI-compressed paragraph instead of the full specification."""
    samples: int = 1
    beam: int = 3
    max_nodes: int = 1
    """When greater than 1, generation is a best-first search over Articraft runs.

    One is the cheap path the sweep uses: a single authoring loop. More than one is what
    the plan's `--beam / --samples / --max-nodes` flags are for, and it costs that many
    full authoring runs, so it is opt-in.
    """

    def __post_init__(self) -> None:
        if not self.model_id and self.provider == "gpugeek":
            self.model_id = DEFAULT_GPUGEEK_MODEL


def prepare(case: BenchCase, case_dir: Path, *, request_text: str | None = None) -> Path:
    """Write the generation inputs, without generating anything.

    Useful on its own — it is how you check what the model will be shown before spending
    an hour of API time on 64 of them.
    """
    case_dir = Path(case_dir).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "request.md").write_text((request_text or case.to_asset_request()) + "\n")
    case.to_grounding_spec().write(case_dir / "grounding-spec.json")
    case_copy = case_dir / "case"
    case_copy.mkdir(exist_ok=True)
    (case_copy / "input.md").write_text(case.input_path.read_text())
    return case_dir


def generate(
    case: BenchCase,
    case_dir: Path,
    *,
    settings: GenerationSettings | None = None,
) -> Path:
    """Generate the asset for one case. Returns the asset directory."""
    from amx.asset.generate import generate_asset  # noqa: PLC0415
    from amx.asset.spec import AssetRequest  # noqa: PLC0415

    case_dir = Path(case_dir).resolve()
    settings = settings or GenerationSettings()
    prompt = case.to_asset_request()
    if settings.concise_prompt:
        from amx.bench.concise_prompt import concise_prompt  # noqa: PLC0415

        prompt = concise_prompt(
            case,
            case_dir,
            provider=settings.provider,
            model_id=settings.model_id,
            thinking_level=settings.thinking_level,
        )
    prepare(case, case_dir, request_text=prompt)
    spec = case.to_grounding_spec()
    if settings.grounding:
        try:
            from amx.skills.library import SkillLibrary  # noqa: PLC0415

            SkillLibrary().activate()
        except Exception:  # noqa: BLE001 — a missing library must not block generation
            pass

    request = AssetRequest(
        asset_id=_asset_id(case),
        prompt=prompt,
        max_turns=settings.max_turns,
    )
    error: Exception | None = None
    bundle = None
    try:
        if settings.max_nodes > 1 and settings.grounding:
            from amx.asset.search_generate import generate_with_search  # noqa: PLC0415
            from amx.search import SearchConfig  # noqa: PLC0415

            bundle = generate_with_search(
                request,
                spec=spec,
                output_root=case_dir / "asset",
                search_dir=case_dir / "agent" / "search",
                provider=settings.provider,
                model_id=settings.model_id,
                thinking_level=settings.thinking_level,
                visual_blocks=settings.visual_blocks,
                config=SearchConfig(
                    max_nodes=settings.max_nodes,
                    samples=max(1, settings.samples),
                    beam=max(1, settings.beam),
                ),
            )
        else:
            bundle = generate_asset(
                request,
                provider=settings.provider,
                model_id=settings.model_id,
                thinking_level=settings.thinking_level,
                output_root=case_dir / "asset",
                grounding_spec=spec if settings.grounding else None,
                visual_blocks=settings.visual_blocks,
            )
    except Exception as raised:  # noqa: BLE001 — traces still have to land
        error = raised
    _export_agent_trace(case_dir)
    if error is not None:
        raise error
    assert bundle is not None
    return Path(bundle.mjcf_path).parent


def _export_agent_trace(case_dir: Path) -> None:
    """Expand Articraft's JSONL even when the authoring run itself failed.

    Hitting the turn budget is a normal outcome, not a missing transcript. The user asked
    for every ReAct turn on disk; skipping the export because Articraft returned 2 would
    throw away the only evidence of what the loop actually did.
    """
    from amx.trace_export import export_trace  # noqa: PLC0415

    trace_dir = _trace_dir(Path(case_dir) / "asset")
    if trace_dir is None:
        return
    try:
        export_trace(trace_dir, Path(case_dir) / "agent")
        trajectory = trace_dir / "trajectory.jsonl"
        dest = Path(case_dir) / "agent" / "trajectory.jsonl"
        if trajectory.is_file() and not dest.exists():
            dest.write_bytes(trajectory.read_bytes())
    except Exception as error:  # noqa: BLE001
        (Path(case_dir) / "agent-export-error.txt").write_text(f"{error}\n")


def judge(
    case: BenchCase,
    case_dir: Path,
    *,
    vision: bool = True,
) -> Scorecard:
    """Measure and score whatever asset is sitting in `case_dir`.

    The rubric is compiled from the case's `input.md`, and a copy of the one that was
    used lands beside the scorecard — a score nobody can reproduce the answer key for is
    not a result.
    """
    case_dir = Path(case_dir).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)

    rubric = load_rubric(case)
    rubric.write(case_dir / "rubric.json")
    asset, build_note = _load_asset(case, case_dir)
    card = judge_asset(
        rubric,
        asset,
        work_dir=case_dir / "checks",
        visual_client=_client() if vision else None,
        build_note=build_note,
    )
    card.write(case_dir / "scorecard.json")
    (case_dir / "scorecard.txt").write_text(card.to_text() + "\n")
    return card


def run_case(
    case: BenchCase,
    run_dir: Path,
    *,
    settings: GenerationSettings | None = None,
    generate_asset_first: bool = True,
    vision: bool = True,
) -> Scorecard:
    """Generate then judge one case, recording a failure as a scorecard rather than raising."""
    case_dir = Path(run_dir).resolve() / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    print(f"running {case.case_id}", flush=True)
    started = time.monotonic()

    if generate_asset_first:
        try:
            generate(case, case_dir, settings=settings)
        except Exception as error:  # noqa: BLE001
            (case_dir / "generation-error.txt").write_text(
                f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}"
            )

    _record_generation(case_dir, settings)
    card = judge(case, case_dir, vision=vision)
    (case_dir / "timing.json").write_text(
        json.dumps({"elapsed_s": round(time.monotonic() - started, 1)}, indent=2) + "\n"
    )
    return card


def _record_generation(case_dir: Path, settings: GenerationSettings | None) -> None:
    """How many turns the authoring loop used, and whether it ran out of them.

    A case that stops because it hit the ceiling stops at an arbitrary point, and that
    is the largest remaining source of score variance between two runs of the same
    case — larger than anything in the rubric. Inferring it afterwards from the trace
    is possible but nobody does it, so it is recorded here as a fact about the run.
    """
    budget = (settings or GenerationSettings()).max_turns
    used = len(list((case_dir / "agent" / "turns").glob("turn-*")))
    try:
        (case_dir / "generation.json").write_text(
            json.dumps(
                {
                    "turn_budget": budget,
                    "turns_used": used,
                    "budget_exhausted": used >= budget,
                    "asset_submitted": (case_dir / "asset" / "model.py").is_file(),
                },
                indent=2,
            )
            + "\n"
        )
    except OSError:
        pass


def sweep(
    cases: list[BenchCase],
    run_dir: Path,
    *,
    settings: GenerationSettings | None = None,
    workers: int = 1,
    resume: bool = False,
    generate_asset_first: bool = True,
    vision: bool = True,
) -> list[Scorecard]:
    """Run many cases, optionally in parallel, and write a leaderboard.

    Workers are threads rather than processes because almost all of the wall time is spent
    waiting on the API, and threads keep the whole sweep in one process where a single
    Ctrl-C stops it and the partial results on disk are still complete per case.
    """
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        case
        for case in cases
        if not (resume and (run_dir / case.case_id / "scorecard.json").is_file())
    ]
    skipped = len(cases) - len(pending)
    if skipped:
        print(f"resuming: {skipped} case(s) already scored", flush=True)

    cards: list[Scorecard] = []
    if workers <= 1:
        for index, case in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] {case.case_id} {case.asset_class}", flush=True)
            cards.append(
                run_case(
                    case,
                    run_dir,
                    settings=settings,
                    generate_asset_first=generate_asset_first,
                    vision=vision,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_case,
                    case,
                    run_dir,
                    settings=settings,
                    generate_asset_first=generate_asset_first,
                    vision=vision,
                ): case
                for case in pending
            }
            for done, future in enumerate(as_completed(futures), start=1):
                case = futures[future]
                try:
                    card = future.result()
                except Exception as error:  # noqa: BLE001
                    print(f"[{done}/{len(pending)}] {case.case_id} crashed: {error}", flush=True)
                    continue
                cards.append(card)
                print(
                    f"[{done}/{len(pending)}] {case.case_id} "
                    f"{card.score:.1f}/{card.total_score:.0f} {card.status}",
                    flush=True,
                )

    cards.extend(_read_existing(run_dir, cases, [card.case_id for card in cards]))
    write_leaderboard(cards, run_dir)
    return sorted(cards, key=lambda card: card.case_id)


def write_leaderboard(cards: list[Scorecard], run_dir: Path) -> Path:
    """A single table of every case, plus the aggregates that actually mean something.

    Reported alongside the mean score: how many certified, and how many are blocked rather
    than failed. A mean on its own can look respectable while nothing at all certifies,
    and that distinction is the point of the rubric's status logic.
    """
    ordered = sorted(cards, key=lambda card: (-_reached(card), card.case_id))
    lines = [
        f"{len(cards)} case(s)",
        "",
        f"{'case':<10} {'class':<28} {'score':>7} {'reach':>7} {'of':>6}  status",
        "-" * 84,
    ]
    for card in ordered:
        lines.append(
            f"{card.case_id:<10} {card.asset_class[:28]:<28} "
            f"{card.score:>6.1f} {_reached(card) * 100:>6.1f}% {card.achievable:>5.0f}  "
            f"{card.status}"
        )
    if cards:
        certified = sum(1 for card in cards if card.certified)
        blocked = sum(1 for card in cards if card.status == "blocked_dependency")
        invalid = sum(1 for card in cards if card.status == "invalid_test")
        mean = sum(card.score for card in cards) / len(cards)
        reached = sum(_reached(card) for card in cards) / len(cards)
        lines.extend(
            [
                "-" * 84,
                f"mean score {mean:.1f}/100   mean reached {reached * 100:.1f}% of what "
                f"the rubric leaves reachable",
                f"certified {certified}/{len(cards)}   blocked {blocked}   invalid {invalid}",
                "",
                *_leaderboard_note(),
            ]
        )
    text = "\n".join(lines)
    (run_dir / "leaderboard.txt").write_text(text + "\n")
    (run_dir / "leaderboard.json").write_text(
        json.dumps([card.as_dict() for card in ordered], indent=2, ensure_ascii=False) + "\n"
    )
    return run_dir / "leaderboard.txt"


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #


def _leaderboard_note() -> list[str]:
    """What a reader has to know before taking the count of certifications seriously."""
    return [
        "Three axes. A failed item costs its own weight; an asset MuJoCo will not load",
        "scores near zero because the items that need the model cannot be measured, not",
        "because a gate wiped the rest. The `of` column is what this case leaves reachable",
        "after its unmeasurable targets are subtracted, so the count of certifications",
        "means what it says rather than being unreachable by construction.",
    ]


def _reached(card: Scorecard) -> float:
    """Fraction of the reachable credit this submission took."""
    ceiling = card.achievable or card.total_score
    return card.score / ceiling if ceiling > 0 else 0.0


def _asset_id(case: BenchCase) -> str:
    """`BEA-001` is not a valid asset id; `bea_001_beaker` is."""
    return case.slug.lower().replace("-", "_").replace(".", "_")


def _read_spec(case: BenchCase, case_dir: Path) -> GroundingSpec:
    """Prefer the spec that was written when the asset was generated.

    Re-deriving it would usually give the same answer, but if the derivation has changed
    since the run then the judge would be measuring against something the generator was
    never shown, which is the exact failure this whole separation exists to prevent.
    """
    path = case_dir / "grounding-spec.json"
    if path.is_file():
        try:
            return GroundingSpec.read(path)
        except ValueError:
            pass
    return case.to_grounding_spec()


def _load_asset(case: BenchCase, case_dir: Path) -> tuple[GroundedAsset | None, str]:
    model = _find_model(case_dir / "asset")
    if model is None:
        return None, "no asset was submitted for this case"
    try:
        return build_grounded_asset(model, asset_id=case.case_id), ""
    except Exception as error:  # noqa: BLE001
        return None, f"the submitted model did not compile: {type(error).__name__}: {error}"


def _find_model(asset_root: Path) -> Path | None:
    """The submitted model, and only that.

    This used to fall back to the newest `model.py` anywhere under `asset/`, which meant
    Articraft's own working directories: a staging copy from a run that never finished, a
    `model.last-good.py` from twenty turns before the end. Those are drafts. Scoring the
    newest file that happens to exist reported a number for a case where nothing was ever
    submitted, and the number was not a measurement of anything the author stood behind.
    """
    direct = asset_root / "model.py"
    return direct if direct.is_file() else None


def _trace_dir(asset_root: Path) -> Path | None:
    candidates = sorted(asset_root.rglob("trajectory.jsonl"))
    return candidates[-1].parent if candidates else None


def _client() -> LlmClient | None:
    try:
        client = LlmClient()
    except ValueError:
        return None
    return client if client.available else None


def _read_existing(
    run_dir: Path, cases: list[BenchCase], already: list[str]
) -> list[Scorecard]:
    """Pull scorecards written by an earlier, interrupted sweep back into the leaderboard."""
    recovered = []
    for case in cases:
        if case.case_id in already:
            continue
        path = run_dir / case.case_id / "scorecard.json"
        if not path.is_file():
            continue
        try:
            payload: dict[str, Any] = json.loads(path.read_text())
        except ValueError:
            continue
        recovered.append(
            Scorecard(
                case_id=payload.get("case_id", case.case_id),
                asset_class=payload.get("asset_class", case.asset_class),
                score=float(payload.get("score") or 0.0),
                total_score=float(payload.get("total_score") or 100.0),
                achievable=float(payload.get("achievable") or 0.0),
                status=str(payload.get("status") or "failed"),
                reasons=list(payload.get("reasons") or []),
                axes=dict(payload.get("categories") or {}),
            )
        )
    return recovered


__all__ = [
    "GenerationSettings",
    "discover",
    "generate",
    "judge",
    "load_case",
    "prepare",
    "run_case",
    "sweep",
    "write_leaderboard",
]
