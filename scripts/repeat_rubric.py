"""How much of the score moves when nothing about the asset changes.

The old benchmark's same-case repeat spread averaged 9.5 points, with BAL-001 going
46.7 then 20.9 and SPC-001 going 55 then 30. Until that number is small, no comparison
between two frameworks means anything, because the gap between them was smaller than
the gap between one framework and itself.

The spread has two sources and they need separating, so this measures them separately:

    --judge-only    re-score the same file N times. Any spread here is the rubric's own,
                    and it must be exactly zero — there is no model in the scoring path
                    and MuJoCo from a fixed pose is deterministic.

    (default)       author the asset N times and score each. The spread here is the
                    authoring loop's, and it is what a turn budget and a decoding
                    temperature control.

    uv run python scripts/repeat_rubric.py --judge-only --run articraft-concise-opus5-v2
    uv run python scripts/repeat_rubric.py --cases BAL-001 SPC-001 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amx.bench.case import load_case  # noqa: E402
from amx.bench.compiler import load_rubric  # noqa: E402
from amx.bench.judge import judge_asset  # noqa: E402
from amx.grounding.build import build_grounded_asset  # noqa: E402
from amx.paths import PROJECT_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_rubric import resolve_model  # noqa: E402

CASES = PROJECT_ROOT / "3D_asset_cases"
RUNS = PROJECT_ROOT / "runs" / "bench"
OUT = PROJECT_ROOT / "runs" / "rubric-repeat"

DEFAULT_CASES = (
    "BAL-001",
    "SPC-001",
    "BEA-001",
    "CTU-001",
    "MWP-001",
    "PIP-001",
    "RAC-001",
    "VOR-001",
)
"""The eight the plan names, chosen because two of them moved by twenty-five points."""


def judge_repeatedly(case_id: str, run: str, repeats: int) -> list[float]:
    case = load_case(CASES, case_id)
    rubric = load_rubric(case)
    model, _ = resolve_model(RUNS / run / case_id)
    if model is None:
        return []

    asset = None
    note = ""
    try:
        asset = build_grounded_asset(model, asset_id=f"repeat-{case_id}".lower())
    except Exception as error:  # noqa: BLE001
        # An asset that will not compile scores zero, and it has to score zero every
        # time. That is a repeat measurement, not a reason to skip the case.
        note = f"{type(error).__name__}: {error}"
    return [
        judge_asset(
            rubric,
            asset,
            work_dir=OUT / run / case_id / f"pass-{index}",
            build_note=note,
        ).score
        for index in range(repeats)
    ]


def generate_repeatedly(case_id: str, repeats: int, **settings: Any) -> list[float]:
    from amx.bench.run import GenerationSettings, run_case  # noqa: PLC0415

    case = load_case(CASES, case_id)
    scores = []
    for index in range(repeats):
        card = run_case(
            case,
            OUT / "generated" / f"pass-{index}",
            settings=GenerationSettings(**settings),
        )
        scores.append(card.score)
    return scores


def turn_budget(run: str) -> dict[str, Any]:
    """Where the authoring loop stopped, across a whole run.

    This is the other half of the reproducibility question and the half no seed can
    fix. A loop that stops because it ran out of turns stops wherever it happened to
    be, so two runs of the same case are being scored at two different points in the
    same unfinished piece of work. Reporting the judge's spread without this number
    would imply the remaining variance is small, and it is not.
    """
    cases = [
        directory
        for directory in sorted((RUNS / run).iterdir())
        if directory.is_dir() and (directory / "agent" / "turns").is_dir()
    ]
    turns = {
        directory.name: len(list((directory / "agent" / "turns").glob("turn-*")))
        for directory in cases
    }
    budget = max(turns.values(), default=0)
    exhausted = [name for name, count in turns.items() if count >= budget > 0]
    submitted = [d.name for d in cases if (d / "asset" / "model.py").is_file()]
    return {
        "run": run,
        "cases_with_a_trace": len(cases),
        "apparent_budget": budget,
        "stopped_at_the_ceiling": len(exhausted),
        "reached_a_clean_finish": len(submitted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--run", default="articraft-concise-opus5-v2")
    parser.add_argument("--provider", default="gpugeek")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args()

    results: dict[str, list[float]] = {}
    for case_id in args.cases:
        if args.judge_only:
            scores = judge_repeatedly(case_id, args.run, args.repeats)
        else:
            scores = generate_repeatedly(
                case_id,
                args.repeats,
                provider=args.provider,
                model_id=args.model_id,
                max_turns=args.max_turns,
            )
        if not scores:
            print(f"{case_id:<10} nothing to score", flush=True)
            continue
        results[case_id] = scores
        print(
            f"{case_id:<10} {'  '.join(f'{score:6.1f}' for score in scores)}   "
            f"spread {max(scores) - min(scores):5.1f}",
            flush=True,
        )

    if not results:
        return 1
    spreads = [max(scores) - min(scores) for scores in results.values()]
    budget = turn_budget(args.run) if args.judge_only else {}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "repeat.json").write_text(
        json.dumps(
            {
                "mode": "judge-only" if args.judge_only else "end-to-end",
                "repeats": args.repeats,
                "scores": results,
                "mean_spread": mean(spreads),
                "worst_spread": max(spreads),
                "authoring": budget,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"\nmean spread {mean(spreads):.2f}  worst {max(spreads):.2f}  "
        f"sd of spreads {pstdev(spreads):.2f}"
    )
    if max(spreads) >= 5.0:
        print("verdict: NOT reproducible; a score difference under 5 points means nothing yet")
        return 2

    if not budget:
        print("verdict: the whole pipeline repeats within 5 points")
        return 0

    print(
        f"verdict: the judge contributes no variance at all.\n"
        f"         The remaining source is the authoring loop: of {budget['cases_with_a_trace']} "
        f"cases in {args.run}, {budget['stopped_at_the_ceiling']} stopped at the "
        f"{budget['apparent_budget']}-turn ceiling and "
        f"{budget['reached_a_clean_finish']} reached a clean finish.\n"
        f"         A loop that runs out of turns stops wherever it happens to be, so an "
        f"end-to-end repeat is\n"
        f"         measuring that, not the rubric. Raise the completion rate before "
        f"reading anything into a\n"
        f"         difference of a few points."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
