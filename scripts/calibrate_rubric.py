"""Re-judge finished runs and check the rubric against its falsification criteria.

A rubric nobody has tried to break is an opinion, not a measurement. The claims this one
was written to make good on are all statements about ordering — an asset MuJoCo will not
load should not outscore one it will, a single lump should not outscore an assembly, our
pipeline should outscore the plain baseline — and every one of them is checkable against
runs already sitting on disk.

Nothing here writes into the runs being read. Evidence and scorecards land under
`runs/rubric-calibration/<run>/`, so a calibration pass can be repeated, compared or
thrown away without touching the experiments it is reading.

    uv run python scripts/calibrate_rubric.py articraft-concise-opus5-v2 newframework100-opus5
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amx.bench.case import discover  # noqa: E402
from amx.bench.compiler import load_rubric  # noqa: E402
from amx.bench.judge import judge_asset  # noqa: E402
from amx.grounding.build import build_grounded_asset  # noqa: E402
from amx.paths import PROJECT_ROOT  # noqa: E402

RUNS = PROJECT_ROOT / "runs" / "bench"
OUT = PROJECT_ROOT / "runs" / "rubric-calibration"
CASES = PROJECT_ROOT / "3D_asset_cases"


@dataclass
class Row:
    """One case under one run."""

    run: str
    case_id: str
    asset_class: str
    source: str
    """How much the author stood behind the file that was scored; see `resolve_model`."""
    score: float
    achievable: float
    status: str
    loads: bool
    bodies: int
    joints: int
    zero_mass: bool
    disconnected: bool
    failed_gates: list[str] = field(default_factory=list)
    note: str = ""


def resolve_model(case_dir: Path) -> tuple[Path | None, str]:
    """The model this case should be judged on, and how much the author stood behind it.

    A finished authoring run leaves `asset/model.py`. A run that errored or ran out of
    turns leaves only Articraft's own revision history, and the last revision in it is
    the best thing the loop ever had.

    Both are scored, because a calibration pass that skipped the unfinished runs would
    be reading the rubric against the easiest tenth of the corpus. But they are not the
    same thing, so the answer says which it was rather than quietly substituting one for
    the other — a score against `staging` is a measurement of a draft, and any ordering
    claim that rests on one should be read with that in view.
    """
    submitted = case_dir / "asset" / "model.py"
    if submitted.is_file():
        return submitted, "submitted"
    drafts = sorted((case_dir / "asset").rglob("model.py"), key=lambda p: p.stat().st_mtime)
    if drafts:
        return drafts[-1], "revision" if "/revisions/" in str(drafts[-1]) else "staging"
    return None, "none"


def judge_one(run: str, case, case_dir: Path) -> Row:
    rubric = load_rubric(case)
    model, source = resolve_model(case_dir)
    work = OUT / run / case.case_id

    asset = None
    note = ""
    if model is None:
        note = "nothing was produced for this case"
    else:
        try:
            asset = build_grounded_asset(model, asset_id=f"{run}-{case.case_id}".lower())
        except Exception as error:  # noqa: BLE001
            note = f"{type(error).__name__}: {error}"

    card = judge_asset(rubric, asset, work_dir=work, build_note=note)
    card.write(work / "scorecard.json")
    (work / "scorecard.txt").write_text(card.to_text() + "\n")

    items = {item.item_id: item for item in card.items}
    load = items.get("PHY-LOAD")
    detail = load.detail if load else {}
    return Row(
        run=run,
        case_id=case.case_id,
        asset_class=case.asset_class,
        source=source,
        score=card.score,
        achievable=card.achievable,
        status=card.status,
        loads=bool(load and load.status == "passed"),
        bodies=int(detail.get("bodies") or 0),
        joints=int(detail.get("joints") or 0),
        zero_mass=bool(items.get("PART-MASS") and items["PART-MASS"].status == "failed"),
        disconnected=bool(
            items.get("PHY-CONNECTED") and items["PHY-CONNECTED"].status == "failed"
        ),
        failed_gates=[gate.gate_id for gate in card.gates if gate.status == "failed"],
        note=note,
    )


# --------------------------------------------------------------------------- #
# the criteria
# --------------------------------------------------------------------------- #


def falsification(
    rows: list[Row], baseline: str, primary: str | None = None
) -> list[tuple[str, bool, str]]:
    """Each criterion answered yes or no, with the number that answered it."""
    out: list[tuple[str, bool, str]] = []
    base = [row for row in rows if row.run == baseline]

    unloadable = [row for row in base if not row.loads]
    worst = max((row.score for row in unloadable), default=0.0)
    out.append((
        "an asset MuJoCo will not load scores nothing",
        worst == 0.0,
        f"{len(unloadable)} unloadable, highest score {worst:.1f}",
    ))

    loadable = [row for row in base if row.loads]
    weightless = [row for row in loadable if row.zero_mass]
    sound = [row for row in loadable if not row.zero_mass]
    if weightless and sound:
        out.append((
            "assets with weightless parts score below sound ones",
            mean(row.score for row in weightless) < mean(row.score for row in sound),
            f"{mean(row.score for row in weightless):.1f} vs "
            f"{mean(row.score for row in sound):.1f}",
        ))
    else:
        # MuJoCo will not compile a moving body with no mass, so the weightless assets
        # the audit found are not among the loadable ones at all — they fail PHY-LOAD
        # and any other item that needs the model, which is the stronger result.
        out.append((
            "no loadable asset gets away with a weightless part",
            not weightless,
            f"{len(weightless)} weightless among {len(loadable)} that load",
        ))

    # A lump is only degenerate relative to what the case asked for. A beaker is one
    # body because a beaker is one piece of glass; a centrifuge that came out as one
    # body has collapsed. Comparing the two says nothing, so this compares submissions
    # only within the cases that require something to move.
    mechanisms = {
        case_id
        for case_id in {row.case_id for row in rows}
        if any(item.axis == "operability" for item in _rubric_items(case_id))
    }
    collapsed = [row for row in loadable if row.case_id in mechanisms and row.bodies <= 1]
    intact = [row for row in loadable if row.case_id in mechanisms and row.bodies > 1]
    if intact:
        ceiling = mean(row.score for row in intact)
        worst = max((row.score for row in collapsed), default=0.0)
        out.append((
            "a mechanism that collapsed to one body does not outscore a real assembly",
            worst <= ceiling,
            f"{len(collapsed)} collapsed (best {worst:.1f}) vs {len(intact)} intact "
            f"(mean {ceiling:.1f})",
        ))

    others = sorted({row.run for row in rows} - {baseline})
    for run in others:
        group = [row for row in rows if row.run == run]
        out.append((
            f"{run} scores above the baseline overall",
            mean(row.score for row in group) > mean(row.score for row in base),
            f"{mean(row.score for row in group):.1f} vs {mean(row.score for row in base):.1f}"
            f" (loads {sum(row.loads for row in group)}/{len(group)} vs "
            f"{sum(row.loads for row in base)}/{len(base)})",
        ))

    # The two named orderings are claims about the framework under test against the
    # baseline. Other runs of our own pipeline are reported beside them for context, not
    # asserted on: an older one produced a media bottle that does not compile, and a
    # rubric that scored it anyway would be the bug rather than the evidence.
    primary = primary if primary in others else (others[-1] if others else None)
    for case_id in ("MED-001", "RAC-001"):
        pair = {row.run: row for row in rows if row.case_id == case_id}
        theirs = pair.get(baseline)
        ours = pair.get(primary) if primary else None
        if not (ours and theirs):
            continue
        context = "  ".join(
            f"[{run} {pair[run].score:.1f}"
            + ("" if pair[run].loads else ", will not compile")
            + "]"
            for run in others
            if run != primary and run in pair
        )
        out.append((
            f"{case_id}: {primary} is not ranked below the baseline",
            ours.score >= theirs.score,
            f"{ours.score:.1f} vs {theirs.score:.1f}  {context}".rstrip(),
        ))
    return out


def _rubric_items(case_id: str):
    from amx.bench.case import load_case  # noqa: PLC0415

    try:
        return load_rubric(load_case(CASES, case_id)).items
    except Exception:  # noqa: BLE001 - a case that will not load has no items to offer
        return []


def report(rows: list[Row], baseline: str, primary: str | None = None) -> str:
    lines = [
        f"{'run':<30} {'case':<10} {'src':<10} {'score':>6} {'of':>5} "
        f"{'bod':>4} {'jnt':>4}  status",
        "-" * 94,
    ]
    for row in sorted(rows, key=lambda row: (row.run, row.case_id)):
        lines.append(
            f"{row.run[:30]:<30} {row.case_id:<10} {row.source:<10} "
            f"{row.score:>6.1f} {row.achievable:>5.0f} {row.bodies:>4} {row.joints:>4}  "
            f"{row.status}"
        )

    lines.append("")
    for run in sorted({row.run for row in rows}):
        group = [row for row in rows if row.run == run]
        loads = sum(row.loads for row in group)
        submitted = sum(1 for row in group if row.source == "submitted")
        lines.append(
            f"{run:<30} n={len(group):<3} loads={loads}/{len(group)} "
            f"({loads / len(group) * 100:.0f}%)  submitted={submitted}  "
            f"mean {mean(row.score for row in group):.1f}"
        )

    lines.extend(["", "falsification criteria", "-" * 94])
    for name, passed, evidence in falsification(rows, baseline, primary):
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {name:<56} {evidence}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run directory names under runs/bench")
    parser.add_argument("--baseline", default="articraft-concise-opus5-v2")
    parser.add_argument(
        "--primary",
        default="newframework100-opus5",
        help="The run under test; the named orderings are asserted on this one",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Cases to leave alone — a sweep still writing into one must not be read",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Re-read rows.json and re-run the criteria without re-judging anything",
    )
    args = parser.parse_args()

    if args.report_only:
        rows = [Row(**payload) for payload in json.loads((OUT / "rows.json").read_text())]
        text = report(rows, args.baseline, args.primary)
        (OUT / "calibration.txt").write_text(text + "\n")
        print(text)
        return 0

    cases = {case.case_id: case for case in discover(CASES)}
    jobs = []
    for run in args.runs:
        for directory in sorted((RUNS / run).iterdir()):
            if not directory.is_dir() or directory.name not in cases:
                continue
            if args.only and directory.name not in args.only:
                continue
            if directory.name in set(args.skip):
                continue
            jobs.append((run, cases[directory.name], directory))

    print(f"{len(jobs)} case-run(s), {args.workers} worker(s)", flush=True)
    rows: list[Row] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, *job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            run, case, _ = futures[future]
            try:
                row = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"[{done}/{len(jobs)}] {run}/{case.case_id} crashed: {error}", flush=True)
                continue
            rows.append(row)
            print(
                f"[{done}/{len(jobs)}] {run}/{row.case_id} {row.score:.1f} "
                f"({row.source}, {row.status})",
                flush=True,
            )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False) + "\n"
    )
    text = report(rows, args.baseline, args.primary)
    (OUT / "calibration.txt").write_text(text + "\n")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
