"""Where a run's turn budget actually went.

Every case in `newframework100-opus5` hit its hundred-turn ceiling, and the obvious
explanations were both wrong. Wall time is not the problem: ninety percent of it is
spent waiting on the model, so making `compile_model` faster buys nothing. Nor is the
grounding deadlock the main cost: the cases that were never deadlocked also ran to the
ceiling. What the budget went on only shows up when turns are attributed to tools by
*turn* rather than by call, which is what this does.

The numbers that came out of the baseline, and the ones worth watching:

    replace                 44.5% of turns, carrying 1.03 edits each
    compile_model           13.2% of turns, 95% of them straight after a clean edit
    probe_model             15.8% of turns, one measurement at a time
    multi-tool turns        8.2%

The `compile_model` line reads like waste and is not: an edit tool only parses the file,
while `compile_model` builds the geometry, so the follow-up is a different question with
a different answer. What it measures is turns that a single batched turn could have held
both halves of, which is why it is reported next to the multi-tool share.

Each of those is a tool contract problem rather than a model problem, so each should
move when the contract changes. Re-run this after a sweep and compare.

    uv run python scripts/profile_turns.py newframework100-opus5
    uv run python scripts/profile_turns.py --json newframework100-opus5 opus5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amx.paths import PROJECT_ROOT  # noqa: E402
from amx.trace_export import Turn, read_trace  # noqa: E402

RUNS = PROJECT_ROOT / "runs" / "bench"

EDIT_TOOLS = frozenset({"replace", "write_file", "apply_patch"})
"""Tools that mutate `model.py`. Turns spent here are the single largest line item."""

GATE_MARKER = "<grounding_required>"
_BULLET = re.compile(r"^- (.+)$", re.M)


def find_traces(run: str) -> dict[str, Path]:
    """Case id -> its newest `trajectory.jsonl`.

    A case that was retried has more than one, and the last write is the one whose
    outcome the leaderboard reports. Both the staging tree and the promoted record tree
    are searched because which one survives depends on whether the run finished.

    The promoted copy is zstd-compressed, and missing the `.zst` suffix made this whole
    script blind to exactly the runs worth measuring: a case that failed leaves a plain
    `trajectory.jsonl` in staging, a case that succeeded leaves only the compressed one.
    Every turn count reported here came from runs that produced no asset.
    """
    found: dict[str, Path] = {}
    root = RUNS / run
    if not root.is_dir():
        raise FileNotFoundError(f"no run at {root}")
    for case in sorted(root.iterdir()):
        if not case.is_dir():
            continue
        candidates = [
            *case.glob("asset/articraft/cache/runs/*/staging/*/traces/trajectory.jsonl"),
            *case.glob("asset/articraft/records/*/revisions/*/traces/trajectory.jsonl"),
            *case.glob("asset/articraft/records/*/revisions/*/traces/trajectory.jsonl.zst"),
        ]
        if candidates:
            found[case.name] = max(candidates, key=lambda path: path.stat().st_mtime)
    return found


def _payload(message: dict[str, Any]) -> dict[str, Any]:
    """A tool result's full parsed body.

    `trace_export._result_text` deliberately unwraps to the `result` field, which is what
    a reader wants and drops the `compilation` block an edit tool attaches. Whether that
    block says `success` is exactly what decides if the next `compile_model` was needed.
    """
    content = message.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return {}
    return content if isinstance(content, dict) else {}


def _tool_names(turn: Turn) -> list[str]:
    return [str(call.get("name") or "?") for call in turn.tool_calls]


def _owner(names: list[str]) -> str:
    """Which tool to charge a turn to.

    A turn calling one tool twice is that tool's. A turn genuinely mixing tools is its own
    category rather than being charged to whichever call happened to come first, because
    mixed turns are the behaviour we are trying to encourage and hiding them inside the
    single-tool counts would make the fix look like it did nothing.
    """
    if not names:
        return "<finish attempt>"
    unique = sorted(set(names))
    return unique[0] if len(unique) == 1 else "+".join(unique)


def _edit_compiled_clean(turn: Turn) -> bool:
    """Did this turn's edits already compile successfully?"""
    for message in turn.tool_results:
        if message.get("name") not in EDIT_TOOLS:
            continue
        compilation = _payload(message).get("compilation")
        if isinstance(compilation, dict) and compilation.get("status") == "success":
            return True
    return False


def _edits_in(turn: Turn) -> int:
    """How many separate edits this turn actually applied.

    Counting `replace` calls instead undercounts a batched one, which is the whole point
    of the `edits` array: a turn that fixes four things is one call and four edits, and
    reporting it as one made the batching look unused when it was not.
    """
    total = 0
    for call in turn.tool_calls:
        if call.get("name") not in EDIT_TOOLS:
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        batched = (arguments or {}).get("edits")
        total += len(batched) if isinstance(batched, list) and batched else 1
    return total


def _compile_succeeded(turn: Turn) -> bool:
    for message in turn.tool_results:
        if message.get("name") != "compile_model":
            continue
        if "status=success" in json.dumps(_payload(message).get("result") or ""):
            return True
    return False


def _budget(path: Path) -> int | None:
    """The `max_turns` this trace was run under, from the run manifest beside it."""
    manifest = path.parents[3] / "run.json"
    if not manifest.is_file():
        return None
    try:
        settings = json.loads(manifest.read_text()).get("settings_summary") or {}
    except json.JSONDecodeError:
        return None
    budget = settings.get("max_turns")
    return int(budget) if isinstance(budget, int) else None


@dataclass
class CaseProfile:
    """One case's turn accounting."""

    case_id: str
    turns: int = 0
    budget: int | None = None
    owners: Counter[str] = field(default_factory=Counter)
    edit_turns: int = 0
    edit_calls: int = 0
    multi_tool_turns: int = 0
    compile_turns: int = 0
    redundant_compile_turns: int = 0
    gate_blocks: int = 0
    repeated_block: int = 0
    """How many times the single most-repeated set of gate failures came back."""
    first_clean_compile: int | None = None
    turns_after_clean_compile: int = 0

    @property
    def exhausted(self) -> bool:
        return self.budget is not None and self.turns >= self.budget


def profile_case(case_id: str, path: Path) -> CaseProfile:
    trace = read_trace(path)
    profile = CaseProfile(case_id=case_id, budget=_budget(path))
    fingerprints: Counter[tuple[str, ...]] = Counter()
    previous_edit_was_clean = False

    for turn in trace.turns:
        profile.turns += 1
        names = _tool_names(turn)
        profile.owners[_owner(names)] += 1

        if len(set(names)) > 1:
            profile.multi_tool_turns += 1

        for text in turn.injected:
            if GATE_MARKER in text:
                profile.gate_blocks += 1
                fingerprints[tuple(sorted(_BULLET.findall(text)))] += 1

        if any(name in EDIT_TOOLS for name in names) and len(set(names)) == 1:
            profile.edit_turns += 1
            profile.edit_calls += _edits_in(turn)

        if names == ["compile_model"]:
            profile.compile_turns += 1
            # Not a wasted turn: the edit tool parsed the file, this builds the geometry.
            # It is a turn that a batched call could have folded into the edit itself.
            if previous_edit_was_clean:
                profile.redundant_compile_turns += 1

        if profile.first_clean_compile is None and _compile_succeeded(turn):
            profile.first_clean_compile = turn.number

        previous_edit_was_clean = any(
            name in EDIT_TOOLS for name in names
        ) and _edit_compiled_clean(turn)

    if profile.first_clean_compile is not None and trace.turns:
        profile.turns_after_clean_compile = (
            trace.turns[-1].number - profile.first_clean_compile
        )
    if fingerprints:
        profile.repeated_block = fingerprints.most_common(1)[0][1]
    return profile


@dataclass
class RunProfile:
    run: str
    cases: list[CaseProfile]

    @property
    def total_turns(self) -> int:
        return sum(case.turns for case in self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "cases": len(self.cases),
            "total_turns": self.total_turns,
            "median_turns": median([case.turns for case in self.cases]) if self.cases else 0,
            "exhausted": sum(1 for case in self.cases if case.exhausted),
            "edits_per_edit_turn": self._edits_per_turn(),
            "multi_tool_share": self._share(sum(c.multi_tool_turns for c in self.cases)),
            "redundant_compile_share": self._redundant_share(),
            "deadlocked_cases": sum(1 for case in self.cases if case.repeated_block >= 3),
            "turn_share": self._turn_share(),
            "per_case": [
                {
                    "case": case.case_id,
                    "turns": case.turns,
                    "first_clean_compile": case.first_clean_compile,
                    "turns_after_clean_compile": case.turns_after_clean_compile,
                    "gate_blocks": case.gate_blocks,
                    "repeated_block": case.repeated_block,
                }
                for case in self.cases
            ],
        }

    def _share(self, turns: int) -> float:
        return round(turns / self.total_turns * 100, 1) if self.total_turns else 0.0

    def _edits_per_turn(self) -> float:
        turns = sum(case.edit_turns for case in self.cases)
        calls = sum(case.edit_calls for case in self.cases)
        return round(calls / turns, 2) if turns else 0.0

    def _redundant_share(self) -> float:
        compiles = sum(case.compile_turns for case in self.cases)
        redundant = sum(case.redundant_compile_turns for case in self.cases)
        return round(redundant / compiles * 100, 1) if compiles else 0.0

    def _turn_share(self) -> dict[str, float]:
        owners: Counter[str] = Counter()
        for case in self.cases:
            owners.update(case.owners)
        return {name: self._share(turns) for name, turns in owners.most_common()}

    def to_text(self) -> str:
        summary = self.to_dict()
        lines = [
            f"=== {self.run} ===",
            f"  {summary['cases']} cases, {summary['total_turns']} turns, "
            f"median {summary['median_turns']:.0f}, "
            f"budget exhausted {summary['exhausted']}/{summary['cases']}",
            "",
            "  turn share by tool:",
        ]
        lines += [
            f"    {name:<44} {share:>5.1f}%"
            for name, share in summary["turn_share"].items()
            if share >= 0.2
        ]
        clean = [c.first_clean_compile for c in self.cases if c.first_clean_compile]
        after = [c.turns_after_clean_compile for c in self.cases if c.first_clean_compile]
        lines += [
            "",
            f"  edits per edit turn        {summary['edits_per_edit_turn']:.2f}"
            "   (1.0 means every edit cost a turn)",
            f"  multi-tool turns           {summary['multi_tool_share']:.1f}%",
            f"  compiles split off an edit  {summary['redundant_compile_share']:.1f}%"
            "   (lone compile_model right after a clean edit; batchable, not wasted)",
            f"  deadlocked cases           {summary['deadlocked_cases']}"
            "   (same gate failures returned 3+ times)",
        ]
        if clean:
            lines += [
                f"  first clean compile        turn {median(clean):.0f} (median)",
                f"  turns spent after that     {median(after):.0f} (median)",
            ]
        worst = sorted(self.cases, key=lambda case: -case.repeated_block)[:5]
        if worst and worst[0].repeated_block >= 3:
            lines += ["", "  worst deadlocks:"]
            lines += [
                f"    {case.case_id:<10} {case.turns:>3} turns, "
                f"blocked {case.gate_blocks:>3}x, same failures {case.repeated_block:>3}x"
                for case in worst
                if case.repeated_block >= 3
            ]
        return "\n".join(lines)


def profile_run(run: str) -> RunProfile:
    traces = find_traces(run)
    return RunProfile(
        run=run,
        cases=[profile_case(case_id, path) for case_id, path in traces.items()],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directory names under runs/bench")
    parser.add_argument("--json", action="store_true", help="emit the numbers as JSON")
    args = parser.parse_args(argv)

    profiles = [profile_run(run) for run in args.runs]
    if args.json:
        print(json.dumps([profile.to_dict() for profile in profiles], indent=2))
    else:
        print("\n\n".join(profile.to_text() for profile in profiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
