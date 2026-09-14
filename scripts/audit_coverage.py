"""What the benchmark actually grades, and what it only appears to.

A score is only worth reading if the items behind it were measured. 244 of 1572 rubric
items across the corpus are `not_scorable`, and lumping them together hides the only
distinction that matters:

* The benchmark does not know the number. `input.md` says `value: null`, `source_class: U`
  and gives an `unknown_reason`. Declining to grade is correct, and no code change can fix
  it — only a source document can.
* The judge cannot take the measurement. The number is right there in the specification
  and nothing reads it. That is a missing capability wearing the same label as an honest
  unknown, and it is why PCR-001 scored 91.8 with wells 23% too narrow to hold a tube.

The second kind is the work list. Run this before and after changing the derivation; the
totals are the acceptance criteria.

    uv run python scripts/audit_coverage.py
    uv run python scripts/audit_coverage.py --json before.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amx.bench.case import BenchCase, discover  # noqa: E402
from amx.bench.compiler import build_rubric  # noqa: E402
from amx.paths import PROJECT_ROOT  # noqa: E402

CASES = PROJECT_ROOT / "3D_asset_cases"

# Reason text -> which bucket the item belongs in. Matched as a substring against
# `params.reason`, longest first, so a more specific phrase wins.
BENCHMARK_UNKNOWN = ("records no target for this dimension",)
DEVICE_PROPERTY = ("is a device property, not geometry",)
JUDGE_CANNOT = (
    "the name does not say how to measure this",
    "is not separable from the part it is cut into",
    "needs a repeated-feature extractor",
    "the cavity integration reports the whole part's capacity",
    "quoted in a state the asset is not exported in",
)


@dataclass
class CaseAudit:
    case_id: str
    items: int = 0
    scorable: int = 0
    benchmark_unknown: int = 0
    device_property: int = 0
    judge_cannot: int = 0
    unclassified: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of items that carry a measurement, counting honest unknowns against it.

        Reported without excusing anything: a case whose specification is half unknown is
        a weak case even though the judge is behaving correctly, and hiding that behind a
        denominator that subtracts the unknowns is how 91.8/100 came to mean nothing.
        """
        return 100.0 * self.scorable / self.items if self.items else 0.0

    @property
    def fixable(self) -> int:
        return self.judge_cannot + len(self.unclassified)


def classify(reason: str) -> str:
    for phrase in BENCHMARK_UNKNOWN:
        if phrase in reason:
            return "benchmark_unknown"
    for phrase in DEVICE_PROPERTY:
        if phrase in reason:
            return "device_property"
    for phrase in JUDGE_CANNOT:
        if phrase in reason:
            return "judge_cannot"
    return "unclassified"


def coverage_gaps(case: BenchCase) -> list[str]:
    """Checks this case needs that the derivation does not produce.

    Named separately from the item census because these leave no `not_scorable` item
    behind — the requirement simply never becomes an item, which is the harder kind of
    gap to notice. `CMP-1 'Tube wells'` on PCR-001 is the archetype: a required component
    that the topology check skips and the rubric never mentions.
    """
    spec = case.to_grounding_spec()
    gaps = []

    graded = {item.primitive for item in build_rubric(case).items}
    for component in spec.components:
        if component.kind != "cavity":
            continue
        # A cavity component is covered when something grades it: a count for the ones
        # stated in quantity, a volume or a hollowness check for the single ones.
        covered = "feature_count" in graded if component.quantity > 1 else "cavity_volume" in graded
        if not covered:
            gaps.append(f"cavity component ({component.name[:34]!r}) is graded by nothing")

    raw = case.spec.get("dimensions") or {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        lowered = name.lower()
        stated = {target.name for target in spec.dimensions}
        if name in stated:
            continue
        if any(word in lowered for word in ("hole", "well", "socket", "aperture", "bore")):
            gaps.append(f"stated {name}={value} is a clear bore and is not measured")
        elif any(word in lowered for word in ("pitch", "spacing")):
            gaps.append(f"stated {name}={value} is a repeated-feature spacing and is not measured")
    return gaps


def audit(case: BenchCase) -> CaseAudit:
    result = CaseAudit(case_id=case.case_id)
    for item in build_rubric(case).items:
        result.items += 1
        if item.primitive != "not_scorable":
            result.scorable += 1
            continue
        bucket = classify(item.params.reason or "")
        if bucket == "unclassified":
            result.unclassified.append(f"{item.id}: {(item.params.reason or '')[:70]}")
        else:
            setattr(result, bucket, getattr(result, bucket) + 1)
    result.gaps = coverage_gaps(case)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Also write the totals here, for diffing")
    parser.add_argument("--only", nargs="*", help="Case ids; default is all")
    args = parser.parse_args()

    cases = discover(CASES)
    if args.only:
        wanted = set(args.only)
        cases = [case for case in cases if case.case_id in wanted]

    audits = [audit(case) for case in cases]
    audits.sort(key=lambda a: (a.coverage, -a.fixable))

    print(f"{'case':11} {'items':>5} {'scored':>6} {'cover':>6} "
          f"{'unknown':>7} {'device':>6} {'FIXABLE':>7}  gaps")
    print("-" * 96)
    for a in audits:
        gaps = f"{len(a.gaps)}" if a.gaps else ""
        print(f"{a.case_id:11} {a.items:5} {a.scorable:6} {a.coverage:5.1f}% "
              f"{a.benchmark_unknown:7} {a.device_property:6} {a.fixable:7}  {gaps}")

    items = sum(a.items for a in audits)
    scorable = sum(a.scorable for a in audits)
    unknown = sum(a.benchmark_unknown for a in audits)
    device = sum(a.device_property for a in audits)
    fixable = sum(a.fixable for a in audits)
    print("-" * 96)
    print(f"{len(audits)} cases, {items} items, {scorable} scored "
          f"({100.0 * scorable / max(items, 1):.1f}%)")
    print(f"  benchmark unknown  {unknown:4}   the specification says `value: null`; not a code defect")
    print(f"  device property    {device:4}   rpm, temperature, area: no mesh can confirm these")
    print(f"  FIXABLE            {fixable:4}   the number is stated and nothing measures it")

    gap_kinds: Counter[str] = Counter()
    for a in audits:
        for gap in a.gaps:
            kind = gap.split(" is ")[-1] if " is " in gap else gap.split(" (")[0]
            gap_kinds[kind] += 1
    print()
    print("requirements that never become an item at all:")
    for kind, count in gap_kinds.most_common():
        print(f"  {count:4}  {kind}")

    worst = [a for a in audits if a.fixable]
    if worst:
        print()
        print("unclassified reasons (extend the buckets above, or fix the derivation):")
        seen: set[str] = set()
        for a in worst:
            for line in a.unclassified:
                text = line.split(": ", 1)[-1]
                if text not in seen:
                    seen.add(text)
                    print(f"  {a.case_id}  {line}")

    if args.json:
        args.json.write_text(json.dumps({
            "cases": len(audits),
            "items": items,
            "scorable": scorable,
            "benchmark_unknown": unknown,
            "device_property": device,
            "fixable": fixable,
            "gaps": sum(len(a.gaps) for a in audits),
            "per_case": {a.case_id: {"items": a.items, "scorable": a.scorable,
                                     "fixable": a.fixable, "gaps": a.gaps} for a in audits},
        }, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
