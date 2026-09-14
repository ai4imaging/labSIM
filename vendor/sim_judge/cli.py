"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from sim_judge.defaults import DEFAULT_THRESHOLDS
from sim_judge.judge import VERDICT_FAIL, VERDICT_INCONCLUSIVE, JudgeOptions, judge_case
from sim_judge.loader.case_bundle import BundleError
from sim_judge.world.timeline import TimelineError

#: Exit codes: 0 pass, 1 judged as failing, 2 inconclusive, 3 bad input.
EXIT_PASS, EXIT_FAIL, EXIT_INCONCLUSIVE, EXIT_BAD_INPUT = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim_judge",
        description=(
            "replay a simulation recording, detect implausible physics such as floating "
            "or penetrating geoms, and print a diagnostic report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m sim_judge correct_case_0000\n"
            "  python -m sim_judge failed_case_0001_x_plus_2mm --format json -o report.json\n"
            "  python -m sim_judge some_case --stride 20        # coarse scan, quick trial run\n"
            "  python -m sim_judge some_case --strict           # generic hints count as failures\n"
        ),
    )
    parser.add_argument("case_dir", type=Path, help="case directory to judge")
    parser.add_argument(
        "--format",
        choices=("text", "json", "both"),
        default="text",
        help="report format, default text",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="write the report to a file instead of stdout"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "replay sampling stride, default 1 (check every step). a larger value speeds up "
            "a coarse scan but misses transient events"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="judge only the first N steps, for a quick trial run",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="count hints from the generic geometric criteria as failures too",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the sha256 integrity check",
    )
    parser.add_argument(
        "--penetration-ratio",
        type=float,
        default=None,
        help=(
            "relative threshold for the generic penetration criterion: penetration depth "
            "over the half-thickness of the thinner geom. "
            f"default {DEFAULT_THRESHOLDS.penetration_ratio_warning}"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    thresholds = DEFAULT_THRESHOLDS
    if args.penetration_ratio is not None:
        thresholds = replace(thresholds, penetration_ratio_warning=args.penetration_ratio)

    options = JudgeOptions(
        stride=max(1, args.stride),
        strict=args.strict,
        verify_integrity=not args.no_verify,
        thresholds=thresholds,
        max_steps=args.max_steps,
    )

    try:
        report = judge_case(args.case_dir, options)
    except (BundleError, TimelineError) as error:
        print(f"bad input: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT

    rendered = _render(report, args.format)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"report written to {args.output} (verdict: {report.verdict})", file=sys.stderr)
    else:
        print(rendered)

    if report.verdict == VERDICT_FAIL:
        return EXIT_FAIL
    if report.verdict == VERDICT_INCONCLUSIVE:
        return EXIT_INCONCLUSIVE
    return EXIT_PASS


def _render(report, style: str) -> str:
    if style == "json":
        return report.to_json()
    if style == "text":
        return report.to_text()
    return report.to_text() + "\n\n" + report.to_json()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
