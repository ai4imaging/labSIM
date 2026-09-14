"""Benchmark cases, compiled rubrics, and the fail-closed judge.

The public surface is small on purpose. A case is a directory with `input.md`. The
compiler turns that into a `Rubric`. The judge scores an asset against it. Generation
and sweeping sit on top of those two.
"""

from amx.bench.case import BenchCase, discover, load_case
from amx.bench.compiler import build_rubric, load_rubric, write_case_artifacts
from amx.bench.judge import Scorecard, judge_asset
from amx.bench.rubric import Gate, Rubric, RubricItem
from amx.bench.run import generate, judge, run_case, sweep

__all__ = [
    "BenchCase",
    "Gate",
    "Rubric",
    "RubricItem",
    "Scorecard",
    "build_rubric",
    "discover",
    "generate",
    "judge",
    "judge_asset",
    "load_case",
    "load_rubric",
    "run_case",
    "sweep",
    "write_case_artifacts",
]
