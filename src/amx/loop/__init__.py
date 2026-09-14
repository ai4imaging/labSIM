"""Closing the loop: judge an episode, route what failed, change the design, run again.

This is the part that makes the other two worth having. Part 1 produces an asset and part 2
produces a cell and a motion, but neither can tell whether the result is any good. A
physics judge can, and the useful thing about its findings is that they say what went wrong
in measurable terms — which body, which step, by how much, against what limit.

`judge.py` runs `sim_judge` and decides what each finding points at. `repair.py` turns a
set of findings into a small typed patch. `loop.py` runs that cycle a bounded number of
times and keeps the best round.
"""

from amx.loop.judge import Judgement, judge
from amx.loop.loop import Design, LoopConfig, LoopResult, Round, materialise, run_loop
from amx.loop.repair import (
    LayoutPatch,
    ParameterPatch,
    PatchRejected,
    RepairPatch,
    WaypointPatch,
    apply_patch,
    propose_repair,
)

__all__ = [
    "Design",
    "Judgement",
    "LayoutPatch",
    "LoopConfig",
    "LoopResult",
    "ParameterPatch",
    "PatchRejected",
    "RepairPatch",
    "Round",
    "WaypointPatch",
    "apply_patch",
    "judge",
    "materialise",
    "propose_repair",
    "run_loop",
]
