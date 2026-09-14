"""Mapping from step index to protocol step, action and time.

For a report to answer when and during which step something happened, the raw step
indices 0..N have to be translated into task semantics. There are two sources:

* ``cases[].actions`` in ``task-execution.json``, which gives each action's
  ``first_step`` / ``step_count`` / ``step_id``. This is the action-level partition
  (``release``, ``close_lid``, and so on).
* The ``phase`` column of each ``steps-*.npz``, which gives the protocol step id per
  step and serves as a cross-check.

One ``task-execution.json`` usually records several cases from the same batch, while
a single directory holds exactly one trace. We match them exactly, via
``identity.evaluation_case_sha256`` in ``raw-trace.json`` against each case's
``realization.case_sha256``, and never guess from ordering.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence


class TimelineError(Exception):
    """The execution record does not line up with the recorded trace."""


@dataclass(frozen=True, slots=True)
class ActionSpan:
    """The half-open step interval one action occupies."""

    action_id: str
    step_id: str
    first_step: int
    step_count: int

    @property
    def stop_step(self) -> int:
        return self.first_step + self.step_count

    def contains(self, step: int) -> bool:
        return self.first_step <= step < self.stop_step


@dataclass(frozen=True, slots=True)
class StepLocation:
    """The complete semantic location of a single step index."""

    step_index: int
    time_s: float
    step_id: str
    action_id: str

    def describe(self) -> str:
        return f"step {self.step_index} / {self.time_s:.3f} s / {self.step_id} - {self.action_id}"


class Timeline:
    """Lookup table from step index to :class:`StepLocation`."""

    def __init__(self, spans: Sequence[ActionSpan], timestep_s: float, step_count: int) -> None:
        if not spans:
            raise TimelineError("the execution record contains no action spans at all")
        self._spans = tuple(sorted(spans, key=lambda s: s.first_step))
        self._starts = [s.first_step for s in self._spans]
        self.timestep_s = timestep_s
        self.step_count = step_count
        self._validate()

    @property
    def spans(self) -> tuple[ActionSpan, ...]:
        return self._spans

    def locate(self, step_index: int) -> StepLocation:
        span = self.span_of(step_index)
        return StepLocation(
            step_index=step_index,
            time_s=step_index * self.timestep_s,
            step_id=span.step_id if span else "",
            action_id=span.action_id if span else "",
        )

    def span_of(self, step_index: int) -> ActionSpan | None:
        position = bisect_right(self._starts, step_index) - 1
        if position < 0:
            return None
        span = self._spans[position]
        return span if span.contains(step_index) else None

    def action_id(self, step_index: int) -> str:
        span = self.span_of(step_index)
        return span.action_id if span else ""

    def step_id(self, step_index: int) -> str:
        span = self.span_of(step_index)
        return span.step_id if span else ""

    def summarize(self) -> list[dict[str, Any]]:
        """Summarise by protocol step, for the report header to show the overall
        time structure."""
        summary: list[dict[str, Any]] = []
        for span in self._spans:
            if summary and summary[-1]["step_id"] == span.step_id:
                entry = summary[-1]
                entry["stop_step"] = span.stop_step
                entry["actions"].append(span.action_id)
            else:
                summary.append(
                    {
                        "step_id": span.step_id,
                        "first_step": span.first_step,
                        "stop_step": span.stop_step,
                        "actions": [span.action_id],
                    }
                )
        return summary

    def _validate(self) -> None:
        cursor = self._spans[0].first_step
        for span in self._spans:
            if span.first_step != cursor:
                raise TimelineError(
                    f"action {span.action_id} starts at step {span.first_step}, "
                    f"but the previous action ended at step {cursor}: the spans are not contiguous"
                )
            cursor = span.stop_step
        if cursor != self.step_count:
            raise TimelineError(
                f"the action spans add up to {cursor} steps while the recording has "
                f"{self.step_count} steps, so the two disagree"
            )


def build_timeline(
    task_execution: Mapping[str, Any],
    trace_manifest: Mapping[str, Any],
) -> tuple[Timeline, str | None]:
    """Build the timeline and return the ``case_sha256`` of this directory's trace.

    That ``case_sha256`` is also what
    :func:`~sim_judge.world.naming.load_geometry_bindings` uses to pick the right
    geometry bindings table.
    """
    case_sha256 = (trace_manifest.get("identity") or {}).get("evaluation_case_sha256")
    case = _select_case(task_execution, case_sha256)

    spans = [
        ActionSpan(
            action_id=str(a.get("action_id", "")),
            step_id=str(a.get("step_id", "")),
            first_step=int(a["first_step"]),
            step_count=int(a["step_count"]),
        )
        for a in case.get("actions") or []
    ]
    timeline = Timeline(
        spans=spans,
        timestep_s=float(trace_manifest.get("timestep_s", 0.0)),
        step_count=int(trace_manifest.get("step_count", 0)),
    )
    return timeline, case_sha256


def _select_case(task_execution: Mapping[str, Any], case_sha256: str | None) -> Mapping[str, Any]:
    cases = task_execution.get("cases") or []
    if not cases:
        raise TimelineError("task-execution.json contains no cases")
    if case_sha256:
        for case in cases:
            if (case.get("realization") or {}).get("case_sha256") == case_sha256:
                return case
        raise TimelineError(
            f"the case digest {case_sha256[:16]}... declared by the recording manifest is not "
            f"among the {len(cases)} cases in task-execution.json, so the action spans "
            "cannot be determined"
        )
    if len(cases) > 1:
        raise TimelineError(
            "the recording manifest declares no case digest while the execution record holds "
            "several cases, so there is no way to tell which one the trace belongs to"
        )
    return cases[0]


def cross_check_phases(timeline: Timeline, observed: Mapping[int, str]) -> list[str]:
    """Compare the per-step ``phase`` logged in the recording against the
    ``step_id`` inferred from the action spans.

    Returns a list describing the mismatches; empty means the two sources agree
    completely.
    """
    problems: list[str] = []
    for step_index, phase in observed.items():
        expected = timeline.step_id(step_index)
        if phase and expected and phase != expected:
            problems.append(
                f"step {step_index}: the recording says {phase}, the action spans imply {expected}"
            )
    return problems
