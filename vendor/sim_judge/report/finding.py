"""Core data types for diagnostic findings.

Three layers:

``Subject``      -- describes "who and who", i.e. the geoms / bodies / joints involved.
``Observation``  -- describes "one hit on one step", emitted step by step by the detectors.
``Finding``      -- describes "one continuous event", built by the aggregator from
                    Observations of the same origin. This is what the final report lists,
                    answering "when, in which phase, who and who, how bad".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Mapping


class Severity(str, Enum):
    """Severity of a finding.

    ``FAILURE`` is granted only to findings that violate an explicit declaration in
    bound-operation.json; it decides the overall verdict.
    ``WARNING`` covers physical implausibility inferred from generic geometric criteria,
    on which the policy takes no position.
    ``INFO`` covers quantitative observations offered for human reference.
    """

    INFO = "info"
    WARNING = "warning"
    FAILURE = "failure"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank < other.rank


_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.FAILURE: 2}


class Tier(str, Enum):
    """Origin of the criterion. Lets a reader see at a glance whether a finding is a
    contract violation or a common-sense inference."""

    DECLARED = "declared"  # threshold comes from bound-operation.json
    GENERIC = "generic"  # threshold comes from the generic geometric criteria in defaults.py


@dataclass(frozen=True, slots=True)
class Subject:
    """The objects a problem involves. Pairwise problems fill both sides a/b, single-object
    problems fill side a only."""

    kind: str  # "geom_pair" | "body" | "joint" | "scene"
    a_name: str
    a_entity: str | None = None
    a_body: str | None = None
    b_name: str | None = None
    b_entity: str | None = None
    b_body: str | None = None

    @property
    def identity(self) -> tuple[str, str, str]:
        """Stable identity used for aggregation. Pairwise problems sort the two sides so
        that (A,B) and (B,A) count as the same object."""
        if self.b_name is None:
            return (self.kind, self.a_name, "")
        lo, hi = sorted((self.a_name, self.b_name))
        return (self.kind, lo, hi)

    def describe(self) -> str:
        left = _label(self.a_name, self.a_entity)
        if self.b_name is None:
            return left
        return f"{left} <-> {_label(self.b_name, self.b_entity)}"


def _label(name: str, entity: str | None) -> str:
    return f"{name}[{entity}]" if entity else name


@dataclass(frozen=True, slots=True)
class Observation:
    """One detector hit on one step.

    A detector only decides "is there a problem on this step"; merging hits into intervals
    is the aggregator's job.
    """

    code: str
    severity: Severity
    tier: Tier
    step_index: int
    subject: Subject
    metrics: Mapping[str, float]
    peak_metric: str
    """Key in ``metrics`` used to pick the peak step. Larger means more severe."""
    thresholds: Mapping[str, float] = field(default_factory=dict)
    rule_id: str | None = None
    min_duration_steps: int = 1
    """How many steps an event must last to be worth reporting. Filters out momentary
    numerical jitter."""

    merge_gap_steps: int | None = None
    """Largest gap allowed between hits of this item. ``None`` means use the aggregator's
    global setting.

    A detector that samples on a fixed stride must state its own stride, otherwise a
    persistent condition gets sliced into many disconnected single-step events.
    """

    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def group_key(self) -> tuple[Any, ...]:
        return (self.code, self.subject.identity, self.rule_id)

    @property
    def peak_value(self) -> float:
        return float(self.metrics[self.peak_metric])


@dataclass(frozen=True, slots=True)
class Finding:
    """One entry in the report, covering a continuous interval of the same problem."""

    code: str
    severity: Severity
    tier: Tier
    subject: Subject

    first_step: int
    last_step: int
    time_range_s: tuple[float, float]
    step_ids: tuple[str, ...]
    action_ids: tuple[str, ...]

    peak_step: int
    peak_time_s: float
    peak_step_id: str
    peak_action_id: str
    peak_metric: str
    metrics: Mapping[str, float]
    """Every metric on the peak step."""

    thresholds: Mapping[str, float]
    observation_count: int
    summary: str
    rule_id: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_steps(self) -> int:
        return self.last_step - self.first_step + 1

    def sort_key(self) -> tuple[Any, ...]:
        return (-self.severity.rank, self.first_step, self.code, self.subject.identity)
