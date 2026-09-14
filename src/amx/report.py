"""The one finding type the whole pipeline reports problems with.

Asset grounding, manufacturability checks and the simulation judge all produce the same
shape, so the repair router has one thing to read and a run's `findings.json` means the
same thing wherever it came from. It deliberately mirrors `sim_judge.Finding`: `code`,
`severity`, a subject, measurements and the thresholds they were compared against.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    FAILURE = "failure"
    WARNING = "warning"
    INFO = "info"


class RepairTarget(StrEnum):
    """What has to change to address a finding. This is what routes a repair."""

    TRAJECTORY = "trajectory"
    """The operation plan: waypoints, durations, grip widths."""

    FIXTURE = "fixture"
    """A bench-mounted co-designed part's parameters."""

    TOOL = "tool"
    """An arm-mounted co-designed part's parameters."""

    LAYOUT = "layout"
    """Where an asset or fixture sits on the bench."""

    ASSET = "asset"
    """The generated asset itself; needs part 1 to be re-run."""

    NONE = "none"
    """Nothing to repair — informational, or a defect in the inputs."""


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: Severity
    summary: str
    subject: str = ""
    repair_target: RepairTarget = RepairTarget.NONE
    metrics: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)

    def line(self) -> str:
        head = f"[{self.severity.value.upper():7}] {self.code}"
        if self.subject:
            head += f"  {self.subject}"
        measured = ", ".join(f"{k}={v:.6g}" for k, v in self.metrics.items())
        return f"{head}\n           {self.summary}" + (f"\n           {measured}" if measured else "")


class Report(BaseModel):
    """A named group of findings with a pass/fail verdict."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    subject: str
    findings: list[Finding] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.FAILURE]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def verdict(self) -> Literal["PASS", "FAIL"]:
        return "PASS" if self.passed else "FAIL"

    def extend(self, other: "Report") -> Self:
        self.findings.extend(other.findings)
        self.notes.extend(other.notes)
        return self

    def to_text(self) -> str:
        header = f"{self.kind} / {self.subject}: {self.verdict}"
        counts = (
            f"  {len(self.failures)} failure(s), {len(self.warnings)} warning(s), "
            f"{len(self.findings)} finding(s) total"
        )
        body = [f.line() for f in self.findings] or ["  (no findings)"]
        notes = [f"  note: {n}" for n in self.notes]
        return "\n".join([header, counts, "", *body, *notes])

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
        return path


def interval_finding(
    *,
    code: str,
    subject: str,
    measured: float,
    interval: tuple[float, float] | None,
    unit: str,
    repair_target: RepairTarget,
    what: str,
) -> Finding | None:
    """Compare a measurement against an inclusive interval, or return None if unspecified."""
    if interval is None:
        return None
    low, high = interval
    if low <= measured <= high:
        return Finding(
            code=code,
            severity=Severity.INFO,
            subject=subject,
            summary=f"{what} is {measured:.6g} {unit}, inside the specified {low:.6g}–{high:.6g}.",
            metrics={"measured": measured},
            thresholds={"minimum": low, "maximum": high},
        )
    direction = "below" if measured < low else "above"
    return Finding(
        code=code,
        severity=Severity.FAILURE,
        subject=subject,
        summary=(
            f"{what} is {measured:.6g} {unit}, {direction} the specified "
            f"{low:.6g}–{high:.6g} {unit}."
        ),
        repair_target=repair_target,
        metrics={"measured": measured},
        thresholds={"minimum": low, "maximum": high},
    )
