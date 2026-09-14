"""Turning a report into one number, so candidates can be ordered.

A search needs a scalar, and a report is a list of findings. Collapsing one into the
other is where a search is usually won or lost, because the obvious collapse — count the
failures — is nearly useless as a gradient. It cannot tell a diameter that is 6% out from
one that is 300% out, so a model that halves its error looks exactly as good as one that
did nothing, and the search has nothing to climb.

So a failing finding earns partial credit wherever its metrics allow the shortfall to be
measured. A dimension 6% out against a 5% tolerance scores nearly as well as a passing
one; the same dimension 300% out scores zero. The score is bounded below 1.0 for any
failure, so no amount of near-misses ever ties with actually passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from amx.report import Finding, Report, Severity

WARNING_CREDIT = 0.75
"""A warning is usually "this could not be measured", which is a real loss of information
but not evidence of a defect."""

MAX_FAILING_CREDIT = 0.9
"""Ceiling on partial credit. A failure that is nearly passing must still lose to a pass,
or the search will settle one step short of the target for ever."""

BLOCKING_CODES = frozenset(
    {
        "G-UNMEASURABLE",
        "G-CAVITY-MISSING",
        "G-STABILITY-DIVERGED",
        "G-PROBE-DIVERGED",
    }
)
"""Findings that mean the rest of the report is uninformative rather than merely bad.

An asset that will not load has not scored badly on twelve checks; it has failed to
answer any of them, and the twelve zeroes that follow are an artefact. Scoring these at
zero overall keeps the search from preferring a broken candidate that happens to trip
fewer checks.
"""


@dataclass(frozen=True)
class Score:
    """A scalar in [0, 1] and enough detail to explain it."""

    value: float
    credits: dict[str, float] = field(default_factory=dict)
    failures: int = 0
    warnings: int = 0
    total: int = 0
    blocked: bool = False

    def __lt__(self, other: "Score") -> bool:
        return self.value < other.value

    def summary(self) -> str:
        if self.blocked:
            return f"{self.value:.3f} (blocked: the asset could not be measured)"
        return (
            f"{self.value:.3f} over {self.total} checks "
            f"({self.failures} failing, {self.warnings} unmeasured)"
        )


def score_report(report: Report) -> Score:
    """Mean credit across every finding, with blocking failures short-circuiting to zero."""
    findings = report.findings
    if not findings:
        # Nothing was checked. Not a pass and not a failure; treat it as unknown-but-neutral
        # so a node with no applicable checks does not outrank one that genuinely passed.
        return Score(value=0.5, total=0)

    if any(finding.code in BLOCKING_CODES for finding in findings):
        return Score(
            value=0.0,
            failures=len(report.failures),
            warnings=len(report.warnings),
            total=len(findings),
            blocked=True,
        )

    credits = {}
    for index, finding in enumerate(findings):
        key = f"{finding.code}:{finding.subject or index}"
        credits[key] = finding_credit(finding)
    value = sum(credits.values()) / len(credits)
    return Score(
        value=value,
        credits=credits,
        failures=len(report.failures),
        warnings=len(report.warnings),
        total=len(findings),
    )


def finding_credit(finding: Finding) -> float:
    """How nearly this one finding is satisfied, in [0, 1]."""
    if finding.severity is Severity.INFO:
        return 1.0
    if finding.severity is Severity.WARNING:
        return WARNING_CREDIT

    graded = _graded_credit(finding)
    return 0.0 if graded is None else min(MAX_FAILING_CREDIT, max(0.0, graded))


def _graded_credit(finding: Finding) -> float | None:
    """Partial credit from whatever the finding measured, or None if it measured nothing."""
    metrics, thresholds = finding.metrics, finding.thresholds

    # A dimension: credit decays linearly from the tolerance out to the hard-fail bound.
    error = metrics.get("relative_error")
    tolerance = thresholds.get("tolerance_rel")
    hard = thresholds.get("hard_fail_rel")
    if error is not None and tolerance is not None and hard is not None and hard > tolerance:
        return MAX_FAILING_CREDIT * (hard - error) / (hard - tolerance)

    # A minimum, such as a cavity that has to hold at least so much.
    for measured_key, minimum_key in (("volume_ml", "minimum_ml"), ("measured", "minimum")):
        measured = metrics.get(measured_key)
        minimum = thresholds.get(minimum_key)
        if measured is not None and minimum and minimum > 0:
            return MAX_FAILING_CREDIT * min(1.0, measured / minimum)

    # A ceiling, such as how far the object may drift. Several may apply at once, and the
    # worst one is what the finding is about.
    ratios = [
        metrics[key] / thresholds[bound]
        for key, bound in (
            ("translation_mm", "max_translation_mm"),
            ("tilt_deg", "max_tilt_deg"),
            ("penetration_mm", "max_penetration_mm"),
            ("return_error_deg", "tolerance_deg"),
        )
        if key in metrics and thresholds.get(bound)
    ]
    if ratios:
        worst = max(ratios)
        # Twice the limit earns nothing; anything between decays linearly.
        return MAX_FAILING_CREDIT * max(0.0, 2.0 - worst)

    return None


def score_reports(reports: list[Report]) -> Score:
    """Score several reports as though they were one, which for a search they are."""
    merged = Report(kind="merged", subject=reports[0].subject if reports else "")
    for report in reports:
        merged.findings.extend(report.findings)
    return score_report(merged)
