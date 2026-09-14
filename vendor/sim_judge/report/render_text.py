"""Plain-text rendering.

Indented as "protocol step -> action -> finding" so that a reader following the task flow
can tell which stage went wrong. The verdict and replay credibility come first, the
per-category counts last.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from sim_judge.report.finding import Finding, Severity, Tier

if TYPE_CHECKING:  # type annotations only; avoids a circular import with the judge module
    from sim_judge.judge import JudgeReport

_SEVERITY_LABEL = {
    Severity.FAILURE: "FAIL",
    Severity.WARNING: "WARN",
    Severity.INFO: "INFO",
}

_TIER_LABEL = {
    Tier.DECLARED: "declared rule",
    Tier.GENERIC: "generic criterion",
}


def render_text(report: JudgeReport, *, width: int = 92) -> str:
    lines: list[str] = []
    rule = "═" * width
    thin = "─" * width

    lines.append(rule)
    lines.append(f"Physics plausibility report   {report.case_dir}")
    lines.append(rule)
    lines.append(f"Verdict: {report.verdict}    {report.verdict_reason}")
    lines.append("")

    lines.extend(_render_replay_block(report))
    lines.append("")
    lines.extend(_render_counts(report))
    lines.append("")

    if not report.findings:
        lines.append("No physically implausible behaviour found.")
        lines.append(rule)
        return "\n".join(lines)

    lines.append(thin)
    lines.append("Findings (grouped by protocol step -> action -> finding)")
    lines.append(thin)
    lines.extend(_render_findings(report.findings))

    if report.notes:
        lines.append("")
        lines.append(thin)
        lines.append("Notes")
        for note in report.notes:
            lines.append(f"  · {note}")

    lines.append(rule)
    return "\n".join(lines)


def _render_replay_block(report: JudgeReport) -> list[str]:
    replay = report.replay_info
    check = replay.get("determinism", {})
    verdict = (
        "consistent"
        if check.get("trustworthy")
        else "inconsistent, the diagnosis may be distorted"
    )
    integrity = report.provenance.get("integrity")
    integrity_label = (
        "not checked" if integrity is None else ("ok" if integrity["ok"] else "mismatch found")
    )
    return [
        "Replay credibility",
        f"  Model source      {replay.get('source')}   engine {replay.get('engine_version')}"
        f" (recorded on {replay.get('recorded_version')})",
        f"  State agreement   {verdict}, max error {check.get('max_state_error', float('nan')):.3e}"
        f" ({check.get('sample_count')} steps sampled, tolerance {check.get('tolerance'):.1e})",
        f"  Steps replayed    {replay.get('frames_examined')} / {replay.get('total_steps')}"
        f"   stride {replay.get('stride')}",
        f"  File integrity    {integrity_label}",
    ]


def _render_counts(report: JudgeReport) -> list[str]:
    counts = report.counts()
    lines = ["Finding counts"]
    lines.append(
        f"  FAIL {counts['failure']}   WARN {counts['warning']}   INFO {counts['info']}"
        f"   ({counts['declared']} from declared rules, {counts['generic']} from generic criteria)"
    )
    if counts["by_code"]:
        for code, number in sorted(counts["by_code"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"    {code:<42s} {number}")
    return lines


def _render_findings(findings: Iterable[Finding]) -> list[str]:
    """Render in order of first appearance, grouped into layers."""
    ordered = sorted(findings, key=lambda f: (f.first_step, -f.severity.rank, f.code))
    lines: list[str] = []
    current_step: str | None = None
    current_action: str | None = None

    for finding in ordered:
        step_id = finding.step_ids[0] if finding.step_ids else "(no step)"
        action_id = finding.action_ids[0] if finding.action_ids else "(no action)"

        if step_id != current_step:
            lines.append("")
            lines.append(f"■ {step_id}")
            current_step, current_action = step_id, None
        if action_id != current_action:
            lines.append(f"  ▸ {action_id}")
            current_action = action_id
        lines.extend(_render_one(finding))
    return lines


def _render_one(finding: Finding) -> list[str]:
    marker = {"failure": "✗", "warning": "!", "info": "·"}[finding.severity.value]
    header = (
        f"    {marker} [{_SEVERITY_LABEL[finding.severity]}] {finding.code}"
        f"    steps {finding.first_step}-{finding.last_step}"
        f" ({finding.time_range_s[0]:.3f}-{finding.time_range_s[1]:.3f} s,"
        f" {finding.duration_steps} steps)"
    )
    lines = [header, f"        Subject   {finding.subject.describe()}"]

    source = _TIER_LABEL[finding.tier]
    if finding.rule_id:
        source += f" · {finding.rule_id}"
    lines.append(f"        Basis     {source}")

    peak = ", ".join(f"{k}={_format(v)}" for k, v in finding.metrics.items())
    lines.append(f"        Peak      step {finding.peak_step} / {finding.peak_time_s:.3f} s   {peak}")

    if finding.thresholds:
        limits = ", ".join(f"{k}={_format(v)}" for k, v in finding.thresholds.items())
        lines.append(f"        Limits    {limits}")

    lines.append(f"        Summary   {finding.summary}")
    return lines


def _format(value: float) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
