"""JSON rendering.

The structure is machine friendly and self-describing: every finding carries its own
thresholds and source rule, so a reader does not have to go back to
``bound-operation.json`` to understand what the verdict was based on.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from sim_judge.report.finding import Finding

if TYPE_CHECKING:  # type annotations only; avoids a circular import with the judge module
    from sim_judge.judge import JudgeReport


def finding_to_dict(finding: Finding) -> dict[str, Any]:
    subject = finding.subject
    return {
        "code": finding.code,
        "severity": finding.severity.value,
        "tier": finding.tier.value,
        "rule_id": finding.rule_id,
        "when": {
            "first_step": finding.first_step,
            "last_step": finding.last_step,
            "duration_steps": finding.duration_steps,
            "time_range_s": [_clean(v) for v in finding.time_range_s],
            "step_ids": list(finding.step_ids),
            "action_ids": list(finding.action_ids),
            "peak_step": finding.peak_step,
            "peak_time_s": _clean(finding.peak_time_s),
            "peak_step_id": finding.peak_step_id,
            "peak_action_id": finding.peak_action_id,
        },
        "who": {
            "kind": subject.kind,
            "a": {"name": subject.a_name, "entity": subject.a_entity, "body": subject.a_body},
            "b": (
                None
                if subject.b_name is None
                else {"name": subject.b_name, "entity": subject.b_entity, "body": subject.b_body}
            ),
        },
        "how_bad": {
            "peak_metric": finding.peak_metric,
            "metrics": {k: _clean(v) for k, v in finding.metrics.items()},
            "thresholds": {k: _clean(v) for k, v in finding.thresholds.items()},
            "observation_count": finding.observation_count,
        },
        "summary": finding.summary,
        "detail": _cleanse(finding.detail),
    }


def render_json(report: JudgeReport, *, indent: int = 2) -> str:
    return json.dumps(report_to_dict(report), ensure_ascii=False, indent=indent)


def report_to_dict(report: JudgeReport) -> dict[str, Any]:
    return {
        "schema": "sim-judge/1.0",
        "case_dir": str(report.case_dir),
        "verdict": report.verdict,
        "verdict_reason": report.verdict_reason,
        "counts": report.counts(),
        "replay": report.replay_info,
        "provenance": report.provenance,
        "timeline": report.timeline_summary,
        "findings": [finding_to_dict(f) for f in report.findings],
        "notes": report.notes,
    }


def _clean(value: Any) -> Any:
    """JSON rejects NaN/Inf, so turn them into strings to avoid emitting an invalid document."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _cleanse(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _cleanse(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cleanse(v) for v in value]
    return _clean(value)
