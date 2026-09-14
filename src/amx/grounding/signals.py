"""Rendering grounding findings in the shape the authoring model already reads.

Articraft answers every `compile_model` call with a `<compile_signals>` block: a
summary, then failures, warnings, notes and suggested next steps. The model has been
trained by its system prompt to read that structure, and giving grounding results a
second, different shape would mean teaching it a second feedback language for no gain.

So this emits `<grounding_signals>` with the same skeleton. The tag differs because the
two answer different questions — one is "does this build", the other is "is this the
object that was asked for" — and conflating them in a transcript makes it impossible to
see which one a turn was responding to.
"""

from __future__ import annotations

from collections.abc import Iterable

from amx.report import Finding, Report, Severity

_MAX_LISTED = 12
"""Findings shown per section. A model that has broken thirty dimensions does not need
thirty lines to know its scale is wrong, and the transcript still has to fit."""


def render_grounding_signals(report: Report, *, check_name: str = "grounding") -> str:
    failures = [f for f in report.findings if f.severity is Severity.FAILURE]
    warnings = [f for f in report.findings if f.severity is Severity.WARNING]
    infos = [f for f in report.findings if f.severity is Severity.INFO]

    summary = _summary_line(check_name, failures, warnings, infos, report.notes)
    if not failures and not warnings:
        parts = ["<grounding_signals>", "<summary>", summary, "</summary>"]
        if report.notes:
            parts.extend(["", "<notes>", _render_notes(report.notes), "</notes>"])
        parts.append("</grounding_signals>")
        return "\n".join(parts)

    parts = ["<grounding_signals>", "<summary>", summary, "</summary>"]
    if failures:
        parts.extend(
            [
                "",
                "<failures>",
                "Failures (must be fixed before you finish):",
                _render_findings(failures),
                "</failures>",
            ]
        )
    if warnings:
        parts.extend(
            [
                "",
                "<warnings>",
                "Warnings (non-blocking):",
                _render_findings(warnings),
                "</warnings>",
            ]
        )
    if report.notes:
        parts.extend(["", "<notes>", _render_notes(report.notes), "</notes>"])

    rules = _response_rules(failures, warnings)
    if rules:
        parts.extend(
            ["", "<response_rules>", "Suggested next steps:\n" + "\n".join(rules), "</response_rules>"]
        )
    parts.append("</grounding_signals>")
    return "\n".join(parts)


def _summary_line(
    check_name: str,
    failures: list[Finding],
    warnings: list[Finding],
    infos: list[Finding],
    notes: list[str],
) -> str:
    status = "failure" if failures else ("warning" if warnings else "success")
    head = (
        f"check={check_name} status={status} failures={len(failures)} "
        f"warnings={len(warnings)} conforming={len(infos)}"
    )
    if failures:
        return f"{head}\nPrimary issue: {failures[0].summary}"
    if warnings:
        return f"{head}\nThe measurements conform; see the warnings for what could not be checked."
    if not infos and notes:
        return f"{head}\nNothing was measurable for this check; see the notes."
    return f"{head}\nEvery stated target was met."


def _render_findings(findings: list[Finding]) -> str:
    lines: list[str] = []
    for finding in findings[:_MAX_LISTED]:
        severity = finding.severity.value.upper()
        subject = f" {finding.subject}" if finding.subject else ""
        lines.append(f"- {severity} [{finding.code}]{subject} {finding.summary}")
        measured = _measurement_line(finding)
        if measured:
            lines.append(f"  {measured}")
    remaining = len(findings) - _MAX_LISTED
    if remaining > 0:
        lines.append(f"- ... and {remaining} more of the same kind")
    return "\n".join(lines)


def _measurement_line(finding: Finding) -> str:
    bits = [f"{key}={value:.6g}" for key, value in finding.metrics.items()]
    bits += [f"{key}<={value:.6g}" for key, value in finding.thresholds.items()]
    return "  ".join(bits)


def _render_notes(notes: Iterable[str]) -> str:
    return "\n".join(f"- NOTE {note}" for note in notes)


def _response_rules(failures: list[Finding], warnings: list[Finding]) -> list[str]:
    """Advice keyed on what actually failed, not a fixed lecture.

    The codes are grouped rather than enumerated: a model that has three dimensions out
    of tolerance needs one instruction about scale, not three.
    """
    rules: list[str] = []
    codes = {finding.code for finding in failures}

    if {"G-DIM", "G-DIM-HARD"} & codes:
        rules.append(
            "- A measured dimension is outside its tolerance. Change the constant that "
            "sets it in `model.py`; do not scale the whole object unless every dimension "
            "is off by the same factor."
        )
    if "G-CAVITY-VOLUME" in codes:
        rules.append(
            "- The internal volume is short. Widen the bore or deepen it, and remember "
            "the usable volume stops at the lowest overflow edge, so a low spout or a "
            "notch in the rim caps it regardless of how tall the wall is."
        )
    if "G-CAVITY-MISSING" in codes:
        rules.append(
            "- No enclosed cavity was found. The interior has to be a real subtraction "
            "from the solid, not a separate inner surface: use a boolean difference so "
            "a cross-section through the wall comes out as a ring."
        )
    if {"G-COMPONENT-MISSING", "G-COMPONENT-PARENT"} & codes:
        rules.append(
            "- A required component is missing or attached to the wrong parent. Name "
            "parts for what they are so the structure check can find them."
        )
    if {"G-STABILITY", "G-STABILITY-DIVERGED"} & codes:
        rules.append(
            "- The object does not sit still on a flat surface. That is usually a base "
            "that is not flat, a centre of mass outside the footprint, or an inertia "
            "that does not match the geometry."
        )
    if "G-PROBE-ESCAPE" in codes:
        rules.append(
            "- The reference probe passed through the floor of the cavity. The bottom is "
            "either open or too thin to collide with; give it real thickness."
        )
    if "G-TILT" in codes:
        rules.append(
            "- The object could not be tilted and returned. Check that nothing is welded "
            "to the world and that the geometry survives the rotation."
        )
    if "G-VISUAL-CLAIM" in codes:
        rules.append(
            "- A required visual feature was not visible in the renders. Add the feature "
            "as real geometry rather than relying on colour."
        )

    if not rules and failures:
        rules.append("- Address the failures above, then call the same check again.")
    if warnings and not failures:
        rules.append(
            "- Nothing is blocking. The warnings record what could not be measured, not "
            "something you did wrong."
        )
    return rules
