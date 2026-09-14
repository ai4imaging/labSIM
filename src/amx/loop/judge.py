"""Running `sim_judge` over a case, and turning its verdict into repair instructions.

`sim_judge` answers one question well: did anything physically implausible happen. What it
deliberately does not say is what to change about the design, because it has no idea what
produced the scene. That is this module's job, and it does it without reading any prose.

Two mechanisms carry the routing. Rules that `amx.sim.policy` generated already state a
`repair_target` — sim_judge preserves it on the rule and reports the `rule_id` alongside
each finding, so a violated declared rule routes itself. Findings from generic criteria
have no rule behind them, so they route by code and by which namespace their subject lives
in: a mount gap belongs to the tool, a body resting inside the bench belongs to the layout,
a massless part belongs to the asset that declared it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sim_judge import JudgeOptions, judge_case
from sim_judge.report.finding import Finding as JudgeFinding

from amx import naming
from amx.report import Finding, RepairTarget, Report, Severity

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"

CODE_ROUTES: dict[str, RepairTarget] = {
    # The arm went somewhere it should not have, or got there in a way physics disliked.
    "C1_MINIMUM_CLEARANCE_VIOLATED": RepairTarget.TRAJECTORY,
    "K1_POSITION_JUMP": RepairTarget.TRAJECTORY,
    "K2_JOINT_LIMIT_VIOLATION": RepairTarget.TRAJECTORY,
    "K3_NON_FINITE_STATE": RepairTarget.TRAJECTORY,
    "K4_CONTACT_NORMAL_FORCE_EXCEEDED": RepairTarget.TRAJECTORY,
    "K4_CONTACT_TANGENTIAL_FORCE_EXCEEDED": RepairTarget.TRAJECTORY,
    "K5_ENGINE_WARNING": RepairTarget.TRAJECTORY,
    "P1_CONTACT_PENETRATION_EXCEEDED": RepairTarget.TRAJECTORY,
    "P1_GEOMETRIC_PENETRATION": RepairTarget.TRAJECTORY,
    "P2_FORBIDDEN_CONTACT": RepairTarget.TRAJECTORY,
    "P3_UNLISTED_ROBOT_DEVICE_CONTACT": RepairTarget.TRAJECTORY,
    "P3_UNLISTED_ROBOT_ENVIRONMENT_CONTACT": RepairTarget.TRAJECTORY,
    "P3_UNLISTED_ROBOT_SELF_CONTACT": RepairTarget.TRAJECTORY,
    "P4_POSSIBLE_TUNNELING": RepairTarget.TRAJECTORY,
    # The part did not end up where the task said it should.
    "R1_RECEIVER_ALIGNMENT_ERROR": RepairTarget.TRAJECTORY,
    "R1_RECEIVER_ALIGNMENT_TRANSIENT": RepairTarget.TRAJECTORY,
    "R3_LABWARE_NOT_SEATED": RepairTarget.TRAJECTORY,
    "R3_LABWARE_STILL_MOVING": RepairTarget.TRAJECTORY,
    "R3_LABWARE_TILTED": RepairTarget.TRAJECTORY,
    "R4_LABWARE_INSERTION_DEPTH_OUT_OF_RANGE": RepairTarget.TRAJECTORY,
    "R5_LABWARE_NOT_TOUCHING_SOCKET_FLOOR": RepairTarget.TRAJECTORY,
    # The socket itself is the wrong size — that is a dimension on the part that made it.
    "R2_RECEIVER_SIDE_CLEARANCE_LOST": RepairTarget.FIXTURE,
    "R2_RECEIVER_SIDE_CLEARANCE_TRANSIENT": RepairTarget.FIXTURE,
    "R2_RECEIVER_SOCKET_TOO_LOOSE": RepairTarget.FIXTURE,
    # Things sitting wrong before anything has moved: a placement problem.
    "F1_UNSUPPORTED_FLOATING_BODY": RepairTarget.LAYOUT,
    "F2_SUPPORT_CHAIN_BROKEN": RepairTarget.LAYOUT,
    "F3_GRAVITY_INCONSISTENT_SUPPORT": RepairTarget.LAYOUT,
    "S4_INITIAL_PENETRATION": RepairTarget.LAYOUT,
    "S6_RESTING_GEOMETRY_INTERPENETRATION": RepairTarget.LAYOUT,
    # Defects in a model rather than in how it was used.
    "S1_MOVABLE_BODY_MASSLESS": RepairTarget.ASSET,
    "S1_MOVABLE_BODY_INERTIALESS": RepairTarget.ASSET,
    "S2_ENTITY_WITHOUT_COLLISION_GEOMETRY": RepairTarget.ASSET,
    "S2_MOVABLE_BODY_VISUAL_ONLY": RepairTarget.ASSET,
    "S3_VISUAL_COLLISION_OFFSET": RepairTarget.ASSET,
    # The tool came loose from the flange, which is the adapter's mating face.
    "S5_TOOL_MOUNT_GAP": RepairTarget.TOOL,
    "S5_TOOL_MOUNT_PENETRATION": RepairTarget.TOOL,
}

NAMESPACE_ROUTES: tuple[tuple[str, RepairTarget], ...] = (
    (naming.TOOL, RepairTarget.TOOL),
    (naming.FIXTURE_PREFIX, RepairTarget.FIXTURE),
)
"""Subjects whose namespace overrides the code's default target.

Only applied to findings about a part's own construction or placement. A robot link
striking a fixture is still a trajectory problem, and the declared rule says so.
"""

NAMESPACE_SENSITIVE_CODES = frozenset(
    {
        "F1_UNSUPPORTED_FLOATING_BODY",
        "F2_SUPPORT_CHAIN_BROKEN",
        "F3_GRAVITY_INCONSISTENT_SUPPORT",
        "S1_MOVABLE_BODY_MASSLESS",
        "S1_MOVABLE_BODY_INERTIALESS",
        "S2_ENTITY_WITHOUT_COLLISION_GEOMETRY",
        "S2_MOVABLE_BODY_VISUAL_ONLY",
        "S3_VISUAL_COLLISION_OFFSET",
        "S4_INITIAL_PENETRATION",
        "S6_RESTING_GEOMETRY_INTERPENETRATION",
    }
)


@dataclass
class Judgement:
    """One case, judged, with everything the loop needs to decide what to do next."""

    case_dir: Path
    verdict: str
    reason: str
    report: Report
    counts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == VERDICT_PASS

    @property
    def conclusive(self) -> bool:
        return self.verdict in (VERDICT_PASS, VERDICT_FAIL)

    def score(self) -> tuple[int, int, int]:
        """Lower is better. Used to keep the best round when none of them passes.

        Failures dominate warnings, and a verdict nobody can trust ranks behind any
        conclusive one regardless of its counts.
        """
        return (
            0 if self.conclusive else 1,
            len(self.report.failures),
            len(self.report.warnings),
        )

    def absorb(self, findings: list[Finding]) -> None:
        """Fold checks made outside `sim_judge` into this verdict.

        Used for the task objective, which the judge only evaluates when the policy declared
        a socket to evaluate it against. Anything absorbed here counts exactly as much as a
        finding the judge raised itself: a failure turns a PASS into a FAIL, because the
        alternative is a verdict that says a run succeeded when the part never moved.
        """
        if not findings:
            return
        self.report.findings.extend(findings)
        failures = [f for f in findings if f.severity is Severity.FAILURE]
        counts = self.counts.setdefault("by_code", {})
        for finding in findings:
            counts[finding.code] = counts.get(finding.code, 0) + 1
        self.counts["total"] = self.counts.get("total", 0) + len(findings)
        self.counts["failure"] = self.counts.get("failure", 0) + len(failures)
        if not failures:
            return
        self.verdict = VERDICT_FAIL
        self.reason = (
            f"{len(failures)} objective check(s) failed: "
            + "; ".join(f.code for f in failures)
        )

    def targets(self) -> list[RepairTarget]:
        """Which parts of the design the failures point at, most-cited first."""
        tally: dict[RepairTarget, int] = {}
        for finding in self.report.failures:
            if finding.repair_target is RepairTarget.NONE:
                continue
            tally[finding.repair_target] = tally.get(finding.repair_target, 0) + 1
        return sorted(tally, key=lambda t: -tally[t])

    def write(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.report.write(directory / "findings.json")
        (directory / "verdict.json").write_text(
            json.dumps(
                {
                    "verdict": self.verdict,
                    "reason": self.reason,
                    "counts": self.counts,
                    "notes": self.notes,
                    "case_dir": str(self.case_dir),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )


def judge(
    case_dir: Path,
    *,
    stride: int = 1,
    max_steps: int | None = None,
    strict: bool = False,
    verify_integrity: bool = False,
) -> Judgement:
    """Judge one case bundle and translate the result into routed findings.

    Integrity verification is off by default: it re-hashes every chunk of the recording,
    which is worth doing once for an acceptance run and is pure overhead in a loop that
    just wrote the files itself.
    """
    case_dir = Path(case_dir)
    raw = judge_case(
        case_dir,
        JudgeOptions(
            stride=stride,
            strict=strict,
            verify_integrity=verify_integrity,
            max_steps=max_steps,
        ),
    )
    rule_targets = _rule_targets(case_dir / "bound-operation.json")
    report = Report(kind="simulation", subject=case_dir.name, notes=list(raw.notes))
    report.findings.extend(_translate(f, rule_targets) for f in raw.findings)

    return Judgement(
        case_dir=case_dir,
        verdict=raw.verdict,
        reason=raw.verdict_reason,
        report=report,
        counts=raw.counts(),
        notes=list(raw.notes),
    )


def _rule_targets(policy_path: Path) -> dict[str, RepairTarget]:
    """`rule_id` to repair target, read from the policy the scene generated."""
    if not policy_path.is_file():
        return {}
    document = json.loads(policy_path.read_text())
    feedback = document.get("runtime_feedback") or {}
    targets: dict[str, RepairTarget] = {}
    for group in ("contact_rules", "clearance_rules"):
        for rule in feedback.get(group) or []:
            rule_id = rule.get("rule_id")
            target = rule.get("repair_target")
            if not rule_id or not target:
                continue
            try:
                targets[str(rule_id)] = RepairTarget(target)
            except ValueError:
                continue
    return targets


def _translate(finding: JudgeFinding, rule_targets: dict[str, RepairTarget]) -> Finding:
    subject = finding.subject.describe()
    return Finding(
        code=finding.code,
        severity=Severity(finding.severity.value),
        summary=finding.summary,
        subject=subject,
        repair_target=_route(finding, rule_targets),
        metrics={k: float(v) for k, v in finding.metrics.items()},
        thresholds={k: float(v) for k, v in finding.thresholds.items()},
        detail={
            "tier": finding.tier.value,
            "rule_id": finding.rule_id,
            "first_step": finding.first_step,
            "last_step": finding.last_step,
            "peak_step": finding.peak_step,
            "peak_step_id": finding.peak_step_id,
            "peak_action_id": finding.peak_action_id,
            "duration_steps": finding.duration_steps,
            "time_range_s": list(finding.time_range_s),
            "action_ids": list(finding.action_ids),
        },
    )


def _route(finding: JudgeFinding, rule_targets: dict[str, RepairTarget]) -> RepairTarget:
    if finding.severity.value == "info":
        return RepairTarget.NONE
    if finding.rule_id and finding.rule_id in rule_targets:
        return rule_targets[finding.rule_id]
    default = CODE_ROUTES.get(finding.code, RepairTarget.TRAJECTORY)
    if finding.code in NAMESPACE_SENSITIVE_CODES:
        return _by_namespace(finding, default)
    return default


def _by_namespace(finding: JudgeFinding, default: RepairTarget) -> RepairTarget:
    names = [
        name
        for name in (
            finding.subject.a_name,
            finding.subject.b_name,
            finding.subject.a_body,
            finding.subject.b_body,
        )
        if name
    ]
    for prefix, target in NAMESPACE_ROUTES:
        if any(name.startswith(prefix) for name in names):
            return target
    # The vendored arm's own links overlap each other at rest — a Robotiq linkage is
    # modelled that way. Nothing in the design can change it, so offering it to the repair
    # step would only invite an invented fix.
    if names and all(naming.is_robot(name) for name in names):
        return RepairTarget.NONE
    return default
