"""Judgement pipeline orchestration: load -> self-check -> replay -> detect -> aggregate -> verdict.

This is the only place that knows the whole flow; the individual layers never call
each other, this module wires them all together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sim_judge.defaults import DEFAULT_THRESHOLDS, GenericThresholds
from sim_judge.detectors import DetectorContext, build_detectors
from sim_judge.loader.case_bundle import CaseBundle, discover_bundle, verify_model_integrity
from sim_judge.loader.model_loader import load_model
from sim_judge.loader.policy import load_policy
from sim_judge.loader.trace_reader import TraceReader
from sim_judge.replay import Replayer
from sim_judge.report.aggregate import EventAggregator, Summarizer
from sim_judge.report.finding import Finding, Severity, Tier
from sim_judge.world.geometry import UpAxis
from sim_judge.world.naming import NameResolver, load_geometry_bindings
from sim_judge.world.timeline import build_timeline, cross_check_phases

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class JudgeOptions:
    """Tunables for the judgement. The defaults are right for almost every case."""

    stride: int = 1
    """Replay sampling stride. 1 checks every step; a larger value speeds up a coarse
    scan but misses transient events shorter than the stride."""

    strict: bool = False
    """Count hints from the generic geometric criteria as failures too. Off by default --
    we don't fail a case on the task author's behalf where their policy declares nothing."""

    verify_integrity: bool = True
    """Verify the sha256 of the model and the chunks. Turning this off saves hashing 43 MB."""

    thresholds: GenericThresholds = DEFAULT_THRESHOLDS
    max_steps: int | None = None
    """Judge only the first N steps. Useful for a quick trial run."""


@dataclass
class JudgeReport:
    """The complete result of one judgement."""

    case_dir: Path
    verdict: str
    verdict_reason: str
    findings: list[Finding]
    replay_info: dict[str, Any]
    provenance: dict[str, Any]
    timeline_summary: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, Any]:
        by_code: dict[str, int] = {}
        for finding in self.findings:
            by_code[finding.code] = by_code.get(finding.code, 0) + 1
        return {
            "total": len(self.findings),
            "failure": sum(f.severity is Severity.FAILURE for f in self.findings),
            "warning": sum(f.severity is Severity.WARNING for f in self.findings),
            "info": sum(f.severity is Severity.INFO for f in self.findings),
            "declared": sum(f.tier is Tier.DECLARED for f in self.findings),
            "generic": sum(f.tier is Tier.GENERIC for f in self.findings),
            "by_code": by_code,
        }

    def to_json(self, *, indent: int = 2) -> str:
        from sim_judge.report.render_json import render_json

        return render_json(self, indent=indent)

    def to_text(self) -> str:
        from sim_judge.report.render_text import render_text

        return render_text(self)


def judge_case(case_dir: str | Path, options: JudgeOptions | None = None) -> JudgeReport:
    """Run the full physics-plausibility judgement over one case directory."""
    options = options or JudgeOptions()
    started = time.perf_counter()

    bundle = discover_bundle(case_dir)
    loaded = load_model(bundle)
    policy = load_policy(bundle.read_bound_operation())
    reader = TraceReader(bundle, verify_chunks=options.verify_integrity)

    task_execution = bundle.read_task_execution()
    manifest = bundle.read_trace_manifest()
    timeline, case_sha256 = build_timeline(task_execution, manifest)
    bindings, binding_note = load_geometry_bindings(task_execution, case_sha256)
    resolver = NameResolver(loaded.model, bindings)
    replayer = Replayer(loaded.model, reader, resolver, timeline)

    notes = _collect_preflight_notes(bundle, loaded, policy, options)
    if binding_note:
        notes.append(binding_note)
    integrity = verify_model_integrity(bundle) if options.verify_integrity else None
    if integrity is not None and not integrity.ok:
        notes.append(
            "The model file checksum does not match the recording manifest; the replay may "
            "not represent the original run."
        )

    determinism = replayer.check_determinism(
        options.thresholds.determinism_sample_count, options.thresholds.determinism_tolerance
    )
    if not determinism.trustworthy:
        notes.append(
            f"Determinism check failed: the largest state deviation over the sampled replay "
            f"was {determinism.max_state_error:.3e}, above the tolerance of "
            f"{determinism.tolerance:.1e}, which means the model or the engine version does "
            f"not match the recording. Treat the diagnostics below as indicative only."
        )

    up_axis = UpAxis.of(loaded.model)
    if up_axis is None:
        notes.append(
            "The model declares no gravity, so there is no vertical direction to work with; "
            "the floating, support and seating criteria (F, R) are disabled entirely."
        )
    context = DetectorContext(
        model=replayer.model,
        resolver=resolver,
        policy=policy,
        timeline=timeline,
        thresholds=options.thresholds,
        replayer=replayer,
        up_axis=up_axis,
        timestep_s=reader.timestep_s or float(loaded.model.opt.timestep),
    )
    findings, frames_examined, observed_phases = _scan(
        context, replayer, [_full_range(reader, options)], options
    )

    if options.stride > 1 and findings:
        windows = _refine_windows(findings, options, reader.step_count)
        findings, refined_frames, _ = _scan(context, replayer, windows, options, stride=1)
        frames_examined += refined_frames
        window_count = f"{len(windows)} suspicious window{'s' if len(windows) > 1 else ''}"
        notes.append(
            f"The coarse scan (stride {options.stride}) flagged {window_count}; each was "
            f"then rescanned step by step, and the numbers in this report come from that "
            f"fine scan."
        )

    phase_conflicts = cross_check_phases(timeline, observed_phases)
    if phase_conflicts:
        notes.append(
            f"The recorded phase disagrees with the action segmentation in "
            f"{len(phase_conflicts)} places, first one: {phase_conflicts[0]}"
        )
    if integrity is not None:
        integrity = integrity.with_chunk_results(reader.chunks_verified, reader.chunk_mismatches)
        if reader.chunk_mismatches:
            notes.append(
                f"These chunks have a checksum that does not match the manifest: "
                f"{', '.join(reader.chunk_mismatches[:5])}"
            )

    verdict, reason = _decide(findings, determinism.trustworthy, options.strict)

    return JudgeReport(
        case_dir=bundle.root,
        verdict=verdict,
        verdict_reason=reason,
        findings=findings,
        replay_info={
            **loaded.describe(),
            "determinism": determinism.to_dict(),
            "total_steps": reader.step_count,
            "frames_examined": frames_examined,
            "stride": options.stride,
            "elapsed_s": round(time.perf_counter() - started, 3),
        },
        provenance={
            "case_sha256": case_sha256,
            "task_id": policy.task_id,
            "run_id": (manifest.get("identity") or {}).get("run_id"),
            "chunk_count": reader.chunk_count,
            "integrity": integrity.to_dict() if integrity else None,
        },
        timeline_summary=timeline.summarize(),
        notes=notes,
    )


def _full_range(reader: TraceReader, options: JudgeOptions) -> tuple[int, int]:
    stop = reader.step_count if options.max_steps is None else min(options.max_steps, reader.step_count)
    return (0, stop)


def _scan(
    context: DetectorContext,
    replayer: Replayer,
    windows: list[tuple[int, int]],
    options: JudgeOptions,
    *,
    stride: int | None = None,
) -> tuple[list[Finding], int, dict[int, str]]:
    """Run one streaming detection pass over the given step ranges.

    The detectors are built fresh here, so separate scans are independent of each other.
    Within a single scan the same detector instances are shared across all windows. The
    stateful detectors that compare against the previous frame (position jump,
    tunnelling) get their continuity reset whenever a new window starts, because the gap
    between two windows is not contiguous in time and a detector carrying state across
    it would report the discontinuity as a real jump. Inside a window those detectors
    scale their criteria by the number of steps actually crossed, so neither the stride
    nor the window boundaries produce false positives.
    """
    detectors = build_detectors(context)
    aggregator = EventAggregator(
        timeline=context.timeline,
        gap_steps=max(context.policy.event_gap_steps, options.thresholds.event_gap_steps),
        summarizer=Summarizer(),
    )

    observed_phases: dict[int, str] = {}
    frames_examined = 0
    effective_stride = options.stride if stride is None else stride

    for index, (start, stop) in enumerate(windows):
        if index:
            for detector in detectors:
                detector.reset_continuity()
        for frame in replayer.iter_frames(stride=effective_stride, start=start, stop=stop):
            frames_examined += 1
            observed_phases[frame.step_index] = frame.step_id
            for detector in detectors:
                aggregator.extend(detector.feed(frame))

    for detector in detectors:
        aggregator.extend(detector.finalize())

    return aggregator.finish(), frames_examined, observed_phases


def _refine_windows(
    findings: list[Finding], options: JudgeOptions, step_count: int
) -> list[tuple[int, int]]:
    """Widen the ranges hit by the coarse scan and merge them into windows to rescan.

    Each window is padded by one coarse stride on either side so that the true start and
    end of the event fall inside it: the coarse scan can only spot an event at one of its
    sample points, and the real onset may be up to one stride earlier. Step 0 is always
    included, because the static-scene checks only run on the initial frame.
    """
    margin = options.stride
    spans = [(0, 1)]
    spans += [
        (max(f.first_step - margin, 0), min(f.last_step + margin + 1, step_count)) for f in findings
    ]
    spans.sort()

    merged: list[tuple[int, int]] = []
    for start, stop in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _decide(findings: list[Finding], trustworthy: bool, strict: bool) -> tuple[str, str]:
    """The verdict rules.

    Violating an explicitly declared rule is the primary way to fail: that is the boundary
    the task author drew. Generic geometric criteria are recorded as hints and do not fail
    a run on their own unless the caller asks for that with ``--strict`` -- with one
    exception. A detector may raise a generic finding to failure level when the
    configuration cannot be physically valid under any policy, the way two bodies sharing
    the same volume with no contact between them cannot. No declaration is needed to know
    that a simulation has stopped describing something real.

    When a run fails on both counts the declared violation leads the headline. It is the
    more specific statement -- it names a requirement this task actually had -- so it tells
    the reader more than a generic criterion does.

    When the replay itself cannot be trusted we refuse to conclude: saying either PASS or
    FAIL at that point would be a guess.
    """
    failures = sorted(
        (f for f in findings if f.severity is Severity.FAILURE),
        key=lambda f: f.tier is not Tier.DECLARED,
    )
    warnings = [f for f in findings if f.severity is Severity.WARNING]

    if not trustworthy:
        return (
            VERDICT_INCONCLUSIVE,
            "The replay does not reproduce the recording, so no trustworthy conclusion is "
            "possible; check the model and the MuJoCo version first.",
        )

    if failures:
        # Most failures come from violating a declared rule, but not all: a generic
        # criterion is allowed to fail a run on its own when the configuration cannot be
        # physically valid whatever the policy says, as with two bodies occupying the same
        # space and no contact anywhere between them. Naming the tiers separately keeps
        # the headline honest about which kind of authority the verdict rests on.
        declared = sum(f.tier is Tier.DECLARED for f in failures)
        generic = len(failures) - declared
        if declared and generic:
            basis = f"{declared} declared rule and {generic} generic criterion failures"
        elif declared:
            basis = f"{declared} declared rule{'s' if declared > 1 else ''} violated"
        else:
            basis = (
                f"{generic} generic criterion failure{'s' if generic > 1 else ''}, "
                "with no declared rule broken"
            )
        top = failures[0]
        return VERDICT_FAIL, f"{basis}; the most severe is {top.code}: {top.summary}"

    if strict and warnings:
        top = warnings[0]
        return (
            VERDICT_FAIL,
            f"Strict mode is on, so generic hints count as failures too: "
            f"{_hints(len(warnings))}, the first being {top.code}: {top.summary}",
        )

    if warnings:
        return (
            VERDICT_PASS,
            f"No declared rule was violated; {_hints(len(warnings))} worth a manual review.",
        )
    return VERDICT_PASS, "No implausible physics found."


def _hints(count: int) -> str:
    """Phrase a count of generic hints with the right agreement.

    Trivial, but the verdict line is the one sentence most readers of this report will
    actually read, and "there are 1 generic geometric hints" undermines it.
    """
    if count == 1:
        return "there is 1 generic geometric hint"
    return f"there are {count} generic geometric hints"


def _collect_preflight_notes(
    bundle: CaseBundle, loaded, policy, options: JudgeOptions
) -> list[str]:
    notes: list[str] = []
    if not bundle.model_is_compiled:
        notes.append(
            "No compiled artifact compiled-model.mjb was found, so closed-scene/scene.xml "
            "was recompiled instead. A recompiled model may differ subtly from the one used "
            "for the recording."
        )
    if loaded.version_matches is False:
        notes.append(
            f"The current MuJoCo {loaded.engine_version} differs from the "
            f"{loaded.recorded_version} used for the recording."
        )
    if not policy.enabled:
        notes.append(
            "runtime_feedback.enabled is false in bound-operation.json, so the declared "
            "rules may be incomplete."
        )
    notes.extend(policy.issues)
    notes.extend(policy.unenforced_contracts())
    if options.stride > 1:
        notes.append(
            f"Scanning coarsely with stride {options.stride}; transient events shorter than "
            f"{options.stride} steps may be missed."
        )
    return notes
