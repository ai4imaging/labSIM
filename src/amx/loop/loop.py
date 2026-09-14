"""The bounded refinement loop.

Build the cell, run the plan, judge it, patch what failed, repeat — for a fixed number of
rounds and no more. That is the whole control structure, and keeping it a plain `for` loop
over immutable designs is deliberate. There is no freeze, no transaction, no receipt, no
manifest: a round is a directory containing the design that produced it and the verdict it
got, and re-running that design reproduces that verdict.

Two things make the loop terminate honestly rather than by exhaustion:

`best-so-far`  Rounds are scored by failure count, so a run that never reaches PASS still
               delivers its best attempt instead of its last one, which is usually a
               different thing.

`acceptance`   The loop scans coarsely, because a stride-20 replay is twenty times cheaper
               and finds all but the briefest events. A round that passes coarsely is
               re-judged step by step before it is believed. Passing the cheap check and
               failing the real one is a normal outcome, and it costs one extra judgement
               rather than a whole extra round.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from amx.codesign.dfm import check_part
from amx.codesign.export import write_geometry
from amx.codesign.parts import AnyPart, Part
from amx.llm import LlmClient
from amx.loop.judge import Judgement, judge
from amx.loop.repair import PatchRejected, RepairPatch, apply_patch, propose_repairs
from amx.report import Finding, RepairTarget, Report, Severity
from amx.sim.outcome import check_destination
from amx.sim.plan import OperationPlan
from amx.sim.run import EpisodeResult, run_episode
from amx.sim.scene import BuiltScene, Workcell, build_scene


class Design(BaseModel):
    """A cell, a plan, and the parts the cell's fixtures are built from.

    This is the loop's unit of state and the thing a round writes to disk. Because the
    parts are carried as parameters rather than as exported directories, a design is small,
    diffable and fully reproducible: `materialise` regenerates every mesh from it.
    """

    model_config = ConfigDict(extra="forbid")

    workcell: Workcell
    plan: OperationPlan
    parts: dict[str, AnyPart] = Field(
        default_factory=dict,
        description="Keyed by part_id, matching the cell's fixture placements.",
    )

    def part_objects(self) -> dict[str, Part]:
        return dict(self.parts)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
        )
        return path

    @classmethod
    def read(cls, path: Path) -> "Design":
        return cls.model_validate(json.loads(Path(path).read_text()))


@dataclass
class LoopConfig:
    """How hard to try, and how carefully to look."""

    max_rounds: int = 4
    scan_stride: int = 20
    """Replay stride for the per-round judgement. `sim_judge` rescans anything it flags."""

    acceptance_stride: int = 1
    """Stride for the confirming judgement of a round that looked like a pass."""

    max_steps: int | None = None
    strict: bool = False
    stop_on_dfm_failure: bool = True
    """Whether an unmakeable part ends the run. Off lets a geometry study continue."""

    candidates: int = 2
    """How many different repairs to try per expansion.

    One is the old greedy chain: propose, apply, hope. Two is enough to matter, because
    the failure modes here are usually ambiguous between "the path is wrong" and "the
    thing is in the wrong place", and one round spent on the wrong reading of that used to
    cost the whole run.
    """

    beam: int = 3
    """How many designs stay eligible for expansion.

    Bounded so the loop cannot quietly turn into breadth-first search over a budget it was
    given for depth.
    """

    backtrack: bool = True
    """Whether a round that made things worse may be abandoned for an earlier one.

    This is the part that makes multiple candidates worth having. Without it the loop
    still walks a single chain and simply picks a better next step.
    """


@dataclass
class Round:
    """One iteration's inputs, outputs and verdict."""

    index: int
    directory: Path
    design: Design
    dfm: Report
    episode: EpisodeResult | None = None
    judgement: Judgement | None = None
    patch: RepairPatch | None = None
    error: str = ""
    parent_index: int | None = None
    """Which round this one was patched from. `None` for the design the run started with.

    Recorded because with backtracking on, round order is no longer lineage: round 5 may
    well be a child of round 2, and a history that does not say so is unreadable.
    """

    origin: str = "seed"
    """Where the patch came from — `heuristic`, `model-0`, and so on."""

    expanded: bool = False

    @property
    def passed(self) -> bool:
        return self.judgement is not None and self.judgement.passed and self.dfm.passed

    def score(self) -> tuple[int, int, int, int]:
        """Lower is better. A round that never ran ranks behind every one that did."""
        if self.judgement is None:
            return (2, len(self.dfm.failures), 10**6, 10**6)
        conclusive, failures, warnings = self.judgement.score()
        return (conclusive, len(self.dfm.failures) + failures, warnings, self.index)

    def summary(self) -> str:
        if self.error:
            return f"round {self.index}: {self.error}"
        verdict = self.judgement.verdict if self.judgement else "NOT RUN"
        counts = (
            f"{len(self.judgement.report.failures)} failure(s), "
            f"{len(self.judgement.report.warnings)} warning(s)"
            if self.judgement
            else "no judgement"
        )
        dfm = "makeable" if self.dfm.passed else f"{len(self.dfm.failures)} DFM failure(s)"
        lineage = (
            "" if self.parent_index is None else f" [from round {self.parent_index}, {self.origin}]"
        )
        return f"round {self.index}: {verdict} — {counts}, {dfm}{lineage}"


@dataclass
class LoopResult:
    """Everything a run produced, and which round to deliver."""

    run_dir: Path
    rounds: list[Round] = field(default_factory=list)
    stopped_because: str = ""

    @property
    def best(self) -> Round | None:
        return min(self.rounds, key=Round.score) if self.rounds else None

    @property
    def passed(self) -> bool:
        best = self.best
        return best is not None and best.passed

    def summary(self) -> str:
        lines = [f"{self.run_dir.name}: {len(self.rounds)} round(s), {self.stopped_because}"]
        lines.extend("  " + r.summary() for r in self.rounds)
        best = self.best
        if best is not None:
            lines.append(f"  best: round {best.index} ({best.directory})")
        return "\n".join(lines)


Repairer = Callable[["Round", Design], RepairPatch]
"""How a round's verdict becomes the next round's design.

The default asks a model. Swapping in a function is what lets the tests drive the loop
deterministically and the offline example run without credentials, and it keeps the loop
itself honest: it never touches an API, it just calls this.
"""


def run_loop(
    design: Design,
    run_dir: Path,
    *,
    config: LoopConfig | None = None,
    client: LlmClient | None = None,
    context: str = "",
    repairer: Repairer | None = None,
) -> LoopResult:
    """Refine `design` until it passes or the round budget runs out.

    The shape is best-first rather than a chain. Each expansion takes the best design that
    has not been expanded yet, proposes `config.candidates` different repairs for it, and
    evaluates each; every evaluation is a round on disk. When a design's children all come
    out worse than it, the frontier simply picks something else next time, which is the
    whole of the backtracking. `max_rounds` counts evaluations, so a run costs the same
    whether it spends its budget going deep or going wide.
    """
    config = config or LoopConfig()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result = LoopResult(run_dir=run_dir, stopped_because="the round budget ran out")

    seed = _execute_round(design, 0, run_dir / "round-00", config)
    result.rounds.append(seed)
    if (stop := _terminal(seed, config)) is not None:
        result.stopped_because = stop
        _write_manifest(result, config)
        return result

    while len(result.rounds) < config.max_rounds:
        parent = _next_to_expand(result.rounds, config)
        if parent is None:
            result.stopped_because = "every design on the frontier has been tried"
            break
        parent.expanded = True

        try:
            proposals = _propose(parent, config, client, context, repairer)
        except Exception as error:  # noqa: BLE001 — a failed proposal ends the run, not the process
            result.stopped_because = f"no repair could be proposed: {error}"
            break
        _write_proposals(parent, proposals)
        if not proposals:
            if _only_asset_failures(parent):
                result.stopped_because = _write_asset_request(run_dir, parent)
                break
            continue

        parent.patch = proposals[0][0]
        stop = ""
        for patch, origin in proposals:
            if len(result.rounds) >= config.max_rounds:
                break
            try:
                workcell, plan, parts = apply_patch(
                    workcell=parent.design.workcell,
                    plan=parent.design.plan,
                    parts=parent.design.part_objects(),
                    patch=patch,
                )
            except PatchRejected as error:
                # One unusable candidate is not a dead run — the others may still apply.
                _note(parent, f"{origin} did not apply: {error}")
                continue
            child_design = Design(workcell=workcell, plan=plan, parts=parts)  # type: ignore[arg-type]
            index = len(result.rounds)
            child = _execute_round(child_design, index, run_dir / f"round-{index:02d}", config)
            child.parent_index = parent.index
            child.origin = origin
            result.rounds.append(child)
            if (terminal := _terminal(child, config)) is not None:
                stop = terminal
                break
        if stop:
            result.stopped_because = stop
            break

    _write_manifest(result, config)
    return result


def _terminal(round_result: Round, config: LoopConfig) -> str | None:
    """Whether this round ends the whole run, and why. `None` to keep going."""
    if round_result.error:
        return round_result.error
    if round_result.passed:
        return f"round {round_result.index} passed"
    if not round_result.dfm.passed and config.stop_on_dfm_failure:
        return f"round {round_result.index} produced a part that cannot be made"
    return None


def _next_to_expand(rounds: list[Round], config: LoopConfig) -> Round | None:
    """The best unexpanded design, from within the beam."""
    ranked = sorted(rounds, key=Round.score)
    frontier = ranked if not config.backtrack else ranked[: max(1, config.beam)]
    if not config.backtrack:
        # Without backtracking the loop stays a chain: only the newest round may grow.
        newest = max(rounds, key=lambda r: r.index)
        return None if newest.expanded or newest.judgement is None else newest
    return next(
        (item for item in frontier if not item.expanded and item.judgement is not None), None
    )


def _propose(
    parent: Round,
    config: LoopConfig,
    client: LlmClient | None,
    context: str,
    repairer: Repairer | None,
) -> list[tuple[RepairPatch, str]]:
    """The candidate repairs for one design.

    A caller-supplied `repairer` stays single-candidate on purpose: it is the seam the
    tests and the offline example drive the loop through, and a deterministic function
    asked for three answers would give the same one three times.
    """
    if repairer is not None:
        patch = repairer(parent, parent.design)
        return [] if patch.empty else [(patch, "repairer")]
    return _next_patches(parent, parent.design, config, client, context)


def _write_proposals(parent: Round, proposals: list[tuple[RepairPatch, str]]) -> None:
    payload = [
        {"origin": origin, "patch": patch.model_dump(mode="json"), "describes": patch.describe()}
        for patch, origin in proposals
    ]
    (parent.directory / "patch.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def _note(round_result: Round, text: str) -> None:
    path = round_result.directory / "notes.txt"
    with path.open("a") as handle:
        handle.write(text + "\n")


def _only_asset_failures(round_result: Round) -> bool:
    """Whether everything left wrong is the 3D asset's fault rather than the design's.

    `S1`/`S2`/`S3` mean a body has no mass, no collision geometry, or geometry that does
    not sit where its visual does. No waypoint and no fixture parameter can fix any of
    that — the asset itself has to be rebuilt. Recognising that is what keeps the loop
    from spending its remaining budget nudging an arm around a body it cannot touch.
    """
    judgement = round_result.judgement
    if judgement is None or not judgement.report.failures:
        return False
    return all(
        finding.repair_target is RepairTarget.ASSET for finding in judgement.report.failures
    )


def _write_asset_request(run_dir: Path, round_result: Round) -> str:
    """Hand the failure back to asset generation, in a form it can be re-run from.

    This is the join between the two halves of the system. Rather than fail with "the
    asset is wrong", the loop writes down which asset, which findings, and what the
    generator would have to hold itself to — which is exactly a `GroundingSpec` fragment,
    so `amx asset generate --grounding-spec` can pick it up unchanged.
    """
    judgement = round_result.judgement
    failures = judgement.report.failures if judgement else []
    subjects = sorted({finding.subject.split("/")[0] for finding in failures})
    payload = {
        "reason": "every remaining failure points at the 3D asset, not at the design",
        "round": round_result.index,
        "assets": subjects,
        "findings": [
            {
                "code": finding.code,
                "subject": finding.subject,
                "summary": finding.summary,
                "metrics": finding.metrics,
            }
            for finding in failures
        ],
        "regenerate_with": {
            "requirements": sorted({_asset_requirement(f.code) for f in failures}),
        },
    }
    (run_dir / "asset-repair-request.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    return (
        f"the asset(s) {', '.join(subjects) or '(unnamed)'} must be regenerated; "
        f"see asset-repair-request.json"
    )


_ASSET_REQUIREMENTS = {
    "S1_MOVABLE_BODY_MASSLESS": "every movable body must carry a mass",
    "S1_MOVABLE_BODY_INERTIALESS": "every movable body must carry an inertia tensor",
    "S2_ENTITY_WITHOUT_COLLISION_GEOMETRY": "every body must have collision geometry",
    "S2_MOVABLE_BODY_VISUAL_ONLY": "a movable body may not be visual-only",
    "S3_VISUAL_COLLISION_OFFSET": "collision geometry must coincide with the visual mesh",
}


def _asset_requirement(code: str) -> str:
    return _ASSET_REQUIREMENTS.get(code, f"fix {code}")


def _execute_round(
    design: Design, index: int, directory: Path, config: LoopConfig
) -> Round:
    """Materialise, run and judge one design. Errors are recorded, not raised."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    design.write(directory / "design.json")

    try:
        built, dfm = materialise(design, directory)
    except Exception as error:  # noqa: BLE001
        report = Report(kind="dfm", subject=f"round {index}")
        report.findings.append(
            Finding(
                code="L-BUILD",
                severity=Severity.FAILURE,
                subject=f"round {index}",
                summary=f"the cell could not be composed: {error}",
                repair_target=RepairTarget.NONE,
            )
        )
        report.write(directory / "dfm.json")
        return Round(
            index=index,
            directory=directory,
            design=design,
            dfm=report,
            error=f"round {index} could not be built: {error}",
        )

    dfm.write(directory / "dfm.json")
    if not dfm.passed and config.stop_on_dfm_failure:
        return Round(index=index, directory=directory, design=design, dfm=dfm)

    try:
        episode = run_episode(built, design.plan, directory / "case", case_id=f"round-{index:02d}")
    except Exception as error:  # noqa: BLE001
        return Round(
            index=index,
            directory=directory,
            design=design,
            dfm=dfm,
            error=f"round {index} could not be executed: {error}",
        )

    judgement = judge(
        episode.case_dir,
        stride=config.scan_stride,
        max_steps=config.max_steps,
        strict=config.strict,
    )
    if judgement.passed and config.scan_stride != config.acceptance_stride:
        # Confirm a coarse pass step by step before believing it.
        judgement = judge(
            episode.case_dir,
            stride=config.acceptance_stride,
            max_steps=config.max_steps,
            strict=config.strict,
        )
    # The judge answers "did anything go wrong"; this answers "did the task happen". Folded
    # in before the verdict is read, so a round cannot pass on a plan that never moved
    # anything — which is what happens when the judge has no detector for the objective.
    judgement.absorb(check_destination(design.plan, episode))
    judgement.write(directory)
    (directory / "episode.txt").write_text(episode.summary() + "\n")

    return Round(
        index=index,
        directory=directory,
        design=design,
        dfm=dfm,
        episode=episode,
        judgement=judgement,
    )


def materialise(design: Design, directory: Path) -> tuple[BuiltScene, Report]:
    """Build every part, check it, and compose the scene that uses it.

    The design names its parts by id; the fixture placements in its workcell point at
    directories this function creates. Rewriting the sources here rather than storing them
    in the design is what lets a design be moved, re-run elsewhere and diffed.
    """
    directory = Path(directory)
    parts_dir = directory / "parts"
    dfm = Report(kind="dfm", subject=design.workcell.workcell_id)

    sources: dict[str, Path] = {}
    for part_id, part in design.parts.items():
        placement = _placement_for(design.workcell, part_id)
        geometry = part.build()
        report = check_part(geometry, mounted_on_arm=placement.mount == "tool")
        dfm.extend(report)
        write_geometry(geometry, parts_dir / part_id, report=report)
        sources[part_id] = parts_dir / part_id

    workcell = design.workcell.model_copy(deep=True)
    for placement in workcell.fixtures:
        if placement.part_id in sources:
            placement.source = sources[placement.part_id]

    built = build_scene(workcell, directory / "scene")
    return built, dfm


def _placement_for(workcell: Workcell, part_id: str):
    try:
        return workcell.fixture(part_id)
    except KeyError:
        raise KeyError(
            f"the design defines part {part_id!r} but the cell places no fixture by that name; "
            f"it places: {', '.join(p.part_id for p in workcell.fixtures) or '(none)'}"
        ) from None


def _next_patches(
    round_result: Round,
    design: Design,
    config: LoopConfig,
    client: LlmClient | None,
    context: str,
) -> list[tuple[RepairPatch, str]]:
    """Ask for repairs, folding manufacturability failures in with the physics ones."""
    judgement = round_result.judgement
    if judgement is None:
        raise RuntimeError("the round produced no judgement to repair from")
    combined = judgement.report.model_copy(deep=True)
    combined.findings.extend(round_result.dfm.failures)
    merged = Judgement(
        case_dir=judgement.case_dir,
        verdict=judgement.verdict,
        reason=judgement.reason,
        report=combined,
        counts=judgement.counts,
        notes=judgement.notes,
    )
    client = client or _tracing_client(round_result)
    context = "\n\n".join(filter(None, [context, _skill_briefing(combined)]))
    return propose_repairs(
        judgement=merged,
        workcell=design.workcell,
        plan=design.plan,
        parts=design.part_objects(),
        count=max(1, config.candidates),
        client=client,
        context=context,
    )


def _skill_briefing(report: Report) -> str:
    """What earlier runs learned about these particular failure codes.

    The same library the asset generator retrieves from. The two halves fail in different
    vocabularies, but a lesson is indexed by the codes it was learned from, so a lesson
    about `R2_RECEIVER_SIDE_CLEARANCE_LOST` reaches whichever half hits that code — and
    nothing else is pulled in, which matters more than breadth here: an irrelevant lesson
    in a repair prompt is an invitation to fix the wrong thing.
    """
    try:
        from amx.skills.library import SkillLibrary  # noqa: PLC0415

        library = SkillLibrary()
        lessons = library.relevant(codes=[f.code for f in report.failures])
        return library.briefing(lessons) if lessons else ""
    except Exception:  # noqa: BLE001 — an unreadable library must not stop a run
        return ""


def _tracing_client(round_result: Round) -> LlmClient | None:
    """A client that writes its exchanges next to the round that prompted them.

    Only built when the caller passed none, so supplying a client keeps full control of
    where — or whether — anything is written.
    """
    try:
        return LlmClient(trace_dir=round_result.directory / "llm")
    except ValueError:
        # No model configured. Let the heuristic candidates carry the round.
        return None


def _write_manifest(result: LoopResult, config: LoopConfig) -> None:
    best = result.best
    payload: dict[str, Any] = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "max_rounds": config.max_rounds,
            "scan_stride": config.scan_stride,
            "acceptance_stride": config.acceptance_stride,
            "max_steps": config.max_steps,
            "strict": config.strict,
            "candidates": config.candidates,
            "beam": config.beam,
            "backtrack": config.backtrack,
        },
        "stopped_because": result.stopped_because,
        "passed": result.passed,
        "best_round": None if best is None else best.index,
        "rounds": [
            {
                "index": r.index,
                "directory": r.directory.name,
                "parent": r.parent_index,
                "origin": r.origin,
                "verdict": r.judgement.verdict if r.judgement else None,
                "failures": len(r.judgement.report.failures) if r.judgement else None,
                "warnings": len(r.judgement.report.warnings) if r.judgement else None,
                "dfm_failures": len(r.dfm.failures),
                "error": r.error or None,
                "patch": r.patch.diagnosis if r.patch else None,
            }
            for r in result.rounds
        ],
    }
    (result.run_dir / "run.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    (result.run_dir / "summary.txt").write_text(result.summary() + "\n")
    if best is not None:
        link = result.run_dir / "best"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(best.directory.name)
