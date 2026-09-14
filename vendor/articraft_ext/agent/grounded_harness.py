"""An Articraft agent that will not finish until the asset is also the right object.

Stock Articraft gates its finish attempt on one thing: does the latest code compile.
That is the right gate for a geometry SDK and the wrong one for this project, because a
solid cylinder of the wrong size compiles perfectly and a beaker with no cavity compiles
perfectly. The agent stops, reports success, and nothing in the loop ever contradicted it.

This subclass adds three things and changes nothing else:

* Three grounding tools the model can call, each answering in the same `<grounding_signals>`
  shape as `<compile_signals>`.
* A finish gate that additionally requires every applicable grounding check to have been
  run against the current revision and to have passed. Stale results do not count: an
  edit invalidates them exactly as it invalidates a compile.
* Renders fed back as images. A tool result is a string on chat-completions providers,
  so the PNGs are attached as a following user message instead — which is why the
  vendored turn loop grew a `_maybe_inject_tool_attachments` hook.

Everything expensive is delegated to `amx.grounding`, which the benchmark judge also
uses. The loop and the grader measure with the same code, or the loop is optimising
something nobody is grading.
"""

from __future__ import annotations

import base64
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from agent.grounding_tools import (
    GROUNDING_TOOL_NAMES,
    PHYSICAL_TOOL,
    PROTOCOL_TOOL,
    VISUAL_TOOL,
)
from agent.harness import ArticraftAgent
from agent.tools import build_tool_registry
from agent.tools.base import ToolResult
from amx.grounding.build import GroundingBuildError, build_grounded_asset
from amx.grounding.checks import check_dimensions, check_protocol, check_topology, check_visual
from amx.grounding.signals import render_grounding_signals
from amx.grounding.spec import GroundingSpec
from amx.paths import ARTICRAFT_DIR
from amx.report import Finding, Report, Severity

logger = logging.getLogger(__name__)

MAX_ATTACHED_RENDERS = 4
"""Renders attached per visual check. Four views is enough to see the silhouette from
every side, and each one costs real context."""

UNSATISFIABLE_AFTER_BLOCKS = 3

THRASHING_AFTER_BLOCKS = 8
"""Refusals in one run, of any kind, after which the run is treated as going nowhere.

`UNSATISFIABLE_AFTER_BLOCKS` counts an identical failure set, and deliberately restarts when the
set changes, so that fixing one thing and breaking another keeps its budget. A model that keeps
changing *which* subset fails never trips it and never stops: PCR-001 was refused 35 times over
one run, alternating an unmeasurable bore with a wrong one and a well count that moved on every
edit, and spent 61% of its turns pressing finish. It submitted nothing and scored zero.

Persistently failing findings are declared unsatisfiable once this is reached. That is safe by
the same argument as the counter above -- they stay `FAILURE` and the judge still grades them, so
the gate stops spending turns while the score does not move -- and a scored asset with honest
failures recorded is worth more than no asset at all.
"""

THRASHING_PERSISTENCE = 0.5
"""How much of the thrashing a finding has to account for to be called the thing blocking it."""
"""Identical refusals to tolerate before a failure is declared unreachable.

Two in a row is a model that has not finished trying. Three is a specification the
geometry cannot satisfy, and `newframework100-opus5` proves the cost of waiting longer:
SYF-001 was handed the same sentence 51 times — a membrane pore rating of 0.22 µm read as
a 0.00 mm bounding-box extent — and spent its entire budget on it. Fourteen of sixty cases
died this way.

This is not forgiveness. The finding stays a `FAILURE`, flagged `unsatisfiable`, and the
judge grades it exactly as it would have. All that changes is that the gate stops spending
turns on it.
"""


class GroundingState:
    """What has been checked, and whether it still applies to the current code.

    Freshness is tracked the same way Articraft tracks compile freshness: against a
    revision counter that every mutating tool bumps. A check that passed two edits ago
    says nothing about what is on disk now, and treating it as though it did is how an
    agent talks itself into finishing.
    """

    def __init__(self) -> None:
        self.reports: dict[str, Report] = {}
        self.revisions: dict[str, int] = {}
        self.revision = 0
        self.unsatisfiable: set[str] = set()
        """Finding keys the model has been refused over and demonstrably cannot fix."""
        self._last_fingerprint: tuple[str, ...] = ()
        self._repeats = 0
        self._counted_at = -1
        """The revision the last refusal was counted at, so a repeat needs a real attempt."""
        self._refusals = 0
        """Refusals in this run, of any kind, for spotting a run that is going nowhere."""
        self._seen: Counter[str] = Counter()
        """How many refusals each finding has blocked, to tell the persistent from the passing."""

    def reset(self) -> None:
        self.reports.clear()
        self.revisions.clear()
        self.revision = 0
        self.unsatisfiable.clear()
        self._last_fingerprint = ()
        self._repeats = 0
        self._counted_at = -1
        self._refusals = 0
        self._seen.clear()

    def mark_mutated(self) -> None:
        self.revision += 1

    def record(self, tool: str, report: Report) -> None:
        self.reports[tool] = self._flag(report)
        self.revisions[tool] = self.revision

    def is_fresh(self, tool: str) -> bool:
        return self.revisions.get(tool) == self.revision

    def stale_or_missing(self, required: list[str]) -> list[str]:
        return [tool for tool in required if not self.is_fresh(tool)]

    def blocking_failures(self, report: Report) -> list[Finding]:
        """The failures still worth refusing a submission over."""
        return [
            finding for finding in report.failures if _key(finding) not in self.unsatisfiable
        ]

    def failing(self, required: list[str]) -> list[Report]:
        return [
            report
            for tool in required
            if (report := self.reports.get(tool)) is not None and self.blocking_failures(report)
        ]

    def note_refusal(self, failing: list[Report]) -> list[str]:
        """Record that these failures blocked a finish attempt.

        Returns the finding keys newly declared unsatisfiable, or an empty list while the
        run still has attempts left.

        What is counted is consecutive *finish attempts*, not turns. A model that submits
        three times running with an identical failure set has said three times that it
        believes it is done, with as many edits in between as it liked. Counting identical
        sets is also what separates a stuck run from a working one: fix one thing and break
        another and the set differs, the count restarts, and the budget is kept.
        """
        fingerprint = tuple(
            sorted(_key(finding) for report in failing for finding in self.blocking_failures(report))
        )
        if not fingerprint:
            self._last_fingerprint, self._repeats = (), 0
            return []

        self._refusals += 1
        self._seen.update(fingerprint)
        if thrashing := self._thrashing():
            return thrashing

        if fingerprint != self._last_fingerprint:
            self._last_fingerprint, self._repeats = fingerprint, 1
            self._counted_at = self.revision
        elif self.revision != self._counted_at:
            # The same failures over again, but the model changed the geometry in between,
            # so this is a second attempt at them and counts as one.
            self._repeats += 1
            self._counted_at = self.revision
        else:
            # Nothing was edited since the last refusal. Submitting again is not another
            # attempt at the problem, it is the same attempt resubmitted — and letting it
            # count would hand a model three presses of the finish button in a row the same
            # verdict as three rounds of genuine work.
            return []

        if self._repeats < UNSATISFIABLE_AFTER_BLOCKS:
            return []

        declared = [key for key in fingerprint if key not in self.unsatisfiable]
        self.unsatisfiable.update(fingerprint)
        # Annotate what is already recorded, so the evidence on disk says the verdict was
        # taken rather than leaving it to be inferred from the run ending.
        for tool, report in self.reports.items():
            self.reports[tool] = self._flag(report)
        return declared

    def _thrashing(self) -> list[str]:
        """Declare what is holding up a run that has stopped converging.

        The identical-set counter cannot see this case: every attempt fails differently, so it
        restarts every time and the run keeps its budget until there is none left. What is stuck
        is not any one failure set but the run, and what is blocking it is whichever findings keep
        coming back across attempts -- not the ones that appeared once while something else moved.
        """
        if self._refusals < THRASHING_AFTER_BLOCKS:
            return []
        persistent = {
            key
            for key, count in self._seen.items()
            if count >= self._refusals * THRASHING_PERSISTENCE
        }
        declared = sorted(persistent - self.unsatisfiable)
        if not declared:
            return []
        self.unsatisfiable.update(persistent)
        for tool, report in self.reports.items():
            self.reports[tool] = self._flag(report)
        return declared

    def _flag(self, report: Report) -> Report:
        """Mark failures ruled unreachable, without softening them.

        The severity stays `FAILURE`. It is tempting to downgrade to a warning, and it
        would be wrong: `judge._finding_credit` awards a warning a flat 0.5 and grades a
        failure on how far off it actually was, so "letting the run finish" would quietly
        hand back half the item's weight for a dimension that was 64% out. The gate stops
        blocking; the score does not move.
        """
        if not self.unsatisfiable:
            return report
        flagged = [
            (
                finding.model_copy(update={"detail": {**finding.detail, "unsatisfiable": True}})
                if finding.severity is Severity.FAILURE
                and _key(finding) in self.unsatisfiable
                and not finding.detail.get("unsatisfiable")
                else finding
            )
            for finding in report.findings
        ]
        if flagged == report.findings:
            return report
        return report.model_copy(update={"findings": flagged})


class GroundedArticraftAgent(ArticraftAgent):
    """`ArticraftAgent` plus grounding tools and a grounding-aware finish gate."""

    def __init__(
        self,
        *args: Any,
        grounding_spec: GroundingSpec | None = None,
        grounding_dir: Path | None = None,
        grounding_blocks: bool = True,
        visual_blocks: bool = False,
        visual_feedback: bool = True,
        llm_client: Any = None,
        **kwargs: Any,
    ) -> None:
        # Set before delegating up: the base constructor builds the system prompt, and
        # the override below has to know whether there is a spec to describe.
        self.grounding_spec = grounding_spec
        self.grounding_dir = Path(grounding_dir) if grounding_dir else None
        self.grounding_blocks = grounding_blocks
        self.visual_blocks = visual_blocks
        self.visual_feedback = visual_feedback
        self.llm_client = llm_client
        self.grounding_state = GroundingState()
        self._pending_attachments: list[tuple[str, Path]] = []

        super().__init__(*args, **kwargs)

        if self.grounding_enabled:
            # Rebuilt rather than patched: the base class picked its tools from the
            # provider alone and did not know a spec was coming.
            self.tool_registry = build_tool_registry(
                self.provider,
                sdk_package=self.sdk_package,
                runtime_limits=self.runtime_limits,
                grounding=True,
            )

    def _build_system_prompt(self, prompt_path: str) -> tuple[Path, str]:
        """Articraft's prompt, plus a section about the tools this subclass adds.

        Appended here rather than compiled into the six generated prompts because the
        grounding tools are conditional on a run having a spec, and a prompt that
        describes tools the model has not been given is worse than saying nothing.
        Overriding the builder rather than the attribute keeps the addition inside
        everything downstream that hashes the prompt.
        """
        path, text = super()._build_system_prompt(prompt_path)
        if not self.grounding_enabled:
            return path, text

        source = ARTICRAFT_DIR / "agent" / "prompts" / "sections" / "grounding_tools.md"
        if not source.is_file():
            logger.warning("%s is missing; the model will not be told the tools exist", source)
            return path, text
        return path, f"{text.rstrip()}\n\n{source.read_text()}"

    # ----------------------------------------------------------------- #
    # state
    # ----------------------------------------------------------------- #

    @property
    def grounding_enabled(self) -> bool:
        return self.grounding_spec is not None and not self.grounding_spec.is_empty()

    def required_checks(self) -> list[str]:
        """Which checks this task can actually be held to, cheapest and surest first.

        A spec with no visual features cannot fail a visual check, and demanding one
        would block the run on a tool that has nothing to say.

        The order is the order failures are worth reading in: what the object measures,
        then what it does when driven, then what it looks like. The visual check is
        required whenever the task states visual features, even though its findings are
        advisory unless `visual_blocks` — requiring it is what puts the renders in front
        of the model, and a part floating unattached under the body is obvious in a render
        and invisible to every measurement.
        """
        if not self.grounding_enabled:
            return []
        spec = self.grounding_spec
        assert spec is not None
        required = []
        if spec.dimensions or spec.components or spec.cavity or spec.total_mass_kg:
            required.append(PHYSICAL_TOOL)
        if spec.operations or spec.stability or spec.probe or spec.tilt:
            required.append(PROTOCOL_TOOL)
        if spec.visual.features:
            required.append(VISUAL_TOOL)
        return required

    def _reset_run_compile_state(self) -> None:
        super()._reset_run_compile_state()
        self.grounding_state.reset()
        self._pending_attachments.clear()

    def _mark_code_mutated(self, tool_name: str) -> None:
        super()._mark_code_mutated(tool_name)
        self.grounding_state.mark_mutated()

    # ----------------------------------------------------------------- #
    # tool interception
    # ----------------------------------------------------------------- #

    async def _execute_tool(self, tool_call: dict) -> tuple[ToolResult, dict]:
        func_name = self._tool_call_name(tool_call)
        if func_name not in GROUNDING_TOOL_NAMES:
            return await super()._execute_tool(tool_call)

        tool_id = tool_call["id"]
        call_type = str(tool_call.get("type", "function"))
        result = await self._execute_grounding(func_name, tool_call_id=tool_id)
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_id,
            "name": func_name,
            "tool_type": call_type if call_type == "custom" else "function",
            "content": json.dumps(
                {k: v for k, v in result.to_dict().items() if k != "tool_call_id"}
            ),
        }
        thought_signature = tool_call.get("thought_signature")
        if thought_signature is not None:
            tool_message["thought_signature"] = thought_signature
        return result, tool_message

    async def _execute_grounding(self, tool: str, *, tool_call_id: str) -> ToolResult:
        if not self.grounding_enabled:
            return ToolResult(
                error="No grounding specification was supplied for this run.",
                tool_call_id=tool_call_id,
            )

        import asyncio  # noqa: PLC0415

        try:
            report = await asyncio.to_thread(self._run_check, tool)
        except GroundingBuildError as error:
            # The asset does not build, so nothing can be measured. This is not the
            # grounding tool's answer to give -- say so and point at compile_model.
            return ToolResult(
                output=(
                    f"<grounding_signals>\n<summary>\ncheck={_label(tool)} status=blocked\n"
                    f"The model could not be built, so nothing could be measured: {error}\n"
                    "Fix the build with `compile_model` first.\n</summary>\n</grounding_signals>"
                ),
                tool_call_id=tool_call_id,
            )
        except Exception as error:  # noqa: BLE001 - a grounding crash must not kill the run
            logger.exception("grounding check %s failed", tool)
            return ToolResult(
                error=f"{_label(tool)} could not be run: {type(error).__name__}: {error}",
                tool_call_id=tool_call_id,
            )

        self.grounding_state.record(tool, report)
        self._persist(tool, report)
        return ToolResult(
            output=render_grounding_signals(report, check_name=_label(tool)),
            tool_call_id=tool_call_id,
        )

    def _run_check(self, tool: str) -> Report:
        """Build the current model and run one check against it. Runs off the event loop."""
        spec = self.grounding_spec
        assert spec is not None
        asset = build_grounded_asset(Path(self.file_path), asset_id=spec.asset_id)

        if tool == PHYSICAL_TOOL:
            report = Report(kind="grounding-physical", subject=spec.asset_id)
            report.extend(check_dimensions(asset, spec))
            report.extend(check_topology(asset, spec))
            return report
        if tool == PROTOCOL_TOOL:
            return check_protocol(asset, spec)

        image_dir = self._render_dir()
        report = check_visual(
            asset,
            spec,
            client=self.llm_client,
            blocking=self.visual_blocks,
            image_dir=image_dir,
        )
        if self.visual_feedback and image_dir is not None:
            self._pending_attachments = [
                (path.stem, path) for path in sorted(image_dir.glob("*.png"))
            ][:MAX_ATTACHED_RENDERS]
        return report

    def _render_dir(self) -> Path | None:
        if self.grounding_dir is None:
            return None
        # Renders go in a per-revision directory so a later check does not overwrite the
        # evidence for an earlier one; the transcript has to stay auditable.
        directory = self.grounding_dir / "renders" / f"rev-{self.grounding_state.revision:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _persist(self, tool: str, report: Report) -> None:
        if self.grounding_dir is None:
            return
        directory = self.grounding_dir / "checks"
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"rev-{self.grounding_state.revision:03d}-{_label(tool)}"
        report.write(directory / f"{stem}.json")
        (directory / f"{stem}.txt").write_text(report.to_text() + "\n")

    def _persist_all(self) -> None:
        """Rewrite the evidence after a downgrade, so what is on disk is what was decided."""
        for tool, report in self.grounding_state.reports.items():
            self._persist(tool, report)

    # ----------------------------------------------------------------- #
    # feeding renders back
    # ----------------------------------------------------------------- #

    def _maybe_inject_tool_attachments(
        self,
        conversation: list[dict],
        *,
        tool_calls: list[dict],
        tool_results: list[ToolResult],
    ) -> None:
        """Attach the renders as a user message, because a tool result cannot carry them.

        On chat-completions providers a `role: tool` message's content is a string. The
        images therefore have to arrive as a separate user turn, immediately after the
        tool result they belong to so the pairing is unambiguous in the transcript.
        """
        if not self._pending_attachments:
            return
        attachments, self._pending_attachments = self._pending_attachments, []

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "<grounding_renders>\n"
                    f"Views of your current model, in order: "
                    f"{', '.join(name for name, _ in attachments)}.\n"
                    "Look at them. A feature can be present in the geometry and still be "
                    "wrong in silhouette, proportion or placement, and that is not "
                    "something the numeric checks can tell you.\n"
                    "</grounding_renders>"
                ),
            }
        ]
        for name, path in attachments:
            try:
                encoded = base64.b64encode(path.read_bytes()).decode()
            except OSError:
                continue
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encoded}",
                    "detail": "high",
                    "name": name,
                }
            )
        if len(content) == 1:
            return

        message = {"role": "user", "content": content}
        conversation.append(message)
        if self.trace_writer:
            # The base64 payloads would make the trace unreadable and enormous; record
            # that the images were sent and where they are on disk instead.
            self.trace_writer.write_message(
                {
                    "role": "user",
                    "content": (
                        f"<grounding_renders>{len(attachments)} images attached: "
                        f"{', '.join(str(p) for _, p in attachments)}</grounding_renders>"
                    ),
                }
            )

    # ----------------------------------------------------------------- #
    # the finish gate
    # ----------------------------------------------------------------- #

    async def _handle_finish_attempt(
        self,
        conversation: list[dict],
        *,
        message: str,
        turn_count: int,
        tool_call_count: int,
        usage: dict[str, int],
        display_turn_override: int | None = None,
    ) -> Optional[Any]:
        if not self.grounding_enabled or not self.grounding_blocks:
            return await super()._handle_finish_attempt(
                conversation,
                message=message,
                turn_count=turn_count,
                tool_call_count=tool_call_count,
                usage=usage,
                display_turn_override=display_turn_override,
            )

        # A build has to come first. Measuring an asset that does not compile is not a
        # different failure, it is the same one seen twice.
        if not self._latest_code_is_fresh():
            self._append_compile_required_reminder(conversation)
            if display_turn_override is not None:
                self.display.current_turn = display_turn_override
            self.display.end_turn(success=True)
            return None

        required = self.required_checks()
        outstanding = self.grounding_state.stale_or_missing(required)
        if outstanding and not await self._verify_before_finishing(conversation, outstanding):
            if display_turn_override is not None:
                self.display.current_turn = display_turn_override
            self.display.end_turn(success=True)
            return None

        outstanding = self.grounding_state.stale_or_missing(required)
        failing = self.grounding_state.failing(required)
        if outstanding or failing:
            # Refusing over the same failure forever is how a budget is lost. Count the
            # repeats first: if this one has now been refused too often to be reachable,
            # `failing` is re-read and may come back empty, and the submission stands.
            if self.grounding_state.note_refusal(failing):
                self._persist_all()
                failing = self.grounding_state.failing(required)
            if outstanding or failing:
                self._append_grounding_required_reminder(conversation, outstanding, failing)
                if display_turn_override is not None:
                    self.display.current_turn = display_turn_override
                self.display.end_turn(success=True)
                return None

        if display_turn_override is not None:
            self.display.current_turn = display_turn_override
        self.display.end_turn(success=True)
        return await self._build_code_valid_result(
            message=message,
            conversation=conversation,
            turn_count=turn_count,
            tool_call_count=tool_call_count,
            usage=usage,
        )

    async def _verify_before_finishing(
        self, conversation: list[dict], outstanding: list[str]
    ) -> bool:
        """Measure the revision being submitted, rather than asking the model to.

        Refusing the finish attempt and telling the model to go call the checks costs a
        turn per check and is where the turn budget went: a hundred turns of edit-compile
        followed by a submission of whatever happened to be on disk, never measured. The
        checks are the harness's to run, so it runs them here. The model still gets every
        failure, and still has to be the one to fix them.

        Returns False when the answer has already been appended to the conversation and
        the finish attempt must be refused now — the model cannot fix a build failure it
        has not been told about.
        """
        import asyncio  # noqa: PLC0415

        for tool in outstanding:
            try:
                report = await asyncio.to_thread(self._run_check, tool)
            except GroundingBuildError as error:
                self._append_verification_blocked_reminder(conversation, tool, str(error))
                return False
            except Exception as error:  # noqa: BLE001 - a crash here must not end the run
                logger.exception("pre-finish grounding check %s failed", tool)
                self._append_verification_blocked_reminder(
                    conversation, tool, f"{type(error).__name__}: {error}"
                )
                return False
            self.grounding_state.record(tool, report)
            self._persist(tool, report)

        # Renders taken during the checks above, if any, still have to reach the model.
        self._maybe_inject_tool_attachments(conversation, tool_calls=[], tool_results=[])
        return True

    def _append_verification_blocked_reminder(
        self, conversation: list[dict], tool: str, detail: str
    ) -> None:
        message = {
            "role": "user",
            "content": (
                "<grounding_required>\n"
                f"Your submission was held back: {_label(tool)} grounding could not be run "
                f"against your current code — {detail}\n"
                "Nothing can be measured until the model builds and exports. Fix that with "
                f"`compile_model`, then call `{tool}` yourself to see the result.\n"
                "</grounding_required>"
            ),
        }
        conversation.append(message)
        if self.trace_writer:
            self.trace_writer.write_message(message)

    def _append_grounding_required_reminder(
        self,
        conversation: list[dict],
        outstanding: list[str],
        failing: list[Report],
    ) -> None:
        lines = [
            "<grounding_required>",
            "The code compiles, but this task is not finished until the object also "
            "matches the specification it was built from. Your submission was measured "
            "and held back.",
        ]
        if outstanding:
            lines.append("")
            lines.append("Not yet verified against your current code:")
            lines.extend(f"- call `{tool}`" for tool in outstanding)
        # Ordered the way the failures are worth working through: what the object
        # measures, then what it does when driven, then what it looks like. Fixing
        # geometry usually changes the motion, so doing it the other way round is wasted
        # work.
        for report in failing:
            # Only what is still worth acting on. Repeating a failure already ruled
            # unreachable would ask for the edit the ruling exists to stop.
            blocking = self.grounding_state.blocking_failures(report)
            lines.append("")
            lines.append(f"Still failing — {report.kind}:")
            lines.extend(f"- {finding.summary}" for finding in blocking[:8])
            remaining = len(blocking) - 8
            if remaining > 0:
                lines.append(f"- ... and {remaining} more of the same kind")
        lines.append("")
        lines.append(
            "Fix these in your model, compile, and finish again. Do not disable a check, "
            "widen a tolerance or declare an overlap intentional to get past it."
        )
        lines.append("</grounding_required>")

        message = {"role": "user", "content": "\n".join(lines)}
        conversation.append(message)
        if self.trace_writer:
            self.trace_writer.write_message(message)


def _label(tool: str) -> str:
    return {
        PHYSICAL_TOOL: "physical",
        PROTOCOL_TOOL: "protocol",
        VISUAL_TOOL: "visual",
    }.get(tool, tool)


def _key(finding: Finding) -> str:
    """What makes two refusals the same refusal.

    Code and subject, not the summary: the summary carries the measured value, which moves
    a little on every edit, so keying on it would read a stuck run as a changing one.
    """
    return f"{finding.code}:{finding.subject}"
