"""Reading a finished search and writing down what it learned.

The evidence is in the tree already. Somewhere along the path from the root to the best
node, a finding that was failing stopped failing, and the edit that made that happen is
sitting right there as a diff between two nodes. That pair — the complaint and the change
that answered it — is a lesson in raw form.

Turning it into a *transferable* lesson is the part a model is needed for, and it is a
narrow job: read a diff and a finding, and say in two sentences what the underlying
mistake was. It is not asked to invent advice, speculate about cases it has not seen, or
decide what is worth remembering. Those are the failure modes that fill a memory with
confident nonsense, so the code decides what is worth remembering and the model only
describes it.

If no model is available the lesson is still recorded, using the finding's own text. It
is a worse lesson and it is still better than nothing.
"""

from __future__ import annotations

import difflib
import logging
import re

from pydantic import BaseModel, ConfigDict, Field

from amx.llm import LlmClient, LlmUnavailable
from amx.report import Finding
from amx.search.tree import Node, SearchTree
from amx.skills.library import Lesson, SkillLibrary

logger = logging.getLogger(__name__)

MAX_DIFF_LINES = 120
"""Diff lines shown to the summariser. A repair that rewrote the whole file is not a
lesson about one mistake, and truncating it keeps the request cheap."""


class LessonDraft(BaseModel):
    """What the summariser is asked for. Deliberately small."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Six to ten words, naming the mistake, not the object.")
    cause: str = Field(description="One or two sentences on why the mistake produces this symptom.")
    remedy: str = Field(description="One or two sentences of what to do instead. Imperative.")
    generalises: bool = Field(
        description=(
            "False if this only applies to this one object — a value specific to this "
            "datasheet, say — and would mislead on anything else."
        )
    )


def learn_from_search(
    tree: SearchTree,
    *,
    library: SkillLibrary,
    asset_class: str = "",
    case_id: str = "",
    client: LlmClient | None = None,
) -> list[Lesson]:
    """Record a lesson for each failure the search actually managed to repair."""
    target = tree.solved() or tree.best()
    if target is None:
        return []

    learned: list[Lesson] = []
    for parent, child in zip(
        tree.lineage(target), tree.lineage(target)[1:], strict=False
    ):
        for finding in _repaired_between(parent, child):
            lesson = _draft(
                finding,
                parent=parent,
                child=child,
                asset_class=asset_class,
                case_id=case_id,
                client=client,
            )
            if lesson is not None:
                learned.append(library.add(lesson))
    return learned


def _repaired_between(parent: Node, child: Node) -> list[Finding]:
    """Failures present in the parent and gone from the child.

    Matched on code and subject together. Code alone would call a diameter fixed when a
    different dimension broke in its place, which is precisely the mistake worth learning
    about and precisely the one this would hide.
    """
    if parent.report is None or child.report is None:
        return []
    still_failing = {(f.code, f.subject) for f in child.report.failures}
    return [
        finding
        for finding in parent.report.failures
        if (finding.code, finding.subject) not in still_failing
    ]


def _draft(
    finding: Finding,
    *,
    parent: Node,
    child: Node,
    asset_class: str,
    case_id: str,
    client: LlmClient | None,
) -> Lesson | None:
    diff = _diff(parent.source, child.source)
    if not diff.strip():
        # The finding cleared without the source changing. That is measurement noise or
        # a check that depends on something other than the model, and neither is a lesson.
        return None

    draft = _summarise(finding, diff, asset_class=asset_class, client=client)
    if draft is not None and not draft.generalises:
        return None

    title = draft.title if draft else _fallback_title(finding)
    return Lesson(
        id=_identifier(finding, title),
        title=title,
        symptom=finding.summary,
        cause=draft.cause if draft else "Not analysed: no summariser was available.",
        remedy=(
            draft.remedy
            if draft
            else "See the diff recorded with this run for the change that resolved it."
        ),
        codes=[finding.code],
        asset_classes=[asset_class] if asset_class else [],
        snippet=_snippet(diff),
        source_case=case_id,
    )


def _summarise(
    finding: Finding,
    diff: str,
    *,
    asset_class: str,
    client: LlmClient | None,
) -> LessonDraft | None:
    client = client or LlmClient()
    try:
        return client.structured(
            purpose="skills.distil",
            system=(
                "You read one automated check failure and the code change that resolved "
                "it, and you write down the underlying mistake so it is not repeated.\n\n"
                "Describe only what the diff shows. Do not speculate about causes it does "
                "not support, and do not restate the check's message back as the cause.\n\n"
                "Set generalises=false when the change is specific to this object — a "
                "particular dimension from its datasheet, a name only it uses — so that "
                "recording it would mislead on anything else."
            ),
            user=(
                f"Object class: {asset_class or 'unspecified'}\n"
                f"Check that was failing: [{finding.code}] {finding.summary}\n"
                f"Measurements: {finding.metrics}\n"
                f"Limits: {finding.thresholds}\n\n"
                f"The change that resolved it:\n```diff\n{diff}\n```"
            ),
            schema=LessonDraft,
        )
    except (LlmUnavailable, Exception) as error:  # noqa: BLE001 - a lesson is worth having anyway
        logger.info("lesson summarisation unavailable (%s); recording the raw finding", error)
        return None


def _diff(before: str, after: str) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=2,
        )
    )
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"... {len(lines) - MAX_DIFF_LINES} more lines"]
    return "\n".join(lines)


def _snippet(diff: str) -> str:
    """The added lines, which are what the fix actually was."""
    added = [line[1:] for line in diff.splitlines() if line.startswith("+") and line[1:3] != "++"]
    return "\n".join(added[:12])


def _fallback_title(finding: Finding) -> str:
    return f"{finding.code}: {finding.summary[:60]}"


def _identifier(finding: Finding, title: str) -> str:
    """A stable id, so the same lesson learned twice merges instead of duplicating."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:48] or "lesson"
    code = re.sub(r"[^a-z0-9]+", "_", finding.code.lower()).strip("_")
    identifier = f"{code}_{slug}"
    return identifier if identifier[0].isalnum() else f"l{identifier}"
