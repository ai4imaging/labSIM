"""Lessons carried from one run to the next.

A run that fails a check, works out why, and fixes it has produced something more
valuable than the asset: it has produced a rule. Without somewhere to put that rule, the
next run on a similar object discovers the same thing from scratch, and the system never
gets better at anything — it only ever gets a particular beaker right.

A lesson is stored as an Articraft example document, frontmatter and all, in a directory
that `ARTICRAFT_EXTRA_EXAMPLE_DIRS` adds to the example corpus. That is deliberate: the
model already has a retrieval tool it knows how to use, already reaches for it when it is
unsure, and already gets the results in a form it reads. A separate memory store would
need its own tool, its own place in the prompt, and would compete with `find_examples`
for the model's attention rather than adding to it.

The cost of that choice is that lessons and curated examples share a ranking. Lessons are
therefore titled and tagged around the *symptom* — "cavity measured as absent",
"diameter reads larger than modelled" — because a symptom is what the model will be
holding when it goes looking.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from amx.paths import PROJECT_ROOT

SKILLS_DIR = PROJECT_ROOT / "skills"
INDEX_NAME = "lessons.jsonl"


class Lesson(BaseModel):
    """One transferable thing learned from a run."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    title: str
    symptom: str
    """What the check reported, in the words it reported it. This is the retrieval key."""
    cause: str
    remedy: str
    codes: list[str] = Field(default_factory=list)
    """Finding codes this lesson bears on, which is what makes retrieval targeted."""
    asset_classes: list[str] = Field(default_factory=list)
    snippet: str = ""
    """A minimal illustration of the fix, if one can be written short enough to be read."""
    source_case: str = ""
    confirmed: int = 1
    """How many separate runs this lesson has held up in. Repeated confirmation is the
    only evidence available that a lesson generalises rather than describing one asset."""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def tags(self) -> list[str]:
        """Retrieval tags. Codes and classes first: those are the discriminating terms."""
        seen: dict[str, None] = {}
        for tag in [
            "lesson",
            "grounding",
            *(code.lower() for code in self.codes),
            *(cls.lower() for cls in self.asset_classes),
            *_keywords(self.symptom),
        ]:
            seen.setdefault(tag, None)
        return list(seen)

    def to_document(self) -> str:
        """Render as an Articraft example document."""
        tag_lines = "\n".join(f"  - {tag}" for tag in self.tags())
        parts = [
            "---",
            f"title: {json.dumps(self.title)}",
            f"description: {json.dumps(self.symptom)}",
            "tags:",
            tag_lines,
            "---",
            f"# {self.title}",
            "",
            "> A lesson recorded from an earlier run, not a curated example. It describes a "
            "mistake that was made and corrected; use it to avoid repeating it.",
            "",
            "## What the check reported",
            "",
            self.symptom,
            "",
            "## Why it happened",
            "",
            self.cause,
            "",
            "## What to do instead",
            "",
            self.remedy,
        ]
        if self.snippet.strip():
            parts += ["", "```python", self.snippet.strip(), "```"]
        parts += [
            "",
            f"_Confirmed in {self.confirmed} run(s). Relevant checks: "
            f"{', '.join(self.codes) or 'unspecified'}._",
            "",
        ]
        return "\n".join(parts)


class SkillLibrary:
    """A directory of lessons, retrievable through Articraft's example search."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or SKILLS_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_NAME

    def all(self) -> list[Lesson]:
        if not self.index_path.is_file():
            return []
        lessons = []
        for line in self.index_path.read_text().splitlines():
            if line.strip():
                lessons.append(Lesson.model_validate_json(line))
        return lessons

    def add(self, lesson: Lesson) -> Lesson:
        """Store a lesson, merging it into an existing one that says the same thing.

        Merging rather than appending matters: the same mistake recurs across cases, and
        a library with eleven near-identical entries for it retrieves worse than one with
        a single entry confirmed eleven times.
        """
        existing = {item.id: item for item in self.all()}
        if lesson.id in existing:
            previous = existing[lesson.id]
            lesson = lesson.model_copy(
                update={
                    "confirmed": previous.confirmed + 1,
                    "created_at": previous.created_at,
                    "codes": sorted(set(previous.codes) | set(lesson.codes)),
                    "asset_classes": sorted(
                        set(previous.asset_classes) | set(lesson.asset_classes)
                    ),
                }
            )
        existing[lesson.id] = lesson
        self._rewrite(list(existing.values()))
        return lesson

    def _rewrite(self, lessons: list[Lesson]) -> None:
        for stale in self.root.glob("*.md"):
            stale.unlink()
        ordered = sorted(lessons, key=lambda item: (-item.confirmed, item.id))
        for lesson in ordered:
            (self.root / f"{lesson.id}.md").write_text(lesson.to_document())
        self.index_path.write_text(
            "".join(lesson.model_dump_json() + "\n" for lesson in ordered)
        )
        refresh_retrieval()

    def activate(self) -> None:
        """Make this library visible to Articraft's `find_examples`.

        Appends rather than replaces, so a caller that has already pointed Articraft at
        another corpus keeps it.
        """
        current = [
            part
            for part in os.environ.get("ARTICRAFT_EXTRA_EXAMPLE_DIRS", "").split(os.pathsep)
            if part.strip()
        ]
        text = str(self.root)
        if text not in current:
            current.append(text)
        os.environ["ARTICRAFT_EXTRA_EXAMPLE_DIRS"] = os.pathsep.join(current)
        refresh_retrieval()

    def relevant(self, *, codes: list[str], asset_class: str = "", limit: int = 5) -> list[Lesson]:
        """Lessons bearing on these finding codes, most-confirmed first.

        Used to put lessons in front of the model at the start of a run, before it has
        any reason to search for them. Matching is on codes rather than on text: at that
        point what is known is which checks a similar task tends to fail, and that is a
        sharper key than any phrasing of the task.
        """
        wanted = {code.upper() for code in codes}
        scored = []
        for lesson in self.all():
            overlap = len(wanted & {code.upper() for code in lesson.codes})
            if not overlap:
                continue
            same_class = asset_class and asset_class.lower() in [
                item.lower() for item in lesson.asset_classes
            ]
            scored.append((overlap, bool(same_class), lesson.confirmed, lesson))
        scored.sort(key=lambda row: row[:3], reverse=True)
        return [row[3] for row in scored[:limit]]

    def briefing(self, lessons: list[Lesson]) -> str:
        """Lessons rendered for injection into a prompt."""
        if not lessons:
            return ""
        parts = [
            "<lessons_from_earlier_runs>",
            "Mistakes made on similar objects before, and what fixed them. These are "
            "observations from real runs, not rules handed down; if one does not apply "
            "here, ignore it.",
            "",
        ]
        for lesson in lessons:
            parts.append(f"- **{lesson.title}**")
            parts.append(f"  Symptom: {lesson.symptom}")
            parts.append(f"  Fix: {lesson.remedy}")
        parts.append("</lessons_from_earlier_runs>")
        return "\n".join(parts)


def refresh_retrieval() -> None:
    """Drop Articraft's example caches so a newly written lesson is retrievable now.

    `load_example_documents` and `load_example_search_index` are keyed on the SDK package
    alone, so neither notices that a corpus directory gained a file.
    """
    try:
        from amx.paths import activate_articraft  # noqa: PLC0415

        activate_articraft()
        from agent.examples import (  # noqa: PLC0415
            load_example_documents,
            load_example_search_index,
        )
    except Exception:  # noqa: BLE001 - without Articraft there is nothing to refresh
        return
    load_example_documents.cache_clear()
    load_example_search_index.cache_clear()


def _keywords(text: str, limit: int = 8) -> list[str]:
    """Distinctive words from a symptom, for tagging."""
    stop = {
        "the", "a", "an", "is", "was", "were", "and", "or", "but", "of", "to", "in", "on",
        "at", "by", "for", "with", "that", "this", "it", "its", "be", "been", "has", "have",
        "not", "no", "than", "from", "as", "which", "into", "out", "up", "down", "so",
    }
    words = re.findall(r"[a-z]{4,}", text.lower())
    seen: dict[str, None] = {}
    for word in words:
        if word not in stop:
            seen.setdefault(word, None)
    return list(seen)[:limit]
