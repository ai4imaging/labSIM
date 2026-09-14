"""Reading `input.md` back into the structure it was generated from.

Each case ships two files. `input.md` is the task — what the asset has to be — and
`rubric.json` is the answer key. They overlap heavily, because the rubric embeds a copy of
every input target it scores against, and it would be less code to read the targets out of
the rubric and skip this module entirely.

That would also quietly break the benchmark. The generator is only ever shown `input.md`,
and the grounding checks it runs on itself have to be derived from the same thing, or the
loop is being held to thresholds the task never stated and a good score stops meaning the
asset is good. So the parse happens here, from the file the model saw, and `rubric.json`
is opened only by the judge.

The markdown was generated from a nested structure and kept its shape, which is what makes
this tractable: indentation is nesting, `` `key`: value `` is a field, `**Item N — `ID`**`
is a list element, and `*(none)*` is an empty list.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^(?P<indent>\s*)-\s+(?P<body>.*)$")
_FIELD = re.compile(r"^`(?P<key>[^`]+)`:\s*(?P<value>.*)$")
_ITEM = re.compile(r"^\*\*Item\s+\d+\s*(?:[—-]\s*)?(?P<label>.*?)\*\*$")
_NONE = "*(none)*"
_NULL = re.compile(r"^`null`")
_TICKED = re.compile(r"^`(?P<inner>[^`]*)`$")


def parse_spec(path: Path) -> dict[str, Any]:
    """`input.md` as nested dicts and lists, keyed by section heading.

    Section headings become snake_case keys, so `## Required Components` is
    `required_components`. Everything below a heading is parsed as one bullet tree.
    """
    text = Path(path).read_text()
    sections: dict[str, Any] = {}
    current: str | None = None
    buffer: list[str] = []

    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading and len(heading.group(1)) == 2:
            if current is not None:
                sections[current] = _parse_bullets(buffer)
            current = _slug(heading.group(2))
            buffer = []
            continue
        if current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = _parse_bullets(buffer)
    return sections


def _parse_bullets(lines: list[str]) -> Any:
    """One indentation-nested bullet tree.

    Returns a dict when the level is made of `key: value` fields and a list when it is
    made of items or bare scalars. A level that is both is treated as a dict, with the
    stray scalars dropped — that only happens in prose asides, which carry no data.
    """
    entries: list[tuple[int, str]] = []
    for line in lines:
        match = _BULLET.match(line)
        if match and match.group("body").strip():
            entries.append((len(match.group("indent")), match.group("body").strip()))
    return _build(entries, 0, len(entries), _indent_of(entries))[0]


def _indent_of(entries: list[tuple[int, str]]) -> int:
    return min((indent for indent, _ in entries), default=0)


def _build(
    entries: list[tuple[int, str]], start: int, end: int, indent: int
) -> tuple[Any, int]:
    fields: dict[str, Any] = {}
    items: list[Any] = []
    index = start

    while index < end:
        level, body = entries[index]
        if level < indent:
            break
        if level > indent:  # Orphaned deeper line; the parent already consumed its block.
            index += 1
            continue

        child_end = index + 1
        while child_end < end and entries[child_end][0] > indent:
            child_end += 1
        has_children = child_end > index + 1

        item = _ITEM.match(body)
        field = _FIELD.match(body)

        if item is not None:
            value, _ = (
                _build(entries, index + 1, child_end, entries[index + 1][0])
                if has_children
                else ({}, child_end)
            )
            if isinstance(value, dict):
                value.setdefault("label", _scalar(item.group("label")))
            items.append(value)
        elif field is not None:
            key = field.group("key")
            raw = field.group("value").strip()
            if has_children:
                nested, _ = _build(entries, index + 1, child_end, entries[index + 1][0])
                fields[key] = nested
            else:
                fields[key] = _scalar(raw)
        elif body == _NONE:
            pass  # An explicit empty list. Nothing to add.
        else:
            items.append(_scalar(body))

        index = child_end

    if fields:
        return fields, index
    return items, index


def _scalar(raw: str) -> Any:
    """One value, with the markdown's own conventions honoured.

    Backticks mark a literal, `` `null` `` is always followed by the same parenthetical
    gloss and means absent, and a bare string is a string. Numbers are converted because
    every threshold in the file arrives this way and a millimetre stored as `"70"` is a
    bug waiting to happen downstream.
    """
    text = raw.strip()
    if not text or text == _NONE:
        return None
    if _NULL.match(text):
        return None
    ticked = _TICKED.match(text)
    if ticked:
        text = ticked.group("inner")
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", heading.strip().lower()).strip("_")
