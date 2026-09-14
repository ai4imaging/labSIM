"""Letting one `replace` call carry every edit the model has already decided on.

Stock `replace` takes one `old_string`/`new_string` pair, so a model with six changes in
mind has to spend six turns, and a hundred-turn budget goes almost half on editing:
`newframework100-opus5` spent 44.5% of its turns in `replace`, carrying 1.03 edits each.
Nothing was wrong with the model's plan — the tool had no way to accept it.

The logic lives here rather than in `vendor/articraft/agent/tools/edit_code.py` because
that tree is only allowed to gain seams (see `vendor/PROVENANCE.md`). The vendored tool
adds one field, one schema fragment and two calls; everything that decides anything is
below, so a re-vendoring is a matter of re-cutting the seam rather than rewriting this.
"""

from __future__ import annotations

from agent.tools.base import ToolParamsModel

SYNTAX_ONLY_NOTE = (
    "Python syntax parses; the geometry is not built. Run `compile_model` for the real "
    "build and QC."
)
"""What an edit tool has and has not established.

The `compilation` field on an edit's result is a `compile(source, "exec")` syntax check,
not a geometry build, and a model reading `"compilation": {"status": "success"}` has no
way to tell the difference. Saying so is cheaper than letting it find out by submitting.
"""

BATCHING_GUIDANCE = (
    "Batching:\n"
    "- Pass `edits` with every change you have already decided on. One call carrying six "
    "edits costs one turn; six calls cost six.\n"
    "- Use `old_string`/`new_string` only for a genuinely single change.\n"
    "- The batch is all-or-nothing: if one entry does not match, none are written and the "
    "reply names the entry that failed.\n"
)
"""The part of the tool description that asks for batches."""

COMPILE_SEPARATELY_NOTE = (
    "This tool only checks that the result parses. It does not build the geometry, so "
    "follow a finished round of edits with `compile_model` — you may call both in the "
    "same turn, and they run in the order you list them."
)
"""Same-turn ordering is guaranteed for every provider we run.

`_tool_calls_are_parallelizable` only returns True for Gemini, so on the GpuGeek gateway
a turn's calls are executed one after another in the order the model listed them. An edit
followed by `compile_model` in one turn therefore compiles the edited file, and saying so
turns two turns into one.
"""

TURN_BUDGET_GUIDANCE = (
    "- A turn is the budget, not an edit. Group every change you have already decided on "
    "into one `replace` call with `edits`, and put `compile_model` in the same turn as the "
    "edits it should check — calls run in the order you list them.\n"
    "- Probe in groups too: one `probe_model` snippet can report every measurement you are "
    "about to check.\n"
)
"""Turn-thrift, stated where the model reads its instructions rather than its tools.

"Small focused edits" is good advice about edit *size* that reads as advice about edit
*count*, and the tool it applied to could only take one at a time. Both are fixed now, so
the guidance says which resource is scarce.
"""

EDITS_PARAMETER = {
    "type": "array",
    "description": (
        "Several edits to apply in one turn, in order. Preferred whenever you have more "
        "than one change in mind."
    ),
    "items": {
        "type": "object",
        "properties": {
            "old_string": {
                "type": "string",
                "description": "Exact literal text to find.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text.",
            },
            "allow_multiple": {
                "type": "boolean",
                "description": (
                    "If false (default), this entry's `old_string` must be unique."
                ),
            },
        },
        "required": ["old_string", "new_string"],
        "additionalProperties": False,
    },
}
"""JSON Schema for the `edits` parameter."""


class ReplaceEdit(ToolParamsModel):
    """One find-and-replace inside a batch."""

    old_string: str
    new_string: str
    allow_multiple: bool = False


def resolve_edits(
    *,
    old_string: str | None,
    new_string: str | None,
    edits: list[ReplaceEdit] | None,
    allow_multiple: bool,
) -> tuple[list[ReplaceEdit], str | None]:
    """Read a call as either the single form or the batch form.

    Returns the edits to apply, or the reason the call is neither. This is checked here
    instead of by a pydantic validator on the params model because the harness renders a
    `ValidationError` by indexing each error's `loc`, and a model-level validator produces
    an empty `loc` — raising would crash the turn rather than tell the model what to fix.
    """
    single = old_string is not None or new_string is not None
    if single and edits:
        return [], "pass old_string/new_string for one edit or edits for several, not both."
    if edits:
        return list(edits), None
    if not single:
        return [], "provide old_string and new_string, or an edits array with at least one entry."
    if old_string is None or new_string is None:
        return [], "a single edit needs both old_string and new_string."
    return [
        ReplaceEdit(
            old_string=old_string,
            new_string=new_string,
            allow_multiple=allow_multiple,
        )
    ], None


def apply_edits(full_code: str, edits: list[ReplaceEdit]) -> tuple[str, str | None]:
    """Apply every edit in order, or none of them.

    Returns the new text, or the original text and the reason the batch was refused. A
    half-applied batch is worse than a refused one: the model would have to work out which
    of its edits landed before it could write another, and the file on disk would match
    neither what it had nor what it asked for.

    Each entry is matched against the result of the ones before it, so a batch may edit a
    passage an earlier entry introduced.
    """
    updated = full_code
    for index, edit in enumerate(edits, start=1):
        where = f"edits[{index}]: " if len(edits) > 1 else ""

        if not edit.old_string:
            # An empty `old_string` means "this file is blank, seed it", which can only be
            # true while nothing has been written yet.
            if updated.strip():
                return full_code, (
                    f"{where}old_string cannot be empty unless model.py is empty. "
                    "Please provide the exact string to replace."
                )
            updated = edit.new_string
            continue

        occurrences = updated.count(edit.old_string)
        if occurrences == 0:
            # Worded to keep the phrase `maybe_inject_edit_code_guidance` looks for, so a
            # failed batch still earns the "reread the file, pick a smaller snippet" nudge.
            return full_code, (
                f"{where}Could not find the old_string in the code. "
                "Make sure the string matches exactly, including whitespace and indentation. "
            )
        if occurrences > 1 and not edit.allow_multiple:
            return full_code, (
                f"{where}The old_string appears {occurrences} times in the code. "
                "Please provide a longer, unique string that appears only once, "
                "or use allow_multiple=true to replace all occurrences."
            )
        updated = updated.replace(
            edit.old_string, edit.new_string, -1 if edit.allow_multiple else 1
        )

    return updated, None
