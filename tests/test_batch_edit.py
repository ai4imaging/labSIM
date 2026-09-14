"""One turn, many edits.

Turn accounting on `newframework100-opus5` (`scripts/profile_turns.py`) put 44.5% of a
hundred-turn budget inside `replace`, carrying 1.03 edits per turn. The model was not
being wasteful: stock `replace` takes one `old_string`/`new_string` pair, so six changes
cost six turns whether or not it had decided on all six at once.

These cover the batch form's contract — all-or-nothing application, and the wording the
harness relies on — plus the seam in the vendored tool that exposes it.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from amx.paths import activate_articraft

CONTRACT = """from __future__ import annotations

RADIUS = 10.0
HEIGHT = 20.0
WIDTH = 30.0


def build_object_model():
    return "draft_model"


def run_tests():
    return None


object_model = build_object_model()
"""


@pytest.fixture(scope="module")
def articraft():
    return activate_articraft()


@pytest.fixture
def model_py(tmp_path: Path) -> Path:
    path = tmp_path / "model.py"
    path.write_text(CONTRACT, encoding="utf-8")
    return path


def _replace(path: Path, params: dict[str, object]):
    from agent.tools.edit_code import ReplaceTool

    async def go():
        tool = ReplaceTool()
        invocation = await tool.build(params)
        invocation.bind_file_path(str(path))
        return await invocation.execute()

    return asyncio.run(go())


def test_one_call_applies_every_edit(articraft, model_py):
    result = _replace(
        model_py,
        {
            "edits": [
                {"old_string": "RADIUS = 10.0", "new_string": "RADIUS = 12.5"},
                {"old_string": "HEIGHT = 20.0", "new_string": "HEIGHT = 22.0"},
                {"old_string": "WIDTH = 30.0", "new_string": "WIDTH = 33.0"},
            ]
        },
    )

    assert result.error is None
    assert result.compilation == {"status": "success", "error": None}
    updated = model_py.read_text(encoding="utf-8")
    assert "RADIUS = 12.5" in updated
    assert "HEIGHT = 22.0" in updated
    assert "WIDTH = 33.0" in updated


def test_a_batch_that_cannot_be_applied_whole_is_not_applied_at_all(articraft, model_py):
    """A half-applied batch leaves the model unable to say what is on disk."""
    result = _replace(
        model_py,
        {
            "edits": [
                {"old_string": "RADIUS = 10.0", "new_string": "RADIUS = 12.5"},
                {"old_string": "NOT_IN_THE_FILE = 1", "new_string": "x = 1"},
            ]
        },
    )

    assert result.error is not None
    assert "edits[2]" in result.error
    assert model_py.read_text(encoding="utf-8") == CONTRACT


def test_a_failed_batch_still_says_what_the_retry_guidance_looks_for(articraft, model_py):
    """`maybe_inject_edit_code_guidance` matches on this phrase to nudge a reread.

    Prefixing the entry number in front of it would be enough to break the match, and the
    model would lose the "read the file again, pick a smaller snippet" hint exactly when a
    batch has made it least sure what the file holds.
    """
    from agent.harness_guidance import GuidanceInjector

    result = _replace(
        model_py,
        {"edits": [{"old_string": "ABSENT", "new_string": "x"}, {"old_string": "y", "new_string": "z"}]},
    )

    phrase = "Could not find the old_string in the code"
    assert phrase in result.error
    assert phrase in inspect.getsource(GuidanceInjector.maybe_inject_edit_code_guidance)


def test_a_later_edit_may_touch_what_an_earlier_one_wrote(articraft, model_py):
    result = _replace(
        model_py,
        {
            "edits": [
                {"old_string": "RADIUS = 10.0", "new_string": "RADIUS = 11.0"},
                {"old_string": "RADIUS = 11.0", "new_string": "RADIUS = 12.0"},
            ]
        },
    )

    assert result.error is None
    assert "RADIUS = 12.0" in model_py.read_text(encoding="utf-8")


def test_each_entry_carries_its_own_uniqueness_rule(articraft, tmp_path):
    path = tmp_path / "model.py"
    path.write_text(CONTRACT.replace("WIDTH = 30.0", "WIDTH = 30.0\nDEPTH = 30.0"), encoding="utf-8")

    ambiguous = _replace(path, {"edits": [{"old_string": "30.0", "new_string": "31.0"}]})
    assert "appears 2 times" in ambiguous.error

    allowed = _replace(
        path, {"edits": [{"old_string": "30.0", "new_string": "31.0", "allow_multiple": True}]}
    )
    assert allowed.error is None
    assert path.read_text(encoding="utf-8").count("31.0") == 2


def test_a_batch_may_not_break_the_model_contract(articraft, model_py):
    result = _replace(
        model_py,
        {
            "edits": [
                {"old_string": "RADIUS = 10.0", "new_string": "RADIUS = 12.5"},
                {"old_string": "\n\nobject_model = build_object_model()", "new_string": ""},
            ]
        },
    )

    assert "object_model = build_object_model()" in result.error
    assert model_py.read_text(encoding="utf-8") == CONTRACT


def test_the_single_edit_form_still_works(articraft, model_py):
    result = _replace(model_py, {"old_string": "RADIUS = 10.0", "new_string": "RADIUS = 1.0"})

    assert result.error is None
    assert result.compilation == {"status": "success", "error": None}
    assert "RADIUS = 1.0" in model_py.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {"old_string": "a", "new_string": "b", "edits": [{"old_string": "c", "new_string": "d"}]},
            id="both-forms",
        ),
        pytest.param({"instruction": "no edit at all"}, id="neither-form"),
        pytest.param({"old_string": "RADIUS = 10.0"}, id="half-a-single-edit"),
    ],
)
def test_an_unreadable_call_is_answered_rather_than_raised(articraft, model_py, params):
    """The harness renders a ValidationError by indexing each error's `loc`.

    A pydantic model-level validator reports an empty `loc`, so raising there would fail
    inside the harness's own error handler and take the turn down with it. Refusing in
    `execute` keeps it a message the model can act on.
    """
    result = _replace(model_py, params)

    assert result.error is not None
    assert model_py.read_text(encoding="utf-8") == CONTRACT


def test_an_edit_says_it_has_not_built_the_geometry(articraft, model_py):
    """`compilation: success` on an edit is a syntax check, and reads like a build."""
    result = _replace(model_py, {"old_string": "RADIUS = 10.0", "new_string": "RADIUS = 1.0"})

    assert "compile_model" in result.output


def test_the_tool_offers_batching_to_the_model(articraft):
    """The capability is worth nothing if the schema does not advertise it."""
    from agent.tools.edit_code import ReplaceTool

    schema = ReplaceTool().schema["function"]
    parameters = schema["parameters"]

    assert "edits" in parameters["properties"]
    # `old_string` was required, which made the batch form unreachable.
    assert parameters["required"] == []
    assert "Batching:" in schema["description"]
    assert "same turn" in schema["description"]


def test_the_seam_keeps_the_batching_logic_out_of_the_vendored_tree(articraft):
    """`vendor/PROVENANCE.md`: the official tree only ever gains a seam."""
    from agent.tools import edit_code

    assert edit_code.apply_edits.__module__ == "agent.batch_edit"
    assert edit_code.resolve_edits.__module__ == "agent.batch_edit"


def test_the_first_turn_guidance_says_a_turn_is_the_budget(articraft):
    """Without this the model is told to "make small focused edits" and nothing else.

    That is good advice about the size of an edit which reads as advice about how many to
    send, and until now the tool could only take one anyway.
    """
    from agent.tools import build_first_turn_runtime_guidance

    guidance = build_first_turn_runtime_guidance("gpugeek")
    assert "A turn is the budget" in guidance
    assert "one `replace` call with `edits`" in guidance
    assert "probe_model" in guidance


def test_probe_model_asks_for_every_measurement_at_once(articraft):
    from agent.tools.probe_model.description import PROBE_MODEL_DESCRIPTION

    assert "answer every question you currently have in one call" in PROBE_MODEL_DESCRIPTION


def test_the_docs_every_run_opens_are_preloaded(articraft):
    """190 turns across 58 cases were spent fetching the same four files.

    Preloading is close to free even in context terms: a doc read on turn five is in the
    conversation for every turn after it, so this front-loads a cost the run was going to
    pay, and only a run that never opens one of them pays anything.
    """
    from agent.workspace_docs import load_sdk_docs_bundle

    from amx.paths import ARTICRAFT_DIR

    preloaded = set(
        load_sdk_docs_bundle(ARTICRAFT_DIR, sdk_package="sdk").default_read_virtual_paths()
    )

    assert {
        "docs/sdk/references/core-types.md",
        "docs/sdk/references/articulated-object.md",
        "docs/sdk/references/geometry/mesh-geometry.md",
        "docs/sdk/references/cadquery/overview.md",
    } <= preloaded
