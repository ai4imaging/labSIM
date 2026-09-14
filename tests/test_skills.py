"""Skill memory: what gets written down, and whether Articraft can then find it.

The load-bearing claim is the second one. Storing lessons is easy; storing them somewhere
the model will actually encounter them is the whole design, and it depends on a vendored
seam (`ARTICRAFT_EXTRA_EXAMPLE_DIRS`) and on clearing two caches that key on nothing
that changed. Both are the kind of thing that breaks silently.
"""

from __future__ import annotations

import pytest

from amx.report import Finding, RepairTarget, Report, Severity
from amx.search.tree import SearchTree
from amx.skills import Lesson, SkillLibrary, learn_from_search


@pytest.fixture
def library(tmp_path) -> SkillLibrary:
    return SkillLibrary(tmp_path / "skills")


def _lesson(**overrides) -> Lesson:
    base = {
        "id": "g_cavity_missing_model_the_interior_as_a_subtraction",
        "title": "Model the interior as a subtraction, not a second surface",
        "symptom": "no enclosed cavity was found: no cross-section comes out as a ring",
        "cause": "An inner shell was added beside the outer one instead of being cut from it.",
        "remedy": "Cut the bore out of the body with a boolean difference.",
        "codes": ["G-CAVITY-MISSING"],
        "asset_classes": ["beaker"],
    }
    base.update(overrides)
    return Lesson(**base)


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #


def test_a_lesson_is_written_as_a_readable_example_document(library):
    library.add(_lesson())
    document = (library.root / f"{_lesson().id}.md").read_text()
    assert document.startswith("---")
    assert "tags:" in document and "- g-cavity-missing" in document
    assert "boolean difference" in document


def test_learning_the_same_lesson_twice_confirms_it_rather_than_duplicating_it(library):
    library.add(_lesson())
    library.add(_lesson(asset_classes=["flask"]))
    stored = library.all()
    assert len(stored) == 1
    assert stored[0].confirmed == 2
    assert stored[0].asset_classes == ["beaker", "flask"]


def test_lessons_are_retrieved_by_the_codes_they_bear_on(library):
    library.add(_lesson())
    library.add(
        _lesson(
            id="g_dim_measure_the_body_not_the_bounding_box",
            title="Measure the body, not the bounding box",
            codes=["G-DIM"],
        )
    )
    matched = library.relevant(codes=["G-DIM"], asset_class="beaker")
    assert [item.codes for item in matched] == [["G-DIM"]]
    assert library.relevant(codes=["G-PROBE-ESCAPE"]) == []


def test_a_briefing_is_empty_when_nothing_applies(library):
    assert library.briefing([]) == ""
    text = library.briefing([_lesson()])
    assert "<lessons_from_earlier_runs>" in text and "boolean difference" in text


# --------------------------------------------------------------------------- #
# retrieval through Articraft
# --------------------------------------------------------------------------- #


def test_articraft_can_find_a_lesson_once_the_library_is_activated(library, monkeypatch):
    """The whole point: a lesson has to come back from the tool the model already uses."""
    monkeypatch.delenv("ARTICRAFT_EXTRA_EXAMPLE_DIRS", raising=False)
    from amx.paths import activate_articraft

    activate_articraft()
    from agent.examples import search_example_documents

    library.add(
        _lesson(
            id="g_cavity_missing_boolean_difference_for_vessel_interiors",
            title="Boolean difference for vessel interiors",
        )
    )
    library.activate()

    matches = search_example_documents(
        "cavity missing boolean difference vessel interior", sdk_package="sdk", limit=5
    )
    titles = [match.doc.title for match in matches]
    assert "Boolean difference for vessel interiors" in titles


def test_a_lesson_written_after_the_index_was_built_is_still_found(library, monkeypatch):
    """The caches key on the SDK package and cannot see a directory that gained a file."""
    monkeypatch.delenv("ARTICRAFT_EXTRA_EXAMPLE_DIRS", raising=False)
    from amx.paths import activate_articraft

    activate_articraft()
    from agent.examples import search_example_documents

    library.activate()
    search_example_documents("anything at all", sdk_package="sdk", limit=1)

    library.add(
        _lesson(
            id="g_probe_escape_give_the_vessel_floor_real_thickness",
            title="Give the vessel floor real thickness",
            codes=["G-PROBE-ESCAPE"],
        )
    )
    matches = search_example_documents(
        "vessel floor real thickness probe", sdk_package="sdk", limit=5
    )
    assert "Give the vessel floor real thickness" in [m.doc.title for m in matches]


# --------------------------------------------------------------------------- #
# learning from a search
# --------------------------------------------------------------------------- #


def _report(*failures: str) -> Report:
    return Report(
        kind="grounding",
        subject="beaker",
        findings=[
            Finding(
                code=code,
                severity=Severity.FAILURE,
                subject="beaker",
                summary=f"{code} is failing",
                repair_target=RepairTarget.ASSET,
            )
            for code in failures
        ],
    )


def test_a_repaired_failure_becomes_a_lesson(tmp_path, library, monkeypatch):
    monkeypatch.setattr(
        "amx.skills.learn._summarise", lambda *a, **k: None, raising=True
    )
    tree = SearchTree(tmp_path / "tree")
    root = tree.add(source="radius = 0.030\n", parent=None)
    tree.record(root, _report("G-CAVITY-MISSING"))
    child = tree.add(source="radius = 0.032\nbore = difference(body, cavity)\n", parent=root)
    tree.record(child, _report())

    learned = learn_from_search(tree, library=library, asset_class="beaker", case_id="BEA-001")
    assert [lesson.codes for lesson in learned] == [["G-CAVITY-MISSING"]]
    assert learned[0].source_case == "BEA-001"
    assert "difference" in learned[0].snippet


def test_a_failure_that_cleared_without_an_edit_is_not_a_lesson(tmp_path, library, monkeypatch):
    """Nothing changed, so nothing was learned; the check simply measured differently."""
    monkeypatch.setattr("amx.skills.learn._summarise", lambda *a, **k: None, raising=True)
    tree = SearchTree(tmp_path / "tree")
    root = tree.add(source="same\n", parent=None)
    tree.record(root, _report("G-STABILITY"))
    child = tree.add(source="same\n", parent=root)
    tree.record(child, _report())

    assert learn_from_search(tree, library=library) == []


def test_swapping_one_failure_for_another_is_not_counted_as_a_repair(
    tmp_path, library, monkeypatch
):
    monkeypatch.setattr("amx.skills.learn._summarise", lambda *a, **k: None, raising=True)
    tree = SearchTree(tmp_path / "tree")
    root = tree.add(source="a = 1\n", parent=None)
    tree.record(root, _report("G-DIM"))
    child = tree.add(source="a = 2\n", parent=root)
    tree.record(child, _report("G-DIM"))

    assert learn_from_search(tree, library=library) == []
