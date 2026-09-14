"""Search behaviour, tested against a synthetic landscape with a known shape.

Using a real asset here would test the compiler and the LLM, not the search. What has to
be pinned down is the search's own conduct: does partial credit give it a gradient to
climb, does it abandon a branch that stops improving, and does it come back to a better
one it had set aside. Those are all statements about the frontier, so the landscape is
made of arithmetic.
"""

from __future__ import annotations

from amx.report import Finding, RepairTarget, Report, Severity
from amx.search import SearchConfig, score_report, search


def _dimension_report(error: float, *, tolerance: float = 0.05, hard: float = 0.2) -> Report:
    """One dimension finding at a stated relative error."""
    passing = error <= tolerance
    return Report(
        kind="test",
        subject="widget",
        findings=[
            Finding(
                code="G-DIM",
                severity=Severity.INFO if passing else Severity.FAILURE,
                subject="widget/diameter",
                summary=f"diameter is {error * 100:.1f}% out",
                repair_target=RepairTarget.NONE if passing else RepairTarget.ASSET,
                metrics={"relative_error": error},
                thresholds={"tolerance_rel": tolerance, "hard_fail_rel": hard},
            )
        ],
    )


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def test_partial_credit_gives_the_search_something_to_climb():
    """Halving the error has to raise the score, or there is no gradient at all."""
    worse = score_report(_dimension_report(0.18)).value
    better = score_report(_dimension_report(0.09)).value
    assert better > worse


def test_passing_always_beats_nearly_passing():
    nearly = score_report(_dimension_report(0.051)).value
    passing = score_report(_dimension_report(0.01)).value
    assert nearly < passing == 1.0


def test_a_hopeless_error_earns_nothing():
    assert score_report(_dimension_report(3.0)).value == 0.0


def test_an_unmeasurable_asset_scores_zero_however_few_checks_it_tripped():
    """A broken asset must not outrank a working one just by answering fewer questions."""
    broken = Report(
        kind="test",
        subject="widget",
        findings=[
            Finding(
                code="G-UNMEASURABLE",
                severity=Severity.FAILURE,
                summary="the asset will not load",
            )
        ],
    )
    assert score_report(broken).value == 0.0
    assert score_report(broken).blocked


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


def _numeric_search(tmp_path, expand, *, target: float = 10.0, **config):
    """Candidates are numbers written as text; the goal is to reach `target`."""

    def evaluate(source: str, node) -> Report:
        value = float(source)
        return _dimension_report(abs(value - target) / target)

    return search(
        root_source="4.0",
        evaluate=evaluate,
        expand=expand,
        run_dir=tmp_path,
        config=SearchConfig(**config),
    )


def test_it_finds_the_answer_and_stops(tmp_path):
    def expand(node, wanted):
        current = float(node.source)
        return [(str(current + step), f"+{step}") for step in (1.0, 3.0)][:wanted]

    result = _numeric_search(tmp_path, expand, max_nodes=20, samples=2)
    assert result.succeeded
    assert float(result.solved.source) == 10.0
    assert result.stopped_because == "a candidate passed every check"


def test_it_backtracks_out_of_a_branch_that_stops_improving(tmp_path):
    """A dead end must cost a bounded amount, not the whole budget."""

    def expand(node, wanted):
        current = float(node.source)
        if current >= 6.0:
            # A cul-de-sac: every step from here makes things worse.
            return [(str(current - 2.0), "regress")][:wanted]
        return [(str(current + 2.0), "+2"), (str(current + 0.5), "+0.5")][:wanted]

    result = _numeric_search(tmp_path, expand, max_nodes=12, samples=2, beam=3)
    exhausted = [n for n in result.tree.nodes.values() if n.status == "exhausted"]
    assert exhausted, "a branch that only regresses should be marked exhausted"
    assert any("backtracking" in n.note for n in exhausted)


def test_a_failing_expander_kills_one_node_rather_than_the_run(tmp_path):
    def expand(node, wanted):
        if node.id == 0:
            raise RuntimeError("the model returned nothing usable")
        return [(str(float(node.source) + 1.0), "+1")][:wanted]

    result = _numeric_search(tmp_path, expand, max_nodes=6)
    assert result.tree.nodes[0].status == "failed"
    assert any("expansion failed" in note for note in result.notes)


def test_an_unevaluable_candidate_is_a_dead_node_rather_than_a_crash(tmp_path):
    def evaluate(source: str, node) -> Report:
        if source == "boom":
            raise ValueError("the asset will not compile")
        return _dimension_report(0.5)

    def expand(node, wanted):
        return [("boom", "broken")][:wanted]

    result = search(
        root_source="4.0",
        evaluate=evaluate,
        expand=expand,
        run_dir=tmp_path,
        config=SearchConfig(max_nodes=4),
    )
    broken = [n for n in result.tree.nodes.values() if n.status == "failed"]
    assert broken and "will not compile" in broken[0].note


def test_the_tree_is_on_disk_while_it_runs_not_only_at_the_end(tmp_path):
    def expand(node, wanted):
        return [(str(float(node.source) + 1.0), "+1")][:wanted]

    result = _numeric_search(tmp_path, expand, max_nodes=5, samples=1)
    assert (tmp_path / "tree.json").is_file()
    assert (tmp_path / "search.txt").is_file()
    for node in result.tree.nodes.values():
        directory = tmp_path / f"node-{node.id:04d}"
        assert (directory / "model.py").read_text() == node.source
        assert (directory / "report.json").is_file()


def test_lineage_reads_back_the_path_that_produced_a_node(tmp_path):
    def expand(node, wanted):
        return [(str(float(node.source) + 2.0), "+2")][:wanted]

    result = _numeric_search(tmp_path, expand, max_nodes=4, samples=1)
    deepest = max(result.tree.nodes.values(), key=lambda n: n.depth)
    chain = result.tree.lineage(deepest)
    assert [n.id for n in chain] == sorted(n.id for n in chain)
    assert chain[0].parent_id is None
