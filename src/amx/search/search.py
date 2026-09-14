"""Best-first search over candidate models, with backtracking.

The reason this exists rather than a plain repair loop: a repair loop is a search of
width one that can never undo anything. When a model's fix for a short cavity is to
scale the whole object, the loop follows it — the next round measures a wider object,
reports the diameter it has now broken, and every subsequent round argues with a mistake
that nothing can reach back and reverse. Greedy descent with no memory of where it came
from is not a bad search strategy so much as an absent one.

What is here instead:

* **Several candidates per expansion.** The same failure is repaired more than once, and
  the attempts differ. Which one was better is a question the checks can answer and the
  model cannot.
* **A frontier, not a cursor.** Any scored node can be expanded next, so a promising
  branch that stalls does not trap the search.
* **Backtracking.** A node whose children all score worse than it did is exhausted, and
  the search returns to the best node it has not finished with.

The expander is a callback. This module does not know whether a candidate comes from an
LLM, from a parametric sweep or from a heuristic, which is what lets part 1 and part 3
share it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from amx.report import Report
from amx.search.tree import Node, SearchTree

logger = logging.getLogger(__name__)

Evaluate = Callable[[str, Node], Report]
"""Given a candidate's source and its node, measure it."""

Expand = Callable[[Node, int], Sequence[tuple[str, str]]]
"""Given a node and how many candidates are wanted, return `(source, origin)` pairs.

Returning fewer than asked for is allowed and means the expander had nothing more to
offer; returning none exhausts the node.
"""


@dataclass
class SearchConfig:
    max_nodes: int = 24
    """Total candidates evaluated, including the root. This is the real budget: every
    node costs a compile, a physics run and usually an LLM call."""

    samples: int = 2
    """Candidates per expansion. Two is the useful minimum — one is a repair loop."""

    beam: int = 3
    """How many nodes stay eligible for expansion. Bounds how far the search can wander
    back into branches it has already had a poor answer from."""

    max_depth: int = 6
    improvement_epsilon: float = 1e-4
    """Below this, a child counts as no better than its parent. Guards against a search
    that spends its budget on rounding noise."""

    stop_when_solved: bool = True
    time_budget_s: float | None = None


@dataclass
class SearchResult:
    tree: SearchTree
    best: Node | None
    solved: Node | None
    stopped_because: str
    evaluated: int = 0
    expansions: int = 0
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.solved is not None

    def to_text(self) -> str:
        head = (
            f"search: {self.evaluated} candidates over {self.expansions} expansions "
            f"in {self.elapsed_s:.1f} s — stopped because {self.stopped_because}"
        )
        if self.solved is not None:
            verdict = f"solved at node {self.solved.id} (depth {self.solved.depth})"
        elif self.best is not None:
            verdict = f"unsolved; best is node {self.best.id} at {self.best.value:.3f}"
        else:
            verdict = "nothing was successfully evaluated"
        return "\n".join([head, verdict, "", self.tree.render()])


def search(
    *,
    root_source: str,
    evaluate: Evaluate,
    expand: Expand,
    run_dir: Path,
    config: SearchConfig | None = None,
) -> SearchResult:
    """Run the search and return the best candidate it found."""
    config = config or SearchConfig()
    tree = SearchTree(Path(run_dir))
    started = time.monotonic()
    evaluated = 0
    expansions = 0
    notes: list[str] = []

    root = tree.add(source=root_source, parent=None, origin="initial")
    _evaluate_into(tree, root, evaluate)
    evaluated += 1

    if root.solved and config.stop_when_solved:
        return _result(tree, "the first candidate already passed", evaluated, expansions,
                       started, notes)

    while True:
        if evaluated >= config.max_nodes:
            reason = f"the budget of {config.max_nodes} candidates was used up"
            break
        if config.time_budget_s is not None and time.monotonic() - started > config.time_budget_s:
            reason = f"the time budget of {config.time_budget_s:.0f} s ran out"
            break

        node = _pick(tree, config)
        if node is None:
            reason = "there was nothing left to expand"
            break

        remaining = config.max_nodes - evaluated
        wanted = max(1, min(config.samples, remaining))
        try:
            candidates = list(expand(node, wanted))
        except Exception as error:  # noqa: BLE001 - an expander failure exhausts a node, not the run
            logger.exception("expanding node %s failed", node.id)
            tree.mark(node, "failed", note=f"expansion failed: {type(error).__name__}: {error}")
            notes.append(f"node {node.id}: expansion failed ({error})")
            continue

        expansions += 1
        if not candidates:
            tree.mark(node, "exhausted", note="the expander had nothing further to try")
            continue

        best_child_value = -1.0
        for source, origin in candidates:
            child = tree.add(source=source, parent=node, origin=origin)
            _evaluate_into(tree, child, evaluate)
            evaluated += 1
            best_child_value = max(best_child_value, child.value)
            if child.solved and config.stop_when_solved:
                tree.mark(node, "expanded")
                return _result(tree, "a candidate passed every check", evaluated, expansions,
                               started, notes)
            if evaluated >= config.max_nodes:
                break

        if best_child_value <= node.value + config.improvement_epsilon:
            # Every attempt from here came back no better. Stop paying for this branch
            # and let the frontier hand the search somewhere else.
            tree.mark(
                node,
                "exhausted",
                note=(
                    f"{len(candidates)} attempt(s) from {node.value:.3f} reached at best "
                    f"{best_child_value:.3f}; backtracking"
                ),
            )
            notes.append(f"node {node.id}: exhausted after no improvement, backtracked")
        else:
            tree.mark(node, "expanded")

    return _result(tree, reason, evaluated, expansions, started, notes)


def _pick(tree: SearchTree, config: SearchConfig) -> Node | None:
    """The most promising node still worth expanding.

    Restricted to the top `beam` by score so the search cannot drift indefinitely back
    into branches it has already had poor answers from, and depth-capped because a very
    deep chain of repairs has usually stopped being a repair of the original problem.
    """
    eligible = [
        node
        for node in tree.nodes.values()
        if node.status == "open" and node.score is not None and node.depth < config.max_depth
    ]
    if not eligible:
        return None
    ranked = sorted(eligible, key=lambda n: (-n.value, n.depth, n.id))
    return ranked[: config.beam][0]


def _evaluate_into(tree: SearchTree, node: Node, evaluate: Evaluate) -> None:
    started = time.monotonic()
    try:
        report = evaluate(node.source, node)
    except Exception as error:  # noqa: BLE001 - an unevaluable candidate is a dead node
        logger.exception("evaluating node %s failed", node.id)
        report = Report(
            kind="search-evaluation",
            subject=f"node-{node.id}",
            notes=[f"evaluation failed: {type(error).__name__}: {error}"],
        )
        tree.record(node, report, elapsed_s=time.monotonic() - started)
        tree.mark(node, "failed", note=str(error))
        return
    tree.record(node, report, elapsed_s=time.monotonic() - started)


def _result(
    tree: SearchTree,
    reason: str,
    evaluated: int,
    expansions: int,
    started: float,
    notes: list[str],
) -> SearchResult:
    result = SearchResult(
        tree=tree,
        best=tree.best(),
        solved=tree.solved(),
        stopped_because=reason,
        evaluated=evaluated,
        expansions=expansions,
        elapsed_s=time.monotonic() - started,
        notes=notes,
    )
    (tree.root / "search.txt").write_text(result.to_text() + "\n")
    return result
