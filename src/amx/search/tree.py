"""The search tree itself: nodes, lineage, and what gets written to disk.

Each node is one candidate `model.py` together with what the grounding checks said about
it. The tree is persisted as it grows rather than at the end, because a search that runs
for an hour and dies on the last expansion should still be inspectable — and because the
question people actually ask of a run is "what did it try, and why did it stop trying
that", which is a shape and not a final answer.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from amx.report import Report
from amx.search.score import Score, score_report

NodeStatus = Literal["open", "expanded", "exhausted", "solved", "failed"]


@dataclass
class Node:
    """One candidate model, and everything known about it."""

    id: int
    parent_id: int | None
    depth: int
    source: str
    """The `model.py` text. Held in memory as well as on disk: a search re-reads a
    parent's source every time it expands it, and these are a few kilobytes each."""

    report: Report | None = None
    score: Score | None = None
    status: NodeStatus = "open"
    origin: str = ""
    """How this candidate came about — which repair was attempted, or which sample it was."""
    children: list[int] = field(default_factory=list)
    note: str = ""
    elapsed_s: float = 0.0

    @property
    def value(self) -> float:
        return self.score.value if self.score else 0.0

    @property
    def solved(self) -> bool:
        return self.report is not None and self.report.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "status": self.status,
            "origin": self.origin,
            "score": round(self.value, 6),
            "score_summary": self.score.summary() if self.score else "not scored",
            "solved": self.solved,
            "failures": [f.summary for f in (self.report.failures if self.report else [])][:10],
            "children": list(self.children),
            "note": self.note,
            "elapsed_s": round(self.elapsed_s, 2),
        }


class SearchTree:
    """Nodes plus their on-disk record.

    `root` gets a `tree.json` that is rewritten after every change, and a `node-NNNN/`
    directory per candidate holding the source it tried, the report it earned and a
    readable rendering of both.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.nodes: dict[int, Node] = {}
        self._next_id = 0
        self.started_at = datetime.now(UTC).isoformat()

    # ------------------------------------------------------------------ #

    def add(
        self,
        *,
        source: str,
        parent: Node | None,
        origin: str = "",
        note: str = "",
    ) -> Node:
        node = Node(
            id=self._next_id,
            parent_id=parent.id if parent else None,
            depth=parent.depth + 1 if parent else 0,
            source=source,
            origin=origin,
            note=note,
        )
        self._next_id += 1
        self.nodes[node.id] = node
        if parent is not None:
            parent.children.append(node.id)
        self._write_node(node)
        self.flush()
        return node

    def record(self, node: Node, report: Report, *, elapsed_s: float = 0.0) -> Node:
        node.report = report
        node.score = score_report(report)
        node.elapsed_s = elapsed_s
        node.status = "solved" if report.passed else node.status
        self._write_node(node)
        self.flush()
        return node

    def mark(self, node: Node, status: NodeStatus, *, note: str = "") -> None:
        node.status = status
        if note:
            node.note = note
        self.flush()

    # ------------------------------------------------------------------ #

    def best(self) -> Node | None:
        scored = [n for n in self.nodes.values() if n.score is not None]
        if not scored:
            return None
        # Ties broken towards the shallower node: it got there with fewer edits, so it is
        # the one a person would rather read and the one less likely to carry incidental
        # damage from a long repair chain.
        return max(scored, key=lambda n: (n.value, -n.depth, -n.id))

    def solved(self) -> Node | None:
        candidates = [n for n in self.nodes.values() if n.solved]
        return min(candidates, key=lambda n: (n.depth, n.id)) if candidates else None

    def lineage(self, node: Node) -> list[Node]:
        chain: list[Node] = []
        current: Node | None = node
        while current is not None:
            chain.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id is not None else None
        return list(reversed(chain))

    def node_dir(self, node: Node) -> Path:
        return self.root / f"node-{node.id:04d}"

    # ------------------------------------------------------------------ #

    def _write_node(self, node: Node) -> None:
        directory = self.node_dir(node)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.py").write_text(node.source)
        if node.report is not None:
            node.report.write(directory / "report.json")
            (directory / "report.txt").write_text(node.report.to_text() + "\n")
        (directory / "node.json").write_text(
            json.dumps(node.to_dict(), indent=2, ensure_ascii=False) + "\n"
        )

    def flush(self) -> None:
        best = self.best()
        payload = {
            "started_at": self.started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "nodes": [self.nodes[i].to_dict() for i in sorted(self.nodes)],
            "best": best.id if best else None,
            "best_score": round(best.value, 6) if best else None,
            "solved": (solved.id if (solved := self.solved()) else None),
        }
        (self.root / "tree.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def adopt(self, node: Node, destination: Path) -> Path:
        """Copy a node's model out of the tree, which is how a search returns an answer."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.node_dir(node) / "model.py", destination)
        return destination

    def render(self) -> str:
        """The tree as indented text. This is what gets read when a run is being explained."""
        lines = [f"search tree: {len(self.nodes)} nodes"]
        roots = [n for n in self.nodes.values() if n.parent_id is None]
        for root in roots:
            self._render_into(lines, root, prefix="")
        best = self.best()
        if best is not None:
            lines.append("")
            lines.append(f"best: node {best.id} at {best.value:.3f} — {best.score.summary()}")
        return "\n".join(lines)

    def _render_into(self, lines: list[str], node: Node, *, prefix: str) -> None:
        marker = {"solved": "OK", "exhausted": "--", "failed": "XX"}.get(node.status, "  ")
        origin = f" {node.origin}" if node.origin else ""
        lines.append(f"{prefix}[{marker}] node {node.id:>3}  {node.value:.3f}{origin}")
        if node.report is not None and node.report.failures:
            lines.append(f"{prefix}       {node.report.failures[0].summary[:110]}")
        for child_id in node.children:
            self._render_into(lines, self.nodes[child_id], prefix=prefix + "    ")
