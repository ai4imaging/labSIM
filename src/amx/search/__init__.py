"""Best-first search over candidate designs, shared by parts 1 and 3.

Part 1 searches over `model.py` revisions scored by grounding; part 3 searches over
repair patches scored by the simulation judge. The frontier, the backtracking and the
persistence are the same problem in both, so they are the same code.
"""

from amx.search.score import Score, finding_credit, score_report, score_reports
from amx.search.search import SearchConfig, SearchResult, search
from amx.search.tree import Node, SearchTree

__all__ = [
    "Node",
    "Score",
    "SearchConfig",
    "SearchResult",
    "SearchTree",
    "finding_credit",
    "score_report",
    "score_reports",
    "search",
]
