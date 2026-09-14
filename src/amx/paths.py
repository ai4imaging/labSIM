"""Project layout and on-demand activation of the vendored Articraft tree."""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VENDOR_DIR = PROJECT_ROOT / "vendor"
ARTICRAFT_DIR = VENDOR_DIR / "articraft"
ARTICRAFT_EXT_DIR = VENDOR_DIR / "articraft_ext"
ROBOTS_DIR = VENDOR_DIR / "robots"

INPUTS_DIR = PROJECT_ROOT / "inputs"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
RUNS_DIR = PROJECT_ROOT / "runs"
ASSETS_DIR = PROJECT_ROOT / "assets"


class ArticraftUnavailable(RuntimeError):
    """The vendored Articraft tree is missing, or its optional dependencies are not installed."""


@cache
def activate_articraft() -> Path:
    """Put the vendored Articraft packages on ``sys.path`` and return its root.

    Articraft exposes bare top-level packages (``sdk``, ``agent``, ``cli``, ``storage``),
    and pulls in heavyweight optional dependencies such as cadquery. Rather than install
    it, we only reach for it when part 1 actually runs.
    """
    root = Path(os.environ.get("ARTICRAFT_ROOT", ARTICRAFT_DIR)).resolve()
    if not (root / "sdk" / "v0" / "__init__.py").is_file():
        raise ArticraftUnavailable(
            f"no Articraft SDK under {root}. Expected {root / 'sdk' / 'v0'} to be populated; "
            "see vendor/PROVENANCE.md."
        )
    # Inserted last-to-first, so the extension tree ends up ahead of the official one on
    # sys.path and its `agent` overlay wins.
    for entry in (root, ARTICRAFT_EXT_DIR):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id
