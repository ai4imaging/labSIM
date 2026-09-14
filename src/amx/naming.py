"""The scene-wide naming convention.

Every geom, body, joint and site in a composed scene carries a prefix that says what it
belongs to. Nothing in the pipeline guesses at roles from geometry: the judge policy
classifies contacts by these prefixes, the repair router decides what to edit from them,
and the scene builder refuses to compose a part that does not use them. Changing a prefix
here changes it everywhere, so they are defined once.
"""

from __future__ import annotations

ROBOT = "robot/"
"""The arm and its gripper, exactly as `scripts/vendor_robots.py` emits them."""

TOOL = "robot/tool/"
"""Anything rigidly mounted on the arm's flange: end-effector adapters and their payload."""

BENCH = "bench/"
"""The static work surface and its frame."""

ASSET_PREFIX = "asset."
"""Instruments and labware produced by part 1. Spelled `asset.<id>/`."""

FIXTURE_PREFIX = "fixture."
"""Bench-mounted co-designed parts from part 2. Spelled `fixture.<id>/`."""

SEPARATOR = "/"


def asset_namespace(asset_id: str) -> str:
    return f"{ASSET_PREFIX}{asset_id}{SEPARATOR}"


def fixture_namespace(part_id: str) -> str:
    return f"{FIXTURE_PREFIX}{part_id}{SEPARATOR}"


def tool_namespace(part_id: str) -> str:
    return f"{TOOL}{part_id}{SEPARATOR}"


def is_robot(name: str) -> bool:
    """True for arm links, false for tool payload — the tool test must come first."""
    return name.startswith(ROBOT) and not name.startswith(TOOL)


def is_tool(name: str) -> bool:
    return name.startswith(TOOL)


def owner_of(name: str) -> str:
    """The namespace a name belongs to, or `""` for unnamespaced scene furniture."""
    for prefix in (TOOL, ROBOT, BENCH):
        if name.startswith(prefix):
            return prefix
    for prefix in (ASSET_PREFIX, FIXTURE_PREFIX):
        if name.startswith(prefix):
            head, _, _ = name.partition(SEPARATOR)
            return head + SEPARATOR
    return ""


def validate(name: str, expected: str) -> str:
    if not name.startswith(expected):
        raise ValueError(f"{name!r} must live under the {expected!r} namespace")
    return name
