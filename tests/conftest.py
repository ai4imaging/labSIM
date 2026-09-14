"""Shared fixtures.

The vendored arm is built by `scripts/vendor_robots.py` rather than checked in, so the
tests that need it are skipped when it is absent instead of failing. A missing arm means
setup has not been run; it does not mean the code is broken.
"""

from __future__ import annotations

import pytest

from amx.paths import ROBOTS_DIR

ROBOT_ID = "ur5e_robotiq85"


@pytest.fixture(scope="session")
def robot_dir():
    directory = ROBOTS_DIR / ROBOT_ID
    if not (directory / "arm.xml").is_file():
        pytest.skip(f"{ROBOT_ID} has not been vendored; run scripts/vendor_robots.py")
    return directory
