"""Resolving what to look at, and deciding what is allowed to move while you look."""

from pathlib import Path

import pytest

from amx import view
from amx.grounding.spec import ComponentTarget, GroundingSpec, OperationTarget


def test_resolve_prefers_persisted_visualization_fallback(tmp_path: Path):
    """Older sweeps have a `visualization/` tree; generation no longer writes one.

    Nothing new lands there — a run that did not finish authoring now leaves no asset at
    all — but the directories already on disk are still worth being able to open.
    """
    case = tmp_path / "BAL-001"
    mjcf = case / "visualization" / "mjcf" / "asset.xml"
    mjcf.parent.mkdir(parents=True)
    mjcf.write_text("<mujoco/>")

    assert view.resolve(case) == mjcf


def test_resolve_tries_older_revision_when_latest_does_not_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = tmp_path / "BAL-001"
    latest = case / "asset" / "model.py"
    older = case / "agent" / "turns" / "turn-001" / "model.py"
    latest.parent.mkdir(parents=True)
    older.parent.mkdir(parents=True)
    latest.write_text("BROKEN = True\n")
    older.write_text("GOOD = True\n")
    expected = tmp_path / "compiled.xml"

    def fake_compile(path: Path, *, asset_id: str) -> Path:
        assert asset_id == "BAL-001"
        if path == latest:
            raise view.ViewError("latest is empty")
        assert path == older
        return expected

    monkeypatch.setattr(view, "_compile", fake_compile)
    assert view.resolve(case) == expected


# --------------------------------------------------------------------------- #
# what --animate drives
# --------------------------------------------------------------------------- #

PANEL = """
<mujoco model="panel">
  <worldbody>
    <body name="housing">
      <geom type="box" size="0.05 0.05 0.02"/>
      <body name="power_button" pos="0 0 0.02">
        <joint name="power_button_joint" type="slide" axis="0 0 1"
               limited="true" range="-0.002 0"/>
        <geom type="box" size="0.005 0.005 0.002"/>
      </body>
      <body name="cooling_fan" pos="0.03 0 0.02">
        <joint name="cooling_fan_joint" type="hinge" axis="0 0 1" limited="false"/>
        <geom type="box" size="0.008 0.002 0.002"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def panel(tmp_path: Path):
    mujoco = pytest.importorskip("mujoco")
    path = tmp_path / "panel.xml"
    path.write_text(PANEL)
    return mujoco.MjModel.from_xml_path(str(path))


def _spec(*operations: OperationTarget) -> GroundingSpec:
    return GroundingSpec(
        asset_id="PAN-001",
        components=[ComponentTarget(id="CMP-1", name="housing", kind="fixed")],
        operations=list(operations),
    )


def test_nothing_is_animated_without_an_operation_contract(panel):
    plan = view.animation_plan(panel, view.articulations(panel), spec=None)
    assert plan.driven == []
    assert "--joint" in plan.note


def test_only_the_joints_the_contract_names_are_driven(panel):
    spec = _spec(
        OperationTarget(
            id="OP-1",
            name="power button",
            kind="press",
            child_hint="power_button",
            expected_joint_types=["slide"],
        )
    )
    plan = view.animation_plan(panel, view.articulations(panel), spec=spec)

    assert [joint.name for joint in plan.driven] == ["power_button_joint"]
    assert not plan.driven[0].spins
    assert "holding 1 joint" in plan.note


def test_an_unlimited_hinge_only_spins_when_the_contract_says_it_turns(panel):
    """`limited="false"` is not a claim that something rotates.

    Reading it as one is what made removable parts — exported as unbounded rotations by
    a bug in the MJCF writer — appear in the viewer as objects spinning on the spot.
    """
    quiet = _spec(
        OperationTarget(
            id="OP-1", name="cooling fan", kind="rotate", child_hint="cooling_fan",
            continuous=False,
        )
    )
    spinning = _spec(
        OperationTarget(
            id="OP-1", name="cooling fan", kind="rotate", child_hint="cooling_fan",
            continuous=True,
        )
    )
    joints = view.articulations(panel)

    assert not view.animation_plan(panel, joints, spec=quiet).driven[0].spins
    assert view.animation_plan(panel, joints, spec=spinning).driven[0].spins


def test_naming_a_joint_overrides_the_contract(panel):
    spec = _spec(
        OperationTarget(id="OP-1", name="power button", kind="press", child_hint="power_button")
    )
    plan = view.animation_plan(
        panel, view.articulations(panel), spec=spec, requested=["cooling_fan_joint"]
    )
    assert [joint.name for joint in plan.driven] == ["cooling_fan_joint"]
