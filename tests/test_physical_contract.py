"""Simulation-ready assets cannot defer mass and inertia until export time."""

from __future__ import annotations

from pathlib import Path

import pytest

from amx.paths import activate_articraft


@pytest.fixture(scope="module", autouse=True)
def _articraft():
    activate_articraft()


def test_test_context_rejects_every_massless_part():
    from sdk import ArticulatedObject, Box, Inertial, TestContext

    model = ArticulatedObject(name="physical_contract")
    body = model.part("body")
    body.visual(Box((0.1, 0.1, 0.1)), name="body")
    lid = model.part("lid")
    lid.visual(Box((0.1, 0.1, 0.01)), name="lid")
    body.inertial = Inertial.from_geometry(Box((0.1, 0.1, 0.1)), mass=1.0)

    ctx = TestContext(model)
    assert not ctx.check_part_inertials()
    report = ctx.report()
    assert not report.passed
    assert report.failures[0].name == "check_part_inertials"
    assert "'lid': missing inertial" in report.failures[0].details


def test_test_context_accepts_positive_definite_inertials():
    from sdk import ArticulatedObject, Box, Inertial, TestContext

    model = ArticulatedObject(name="physical_contract")
    for name, mass in (("body", 1.0), ("lid", 0.1)):
        part = model.part(name)
        proxy = Box((0.1, 0.1, 0.01))
        part.visual(proxy, name=name)
        part.inertial = Inertial.from_geometry(proxy, mass=mass)

    ctx = TestContext(model)
    assert ctx.check_part_inertials()
    assert ctx.report().passed


def test_compiler_owned_qc_blocks_a_missing_inertial(tmp_path: Path):
    from agent.compiler import compile_urdf_report

    script = tmp_path / "model.py"
    script.write_text(
        """
from sdk import ArticulatedObject, Box, TestContext

def build_object_model():
    model = ArticulatedObject(name="massless")
    body = model.part("body")
    body.visual(Box((0.1, 0.1, 0.1)), name="body")
    return model

def run_tests():
    return TestContext(object_model).report()

object_model = build_object_model()
""".lstrip()
    )

    with pytest.raises(RuntimeError, match="check_part_inertials"):
        compile_urdf_report(script, run_checks=True)


def test_button_visual_cannot_be_fused_into_a_panel():
    from sdk import ArticulatedObject, Box, TestContext

    model = ArticulatedObject(name="controls")
    panel = model.part("control_panel")
    panel.visual(Box((0.2, 0.1, 0.01)), name="panel")
    panel.visual(Box((0.02, 0.02, 0.004)), name="start_button")

    ctx = TestContext(model)
    assert not ctx.check_interaction_semantics()
    assert "split each control into its own Part" in ctx.report().failures[0].details


def test_button_part_needs_a_bounded_prismatic_press():
    from sdk import ArticulatedObject, ArticulationType, Box, MotionLimits, TestContext

    model = ArticulatedObject(name="controls")
    panel = model.part("control_panel")
    panel.visual(Box((0.2, 0.1, 0.01)), name="panel")
    button = model.part("start_button")
    button.visual(Box((0.02, 0.02, 0.004)), name="button_cap")
    model.articulation(
        "panel_to_start_button",
        ArticulationType.PRISMATIC,
        parent=panel,
        child=button,
        axis=(0.0, 0.0, -1.0),
        motion_limits=MotionLimits(lower=0.0, upper=0.002, effort=2.0, velocity=0.02),
    )

    ctx = TestContext(model)
    assert ctx.check_interaction_semantics()
    assert ctx.report().passed


def test_lid_cannot_use_an_unbounded_rotor_joint():
    from sdk import ArticulatedObject, ArticulationType, Box, TestContext

    model = ArticulatedObject(name="bad_lid")
    body = model.part("body")
    body.visual(Box((0.2, 0.1, 0.1)), name="body")
    lid = model.part("machine_lid")
    lid.visual(Box((0.2, 0.1, 0.01)), name="lid")
    model.articulation(
        "body_to_lid",
        ArticulationType.CONTINUOUS,
        parent=body,
        child=lid,
    )

    ctx = TestContext(model)
    assert not ctx.check_interaction_semantics()
    assert "machine_lid" in ctx.report().failures[0].details
    assert "continuous" in ctx.report().failures[0].details


def test_named_lid_cannot_be_a_fixed_part():
    from sdk import ArticulatedObject, Box, TestContext

    model = ArticulatedObject(name="fixed_lid")
    body = model.part("body")
    body.visual(Box((0.2, 0.1, 0.1)), name="body")
    lid = model.part("machine_lid")
    lid.visual(Box((0.2, 0.1, 0.01)), name="lid_shell")

    ctx = TestContext(model)
    assert not ctx.check_interaction_semantics()
    assert "machine_lid" in ctx.report().failures[0].details
    assert "no joint" in ctx.report().failures[0].details


def test_rotor_requires_a_real_continuous_joint():
    from sdk import ArticulatedObject, Box, TestContext

    model = ArticulatedObject(name="fixed_rotor")
    body = model.part("body")
    body.visual(Box((0.2, 0.1, 0.1)), name="body")
    rotor = model.part("sample_rotor")
    rotor.visual(Box((0.1, 0.1, 0.01)), name="rotor_disc")

    ctx = TestContext(model)
    assert not ctx.check_interaction_semantics()
    assert "sample_rotor" in ctx.report().failures[0].details
    assert "continuous" in ctx.report().failures[0].details


def test_compile_loop_can_restore_last_good_source(tmp_path: Path):
    from agent.harness_compile import CompileFeedbackLoop

    model = tmp_path / "model.py"
    checkpoint = tmp_path / "model.urdf"
    model.write_text("GOOD = True\n")
    loop = CompileFeedbackLoop(
        file_path=str(model),
        sdk_package="sdk",
        runtime_limits=None,
        checkpoint_urdf_path=checkpoint,
    )
    loop.remember_successful_code()
    model.write_text("BROKEN = True\n")

    assert loop.restore_last_successful_code()
    assert model.read_text() == "GOOD = True\n"
    assert checkpoint.with_name("model.last-good.py").read_text() == "GOOD = True\n"
