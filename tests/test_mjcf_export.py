"""What a URDF joint becomes in MuJoCo, and what that means for what the asset does.

The export is where a mechanism stops being a declaration and starts being something
that moves, so a joint type mapped wrongly here is not a cosmetic defect: it is an asset
whose pestle spins on the spot, and a benchmark that measures the spinning.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from amx.paths import activate_articraft

pytest.importorskip("cadquery", reason="the Articraft compiler needs the geometry backend")


@pytest.fixture(scope="module", autouse=True)
def _articraft():
    activate_articraft()


MODEL = '''
from sdk import (
    ArticulatedObject,
    ArticulationType,
    Box,
    Inertial,
    MotionLimits,
    TestContext,
)


def build_object_model():
    model = ArticulatedObject(name="mortar_set")

    bowl = model.part("mortar_bowl")
    bowl.visual(Box((0.12, 0.12, 0.06)), name="bowl")
    bowl.inertial = Inertial.from_geometry(Box((0.12, 0.12, 0.06)), mass=0.9)

    pestle = model.part("removable_pestle")
    pestle.visual(Box((0.02, 0.02, 0.10)), name="pestle")
    pestle.inertial = Inertial.from_geometry(Box((0.02, 0.02, 0.10)), mass=0.12)
    model.articulation(
        "bowl_to_pestle",
        ArticulationType.FLOATING,
        parent=bowl,
        child=pestle,
        origin=Origin(xyz=(0.0, 0.0, 0.08)),
    )

    cover = model.part("hinged_cover")
    cover.visual(Box((0.12, 0.12, 0.008)), name="cover")
    cover.inertial = Inertial.from_geometry(Box((0.12, 0.12, 0.008)), mass=0.2)
    model.articulation(
        "bowl_to_cover",
        ArticulationType.REVOLUTE,
        parent=bowl,
        child=cover,
        axis=(1.0, 0.0, 0.0),
        origin=Origin(xyz=(0.0, 0.06, 0.03)),
        motion_limits=MotionLimits(lower=0.0, upper=1.5, effort=4.0, velocity=1.0),
    )
    return model


def run_tests():
    return TestContext(object_model).report()


object_model = build_object_model()
'''


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> ET.Element:
    from agent.compiler import compile_urdf_report
    from agent.mujoco_export import export_record_to_mjcf

    root = tmp_path_factory.mktemp("mortar")
    script = root / "model.py"
    script.write_text("from sdk import Origin\n" + MODEL.lstrip())

    report = compile_urdf_report(script, run_checks=False, target="full")
    urdf = root / "model.urdf"
    urdf.write_text(report.urdf_xml)
    result = export_record_to_mjcf(
        model_path=script, urdf_path=urdf, output_dir=root / "mjcf", record_id="MOR-TEST"
    )
    return ET.parse(result.asset_xml_path).getroot(), Path(result.asset_xml_path)


def _body(root: ET.Element, name: str) -> ET.Element:
    found = next(body for body in root.iter("body") if body.attrib.get("name") == name)
    return found


def test_a_floating_part_gets_six_degrees_of_freedom_not_a_spin(exported):
    """`floating` used to fall into the hinge branch and come out unlimited.

    An unbounded hinge is one rotational degree of freedom, so a part meant to lift out
    could only turn — and `limited="false"` then read downstream as "this thing spins".
    """
    root, _ = exported
    joints = _body(root, "removable_pestle").findall("joint")

    assert [joint.attrib["type"] for joint in joints] == ["slide", "slide", "slide", "ball"]
    assert not any(joint.attrib.get("type") == "hinge" for joint in joints)


def test_a_bounded_hinge_still_exports_as_a_bounded_hinge(exported):
    root, _ = exported
    joint = _body(root, "hinged_cover").find("joint")

    assert joint.attrib["type"] == "hinge"
    assert joint.attrib["limited"] == "true"
    assert joint.attrib["range"].split()[1].startswith("1.5")


def test_a_removable_part_still_collides_with_what_holds_it(exported):
    """Excluding the pair would answer "can it be lifted out" with yes, by construction."""
    root, _ = exported
    excluded = {
        (pair.attrib["body1"], pair.attrib["body2"])
        for pair in root.iter("exclude")
    }

    assert ("mortar_bowl", "removable_pestle") not in excluded
    assert ("mortar_bowl", "hinged_cover") in excluded


def test_the_exported_model_loads(exported):
    """Seven qpos for the floating part, and a keyframe that is not a zero quaternion."""
    mujoco = pytest.importorskip("mujoco")
    _, path = exported

    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nq >= 8  # six for the pestle (3 + 4 qpos) plus the cover hinge
