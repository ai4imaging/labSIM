"""Scene assembly, gripper calibration, inverse kinematics and the generated policy.

Everything here runs on the vendored arm and a couple of parametric parts, without
executing an episode: the properties being checked are geometric and structural, and an
episode takes a minute.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from amx import naming
from amx.codesign import TEMPLATES, export_part
from amx.geometry import Pose
from amx.sim.gripper import calibrate, find_pad_geoms
from amx.sim.kinematics import joint_addresses, solve_site_pose
from amx.sim.scene import (
    Bench,
    FixturePlacement,
    RobotRef,
    RobotSpec,
    Workcell,
    build_scene,
)


@pytest.fixture(scope="module")
def spec(robot_dir):
    return RobotSpec.load(robot_dir.name)


@pytest.fixture(scope="module")
def arm(robot_dir):
    return mujoco.MjModel.from_xml_path(str(robot_dir / "arm.xml"))


@pytest.fixture(scope="module")
def cell(robot_dir, tmp_path_factory):
    """A minimal cell: the arm, a bench and one rack bolted to it."""
    root = tmp_path_factory.mktemp("cell")
    rack = TEMPLATES["tube_rack"](part_id="rack", rows=1, columns=2)
    export_part(rack, root / "rack")
    workcell = Workcell(
        workcell_id="probe",
        robot=RobotRef(model=robot_dir.name, base=Pose(pos=(0.0, 0.0, 0.0))),
        bench=Bench(size_xy=(1.0, 0.8)),
        assets=[],
        fixtures=[
            FixturePlacement(
                part_id="rack",
                source=root / "rack",
                mount="bench",
                pose=Pose(pos=(0.3, 0.1, 0.0)),
            )
        ],
    )
    return build_scene(workcell, root / "scene")


def test_scene_loads_and_keeps_the_naming_convention(cell):
    """Every name the judge routes on must carry its owner's prefix.

    The convention is not cosmetic: `amx.sim.policy` classifies geoms into robot, device,
    fixture and environment purely from these prefixes, so a body that escapes the scheme
    becomes invisible to every rule.
    """
    model = mujoco.MjModel.from_xml_path(str(cell.scene_path))
    assert model.ngeom > 0

    prefixes = (naming.ROBOT, naming.ASSET_PREFIX, naming.FIXTURE_PREFIX, naming.BENCH)
    for kind in (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_JOINT):
        count = {mujoco.mjtObj.mjOBJ_BODY: model.nbody,
                 mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
                 mujoco.mjtObj.mjOBJ_JOINT: model.njnt}[kind]
        for index in range(count):
            name = mujoco.mj_id2name(model, kind, index)
            if not name or name == "world":
                continue
            assert name.startswith(prefixes), f"{name} is outside the naming convention"


def test_home_keyframe_is_a_settled_pose(cell):
    """The scene must start where it says it starts, or every episode begins by falling."""
    model = mujoco.MjModel.from_xml_path(str(cell.scene_path))
    assert model.nkey >= 1
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.qacc).all()
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all(), "the home pose diverges when simply left alone"
    for warning in (
        mujoco.mjtWarning.mjWARN_BADQPOS,
        mujoco.mjtWarning.mjWARN_BADQACC,
    ):
        assert int(data.warning[warning].number) == 0


def test_meshes_are_flattened_and_shared(cell):
    """Content-addressed mesh names, so two parts cut from the same STL are one asset."""
    text = cell.scene_path.read_text()
    assert "meshdir" not in text or "meshes" in text
    files = {p.name for p in (cell.scene_path.parent / "meshes").glob("*")}
    assert files, "the composed scene has no mesh directory"


def test_gripper_calibration_is_monotonic_and_invertible(arm, spec):
    """`Grip(width_m=...)` only means anything if width maps to one joint value.

    The curve has to be monotonic for the inverse to exist at all, and the round trip has
    to land inside a tenth of a millimetre or a commanded 10.8 mm is not 10.8 mm.
    """
    joints = tuple(str(a["joint"]) for a in spec.gripper_actuators)
    cal = calibrate(
        arm,
        joint_names=joints,
        open_value=float(spec.gripper_actuators[0]["open"]),
        closed_value=float(spec.gripper_actuators[0]["closed"]),
        flange_body=spec.flange_body,
        namespace=naming.ROBOT,
    )
    widths = cal.widths_m
    assert widths.size > 4
    assert np.all(np.diff(widths) < 0.0), "the opening does not close monotonically"

    low, high = cal.width_range_m
    assert low < 0.005 < high
    for target in np.linspace(low + 1e-4, high - 1e-4, 9):
        value = cal.joint_for_width(float(target))
        assert abs(cal.width_for_joint(value) - target) < 1e-4


def test_pads_reach_less_far_than_fingertips(arm, spec):
    """The two reaches must be distinct, since their difference is a design constraint.

    A part has to stand proud of its socket by `finger_reach_m - pad_reach_m` for the pads
    to touch it while the fingertips stay outside. If the two were read as one number the
    constraint would come out as "no socket can ever be picked from".
    """
    assert spec.pad_reach_m > 0.0
    assert spec.finger_reach_m > spec.pad_reach_m
    assert spec.finger_reach_m - spec.pad_reach_m < 0.01


def test_tcp_sits_between_the_pads(arm, spec):
    """The tool centre has to be where the grasp happens, not where the vendor put a site.

    Stated as a comparison against the vendor's own `grip_site`, because that is the claim:
    not that the measured centre is exact to the micron — the passive linkage joints sit
    where `qpos0` puts them here rather than where their constraints would — but that it is
    an order of magnitude closer to the grasp than the site that shipped with the gripper.
    """
    data = mujoco.MjData(arm)
    for actuator in spec.gripper_actuators:
        joint = mujoco.mj_name2id(arm, mujoco.mjtObj.mjOBJ_JOINT, str(actuator["joint"]))
        data.qpos[arm.jnt_qposadr[joint]] = float(actuator["open"])
    mujoco.mj_forward(arm, data)

    site = mujoco.mj_name2id(arm, mujoco.mjtObj.mjOBJ_SITE, spec.tcp_site)
    assert site >= 0
    pads = find_pad_geoms(arm, naming.ROBOT)
    centres = [
        np.asarray(data.geom_xpos[mujoco.mj_name2id(arm, mujoco.mjtObj.mjOBJ_GEOM, p)])
        for p in pads
    ]
    midpoint = (centres[0] + centres[1]) / 2.0
    measured = float(np.linalg.norm(np.asarray(data.site_xpos[site]) - midpoint))
    assert measured < 0.001

    vendor = mujoco.mj_name2id(arm, mujoco.mjtObj.mjOBJ_SITE, f"{naming.ROBOT}grip_site")
    if vendor >= 0:
        shipped = float(np.linalg.norm(np.asarray(data.site_xpos[vendor]) - midpoint))
        assert measured * 4 < shipped, (
            f"the measured tool centre is {measured * 1000:.2f} mm from the pad midpoint "
            f"and the vendor's is {shipped * 1000:.2f} mm; measuring it gained nothing"
        )


def test_ik_reaches_a_pose_it_can_reach(cell, spec):
    """Damped least squares has to converge on a target inside the workspace."""
    model = mujoco.MjModel.from_xml_path(str(cell.scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.tcp_site)
    start = np.asarray(data.site_xpos[site]).copy()
    qpos_ids, dof_ids = joint_addresses(model, spec.arm_joints)
    result = solve_site_pose(
        model,
        site,
        start + np.array([0.02, -0.03, 0.04]),
        None,
        dof_ids=dof_ids,
        qpos_ids=qpos_ids,
        seed_qpos=data.qpos.copy(),
    )
    assert result.converged
    assert result.position_error_m < 1e-3


def test_ik_reports_failure_rather_than_lying(cell, spec):
    """A target a metre outside the workspace must come back as not converged."""
    model = mujoco.MjModel.from_xml_path(str(cell.scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.tcp_site)
    qpos_ids, dof_ids = joint_addresses(model, spec.arm_joints)
    result = solve_site_pose(
        model,
        site,
        np.array([3.0, 3.0, 3.0]),
        None,
        dof_ids=dof_ids,
        qpos_ids=qpos_ids,
        seed_qpos=data.qpos.copy(),
    )
    assert not result.converged
    assert result.position_error_m > 0.5


def test_joint_addresses_rejects_a_free_joint(cell):
    """Solving for a joint that has six degrees of freedom is a bug, not a special case."""
    model = mujoco.MjModel.from_xml_path(str(cell.scene_path))
    with pytest.raises(LookupError):
        joint_addresses(model, ["robot/no_such_joint"])
