"""The generated policy, the objective checks, and the loop's bookkeeping.

These are the tests that guard against the worst failure this system can have, which is not
a crash but a false PASS: a run that reports success while the tube never left the bench.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from amx import naming
from amx.codesign import TEMPLATES, export_part
from amx.geometry import Pose
from amx.report import RepairTarget, Severity
from amx.sim.outcome import check_destination
from amx.sim.plan import Destination, Grip, Hold, Move, OperationPlan, PoseTarget
from amx.sim.policy import build_policy, geometry_bindings
from amx.sim.run import BodyPose, EpisodeResult
from amx.sim.scene import (
    Bench,
    FixturePlacement,
    Placement,
    RobotRef,
    Workcell,
    build_scene,
)

TOOL_DOWN = (3.14159, 0.0, 0.0)


def _plan(**destination) -> OperationPlan:
    """A pick-and-place with the shape the policy generator expects: grip, carry, release."""
    return OperationPlan(
        plan_id="probe",
        protocol_id="probe",
        manipulated_asset="tube",
        actions=[
            Move(step_id="step.p01.approach", duration_s=0.5,
                 target=PoseTarget(pos=(0.3, 0.1, 0.3), euler=TOOL_DOWN)),
            Grip(step_id="step.p02.grasp", duration_s=0.2, width_m=0.010),
            Move(step_id="step.p03.lift", duration_s=0.5,
                 target=PoseTarget(pos=(0.3, 0.1, 0.4), euler=TOOL_DOWN)),
            Grip(step_id="step.p04.release", duration_s=0.2, width_m=0.030),
            Hold(step_id="step.p04.release", duration_s=0.2),
        ],
        destination=Destination(site="fixture.rack/well.r0c0", **destination)
        if destination
        else None,
    )


@pytest.fixture(scope="module")
def built(robot_dir, tmp_path_factory):
    """A cell with a rack and a tube-sized asset, so the policy has both to write about."""
    root = tmp_path_factory.mktemp("judged")
    rack = TEMPLATES["tube_rack"](part_id="rack", rows=1, columns=1)
    export_part(rack, root / "rack")

    # A standalone MJCF standing in for a compiled Articraft asset: the policy only reads
    # names and geometry, so a plain cylinder exercises it exactly as a real tube does.
    asset = root / "tube"
    (asset / "meshes").mkdir(parents=True)
    (asset / "asset.xml").write_text(
        """<mujoco model="tube">
  <worldbody>
    <body name="root_body">
      <freejoint name="root"/>
      <geom name="barrel" type="cylinder" size="0.0054 0.018" pos="0 0 0.018" mass="0.0013"/>
    </body>
  </worldbody>
</mujoco>"""
    )

    workcell = Workcell(
        workcell_id="judged",
        robot=RobotRef(model=robot_dir.name),
        bench=Bench(size_xy=(1.0, 0.8)),
        assets=[Placement(asset_id="tube", source=asset, pose=Pose(pos=(0.3, 0.1, 0.05)))],
        fixtures=[
            FixturePlacement(
                part_id="rack", source=root / "rack", mount="bench",
                pose=Pose(pos=(0.3, 0.1, 0.0)),
            )
        ],
    )
    return build_scene(workcell, root / "scene")


@pytest.fixture(scope="module")
def model(built):
    return mujoco.MjModel.from_xml_path(str(built.scene_path))


def test_sim_judge_accepts_the_generated_policy(built, model, tmp_path):
    """The generated document has to survive `sim_judge`'s own loader, not just look right.

    `load_policy` is forgiving — missing sections silently do nothing — which means a policy
    can be wrong in a way that produces a clean PASS. Loading it and reading the parsed
    result back is the only way to know the rules arrived.
    """
    from sim_judge.loader.policy import load_policy

    document = build_policy(built, _plan(), model, task_id="probe")
    # Round-tripped through JSON first, since that is how the judge receives it and it is
    # where a stray numpy scalar or Path would surface.
    path = tmp_path / "bound-operation.json"
    path.write_text(json.dumps(document, indent=2))

    policy = load_policy(json.loads(path.read_text()))
    assert policy.enabled
    assert policy.contact_rules, "no contact rule survived loading"
    assert policy.unlisted.allowed_robot_device_body_pairs, "the grasp allowlist is empty"
    assert policy.bench_top_geom == f"{naming.BENCH}top"
    assert policy.physical_entities, "no entity is subject to the completeness checks"
    assert policy.release_phase_names, "the judge cannot tell when the part was let go"


def test_the_fingers_are_allowed_and_the_rest_of_the_arm_is_not(built, model):
    """The two halves of one statement: pads may touch the part, links may not.

    Getting this wrong is silent in both directions. Forbid `robot/*` and every good grasp
    reports as a collision; allow it and an elbow through the labware goes unnoticed.
    """
    rules = build_policy(built, _plan(), model, task_id="probe")["runtime_feedback"][
        "contact_rules"
    ]
    allowed = {r["geom_a"] for r in rules if r["disposition"] == "allowed"}
    forbidden = {r["geom_a"] for r in rules if r["disposition"] == "forbidden"}

    assert any("fingerpad" in name for name in allowed)
    assert not any("fingerpad" in name for name in forbidden)
    assert f"{naming.ROBOT}forearm_col" in forbidden
    assert f"{naming.ROBOT}*" not in forbidden or all(
        naming.BENCH in r["geom_b"] for r in rules
        if r["disposition"] == "forbidden" and r["geom_a"] == f"{naming.ROBOT}*"
    ), "a blanket robot prohibition would override the grasp allowance"


def test_grasp_allowance_scales_with_the_part(built, model):
    """A penetration limit has to be relative, or it means different things on different parts."""
    rules = build_policy(built, _plan(), model, task_id="probe")["runtime_feedback"][
        "contact_rules"
    ]
    grasp = [r for r in rules if r["rule_id"].startswith("grasp-")]
    assert grasp
    limit = grasp[0]["maximum_penetration_m"]
    # The stand-in tube is 10.8 mm across, so a fifth of it is a shade over 2 mm.
    assert 0.0015 < limit < 0.003
    for rule in grasp:
        assert rule["maximum_penetration_m"] == limit


def test_every_geom_is_bound_to_an_entity(model):
    """A finding about `geom 47` is not actionable, so the binding table has to be complete."""
    bindings = {b["geom_name"] for b in geometry_bindings(model)}
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if name and name != "world":
            assert name in bindings, f"{name} belongs to no entity"


def _episode(start, final, *, diverged=None) -> EpisodeResult:
    return EpisodeResult(
        case_dir=None,
        scene_path=None,
        step_count=100,
        duration_s=0.2,
        spans=[],
        subject_start=BodyPose(position=np.asarray(start, float), quaternion=np.array([1, 0, 0, 0.0])),
        subject_final=BodyPose(position=np.asarray(final, float), quaternion=np.array([1, 0, 0, 0.0])),
        destination_xy=(0.3, 0.1),
        diverged_at_step=diverged,
    )


def test_a_part_that_never_moved_is_a_failure():
    """The check that exists because the judge once passed a run where nothing happened.

    `sim_judge` only evaluates the objective when the policy declared a socket to evaluate
    against; with no socket it reports a clean PASS on an episode in which the tube stood
    still for fifteen seconds. Whatever else changes, this must stay a failure.
    """
    plan = _plan(
        tolerance_xy_m=0.004,
        seated_z_range_m=(0.019, 0.033),
        minimum_transfer_distance_m=0.05,
    )
    findings = check_destination(plan, _episode((0.0, 0.0, 0.021), (0.0, 0.0, 0.021)))
    failures = [f for f in findings if f.severity is Severity.FAILURE]
    assert any(f.code == "O-TRANSFER" for f in failures)


def test_a_diverged_run_is_not_measured():
    """After a divergence MuJoCo resets the state, so the final pose is the initial pose.

    Measuring it reports that the part never moved — plausible, specific and false. The
    only honest answer is that the outcome is unknown, and that has to be a failure rather
    than a warning, or a diverged run can pass.
    """
    plan = _plan(tolerance_xy_m=0.004, seated_z_range_m=(0.019, 0.033))
    findings = check_destination(
        plan, _episode((0.0, 0.0, 0.021), (0.0, 0.0, 0.021), diverged=4000)
    )
    codes = {f.code for f in findings if f.severity is Severity.FAILURE}
    assert codes == {"O-DIVERGED"}


def test_a_completed_transfer_passes():
    """The other direction: a part that arrived must not be reported as a failure."""
    plan = _plan(
        tolerance_xy_m=0.004,
        seated_z_range_m=(0.019, 0.033),
        minimum_transfer_distance_m=0.05,
        maximum_tilt_rad=0.2,
    )
    findings = check_destination(
        plan, _episode((0.0, 0.0, 0.021), (0.3005, 0.1002, 0.0213))
    )
    assert findings
    assert not [f for f in findings if f.severity is Severity.FAILURE]


def test_no_destination_means_no_opinion():
    """A plan is allowed to be an exploratory poke, and then there is nothing to check."""
    assert check_destination(_plan(), _episode((0, 0, 0), (0, 0, 0))) == []


def test_findings_carry_a_repair_target():
    """The loop routes on this field, so a finding without one cannot be acted upon."""
    plan = _plan(tolerance_xy_m=0.004, seated_z_range_m=(0.019, 0.033))
    findings = check_destination(plan, _episode((0, 0, 0.021), (0.4, 0.3, 0.002)))
    for finding in findings:
        assert isinstance(finding.repair_target, RepairTarget)
