"""Generating the judging policy from the workcell and the plan.

`sim_judge` deliberately hard-codes no task thresholds: what counts as a forbidden contact,
how much clearance must be kept, and when a part is properly seated all come from
`bound-operation.json`. Writing that file by hand for every scene is how the previous
version of this pipeline ended up with a two-thousand-line policy tied to one protocol.

Here it is derived. The naming convention in `amx.naming` says which namespace every geom
belongs to, the plan says which steps the gripper is closed during, and the destination says
where the part is supposed to end up — that is enough to generate the rules mechanically.
Every rule carries a `repair_target`, which is what lets `amx.loop` route a failure to the
thing that has to change without interpreting the finding's prose.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from amx import naming
from amx.report import RepairTarget
from amx.sim.gripper import find_pad_geoms
from amx.sim.plan import OperationPlan
from amx.sim.scene import BuiltScene

# Two arm links brushing the bench is a collision; a fingertip resting on a part is not.
BENCH_CONTACT_PENETRATION_M = 0.0005
GRASP_CRUSH_FRACTION = 0.2
"""How far into a part a grasp may press, as a fraction of the part's own width.

A fraction rather than a distance, because a distance cannot mean the same thing twice. Two
millimetres is a fifth of the way through a microcentrifuge tube and a twentieth of the way
through a 50 mL conical; whatever value is chosen, it is either lenient for one or absurd
for the other, and it has to be re-picked for every part that enters the cell.

A fifth of the width is a statement about the part instead: thin-walled labware squeezed
that far has stopped being round. It also survives the fact that a rigid-body solver always
sinks further than commanded — the sink scales with the interference, which scales with the
part — so the rule stays a statement about crushing rather than about MuJoCo.
"""

GRASP_PENETRATION_FLOOR_M = 0.0005
"""A lower bound for the above, so a very small part does not get an unmeasurably tight rule."""

ASSET_CLEARANCE_M = 0.002
JOINT_LIMIT_TOLERANCE_RAD = 0.02
MIN_MOVABLE_MASS_KG = 1e-6
MIN_MOVABLE_INERTIA = 1e-12


def build_policy(
    built: BuiltScene,
    plan: OperationPlan,
    model: mujoco.MjModel,
    *,
    task_id: str,
) -> dict[str, Any]:
    """Produce the `bound-operation.json` document for one case."""
    grip_steps = plan.grip_step_ids() or ["*"]
    fingerpads = _finger_geoms(model)
    other_links = _non_gripping_geoms(model)
    manipulated_body = _manipulated_body(built, plan, model)

    document: dict[str, Any] = {
        "task_id": task_id,
        "schema_version": "1.0",
        "generated_by": "amx.sim.policy",
        "runtime_feedback": {
            "enabled": True,
            "robot_geom_prefix": naming.ROBOT,
            "tool_body_prefix": naming.TOOL,
            "device_geom_prefix": naming.ASSET_PREFIX,
            "environment_geom_patterns": [f"{naming.BENCH}*"],
            "forbid_unlisted_robot_device_contacts": True,
            "forbid_unlisted_robot_environment_contacts": True,
            "forbid_unlisted_robot_tool_contacts": False,
            "event_gap_steps": 2,
            "clearance_sample_stride_steps": 5,
            "fail_on_engine_warning": True,
            "contact_rules": _contact_rules(
                built, plan, grip_steps, fingerpads, other_links, model
            ),
            "clearance_rules": _clearance_rules(built, plan),
        },
        "scene_semantics": {
            "workcell": {
                "bench_top_geom": f"{naming.BENCH}top",
                "supported_bodies": _supported_bodies(built, plan),
            },
            "physics_completeness": {
                "minimum_movable_body_mass_kg": MIN_MOVABLE_MASS_KG,
                "minimum_movable_body_inertia_kg_m2": MIN_MOVABLE_INERTIA,
                "physical_entities": _physical_entities(built),
                # The last pattern is the co-designed parts' display mesh, which is named
                # plainly `visual` and carries `contype="0"`. The judge measures overlap
                # geometrically, without consulting contype, so an unexempted display mesh
                # reports the socket it is a picture of as interpenetrating whatever is
                # correctly seated inside it.
                "visual_exemptions": [
                    {"geom_patterns": ["*__visual_*", "*_visual", "*_vis", "*/visual"]}
                ],
            },
        },
        "device": {
            "allowed_robot_device_body_pairs": _grasp_body_pairs(built, fingerpads, model),
        },
        "joint_limit_tolerance": JOINT_LIMIT_TOLERANCE_RAD,
        "release_phase_names": plan.release_step_ids(),
        "retention_phase_names": grip_steps if grip_steps != ["*"] else [],
        "final_phase_name": plan.actions[-1].step_id,
    }

    if plan.destination is not None and manipulated_body:
        document["labware"] = _labware(built, plan, manipulated_body, model)
        receiver = _receiver(plan, manipulated_body, model)
        if receiver is not None:
            document["scene_semantics"]["receiver"] = receiver

    tool_mount = _tool_mount(built, model)
    if tool_mount is not None:
        document["tool_mount_integrity"] = tool_mount

    return document


def geometry_bindings(model: mujoco.MjModel) -> list[dict[str, str]]:
    """Map every named geom to the domain entity that owns it.

    The judge uses this to say "the tube rack" rather than "geom 47" in a finding, and it
    falls straight out of the naming convention.
    """
    bindings: list[dict[str, str]] = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if not name:
            continue
        owner = naming.owner_of(name)
        if not owner:
            continue
        bindings.append({"geom_name": name, "entity_id": f"entity.{owner.rstrip('/.')}"})
    return bindings


def _contact_rules(
    built: BuiltScene,
    plan: OperationPlan,
    grip_steps: list[str],
    fingerpads: list[str],
    other_links: list[str],
    model: mujoco.MjModel,
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []

    rules.append(
        {
            "rule_id": "arm-vs-bench",
            "disposition": "forbidden",
            "geom_a": f"{naming.ROBOT}*",
            "geom_b": f"{naming.BENCH}top",
            "phases": ["*"],
            "maximum_penetration_m": BENCH_CONTACT_PENETRATION_M,
            "message_en": (
                "An arm link touched the bench top. The trajectory passes too close to the "
                "work surface; raise the affected waypoint or approach from above."
            ),
            "repair_target": RepairTarget.TRAJECTORY.value,
        }
    )

    for asset_id in built.asset_namespaces:
        namespace = built.asset_namespaces[asset_id]
        crush = _crush_allowance(namespace, model)
        for pad in fingerpads:
            rules.append(
                {
                    "rule_id": f"grasp-{asset_id}-{_short(pad)}",
                    "disposition": "allowed",
                    "geom_a": pad,
                    "geom_b": f"{namespace}*",
                    "phases": grip_steps,
                    "maximum_penetration_m": crush,
                    "message_en": (
                        f"The gripper pressed into {asset_id} further than a grasp should. "
                        "Either the commanded width is narrower than the part can take, or the "
                        "fingers are closing on a face they cannot grip."
                    ),
                    "repair_target": RepairTarget.TRAJECTORY.value,
                }
            )
        # Named link by link rather than as `robot/*`, because the judge treats a
        # prohibition as a hard constraint that overrides any allowance: a rule forbidding
        # `robot/*` from touching the asset also forbids the fingers, and then every
        # successful grasp is reported as the arm colliding with what it just picked up.
        # What this rule means is "with anything but the hand", and that has to be spelt out.
        for link in other_links:
            rules.append(
                {
                    "rule_id": f"arm-vs-{asset_id}-{_short(link)}",
                    "disposition": "forbidden",
                    "geom_a": link,
                    "geom_b": f"{namespace}*",
                    "phases": ["*"],
                    "maximum_penetration_m": BENCH_CONTACT_PENETRATION_M,
                    "message_en": (
                        f"{link} struck {asset_id}. Only the gripper's fingers are meant to "
                        "touch it; move the waypoint or change the approach axis."
                    ),
                    "repair_target": RepairTarget.TRAJECTORY.value,
                }
            )

    for part_id, namespace in built.fixture_namespaces.items():
        placement = built.workcell.fixture(part_id)
        target = RepairTarget.TOOL if placement.mount == "tool" else RepairTarget.FIXTURE
        rules.append(
            {
                "rule_id": f"fixture-{part_id}-vs-bench",
                "disposition": "allowed" if placement.mount == "bench" else "forbidden",
                "geom_a": f"{namespace}*",
                "geom_b": f"{naming.BENCH}top",
                "phases": ["*"],
                "maximum_penetration_m": BENCH_CONTACT_PENETRATION_M,
                "message_en": (
                    f"The co-designed part {part_id} is interfering with the bench. A "
                    "bench-mounted part should rest on the surface, not sink into it; an "
                    "arm-mounted one should never reach it."
                ),
                "repair_target": target.value,
            }
        )
        for asset_id, asset_namespace in built.asset_namespaces.items():
            rules.append(
                {
                    "rule_id": f"fixture-{part_id}-vs-{asset_id}",
                    "disposition": "allowed",
                    "geom_a": f"{namespace}*",
                    "geom_b": f"{asset_namespace}*",
                    "phases": ["*"],
                    "maximum_penetration_m": _crush_allowance(asset_namespace, model),
                    "message_en": (
                        f"{part_id} is penetrating {asset_id} rather than mating with it. The "
                        "part's clearance is too tight for the interface it was built against."
                    ),
                    "repair_target": target.value,
                }
            )

    assets = list(built.asset_namespaces.items())
    for index, (first_id, first_ns) in enumerate(assets):
        for second_id, second_ns in assets[index + 1 :]:
            rules.append(
                {
                    "rule_id": f"{first_id}-vs-{second_id}",
                    "disposition": "forbidden",
                    "geom_a": f"{first_ns}*",
                    "geom_b": f"{second_ns}*",
                    "phases": ["*"],
                    "maximum_penetration_m": BENCH_CONTACT_PENETRATION_M,
                    "message_en": (
                        f"{first_id} and {second_id} collided. Two instruments are laid out too "
                        "close together, or one was carried into the other."
                    ),
                    "repair_target": RepairTarget.LAYOUT.value,
                }
            )
    return rules


def _clearance_rules(built: BuiltScene, plan: OperationPlan) -> list[dict[str, Any]]:
    """Keep the arm away from instruments it is not currently working on.

    Only generated for assets the plan does not manipulate: the one being handled is
    supposed to be approached and touched, so a clearance floor on it would fail every
    successful episode.
    """
    rules: list[dict[str, Any]] = []
    for asset_id, namespace in built.asset_namespaces.items():
        if asset_id == plan.manipulated_asset:
            continue
        rules.append(
            {
                "rule_id": f"keep-clear-{asset_id}",
                "body_a": f"{naming.ROBOT}*",
                "body_b": f"{namespace}*",
                "minimum_clearance_m": ASSET_CLEARANCE_M,
                "geom_groups": [0, 3],
                "phases": ["*"],
                "message_en": (
                    f"The arm came within {ASSET_CLEARANCE_M * 1000:.1f} mm of {asset_id}, which "
                    "this plan never handles. Route the trajectory around it."
                ),
                "repair_target": RepairTarget.TRAJECTORY.value,
            }
        )
    return rules


def _supported_bodies(built: BuiltScene, plan: OperationPlan) -> list[dict[str, Any]]:
    """Assets welded to the bench are declared as resting on it.

    Free assets are left out: they are supposed to be lifted, so a support requirement on
    them would fail the moment the plan works.
    """
    bodies: list[dict[str, Any]] = []
    for placement in built.workcell.assets:
        if placement.attachment != "fixed":
            continue
        bodies.append(
            {
                "body": f"{placement.namespace}*",
                "component_id": f"entity.{placement.namespace.rstrip('/.')}",
                "support_geom": f"{naming.BENCH}top",
                "maximum_gap_m": 0.002,
                "maximum_penetration_m": 0.001,
            }
        )
    return bodies


def _physical_entities(built: BuiltScene) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for asset_id, namespace in built.asset_namespaces.items():
        entities.append(
            {
                "entity_id": f"entity.{namespace.rstrip('/.')}",
                "body_patterns": [f"{namespace}*"],
                "require_contact_geometry": True,
                "require_mass_inertia_if_movable": True,
            }
        )
    for part_id, namespace in built.fixture_namespaces.items():
        entities.append(
            {
                "entity_id": f"entity.{namespace.rstrip('/.')}",
                "body_patterns": [f"{namespace}*"],
                "require_contact_geometry": True,
                "require_mass_inertia_if_movable": True,
            }
        )
    return entities


def _labware(
    built: BuiltScene, plan: OperationPlan, manipulated_body: str, model: mujoco.MjModel
) -> dict[str, Any]:
    destination = plan.destination
    assert destination is not None
    centre = _site_xy(model, destination.site, built)
    return {
        "body": manipulated_body,
        "contact_geom": f"{built.asset_namespaces[plan.manipulated_asset]}*",
        "free_joint": f"{built.asset_namespaces[plan.manipulated_asset]}root",
        "maximum_tilt_rad": destination.maximum_tilt_rad,
        "maximum_final_speed_m_s": 0.01,
        "receiver_center_xy_m": list(centre),
        "maximum_receiver_xy_error_m": destination.tolerance_xy_m,
        "seated_center_z_range_m": list(destination.seated_z_range_m),
        "minimum_transfer_distance_m": destination.minimum_transfer_distance_m,
    }


def _receiver(
    plan: OperationPlan, manipulated_body: str, model: mujoco.MjModel
) -> dict[str, Any] | None:
    destination = plan.destination
    assert destination is not None
    if not destination.socket_floor_geom:
        return None
    return {
        "interface_id": f"interface.{plan.manipulated_asset}.seated",
        "labware_body": manipulated_body,
        "labware_geom": f"{naming.asset_namespace(plan.manipulated_asset)}*",
        "socket_site": destination.site,
        "socket_floor_geom": destination.socket_floor_geom,
        "socket_wall_geoms": destination.socket_wall_geoms,
        "inner_radius_m": destination.inner_radius_m,
        "minimum_side_clearance_m": 0.0002,
        "maximum_side_clearance_m": destination.inner_radius_m,
        "minimum_insertion_depth_m": destination.minimum_insertion_depth_m,
        "maximum_axis_tilt_rad": destination.maximum_tilt_rad,
        "maximum_interface_alignment_error_m": destination.tolerance_xy_m,
        "require_floor_contact": destination.minimum_insertion_depth_m > 0.0,
    }


def _tool_mount(built: BuiltScene, model: mujoco.MjModel) -> dict[str, Any] | None:
    """Constrain the integrity of arm-mounted tooling, when there is any."""
    tool_parts = [p for p in built.workcell.fixtures if p.mount == "tool"]
    if not tool_parts:
        return None
    part = tool_parts[0]
    return {
        "mount_body": built.robot.flange_body,
        "tool_root_body": f"{part.namespace}*",
        "tool_geom_prefix": part.namespace,
        "maximum_self_penetration_m": 0.001,
        "mount_geom_pairs": [],
        "clearance_geom_pairs": [],
        "allowed_self_collision_geom_pairs": [],
    }


def _manipulated_body(
    built: BuiltScene, plan: OperationPlan, model: mujoco.MjModel
) -> str:
    if not plan.manipulated_asset:
        return ""
    if plan.manipulated_body:
        return plan.manipulated_body
    namespace = built.asset_namespaces.get(plan.manipulated_asset)
    if namespace is None:
        raise KeyError(
            f"the plan manipulates {plan.manipulated_asset!r}, which is not in the workcell"
        )
    # The asset's root body is the one whose parent is the world.
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name.startswith(namespace) and model.body_parentid[body] == 0:
            return name
    return f"{namespace}*"


def _site_xy(model: mujoco.MjModel, site: str, built: BuiltScene) -> tuple[float, float]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if site_id < 0:
        raise KeyError(f"the destination names site {site!r}, which the scene does not define")
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    position = data.site_xpos[site_id]
    return (float(position[0]), float(position[1]))


def _crush_allowance(namespace: str, model: mujoco.MjModel) -> float:
    """`GRASP_CRUSH_FRACTION` of how wide the part is across its narrowest axis.

    The narrowest axis, because that is the one something closes on: a tube standing on a
    bench is gripped across its diameter, not along its length, and scaling the rule by the
    length would let a gripper press 8 mm into a 10 mm tube.

    Measured from the collision geometry rather than from a declared datasheet, so it also
    covers parts the caller never described — and it costs nothing, because the compiled
    model is already here.
    """
    extents = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if not name or not name.startswith(namespace) or int(model.geom_contype[geom]) == 0:
            continue
        extents.append(2.0 * np.asarray(model.geom_aabb[geom][3:], dtype=float))
    if not extents:
        return GRASP_PENETRATION_FLOOR_M
    span = np.max(np.asarray(extents), axis=0)
    return max(GRASP_CRUSH_FRACTION * float(span.min()), GRASP_PENETRATION_FLOOR_M)


def _grasp_body_pairs(
    built: BuiltScene, fingerpads: list[str], model: mujoco.MjModel
) -> list[list[str]]:
    """The robot-to-part body pairs the judge should not treat as a stray collision.

    A separate statement from the contact rules, and it has to be made separately: the
    rules govern how *deep* a permitted contact may go, while this governs whether the
    contact is permitted at all. A policy with grasp rules and no entry here reports every
    finger on every part as an unlisted robot-device contact — 8 findings for one clean
    pick, all of them saying the gripper gripped something.

    Stated per finger body against each part as a whole, since which of a part's bodies a
    finger lands on is not something a plan can or should predict.
    """
    bodies = sorted(
        {
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pad)]),
            )
            or ""
            for pad in fingerpads
        }
        - {""}
    )
    return [
        [body, f"{namespace}*"]
        for namespace in built.asset_namespaces.values()
        for body in bodies
    ]


def _non_gripping_geoms(model: mujoco.MjModel) -> list[str]:
    """Every collidable robot geom that is not part of the gripping surface.

    The complement of `_finger_geoms`, so the two together say exactly one thing: a part may
    be touched by the faces that grip it and by nothing else. The knuckles and the gripper
    body are on the near side of that line, which is intended — a knuckle against a tube is
    the arm leaning on it, not a grasp.
    """
    gripping = set(_finger_geoms(model))
    return [
        name
        for geom in range(model.ngeom)
        if int(model.geom_contype[geom]) != 0
        and (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
        and name.startswith(naming.ROBOT)
        and name not in gripping
    ]


def _finger_geoms(model: mujoco.MjModel) -> list[str]:
    """The gripper geoms that are meant to touch the part being grasped.

    Everything collidable on the bodies that carry the pads, rather than the pads alone.
    The two differ, and the difference matters: a Robotiq 85's finger carries a `fingerpad`
    and a `fingertip`, and on something as small as a 10.8 mm tube it is the tip that makes
    contact first. Whitelisting only the pad leaves that contact matching the rule that
    forbids the arm from striking the asset, so every successful grasp reports itself as a
    collision and the loop is handed a failure it cannot act on.
    """
    try:
        pads = find_pad_geoms(model, naming.ROBOT)
    except LookupError:
        return [f"{naming.ROBOT}*finger*"]

    bodies = {
        int(model.geom_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pad)])
        for pad in pads
    }
    return [
        name
        for geom in range(model.ngeom)
        if int(model.geom_bodyid[geom]) in bodies
        and int(model.geom_contype[geom]) != 0
        and (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
    ]


def _short(name: str) -> str:
    return name.rsplit("/", 1)[-1].replace("_collision", "")
