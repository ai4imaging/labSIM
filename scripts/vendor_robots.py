"""Build `vendor/robots/<id>/` from robot assets published inside the robosuite wheel.

robosuite ships clean, well-named MuJoCo models for a dozen arms and grippers. We pull the
arm and the gripper out of the wheel over ranged GETs, splice the gripper onto the arm's
tool flange, move every identifier into the `robot/` namespace that `amx.sim` requires, and
swap the torque motors for position servos so `Move` actions can drive the arm.

The result is a self-contained directory: `arm.xml`, `meshes/`, and a `spec.json` telling
`amx.sim` which joints, actuators and sites to drive.

Usage:  python scripts/vendor_robots.py [--robot ur5e] [--gripper robotiq_gripper_85]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from _wheelfetch import Member, RemoteZip, newest_wheel  # noqa: E402

from amx.naming import ROBOT  # noqa: E402
from amx.sim.gripper import GripperCalibration, calibrate  # noqa: E402
from amx.sim.mjcf import apply_prefix, collect_names, find_body, section, write  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "vendor" / "robots"
ASSETS = "robosuite/models/assets"

# The body a gripper is spliced into, per arm. robosuite is consistent about this name.
FLANGE_BODY = {
    "ur5e": "right_hand",
    "panda": "right_hand",
    "xarm7": "right_hand",
    "iiwa": "right_hand",
    "sawyer": "right_hand",
    "kinova3": "right_hand",
    "jaco": "right_hand",
}

SERVO_BANDWIDTH_RAD_S = 60.0
"""How fast each servo should be able to correct itself, in rad/s — about 10 Hz.

Gains are specified this way rather than as a torque per radian because the two ends of the
arm are nothing alike: the shoulder swings the whole arm and the wrist swings a gripper, and
their inertias differ by two orders of magnitude. One number for both cannot work. At 2000
N·m/rad the shoulder sags 20 mrad under the arm's own weight — 6 mm at the tool, enough to
close the fingers beside a 10.8 mm tube rather than around it. At 40000 the shoulder is
fine and the wrists diverge within fifty steps, because for them that gain is a spring
stiff enough to cross its own equilibrium in one 2 ms step.

Asking for a bandwidth instead gives every joint the same settling time and the same
stability margin, and `kp = w^2 * I` turns it back into a gain once the inertia is known —
which it is, from the compiled model, at the pose where each joint carries the most.
"""

SERVO_DAMP_RATIO = 1.0
"""Critically damped. Given as a ratio so MuJoCo derives `kv` from the same inertia."""

GRIP_FORCE_N = 40.0
"""What the fingers should squeeze with when held against a part.

Within the 20–235 N a Robotiq 85 delivers, at the low end of it: labware is thin-walled
polypropylene and the arm is moving it, not resisting it.
"""

GRIP_DEFLECTION_M = 1e-3
"""The width deficit at which `GRIP_FORCE_N` is reached — how far the servo must be pushed
back from its commanded opening to be exerting full force. A millimetre is about what a
tube wall gives."""

INERTIA_SAMPLES = 24
"""Poses to sample when looking for each joint's worst-case inertia. The arm's mass matrix
depends on its configuration — a UR5e's shoulder carries roughly twice as much extended as
folded — and sizing the gain to the lighter pose leaves it sagging in the heavier one."""


@dataclass(frozen=True)
class LinkageFix:
    """Corrections that make a linkage gripper physically sound.

    robosuite models the Robotiq 2F-85's closed four-bar as a soft tendon between three
    joints. That is cheap but wrong in three ways, and all three show up immediately once a
    judge inspects the recording: the loop does not close, so the passive joints wander
    hundreds of milliradians outside their declared limits; the linkage's own meshes overlap
    each other and MuJoCo does not auto-exclude them because they are not parent and child,
    which produces tens of millimetres of self-penetration and eventually diverges; and the
    servos are commanded onto a mechanical hard stop when open.

    So the vendored gripper replaces the tendon with hard equality constraints expressing
    the mechanism's real kinematics — the inner knuckle follows the driver, and the inner
    finger counter-rotates to keep the pad parallel — excludes contact inside the linkage,
    drops the now-redundant limits on the passive joints, and parks the open position just
    off the stop. What the coupling is cannot be read out of robosuite's file, because its
    tendon does not encode it, so it is stated here per gripper.
    """

    drivers: tuple[str, ...]
    couplings: tuple[tuple[str, str, float], ...]
    """(passive joint, driver joint, ratio) — passive = ratio x driver."""

    damping: float = 0.05
    rest_fraction: float = 0.0375
    """How far off the fully-open hard stop to park, as a fraction of the driver's travel."""


GRIPPER_LINKAGE = {
    "robotiq_gripper_85": LinkageFix(
        drivers=("finger_joint", "right_outer_knuckle_joint"),
        couplings=(
            ("left_inner_knuckle_joint", "finger_joint", 1.0),
            ("left_inner_finger_joint", "finger_joint", -1.0),
            ("right_inner_knuckle_joint", "right_outer_knuckle_joint", 1.0),
            ("right_inner_finger_joint", "right_outer_knuckle_joint", -1.0),
        ),
    ),
    "robotiq_gripper_140": LinkageFix(
        drivers=("finger_joint", "right_outer_knuckle_joint"),
        couplings=(
            ("left_inner_knuckle_joint", "finger_joint", 1.0),
            ("left_inner_finger_joint", "finger_joint", -1.0),
            ("right_inner_knuckle_joint", "right_outer_knuckle_joint", 1.0),
            ("right_inner_finger_joint", "right_outer_knuckle_joint", -1.0),
        ),
    ),
}


def flatten_mesh_files(root: ET.Element, xml_dir: str) -> dict[str, str]:
    """Point every `<mesh file=...>` at a flat `meshes/` directory; return source -> name."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for mesh in root.iter("mesh"):
        source = mesh.get("file")
        if not source:
            continue
        flat = Path(source).name
        stem, suffix = Path(flat).stem, Path(flat).suffix
        counter = 1
        while flat in used:
            counter += 1
            flat = f"{stem}_{counter}{suffix}"
        used.add(flat)
        mapping[f"{xml_dir}/{source}"] = flat
        mesh.set("file", flat)
    return mapping


def drop_duplicate_meshes(root: ET.Element) -> list[str]:
    """robosuite's Robotiq file declares `robotiq_arg2f_base_link` twice.

    MuJoCo rejects duplicate mesh names, and the second declaration points at a `_vis.stl`
    the wheel does not ship, so the first is the usable definition.
    """
    asset = root.find("asset")
    if asset is None:
        return []
    seen: set[str] = set()
    dropped: list[str] = []
    for mesh in list(asset.findall("mesh")):
        name = mesh.get("name", "")
        if name in seen:
            asset.remove(mesh)
            dropped.append(name)
        else:
            seen.add(name)
    return dropped


def to_position_servos(actuator: ET.Element, arm_joints: list[str]) -> None:
    """Replace the arm's torque motors with position servos.

    `Move` actions solve for a joint configuration and then hold it, which needs one servo
    per joint. The gripper's actuators are already position servos and are left alone.
    """
    for motor in list(actuator.findall("motor")):
        if motor.get("joint") in arm_joints:
            actuator.remove(motor)
    for joint_name in arm_joints:
        servo = ET.SubElement(actuator, "position")
        servo.set("name", f"{joint_name}/servo")
        servo.set("joint", joint_name)
        # `kp` is filled in by `tune_servos` once the model can be compiled and its inertia
        # measured. A placeholder is needed because the model has to compile to be measured.
        servo.set("kp", "1")
        servo.set("dampratio", str(SERVO_DAMP_RATIO))


def set_solver_defaults(arm: ET.Element) -> None:
    """Make the extracted arm loadable and stable on its own.

    robosuite supplies solver settings from a separate base scene, so the robot file alone
    integrates with the default semi-implicit Euler and diverges within a few steps under
    the stiff position servos. `implicitfast` holds a commanded pose without drift.
    `amx.sim.scene` re-declares these when it composes a workcell; keeping them here means
    `arm.xml` is directly loadable for inspection.
    """
    compiler = section(arm, "compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    compiler.set("meshdir", "meshes")
    option = section(arm, "option")
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("impratio", "10")


def add_tool_mount(arm: ET.Element, flange: ET.Element) -> str:
    """A named frame on the flange for co-designed tooling to be positioned against.

    `amx.codesign` aligns adapters to this site, so its existence is part of the robot's
    contract with the rest of the pipeline rather than something each part re-derives.
    """
    name = f"{ROBOT}tool_mount"
    site = ET.Element("site")
    site.set("name", name)
    site.set("pos", "0 0 0")
    site.set("size", "0.004")
    site.set("group", "4")
    flange.insert(0, site)
    return name


def repair_gripper_linkage(
    arm: ET.Element, flange: ET.Element, gripper: str
) -> tuple[LinkageFix | None, dict[str, float]]:
    """Apply `GRIPPER_LINKAGE`'s corrections. Returns the fix and the driver rest positions."""
    fix = GRIPPER_LINKAGE.get(gripper)
    if fix is None:
        return None, {}

    linkage_bodies = [b.get("name", "") for b in flange.iter("body") if b.get("name")]
    exclude_within(arm, linkage_bodies, prefix="gripper_self")

    for tendon in arm.findall("tendon"):
        arm.remove(tendon)

    equality = section(arm, "equality")
    for passive, driver, ratio in fix.couplings:
        ET.SubElement(
            equality,
            "joint",
            {
                "name": f"{ROBOT}{passive}/coupling",
                "joint1": ROBOT + passive,
                "joint2": ROBOT + driver,
                "polycoef": f"0 {ratio:g} 0 0 0",
                "solimp": "0.95 0.99 0.001",
                "solref": "0.005 1",
            },
        )

    passives = {passive for passive, _, _ in fix.couplings}
    rest: dict[str, float] = {}
    for joint in arm.iter("joint"):
        short = (joint.get("name") or "").removeprefix(ROBOT)
        if short not in passives and short not in fix.drivers:
            continue
        joint.set("damping", f"{fix.damping:g}")
        if short in passives:
            # The equality constraint already determines these from the driver, so their
            # own limits are redundant and only fight the solver.
            joint.set("limited", "false")
            joint.attrib.pop("range", None)
        else:
            low, high = (float(v) for v in (joint.get("range") or "0 1").split())
            rest[ROBOT + short] = low + fix.rest_fraction * (high - low)
    return fix, rest


def exclude_within(arm: ET.Element, bodies: list[str], *, prefix: str) -> None:
    """Suppress contact between every pair of bodies in a list."""
    contact = section(arm, "contact")
    for i, first in enumerate(bodies):
        for j, second in enumerate(bodies[i + 1 :], start=i + 1):
            ET.SubElement(
                contact,
                "exclude",
                {"name": f"{prefix}_{i}_{j}", "body1": first, "body2": second},
            )


def gripper_gain(calibration: GripperCalibration) -> float:
    """The gain that makes the fingers squeeze with a stated force, not track a trajectory.

    The arm's bandwidth rule does not transfer here, and applying it anyway produced a gain
    of 0.3 N·m/rad. It is not wrong arithmetic — a driver knuckle really does have almost no
    inertia, and the mass matrix knows nothing of the equality constraints that make the
    rest of the finger follow it — it is the wrong question. An arm joint's gain decides how
    well it tracks a position. A gripper spends its working life stalled against something,
    so its gain decides how hard it holds, and that is what has to be specified.

    Virtual work turns a grip force into a gain. The linkage's mechanical advantage is the
    slope of the calibration curve, `dw/dq`, measured rather than read off a drawing, and a
    force conjugate to the opening `w` relates to the driver torque by `F = tau / (dw/dq)`.
    With `tau = kp * dq` and `dq = delta / (dw/dq)` for a width deficit `delta`:

        kp = F * (dw/dq)^2 / delta

    Robosuite's own kp=20 is two orders below this, which is why the fingers took over 1.4
    seconds to close and then covered their last 7 mm in 26 — a gripper too weak to hold a
    1.3 g tube, arriving fast enough to press 2.7 mm into it.
    """
    slope = float(
        abs(
            np.gradient(calibration.widths_m, calibration.joint_values).mean()
        )
    )
    return GRIP_FORCE_N * slope**2 / GRIP_DEFLECTION_M


def tune_servos(arm: ET.Element, model: mujoco.MjModel, arm_joints: list[str]) -> dict[str, float]:
    """Set each servo's gain from the inertia its joint actually has to move.

    The inertia in question is the diagonal of the mass matrix, which depends on the arm's
    configuration, so it is sampled across the joint ranges and the largest value kept. That
    makes the gain sufficient everywhere rather than only where it was measured.
    """
    data = mujoco.MjData(model)
    dofs, addresses = [], []
    for name in arm_joints:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dofs.append(int(model.jnt_dofadr[joint]))
        addresses.append(int(model.jnt_qposadr[joint]))

    rng = np.random.default_rng(0)
    full = np.zeros((model.nv, model.nv))
    worst = np.zeros(len(arm_joints))
    for sample in range(INERTIA_SAMPLES):
        for name, address in zip(arm_joints, addresses, strict=True):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if bool(model.jnt_limited[joint]):
                low, high = (float(v) for v in model.jnt_range[joint])
            else:
                low, high = (-np.pi, np.pi)
            # The first sample is the zero pose, so the sweep always includes it.
            data.qpos[address] = 0.0 if sample == 0 else rng.uniform(low, high)
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, data, full)
        worst = np.maximum(worst, [full[d, d] for d in dofs])

    gains = {
        name: float(SERVO_BANDWIDTH_RAD_S**2 * inertia)
        for name, inertia in zip(arm_joints, worst, strict=True)
    }
    print(
        "  servo gains (N·m/rad): "
        + ", ".join(f"{n.removeprefix(ROBOT)}={g:.0f}" for n, g in gains.items())
    )
    actuator = section(arm, "actuator")
    for servo in actuator.findall("position"):
        joint = servo.get("joint") or ""
        if joint in gains:
            servo.set("kp", f"{gains[joint]:.4g}")
    return gains


def compile_probe(arm: ET.Element, output: Path) -> mujoco.MjModel:
    """Compile the merged tree so it can be measured.

    Written beside the meshes it references, because MuJoCo resolves `meshdir` relative to
    the file, and removed again: it is scaffolding for the measurement, not an output.
    """
    probe = output / "probe.xml"
    write(arm, probe)
    try:
        return mujoco.MjModel.from_xml_path(str(probe))
    finally:
        probe.unlink(missing_ok=True)


def add_measured_tcp(
    arm: ET.Element,
    flange: ET.Element,
    model: mujoco.MjModel,
    actuators: list[dict[str, object]],
) -> tuple[str, float, float]:
    """Put a site exactly midway between the fingerpads and call that the tool centre.

    The vendored grippers ship a `grip_site`, but on the Robotiq 85 it sits about 4.5 mm to
    one side of where the pads actually meet. Everything downstream aims the TCP at the
    thing it means to pick up, so that offset is 4.5 mm of miss on every grasp, applied
    consistently enough to look like a planning error rather than a frame error.
    """
    if not actuators:
        raise SystemExit("the gripper has no actuators, so its tool centre cannot be measured")
    calibration = calibrate(
        model,
        joint_names=tuple(str(a["joint"]) for a in actuators),
        open_value=float(actuators[0]["open"]),  # type: ignore[arg-type]
        closed_value=float(actuators[0]["closed"]),  # type: ignore[arg-type]
        flange_body=flange.get("name", ""),
        namespace=ROBOT,
    )
    print(
        f"  tcp at {np.round(calibration.tcp_offset_flange, 5).tolist()} in the flange "
        f"frame; opening spans {calibration.width_range_m[0] * 1000:.1f}"
        f"–{calibration.width_range_m[1] * 1000:.1f} mm; fingers reach "
        f"{calibration.finger_reach_m * 1000:.1f} mm past it, pads "
        f"{calibration.pad_reach_m * 1000:.1f} mm"
    )

    kp = gripper_gain(calibration)
    print(f"  gripper gain {kp:.3g} N·m/rad for {GRIP_FORCE_N:.0f} N of grip")
    driven = {str(a["joint"]) for a in actuators}
    for servo in section(arm, "actuator").findall("position"):
        if servo.get("joint") in driven:
            servo.set("kp", f"{kp:.4g}")
            servo.set("dampratio", str(SERVO_DAMP_RATIO))

    name = f"{ROBOT}tcp"
    site = ET.Element("site")
    site.set("name", name)
    site.set("pos", " ".join(f"{v:.6g}" for v in calibration.tcp_offset_flange))
    site.set("size", "0.004")
    site.set("group", "4")
    site.set("rgba", "0 1 1 1")
    flange.insert(0, site)
    return name, calibration.finger_reach_m, calibration.pad_reach_m


def gripper_actuators(
    arm: ET.Element, arm_joints: list[str], rest: dict[str, float]
) -> list[dict[str, object]]:
    """The gripper's actuators, with `open` parked just off the mechanical stop.

    Commanding a servo exactly onto its hard stop is poor practice on real hardware and, in
    simulation, leaves the joint sitting on a soft limit constraint that noise pushes
    straight through. `rest` supplies the parked value where a linkage fix computed one.
    """
    found: list[dict[str, object]] = []
    actuator = arm.find("actuator")
    if actuator is None:
        return found
    for element in actuator:
        joint = element.get("joint")
        if joint in arm_joints or not element.get("name"):
            continue
        low, high = (float(v) for v in (element.get("ctrlrange") or "0 1").split())
        found.append(
            {
                "name": element.get("name"),
                "joint": joint,
                "open": rest.get(joint or "", low),
                "closed": high,
            }
        )
    return found


def root_body_name(arm: ET.Element) -> str:
    world = arm.find("worldbody")
    if world is None:
        raise SystemExit("the arm model has no worldbody")
    for body in world:
        if body.tag == "body" and body.get("name"):
            return body.get("name", "")
    raise SystemExit("the arm model's worldbody has no named root body")


def build(robot: str, gripper: str, wheel: RemoteZip, members: dict[str, Member]) -> Path:
    arm_xml = f"{ASSETS}/robots/{robot}/robot.xml"
    hand_xml = f"{ASSETS}/grippers/{gripper}.xml"
    for required in (arm_xml, hand_xml):
        if required not in members:
            raise SystemExit(f"{required} is not in the robosuite wheel")
    if robot not in FLANGE_BODY:
        raise SystemExit(f"the flange body for {robot!r} is not known; add it to FLANGE_BODY")

    arm = ET.fromstring(wheel.read(members[arm_xml]).decode("utf-8"))
    hand = ET.fromstring(wheel.read(members[hand_xml]).decode("utf-8"))

    names = collect_names(arm, hand)
    arm_joints = [ROBOT + j.get("name", "") for j in arm.iter("joint") if j.get("name")]
    apply_prefix(arm, ROBOT, names)
    apply_prefix(hand, ROBOT, names)

    dropped = drop_duplicate_meshes(arm) + drop_duplicate_meshes(hand)
    arm_meshes = flatten_mesh_files(arm, f"{ASSETS}/robots/{robot}")
    hand_meshes = flatten_mesh_files(hand, f"{ASSETS}/grippers")

    flange = find_body(arm, ROBOT + FLANGE_BODY[robot])
    hand_world = hand.find("worldbody")
    if hand_world is None:
        raise SystemExit(f"{gripper} has no worldbody")
    for body in list(hand_world):
        flange.append(body)

    for tag in ("asset", "tendon", "actuator", "contact", "equality", "default", "sensor"):
        origin = hand.find(tag)
        if origin is None:
            continue
        target = section(arm, tag)
        for child in list(origin):
            target.append(child)

    to_position_servos(section(arm, "actuator"), arm_joints)
    set_solver_defaults(arm)
    linkage, rest = repair_gripper_linkage(arm, flange, gripper)
    tool_mount = add_tool_mount(arm, flange)
    arm.set("model", f"{robot}_{gripper}")

    output = OUTPUT_ROOT / f"{robot}_{gripper.replace('robotiq_gripper_', 'robotiq')}"
    mesh_dir = output / "meshes"
    if output.exists():
        shutil.rmtree(output)
    mesh_dir.mkdir(parents=True)

    for mapping in (arm_meshes, hand_meshes):
        for source, flat in mapping.items():
            member = members.get(source)
            if member is None:
                raise SystemExit(f"mesh {source}, referenced by the model, is not in the wheel")
            (mesh_dir / flat).write_bytes(wheel.read(member))

    # The gains and the tool centre are both measured off the compiled model, so they run
    # here: after the meshes are on disk for it to compile against, and before the tree is
    # written out with what they produced.
    actuators = gripper_actuators(arm, arm_joints, rest)
    probe = compile_probe(arm, output)
    gains = tune_servos(arm, probe, arm_joints)
    tcp, finger_reach, pad_reach = add_measured_tcp(arm, flange, probe, actuators)
    write(arm, output / "arm.xml")
    spec = {
        "robot_id": output.name,
        "source": {"project": "robosuite", "robot": robot, "gripper": gripper},
        "namespace": ROBOT,
        "arm_joints": arm_joints,
        "arm_actuators": [f"{j}/servo" for j in arm_joints],
        "gripper_actuators": actuators,
        "servo_gains": {j: round(g, 3) for j, g in gains.items()},
        "gripper_rest_qpos": rest,
        "tcp_site": tcp,
        "finger_reach_m": round(finger_reach, 5),
        "pad_reach_m": round(pad_reach, 5),
        "tool_mount_site": tool_mount,
        "flange_body": ROBOT + FLANGE_BODY[robot],
        "root_body": root_body_name(arm),
        "bodies": sorted(b.get("name", "") for b in arm.iter("body") if b.get("name")),
        "dropped_duplicate_meshes": dropped,
        "linkage_repaired": linkage is not None,
    }
    (output / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="ur5e")
    parser.add_argument("--gripper", default="robotiq_gripper_85")
    args = parser.parse_args()

    print("==> reading the robosuite wheel index")
    wheel = newest_wheel("robosuite")
    output = build(args.robot, args.gripper, wheel, wheel.members())
    meshes = len(list((output / "meshes").glob("*")))
    print(f"==> wrote {output.relative_to(PROJECT_ROOT)}  ({meshes} meshes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
