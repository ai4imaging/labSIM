"""Measuring a gripper, so a grasp can be asked for in millimetres.

A `Grip` action says how wide to hold the fingers, because that is the only thing about a
grasp the caller actually knows: the barrel is 10.8 mm across, so hold at 9 mm and squeeze.
What the gripper needs is a joint angle, and for a Robotiq-style four-bar the two are
related by a linkage nobody wants to re-derive per gripper.

So measure it instead. Sweep the driven joints across their range in a scratch copy of the
model, read the distance between the fingerpads off forward kinematics at each step, and
keep the curve. Inverting it is then a lookup. The same sweep also locates the point midway
between the pads — the only place worth calling the tool centre, and several millimetres
from where the vendored model puts its own `grip_site`.

Nothing here is fitted or tuned. It is a measurement of the model that was loaded, and it
comes out the same every time for that model.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

SWEEP_SAMPLES = 96
"""Steps across the joint range. The width curve is smooth and monotonic, so this is far
more than needed to interpolate it to well under the width of a printed layer."""

PAD_SUBSTRINGS = ("fingerpad", "finger_pad", "finger_tip", "fingertip", "pad")
"""What fingerpads are called, in rough order of how specific the name is. Vendored grippers
disagree, and the alternative is a hand-maintained table per gripper."""


@dataclass(frozen=True)
class GripperCalibration:
    """The measured relationship between a gripper's joint angle and its opening."""

    pad_geoms: tuple[str, str]
    widths_m: np.ndarray
    """Pad separation at each swept joint value, ascending in joint value."""

    joint_values: np.ndarray
    tcp_offset_flange: np.ndarray
    """Midpoint between the pads, in the flange body's frame, with the fingers at rest."""

    finger_reach_m: float
    """How far the fingers extend past the tool centre, along the approach direction.

    The number that decides whether a grasp is geometrically possible at all. To take hold
    of something standing in a socket, the tool centre has to be at least this far above the
    socket's rim, or the fingers are inside the socket wall before the pads are on the part.
    A rack deep enough to hold a tube securely and shallow enough to let this gripper reach
    into it is a real constraint between two parts, and it is only checkable because it is
    measured rather than assumed.
    """

    pad_reach_m: float
    """How far the gripping faces alone extend past the tool centre.

    Slightly less than `finger_reach_m`, and the gap between them is what makes a grasp
    possible at all. The fingers may not enter a socket, but the pads must reach the part —
    so the part has to stand proud of the socket's rim by the difference plus however much
    pad is wanted on it. Reading `finger_reach_m` for both makes every socket look
    unreachable, since it says the pads stop where the fingertips do.
    """

    @property
    def width_range_m(self) -> tuple[float, float]:
        return (float(self.widths_m.min()), float(self.widths_m.max()))

    def joint_for_width(self, width_m: float) -> float:
        """The joint value that opens the pads to `width_m`.

        Widths outside what the gripper can do clamp to the nearest end, which is what a
        real gripper does: ask for less than zero and it simply closes.
        """
        # The sweep runs from open to closed, so width descends as the joint value rises.
        order = np.argsort(self.widths_m)
        return float(np.interp(width_m, self.widths_m[order], self.joint_values[order]))

    def width_for_joint(self, value: float) -> float:
        return float(np.interp(value, self.joint_values, self.widths_m))


def find_pad_geoms(model: mujoco.MjModel, namespace: str) -> tuple[str, str]:
    """The two geoms that do the gripping.

    Looked up by name because the alternative — guessing from the kinematic tree — gets the
    knuckles as often as the pads.
    """
    names = [
        name
        for i in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))
        and name.startswith(namespace)
    ]
    for token in PAD_SUBSTRINGS:
        hits = sorted(n for n in names if token in n)
        if len(hits) == 2:
            return (hits[0], hits[1])
        if len(hits) > 2:
            # More than two matched, so the token was too loose to identify the pads. A
            # narrower one may still work; a wrong pair here would silently mis-measure
            # every grasp, which is worse than not finding them at all.
            continue
    raise LookupError(
        f"cannot find exactly two fingerpad geoms under {namespace!r}: tried "
        f"{PAD_SUBSTRINGS} against {len(names)} candidates. Name the pads so one of those "
        "substrings matches exactly two of them, or the width of a grasp cannot be measured."
    )


def _free_gap(
    model: mujoco.MjModel, data: mujoco.MjData, pad_ids: tuple[int, int]
) -> float:
    """The clear space between the gripping faces, not between the pads' centres.

    A grasp width has to mean the opening something fits through, or the caller is asked to
    know the thickness of the pads to work out whether a 10.8 mm tube goes in.

    Measured between the two inner faces rather than out to the pads' furthest corners. On
    a linkage gripper the pads tilt as they close — 23 deg at grasping range on a Robotiq
    85 — and a tilted pad's corner reaches several millimetres past its face. Taking the
    corner made the gripper look closed a tenth of a radian before it was, so a command to
    hold a 10.8 mm tube left the fingers 24 mm apart and the tube where it stood.
    """
    centres = [np.asarray(data.geom_xpos[i], dtype=float) for i in pad_ids]
    span = centres[1] - centres[0]
    distance = float(np.linalg.norm(span))
    if distance < 1e-9:
        return 0.0
    axis = span / distance
    faces = []
    for index, pad in enumerate(pad_ids):
        half = np.asarray(model.geom_size[pad], dtype=float)
        rotation = np.asarray(data.geom_xmat[pad], dtype=float).reshape(3, 3)
        # A pad is a thin slab, so its thinnest local axis is the one it grips along.
        thin = int(np.argmin(half))
        normal = rotation[:, thin]
        inward = axis if index == 0 else -axis
        if float(normal @ inward) < 0.0:
            normal = -normal
        faces.append(centres[index] + normal * half[thin])
    return max(float((faces[1] - faces[0]) @ axis), 0.0)


def calibrate(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
    open_value: float,
    closed_value: float,
    flange_body: str,
    namespace: str = "robot/",
) -> GripperCalibration:
    """Sweep the gripper and record what it does.

    Kinematics only — `mj_kinematics` on a scratch `MjData`, with no stepping, so this
    neither disturbs a running episode nor depends on the solver.
    """
    data = mujoco.MjData(model)
    pads = find_pad_geoms(model, namespace)
    pad_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in pads)
    addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise LookupError(f"the scene has no gripper joint named {name!r}")
        addresses.append(int(model.jnt_qposadr[joint_id]))

    flange_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, flange_body)
    if flange_id < 0:
        raise LookupError(f"the scene has no flange body named {flange_body!r}")
    hand = _hand_geoms(model, flange_id, namespace)

    values = np.linspace(open_value, closed_value, SWEEP_SAMPLES)
    widths = np.empty(SWEEP_SAMPLES)
    offset = np.zeros(3)
    # Reach is the worst case over the whole sweep, not the value at rest. A Robotiq's
    # fingers swing inwards and downwards together, so they are at their furthest extent
    # when closed — measure only the open pose and every grasp is planned 4 mm too low.
    reach = 0.0
    pad_reach = 0.0
    tcp = np.zeros(3)
    couplings = _joint_couplings(model, addresses)
    for index, value in enumerate(values):
        for address in addresses:
            data.qpos[address] = value
        _apply_couplings(model, data, couplings)
        mujoco.mj_kinematics(model, data)
        widths[index] = _free_gap(model, data, pad_ids)
        midpoint = (data.geom_xpos[pad_ids[0]] + data.geom_xpos[pad_ids[1]]) / 2.0
        if index == 0:
            rotation = data.xmat[flange_id].reshape(3, 3)
            offset = rotation.T @ (midpoint - data.xpos[flange_id])
            # The tool centre is fixed to the flange at the open pose, so that is the datum
            # every later sample is measured from. Measuring against the live midpoint
            # instead makes the datum descend with the fingers it is measuring, and the
            # reach comes out as the fingers' half-length — 18.7 mm here rather than the
            # true 33.2 — because the two move together. That understatement is worth 15 mm
            # of well depth, which is the difference between fingers that clear a rack's rim
            # and fingers that arrive inside its wall.
            tcp = midpoint.copy()
        reach = max(reach, _finger_reach(model, data, flange_id, tcp, hand))
        pad_reach = max(pad_reach, _finger_reach(model, data, flange_id, tcp, list(pad_ids)))

    usable = _closing_travel(widths)
    return GripperCalibration(
        pad_geoms=pads,
        widths_m=widths[:usable],
        joint_values=values[:usable],
        tcp_offset_flange=offset,
        finger_reach_m=reach,
        pad_reach_m=pad_reach,
    )


def _closing_travel(widths: np.ndarray) -> int:
    """How much of the swept range is real closing, as a count of samples.

    A gripper's declared control range can run past the point where its fingers meet. The
    Robotiq 85's does: its `ctrlrange` ends at 0.8 rad, the pads touch at about 0.57, and
    kinematics carries cheerfully on through to 0.8 with the fingers passed through each
    other and separating again on the far side. Sweeping that far leaves a width curve that
    is not monotonic, and inverting a non-monotonic curve returns whichever branch the
    interpolation happens to land on.

    So the usable range stops where the width stops falling, which is where the mechanism
    stops closing.
    """
    for index in range(1, len(widths)):
        if widths[index] >= widths[index - 1]:
            return index
    return len(widths)


def _joint_couplings(
    model: mujoco.MjModel, driver_addresses: list[int]
) -> list[tuple[int, int, np.ndarray, float, float]]:
    """The joint-to-joint equality constraints driven by the given joints.

    A linkage gripper's passive joints are held to its drivers by these constraints, and
    the solver only enforces them while stepping. This sweep does not step — it is pure
    kinematics — so the couplings have to be applied by hand, or the fingers rotate about
    their knuckles while the pads stay stubbornly square and the measured opening belongs
    to no configuration the gripper can actually reach.

    Returns (dependent address, driver address, polynomial, dependent qpos0, driver qpos0).
    """
    found = []
    for constraint in range(model.neq):
        if int(model.eq_type[constraint]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        dependent = int(model.eq_obj1id[constraint])
        driver = int(model.eq_obj2id[constraint])
        if driver < 0:
            continue
        driver_address = int(model.jnt_qposadr[driver])
        if driver_address not in driver_addresses:
            continue
        found.append(
            (
                int(model.jnt_qposadr[dependent]),
                driver_address,
                np.asarray(model.eq_data[constraint][:5], dtype=float),
                float(model.qpos0[int(model.jnt_qposadr[dependent])]),
                float(model.qpos0[driver_address]),
            )
        )
    return found


def _apply_couplings(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    couplings: list[tuple[int, int, np.ndarray, float, float]],
) -> None:
    """Put the passive joints where their constraints say they belong."""
    for address, driver_address, poly, rest, driver_rest in couplings:
        offset = float(data.qpos[driver_address]) - driver_rest
        powers = np.array([1.0, offset, offset**2, offset**3, offset**4])
        data.qpos[address] = rest + float(poly @ powers)


def _hand_geoms(model: mujoco.MjModel, flange_id: int, namespace: str) -> list[int]:
    """Every collidable geom carried by the flange — the hand, whatever it is made of."""
    return [
        geom
        for geom in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
        and name.startswith(namespace)
        and int(model.geom_contype[geom]) != 0
        and _under(model, int(model.geom_bodyid[geom]), flange_id)
    ]


def _finger_reach(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    flange_id: int,
    tcp: np.ndarray,
    hand: list[int],
) -> float:
    """How far the hand's geometry extends past the tool centre, along the tool axis.

    Every geom on the hand is considered, not just the pads, because what stops a grasp is
    whichever part arrives first — on a Robotiq 85 that is the fingertips, which reach past
    the pads they carry.
    """
    axis = np.asarray(data.xmat[flange_id], dtype=float).reshape(3, 3)[:, 2]
    tcp_along = float(axis @ tcp)
    reach = 0.0
    for geom in hand:
        rotation = np.asarray(data.geom_xmat[geom], dtype=float).reshape(3, 3)
        # From the bounding box rather than from `geom_size`, because a mesh's box is not
        # centred on its geom frame and `geom_size` carries only the half-extents. Ignoring
        # that offset understated this gripper's reach by 6 mm, which is a well deep enough
        # for the fingers to arrive inside its wall: the tube then tilted 20 deg on its way
        # out, whipped to half a metre a second, and struck the finger that was holding it.
        box = np.asarray(model.geom_aabb[geom], dtype=float)
        centre = np.asarray(data.geom_xpos[geom], dtype=float) + rotation @ box[:3]
        support = float(np.abs(rotation.T @ axis) @ box[3:])
        reach = max(reach, float(axis @ centre) + support - tcp_along)
    return reach


def _under(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return body == ancestor
