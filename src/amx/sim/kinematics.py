"""Inverse kinematics for driving a site to a pose.

Damped least squares on the site Jacobian. It runs on a scratch `MjData` so the recorded
episode never sees the solver's intermediate configurations, and it returns the residual
rather than raising, because a partially reached target is often still a usable waypoint
and the judge is the thing that decides whether the motion was acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from amx.geometry import orientation_error

DAMPING = 5e-4
MAX_JOINT_STEP = 0.06
"""Radians per iteration. Caps the step near singularities, where the DLS solution is large."""


@dataclass(frozen=True)
class IkResult:
    qpos: np.ndarray
    """Full qpos vector with the solved arm configuration written in."""

    position_error_m: float
    orientation_error_rad: float
    iterations: int
    converged: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "position_error_m": round(self.position_error_m, 8),
            "orientation_error_rad": round(self.orientation_error_rad, 8),
            "iterations": self.iterations,
            "converged": self.converged,
        }


def solve_site_pose(
    model: mujoco.MjModel,
    site_id: int,
    target_pos: np.ndarray,
    target_mat: np.ndarray | None,
    *,
    dof_ids: np.ndarray,
    qpos_ids: np.ndarray,
    seed_qpos: np.ndarray,
    position_tolerance_m: float = 1e-4,
    orientation_tolerance_rad: float = 1e-3,
    max_iterations: int = 400,
) -> IkResult:
    """Drive `site_id` onto the target, moving only the joints named by `dof_ids`.

    `target_mat` may be None, in which case orientation is left unconstrained and the
    solver only has to satisfy three of six task-space dimensions.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = seed_qpos
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    lower = model.jnt_range[:, 0]
    upper = model.jnt_range[:, 1]
    limited = model.jnt_limited.astype(bool)

    rows = 3 if target_mat is None else 6
    position_error = float("inf")
    rotation_error = 0.0
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)

        delta = np.zeros(rows)
        delta[:3] = target_pos - scratch.site_xpos[site_id]
        position_error = float(np.linalg.norm(delta[:3]))
        if target_mat is not None:
            rotation = orientation_error(scratch.site_xmat[site_id].reshape(3, 3), target_mat)
            delta[3:] = rotation
            rotation_error = float(np.linalg.norm(rotation))
        if position_error < position_tolerance_m and rotation_error < orientation_tolerance_rad:
            break

        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        jacobian = jacp[:, dof_ids] if target_mat is None else np.vstack(
            (jacp[:, dof_ids], jacr[:, dof_ids])
        )
        # Damped least squares in the task-space form: J^T (J J^T + lambda I)^-1 e.
        gram = jacobian @ jacobian.T + DAMPING * np.eye(rows)
        step = jacobian.T @ np.linalg.solve(gram, delta)

        largest = float(np.abs(step).max()) if step.size else 0.0
        if largest > MAX_JOINT_STEP:
            step *= MAX_JOINT_STEP / largest

        for offset, (qpos_id, joint_dof) in enumerate(zip(qpos_ids, dof_ids, strict=True)):
            value = scratch.qpos[qpos_id] + step[offset]
            joint_id = int(model.dof_jntid[joint_dof])
            if limited[joint_id]:
                value = min(max(value, lower[joint_id]), upper[joint_id])
            scratch.qpos[qpos_id] = value

    converged = (
        position_error < position_tolerance_m and rotation_error < orientation_tolerance_rad
    )
    return IkResult(
        qpos=scratch.qpos.copy(),
        position_error_m=position_error,
        orientation_error_rad=rotation_error,
        iterations=iteration,
        converged=converged,
    )


def joint_addresses(
    model: mujoco.MjModel, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Map joint names to their `qpos` addresses and velocity DOF indices.

    Only hinge and slide joints are accepted, because both have exactly one address in
    each array and the IK step maths above assumes that.
    """
    qpos_ids: list[int] = []
    dof_ids: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"the scene has no joint named {name!r}")
        # MuJoCo's Python enums are pybind11 enums, not IntEnum, so the model's int has to
        # be compared against int(...) rather than the enum member directly.
        joint_type = int(model.jnt_type[joint_id])
        single_dof = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        if joint_type not in single_dof:
            raise ValueError(f"{name!r} is not a hinge or slide joint and cannot be solved for")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        dof_ids.append(int(model.jnt_dofadr[joint_id]))
    return np.array(qpos_ids, dtype=int), np.array(dof_ids, dtype=int)


def site_reachable(
    model: mujoco.MjModel,
    site_id: int,
    target_pos: np.ndarray,
    *,
    dof_ids: np.ndarray,
    qpos_ids: np.ndarray,
    seed_qpos: np.ndarray,
    tolerance_m: float = 2e-3,
) -> tuple[bool, float]:
    """Position-only reachability probe, used by asset grounding and layout checks."""
    result = solve_site_pose(
        model,
        site_id,
        target_pos,
        None,
        dof_ids=dof_ids,
        qpos_ids=qpos_ids,
        seed_qpos=seed_qpos,
        position_tolerance_m=tolerance_m,
    )
    return result.position_error_m <= tolerance_m, result.position_error_m
