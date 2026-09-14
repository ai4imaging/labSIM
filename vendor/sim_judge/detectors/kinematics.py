"""Class K: kinematic anomalies.

``K1`` teleport: the configuration change within a single step exceeds what the current
velocity can account for.
``K2`` joint limit violation: ``qpos`` crosses ``jnt_range``, with the tolerance taken
from the policy's ``joint_limit_tolerance``.
``K3`` numerical instability: non-finite values appear, or speeds are too large to be
real motion.
``K5`` engine warning: MuJoCo itself raised a warning during ``mj_forward`` (contact
buffer overflow, solver non-convergence, and the like).

The distinction between K1 and K3 is worth spelling out: K3 catches speeds that are
absurd in themselves, while K1 catches position jumps that do not match the velocity. The
latter is the signature of state being overwritten from outside (a teleport or a reset) —
and in that case the velocity often still looks perfectly normal.
"""

from __future__ import annotations

from collections.abc import Iterable

import mujoco
import numpy as np

from sim_judge.detectors.base import (
    BaseDetector,
    DetectorContext,
    body_subject,
    declared,
    generic,
    joint_subject,
)
from sim_judge.replay import Frame, free_body_ids
from sim_judge.report.finding import Observation, Severity, Subject

_SCENE_SUBJECT = Subject(kind="scene", a_name="<simulation>")


class KinematicsDetector(BaseDetector):
    """K1 / K2 / K3."""

    code_prefix = "K"

    def __init__(self, context: DetectorContext) -> None:
        super().__init__(context)
        self._timestep = float(self.model.opt.timestep)
        self._tolerance = max(self.policy.joint_limit_tolerance, self.limits.joint_limit_soft_allowance)
        self._limited_joints = [
            j for j in range(self.model.njnt) if bool(self.model.jnt_limited[j])
        ]
        self._free_bodies = free_body_ids(self.model)
        angular = _angular_dof_mask(self.model)
        self._angular_dof_indices = np.flatnonzero(angular)
        self._linear_dof_indices = np.flatnonzero(~angular)
        self._previous_positions: dict[int, np.ndarray] = {}
        self._previous_velocities: dict[int, np.ndarray] = {}
        self._previous_step: int | None = None

    def feed(self, frame: Frame) -> Iterable[Observation]:
        yield from self._check_finiteness_and_speed(frame)
        yield from self._check_joint_limits(frame)
        yield from self._check_teleport(frame)
        self._previous_step = frame.step_index

    def reset_continuity(self) -> None:
        self._previous_positions.clear()
        self._previous_velocities.clear()
        self._previous_step = None

    # -- K3 ----------------------------------------------------------------

    def _check_finiteness_and_speed(self, frame: Frame) -> Iterable[Observation]:
        if not (np.all(np.isfinite(frame.qpos)) and np.all(np.isfinite(frame.qvel))):
            # A non-finite value is not a matter of degree; under any threshold it means
            # the simulation has already blown up.
            yield generic(
                code="K3_NON_FINITE_STATE",
                step=frame.step_index,
                subject=_SCENE_SUBJECT,
                metrics={"non_finite_count": float(np.count_nonzero(~np.isfinite(frame.qvel)))},
                peak_metric="non_finite_count",
                thresholds={},
                severity=Severity.FAILURE,
            )
            return

        speeds = np.abs(frame.qvel)
        for dofs, code, metric, limit in (
            (self._linear_dof_indices, "K3_LINEAR_SPEED_EXPLOSION", "speed_m_s", self.limits.max_linear_speed_m_s),
            (self._angular_dof_indices, "K3_ANGULAR_SPEED_EXPLOSION", "speed_rad_s", self.limits.max_angular_speed_rad_s),
        ):
            if dofs.size == 0:
                continue
            worst = int(dofs[np.argmax(speeds[dofs])])
            value = float(speeds[worst])
            if value <= limit:
                continue
            yield generic(
                code=code,
                step=frame.step_index,
                subject=joint_subject(self.model, int(self.model.dof_jntid[worst])),
                metrics={metric: value},
                peak_metric=metric,
                thresholds={f"maximum_{metric}": limit},
                severity=Severity.FAILURE,
                detail={"dof_index": worst},
            )

    # -- K2 ----------------------------------------------------------------

    def _check_joint_limits(self, frame: Frame) -> Iterable[Observation]:
        for joint_id in self._limited_joints:
            address = int(self.model.jnt_qposadr[joint_id])
            value = float(frame.qpos[address])
            lower, upper = (float(v) for v in self.model.jnt_range[joint_id])
            violation = max(lower - value, value - upper)
            if violation <= self._tolerance:
                continue
            yield declared(
                code="K2_JOINT_LIMIT_VIOLATION",
                step=frame.step_index,
                subject=joint_subject(self.model, joint_id),
                metrics={"position": value, "violation": violation},
                peak_metric="violation",
                thresholds={"lower": lower, "upper": upper, "tolerance": self._tolerance},
                rule_id="joint_limit_tolerance",
            )

    # -- K1 ----------------------------------------------------------------

    def _check_teleport(self, frame: Frame) -> Iterable[Observation]:
        elapsed = self._timestep * (
            1 if self._previous_step is None else max(frame.step_index - self._previous_step, 1)
        )
        for body_id in self._free_bodies:
            position = frame.xpos[body_id].copy()
            dof_start = int(self.model.body_dofadr[body_id])
            velocity = frame.qvel[dof_start : dof_start + 3].copy()

            previous_position = self._previous_positions.get(body_id)
            previous_velocity = self._previous_velocities.get(body_id)
            self._previous_positions[body_id] = position
            self._previous_velocities[body_id] = velocity
            if previous_position is None or previous_velocity is None:
                continue

            travelled = float(np.linalg.norm(position - previous_position))
            # Use the larger of the velocities from the two steps as the reference, so a
            # body that happens to be accelerating is not flagged by mistake.
            reference_speed = max(
                float(np.linalg.norm(previous_velocity)), float(np.linalg.norm(velocity))
            )
            allowed = (
                self.limits.teleport_velocity_slack_ratio * reference_speed * elapsed
                + self.limits.teleport_absolute_slack_m
            )
            if travelled <= allowed:
                continue

            yield generic(
                code="K1_POSITION_JUMP",
                step=frame.step_index,
                subject=body_subject(self.resolver, body_id),
                metrics={
                    "displacement_m": travelled,
                    "explainable_m": reference_speed * elapsed,
                    "speed_m_s": reference_speed,
                },
                peak_metric="displacement_m",
                thresholds={"displacement_m": allowed, "elapsed_s": elapsed},
            )


class EngineWarningDetector(BaseDetector):
    """K5: warnings raised by the MuJoCo engine itself during replay.

    These warnings (contact buffer full, solver divergence, NaN in the configuration, and
    so on) mean the physics solve itself is no longer trustworthy, which is more
    fundamental than any geometric threshold. The policy declares whether to tolerate them
    via ``fail_on_engine_warning``.
    """

    code_prefix = "K"

    def __init__(self, context: DetectorContext) -> None:
        super().__init__(context)
        self._fatal = self.policy.fail_on_engine_warning

    def feed(self, frame: Frame) -> Iterable[Observation]:
        for name in frame.warnings:
            yield generic(
                code="K5_ENGINE_WARNING",
                step=frame.step_index,
                subject=Subject(kind="scene", a_name=name),
                metrics={"count": 1.0},
                peak_metric="count",
                thresholds={},
                severity=Severity.FAILURE if self._fatal else Severity.WARNING,
                detail={"warning": name},
            )


def _angular_dof_mask(model: mujoco.MjModel) -> np.ndarray:
    """Mark which degrees of freedom are rotational.

    Of a free joint's six DOFs, the first three are translation in the world frame and the
    last three are rotation in the body frame; slide joints are translational, while hinge
    and ball joints are rotational. Separate thresholds are the only ones that make sense
    here — rad/s and m/s cannot be compared against the same number.
    """
    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    ball_type = int(mujoco.mjtJoint.mjJNT_BALL)
    slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)

    mask = np.zeros(model.nv, dtype=bool)
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        dof = int(model.jnt_dofadr[joint_id])
        if joint_type == free_type:
            mask[dof + 3 : dof + 6] = True
        elif joint_type == ball_type:
            mask[dof : dof + 3] = True
        elif joint_type != slide_type:  # hinge
            mask[dof] = True
    return mask
