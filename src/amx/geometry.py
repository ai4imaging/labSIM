"""Poses and the small amount of rotation maths the pipeline needs."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
"""MuJoCo ordering: (w, x, y, z)."""


class Pose(BaseModel):
    """A rigid placement, expressed the way MJCF spells it."""

    model_config = ConfigDict(extra="forbid")

    pos: Vec3 = (0.0, 0.0, 0.0)
    euler: Vec3 = (0.0, 0.0, 0.0)
    """Intrinsic XYZ Euler angles in radians. MJCF's default `eulerseq` is `xyz`."""

    def quat(self) -> Quat:
        return euler_to_quat(self.euler)

    def matrix(self) -> np.ndarray:
        """4x4 homogeneous transform."""
        out = np.eye(4)
        out[:3, :3] = quat_to_matrix(self.quat())
        out[:3, 3] = self.pos
        return out

    def translated(self, delta: Vec3) -> "Pose":
        return Pose(
            pos=(self.pos[0] + delta[0], self.pos[1] + delta[1], self.pos[2] + delta[2]),
            euler=self.euler,
        )

    def mjcf(self) -> dict[str, str]:
        attrs = {"pos": " ".join(f"{v:.6g}" for v in self.pos)}
        if any(abs(v) > 1e-12 for v in self.euler):
            attrs["euler"] = " ".join(f"{v:.6g}" for v in self.euler)
        return attrs


class PoseTarget(BaseModel):
    """Where a `Move` action should put a site.

    Orientation is optional: most wetlab motions only constrain the approach axis, and
    leaving it free gives the solver a redundancy to spend on staying inside joint limits.
    """

    model_config = ConfigDict(extra="forbid")

    pos: Vec3
    euler: Vec3 | None = None
    frame: str | None = Field(
        default=None,
        description="Name of a site or body to interpret `pos`/`euler` relative to. World frame if unset.",
    )


def euler_to_quat(euler: Vec3) -> Quat:
    """Intrinsic XYZ Euler angles to a (w, x, y, z) quaternion."""
    half = [v * 0.5 for v in euler]
    cr, cp, cy = (math.cos(h) for h in half)
    sr, sp, sy = (math.sin(h) for h in half)
    return (
        cr * cp * cy - sr * sp * sy,
        sr * cp * cy + cr * sp * sy,
        cr * sp * cy - sr * cp * sy,
        cr * cp * sy + sr * sp * cy,
    )


def quat_to_matrix(quat: Quat) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(matrix: np.ndarray) -> Quat:
    """Shepperd's method: pick the largest diagonal term to stay away from the singularity."""
    m = np.asarray(matrix, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)


def orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Rotation vector taking `current` to `desired`, as a 3-vector in world axes."""
    relative = desired @ current.T
    w, x, y, z = matrix_to_quat(relative)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(norm, w)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return np.array([x, y, z]) * (angle / norm)


def smoothstep(fraction: float) -> float:
    """C1-continuous 0->1 ramp. Zero velocity at both ends keeps servo targets gentle."""
    t = min(1.0, max(0.0, fraction))
    return t * t * (3.0 - 2.0 * t)
