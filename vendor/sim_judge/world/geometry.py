"""Name-agnostic pure geometry helpers: vertical direction, support height and
characteristic geom dimensions.

Every function here deliberately avoids assuming that world +Z is up. The judge has
to hold for arbitrary scenes, and "up" is a physical quantity defined by gravity,
not the index of a coordinate axis -- hard-coding it would make the whole class-F
criteria produce numbers that look plausible but are completely wrong on a model
whose gravity points along -Y.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

# MuJoCo geom type constants, following the mjtGeom enum.
PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
HFIELD = int(mujoco.mjtGeom.mjGEOM_HFIELD)
SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
MESH = int(mujoco.mjtGeom.mjGEOM_MESH)

_GRAVITY_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class UpAxis:
    """The scene's "up" direction, defined as the negation of gravity.

    As long as the model declares a non-zero gravity, "up" is well determined and
    entirely independent of how the coordinate axes happen to be numbered. Under
    zero gravity (a space scene, say) there is nothing to define it from, so
    :meth:`of` returns ``None`` and every criterion that depends on the vertical
    direction switches off, rather than computing against a guessed direction.
    """

    direction: np.ndarray
    """World-frame unit vector pointing up."""

    gravity_m_s2: float
    """Magnitude of the gravitational acceleration, always positive."""

    @classmethod
    def of(cls, model: mujoco.MjModel) -> UpAxis | None:
        gravity = np.asarray(model.opt.gravity, dtype=np.float64)
        magnitude = float(np.linalg.norm(gravity))
        if magnitude < _GRAVITY_EPS:
            return None
        return cls(direction=-gravity / magnitude, gravity_m_s2=magnitude)

    def height(self, position: np.ndarray) -> float:
        """Height of a point along the vertical, i.e. the projection of the position
        vector onto the "up" direction."""
        return float(np.asarray(position, dtype=np.float64) @ self.direction)

    def component(self, vector: np.ndarray) -> float:
        """Vertical component of a vector. Positive is upward."""
        return float(np.asarray(vector, dtype=np.float64) @ self.direction)


def local_support(geom_type: int, size: np.ndarray, local_direction: np.ndarray) -> float:
    """Support function of a geom along a given unit direction, in the geom's own
    local frame.

    That is, "starting from the geom origin, how far does the surface reach in this
    direction". Every height and half-extent in this module is derived from it, so
    supporting a new geom type only takes one more branch here and the rest of the
    criteria generalise along with it.

    The meaning of MuJoCo's ``geom_size`` varies by type, and the branches interpret
    it accordingly: a sphere carries only a radius; capsules and cylinders carry
    (radius, half-length) about the local z axis; ellipsoids, boxes and mesh
    bounding boxes carry three half-axis lengths.
    """
    v = local_direction
    if geom_type in (PLANE, HFIELD):
        return float("inf")  # semi-infinite, so there is no "farthest point"
    if geom_type == SPHERE:
        return float(size[0])
    if geom_type == CAPSULE:  # sphere swept along a segment: project the segment, add the radius
        return abs(float(v[2])) * float(size[1]) + float(size[0])
    if geom_type == CYLINDER:  # project axially and radially, separately
        return abs(float(v[2])) * float(size[1]) + float(size[0]) * float(np.hypot(v[0], v[1]))
    if geom_type == ELLIPSOID:
        return float(np.linalg.norm(v * size))
    return float(np.abs(v) @ size)  # boxes and mesh bounding boxes


def support_height(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int, up: UpAxis) -> float:
    """Height of the surface a geom presents when acting as a support, i.e. its
    highest point along the vertical.

    Semi-infinite geoms (planes, height fields) have no meaningful thickness, so we
    just take the height of their origin.
    """
    center = up.height(data.geom_xpos[geom_id])
    geom_type = int(model.geom_type[geom_id])
    if geom_type in (PLANE, HFIELD):
        return center

    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    size = np.abs(np.asarray(model.geom_size[geom_id][:3], dtype=np.float64))
    return center + local_support(geom_type, size, rotation.T @ up.direction)


def min_half_extent(geom_type: int, size: Sequence[float], rbound: float) -> float:
    """Half-extent of a geom along its thinnest direction.

    This is the natural yardstick for penetration depth: a depth that reaches it
    means half the geom's thickness has been eaten into, and twice it means the geom
    has been passed through entirely.
    """
    if geom_type in (PLANE, HFIELD):
        return float("inf")  # semi-infinite: penetration cannot be measured against a thickness
    if geom_type in (SPHERE, CAPSULE):
        return float(size[0])  # for spheres and capsules the thinnest direction is the radius
    if geom_type == CYLINDER:
        return min(float(size[0]), float(size[1]))
    if geom_type in (BOX, ELLIPSOID, MESH):
        positive = [float(v) for v in size if v > 0.0]
        return min(positive) if positive else rbound
    return rbound if rbound > 0.0 else float("inf")


@dataclass(frozen=True, slots=True)
class Extents:
    """Vertical half-height and horizontal half-width of a geom, relative to its own
    origin."""

    horizontal_m: float
    """Largest half-width within the horizontal plane. This is what tells you
    whether something fits into a socket."""

    vertical_m: float
    """Distance from the origin to the end face along the vertical. This is what
    tells you where the bottom face is."""


def extents(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int, up: UpAxis) -> Extents:
    """Vertical half-height and horizontal half-width of a geom.

    Labware is not necessarily an upright cylinder. Reading ``geom_size[0]`` as a
    radius and ``geom_size[1]`` as a half-height only works for cylinders and
    capsules, and only while they stand upright; a box's size is three half-edge
    lengths and a sphere carries a single radius. So everything goes through the
    support function instead: the vertical half-height is the support value along
    "up", and the horizontal half-width is the largest support value over the
    horizontal directions.

    There is no closed form for the horizontal maximum (a box rotating about the
    vertical axis has a half-width that varies non-smoothly with angle), but this
    quantity is computed once at construction time, so densely sampling the
    horizontal circle is fine and its cost is negligible.
    """
    geom_type = int(model.geom_type[geom_id])
    size = np.abs(np.asarray(model.geom_size[geom_id][:3], dtype=np.float64))
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)

    vertical = local_support(geom_type, size, rotation.T @ up.direction)
    horizontal = max(
        local_support(geom_type, size, rotation.T @ direction)
        for direction in _horizontal_directions(up)
    )
    return Extents(horizontal_m=horizontal, vertical_m=vertical)


def _horizontal_directions(up: UpAxis, samples: int = 64) -> np.ndarray:
    """Unit directions spread uniformly around the horizontal plane, shape
    ``(samples, 3)``."""
    # Take any vector not parallel to "up" and cross-product out an orthogonal basis
    # for the horizontal plane.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ up.direction)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(up.direction, seed)
    first /= np.linalg.norm(first)
    second = np.cross(up.direction, first)

    angles = np.linspace(0.0, np.pi, samples, endpoint=False)  # centrally symmetric, so half a turn suffices
    return np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second


def horizontal_offset(up: UpAxis, delta: np.ndarray) -> float:
    """Length of a displacement vector within the horizontal plane, i.e. its norm
    once the vertical component is removed."""
    delta = np.asarray(delta, dtype=np.float64)
    return float(np.linalg.norm(delta - up.component(delta) * up.direction))


def body_point_world(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, local: Sequence[float]
) -> np.ndarray:
    """Transform a point given in body-local coordinates into the world frame."""
    rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    return np.asarray(data.xpos[body_id], dtype=np.float64) + rotation @ np.asarray(
        local, dtype=np.float64
    )


def quaternion_to_axis(quat: np.ndarray, local_axis: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Apply a body quaternion (w,x,y,z) to a local axis to get a world-frame
    direction."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.ascontiguousarray(quat, dtype=np.float64))
    return rot.reshape(3, 3) @ np.asarray(local_axis, dtype=np.float64)
