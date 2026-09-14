"""Shared internal SDK v0 implementation."""

# ruff: noqa: F401
from .articulated_object import ArticulatedObject
from .assets import AssetContext
from .errors import SDKError, ValidationError
from .geometry_scale import scale_geometry_to_size
from .placement import (
    SurfaceFrame,
    SurfaceWrapMapping,
    align_centers,
    part_local_aabb,
    place_on_face,
    place_on_face_uv,
    place_on_surface,
    proud_for_flush_mount,
    surface_frame,
    wrap_mesh_onto_surface,
    wrap_profile_onto_surface,
)
from .testing import TestContext, TestFailure, TestReport
from .types import (
    Articulation,
    ArticulationType,
    Box,
    Cylinder,
    Inertia,
    Inertial,
    Material,
    Mesh,
    Mimic,
    MotionLimits,
    MotionProperties,
    Origin,
    Part,
    Sphere,
    Visual,
)
