from __future__ import annotations

import copy
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.manufacturer_spec_catalog import (
    DeviceCategory,
    ManufacturerObservation,
    ManufacturerProductSpec,
    ManufacturerSpecCatalog,
    rank_products_by_dimensions,
)
from sdk import Inertia, Inertial, MotionProperties, Origin

Confidence = Literal["high", "medium", "low"]
PhysicalSourceKind = Literal[
    "manufacturer_reference",
    "measured",
    "geometry_derived",
    "category_prior",
    "engineering_prior",
    "calibrated",
]
MassPolicy = Literal["retain_model", "constrain_to_reference", "category_prior"]
Vec3 = tuple[float, float, float]
Inertia6 = tuple[float, float, float, float, float, float]


class ResolvedPhysicalScalar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str
    source_kind: PhysicalSourceKind
    confidence: Confidence
    source_ids: list[str] = Field(default_factory=list)
    lower_bound: float | None = None
    upper_bound: float | None = None
    method_zh: str

    @model_validator(mode="after")
    def validate_scalar(self) -> ResolvedPhysicalScalar:
        if not math.isfinite(self.value) or not self.unit.strip() or not self.method_zh.strip():
            raise ValueError("physical scalar value, unit and method must be valid")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("source_ids must be unique")
        if self.lower_bound is not None and self.value < self.lower_bound:
            raise ValueError("physical scalar value is below lower_bound")
        if self.upper_bound is not None and self.value > self.upper_bound:
            raise ValueError("physical scalar value is above upper_bound")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("physical scalar bounds must be ordered")
        return self


class PartPhysicalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mass_kg: ResolvedPhysicalScalar
    mass_fraction: float
    center_xyz_m: Vec3
    inertial_frame_rpy_rad: Vec3 = (0.0, 0.0, 0.0)
    inertia_kg_m2: Inertia6

    @model_validator(mode="after")
    def validate_part(self) -> PartPhysicalSpec:
        if self.mass_kg.value <= 0.0 or not 0.0 < self.mass_fraction <= 1.0:
            raise ValueError("part mass and mass_fraction must be positive")
        if any(not math.isfinite(value) for value in self.inertia_kg_m2):
            raise ValueError("part inertia tensor must be finite")
        if any(value <= 0.0 for value in self.inertia_kg_m2[:3]):
            raise ValueError("part principal inertia entries must be positive")
        return self


class ContactMaterialPhysicalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    friction: Vec3
    restitution: float
    source_kind: PhysicalSourceKind = "engineering_prior"
    confidence: Confidence = "low"
    source_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contact(self) -> ContactMaterialPhysicalSpec:
        if any(value < 0.0 or not math.isfinite(value) for value in self.friction):
            raise ValueError("contact friction must contain three non-negative values")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("contact restitution must be in [0, 1]")
        return self


class JointPhysicalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    damping: float
    friction: float
    stiffness: float | None = None
    spring_reference: float | None = None
    effort_limit: float
    velocity_limit: float
    source_kind: PhysicalSourceKind = "engineering_prior"
    confidence: Confidence = "low"

    @model_validator(mode="after")
    def validate_joint(self) -> JointPhysicalSpec:
        values = [self.damping, self.friction, self.effort_limit, self.velocity_limit]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("joint physics must be finite")
        if self.damping < 0.0 or self.friction < 0.0:
            raise ValueError("joint damping and friction must be non-negative")
        if self.effort_limit <= 0.0 or self.velocity_limit <= 0.0:
            raise ValueError("joint effort and velocity limits must be positive")
        if self.stiffness is not None and self.stiffness < 0.0:
            raise ValueError("joint stiffness must be non-negative")
        return self


class ManufacturerReferenceMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str
    manufacturer: str
    model: str
    dimension_log_rmse: float
    official_dimensions_wdh_m: Vec3
    official_mass_kg: float
    mass_source_id: str
    adopted_as_total_mass_constraint: bool


class PhysicalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    physical_spec_id: str
    asset_id: str
    category: DeviceCategory
    mass_policy: MassPolicy
    total_mass_kg: ResolvedPhysicalScalar
    parts: dict[str, PartPhysicalSpec]
    contact_materials: dict[str, ContactMaterialPhysicalSpec]
    joints: dict[str, JointPhysicalSpec]
    manufacturer_reference: ManufacturerReferenceMatch | None = None
    unresolved_fields: list[str] = Field(default_factory=list)
    notes_zh: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_spec(self) -> PhysicalSpec:
        if not self.parts:
            raise ValueError("physical spec requires at least one part")
        total = sum(part.mass_kg.value for part in self.parts.values())
        if not math.isclose(total, self.total_mass_kg.value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("part masses must sum to total_mass_kg")
        fraction = sum(part.mass_fraction for part in self.parts.values())
        if not math.isclose(fraction, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("part mass fractions must sum to one")
        if len(self.unresolved_fields) != len(set(self.unresolved_fields)):
            raise ValueError("unresolved_fields must be unique")
        return self


def _mass_observation(product: ManufacturerProductSpec) -> ManufacturerObservation:
    return next(item for item in product.observations if item.property_id == "mass_kg")


def _mass_midpoint(observation: ManufacturerObservation) -> float:
    value = observation.value
    if isinstance(value, list):
        return (float(value[0]) + float(value[1])) / 2.0
    return float(value)


def _category_mass_bounds(
    catalog: ManufacturerSpecCatalog, category: DeviceCategory
) -> tuple[float, float, float]:
    masses = [
        _mass_midpoint(_mass_observation(product))
        for product in catalog.products
        if product.category == category
    ]
    ordered = sorted(masses)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2 == 1
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return min(ordered), median, max(ordered)


def _reference_match(
    catalog: ManufacturerSpecCatalog,
    *,
    category: DeviceCategory,
    target_dimensions_wdh_m: Vec3,
    reference_product_id: str | None,
) -> tuple[ManufacturerProductSpec, dict[str, object]]:
    ranked = rank_products_by_dimensions(
        catalog,
        category=category,
        target_dimensions_wdh_m=target_dimensions_wdh_m,
        limit=max(1, len(catalog.products)),
    )
    selected = (
        next(
            (item for item in ranked if item["product_id"] == reference_product_id),
            None,
        )
        if reference_product_id is not None
        else ranked[0]
    )
    if selected is None:
        raise ValueError(
            f"Reference product {reference_product_id!r} is not in category {category!r}."
        )
    product = next(
        product
        for product in catalog.products
        if product.product_id == selected["product_id"]
    )
    return product, selected


def resolve_physical_spec_for_model(
    model: object,
    *,
    physical_spec_id: str,
    asset_id: str,
    category: DeviceCategory,
    target_dimensions_wdh_m: Vec3,
    catalog: ManufacturerSpecCatalog,
    mass_policy: MassPolicy = "retain_model",
    reference_product_id: str | None = None,
    maximum_reference_dimension_log_rmse: float = 0.15,
) -> PhysicalSpec:
    parts = {
        str(part.name): part
        for part in getattr(model, "parts", []) or []
        if isinstance(getattr(part, "name", None), str)
    }
    if not parts or any(getattr(part, "inertial", None) is None for part in parts.values()):
        raise ValueError("all model parts need inertials before physical resolution")
    current_masses = {
        name: float(part.inertial.mass) for name, part in parts.items()
    }
    current_total = sum(current_masses.values())
    if current_total <= 0.0:
        raise ValueError("model total mass must be positive")

    reference_product, match = _reference_match(
        catalog,
        category=category,
        target_dimensions_wdh_m=target_dimensions_wdh_m,
        reference_product_id=reference_product_id,
    )
    mass_observation = _mass_observation(reference_product)
    reference_mass = _mass_midpoint(mass_observation)
    match_error = float(match["dimension_log_rmse"])
    category_min, category_median, category_max = _category_mass_bounds(catalog, category)

    if mass_policy == "constrain_to_reference":
        if match_error > maximum_reference_dimension_log_rmse:
            raise ValueError(
                "Reference dimensions are too distant for a total-mass constraint: "
                f"rmse={match_error:.6f} > {maximum_reference_dimension_log_rmse:.6f}."
            )
        resolved_total = reference_mass
        total_source = "manufacturer_reference"
        total_confidence: Confidence = "medium"
        total_sources = [mass_observation.source_id]
        method = "显式采用尺寸相近的官方参考型号总质量，并按原部件质量比例重新分配。"
    elif mass_policy == "category_prior":
        resolved_total = category_median
        total_source = "category_prior"
        total_confidence = "low"
        total_sources = [catalog.catalog_id]
        method = "采用同类官方样本质量中位数；仅作为缺少具体型号数据时的低置信度先验。"
    else:
        resolved_total = current_total
        total_source = "engineering_prior"
        total_confidence = "low"
        total_sources = [catalog.catalog_id]
        method = "保留资产当前工程质量分配；官方型号仅用于范围检查，不作为复制约束。"

    scale = resolved_total / current_total
    resolved_parts: dict[str, PartPhysicalSpec] = {}
    for name, part in parts.items():
        inertial = part.inertial
        tensor = inertial.inertia
        part_mass = current_masses[name] * scale
        resolved_parts[name] = PartPhysicalSpec(
            mass_kg=ResolvedPhysicalScalar(
                value=part_mass,
                unit="kg",
                source_kind=(
                    "geometry_derived"
                    if mass_policy == "constrain_to_reference"
                    else total_source
                ),
                confidence=total_confidence,
                source_ids=total_sources,
                method_zh="保持现有几何/机构质量比例，并受已选择的总质量策略约束。",
            ),
            mass_fraction=current_masses[name] / current_total,
            center_xyz_m=tuple(float(value) for value in inertial.origin.xyz),
            inertial_frame_rpy_rad=tuple(float(value) for value in inertial.origin.rpy),
            inertia_kg_m2=tuple(
                float(value) * scale
                for value in (
                    tensor.ixx,
                    tensor.iyy,
                    tensor.izz,
                    tensor.ixy,
                    tensor.ixz,
                    tensor.iyz,
                )
            ),
        )

    meta = getattr(model, "meta", {})
    simulation = meta.get("simulation", {}) if isinstance(meta, dict) else {}
    raw_contacts = (
        simulation.get("contact_materials", {}) if isinstance(simulation, dict) else {}
    )
    contact_materials = {
        str(name): ContactMaterialPhysicalSpec(
            friction=tuple(float(value) for value in spec["friction"]),
            restitution=float(spec["restitution"]),
        )
        for name, spec in raw_contacts.items()
        if isinstance(spec, dict)
        and isinstance(spec.get("friction"), (list, tuple))
        and spec.get("restitution") is not None
    }
    joints: dict[str, JointPhysicalSpec] = {}
    for joint in getattr(model, "articulations", []) or []:
        dynamics = getattr(joint, "motion_properties", None)
        limits = getattr(joint, "motion_limits", None)
        if dynamics is None or limits is None:
            continue
        if dynamics.damping is None or dynamics.friction is None:
            continue
        joints[str(joint.name)] = JointPhysicalSpec(
            damping=float(dynamics.damping),
            friction=float(dynamics.friction),
            stiffness=(
                float(dynamics.stiffness) if dynamics.stiffness is not None else None
            ),
            spring_reference=(
                float(dynamics.spring_reference)
                if dynamics.spring_reference is not None
                else None
            ),
            effort_limit=float(limits.effort),
            velocity_limit=float(limits.velocity),
        )

    reference = ManufacturerReferenceMatch(
        product_id=reference_product.product_id,
        manufacturer=reference_product.manufacturer,
        model=reference_product.model,
        dimension_log_rmse=match_error,
        official_dimensions_wdh_m=tuple(
            float(value) for value in match["official_dimensions_wdh_m"]
        ),
        official_mass_kg=reference_mass,
        mass_source_id=mass_observation.source_id,
        adopted_as_total_mass_constraint=mass_policy == "constrain_to_reference",
    )
    unresolved = sorted(
        set(reference_product.not_found_in_cited_sources)
        | {
            "center_of_mass_validation",
            "inertia_tensor_validation",
            "surface_friction_calibration",
            "joint_dynamics_calibration",
        }
    )
    return PhysicalSpec(
        physical_spec_id=physical_spec_id,
        asset_id=asset_id,
        category=category,
        mass_policy=mass_policy,
        total_mass_kg=ResolvedPhysicalScalar(
            value=resolved_total,
            unit="kg",
            source_kind=total_source,
            confidence=total_confidence,
            source_ids=total_sources,
            lower_bound=(category_min if category_min <= resolved_total <= category_max else None),
            upper_bound=(category_max if category_min <= resolved_total <= category_max else None),
            method_zh=method,
        ),
        parts=resolved_parts,
        contact_materials=contact_materials,
        joints=joints,
        manufacturer_reference=reference,
        unresolved_fields=unresolved,
        notes_zh=[
            "厂家参考描述相似商业设备，不代表生成资产必须复制该具体型号。",
            "摩擦、阻尼、质心和惯量在实测或系统辨识前保持低置信度。",
        ],
    )


def apply_physical_spec(model: object, spec: PhysicalSpec) -> object:
    cloned = copy.deepcopy(model)
    parts = {str(part.name): part for part in getattr(cloned, "parts", []) or []}
    joints = {
        str(joint.name): joint for joint in getattr(cloned, "articulations", []) or []
    }
    if set(spec.parts) - set(parts):
        raise ValueError("physical spec references unknown model parts")
    if set(spec.joints) - set(joints):
        raise ValueError("physical spec references unknown model joints")

    for name, part_spec in spec.parts.items():
        tensor = part_spec.inertia_kg_m2
        parts[name].inertial = Inertial(
            mass=part_spec.mass_kg.value,
            inertia=Inertia(
                ixx=tensor[0],
                ixy=tensor[3],
                ixz=tensor[4],
                iyy=tensor[1],
                iyz=tensor[5],
                izz=tensor[2],
            ),
            origin=Origin(
                xyz=part_spec.center_xyz_m,
                rpy=part_spec.inertial_frame_rpy_rad,
            ),
        )
    for name, joint_spec in spec.joints.items():
        joints[name].motion_properties = MotionProperties(
            damping=joint_spec.damping,
            friction=joint_spec.friction,
            stiffness=joint_spec.stiffness,
            spring_reference=joint_spec.spring_reference,
        )

    meta = getattr(cloned, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
        cloned.meta = meta
    simulation = meta.get("simulation")
    if not isinstance(simulation, dict):
        simulation = {}
        meta["simulation"] = simulation
    simulation["contact_materials"] = {
        name: {
            "friction": material.friction,
            "restitution": material.restitution,
        }
        for name, material in spec.contact_materials.items()
    }
    meta["physical_spec"] = spec.model_dump(mode="json")
    return cloned
