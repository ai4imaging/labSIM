from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DeviceCategory = Literal["microcentrifuge", "thermal_cycler", "microplate_reader"]
SourceType = Literal["manufacturer_product_page", "manufacturer_datasheet", "manufacturer_manual"]
ObservationQualifier = Literal[
    "exact",
    "approximate",
    "range",
    "maximum",
    "minimum",
    "boolean",
    "text",
]
ObservationConfidence = Literal["high", "medium"]
ObservationValue = float | str | bool | list[float] | list[str]

_PROPERTY_ID_RE = re.compile(r"^[a-z][a-z0-9_.]*$")
_COMMON_UNITS = {
    "dimensions_wdh_m": "m",
    "mass_kg": "kg",
    "open_lid_height_m": "m",
    "max_speed_rpm": "rpm",
    "max_rcf_g": "g",
    "max_power_consumption_w": "W",
    "max_apparent_power_va": "VA",
    "wavelength_range_nm": "nm",
    "max_block_ramp_deg_c_s": "degC/s",
    "max_sample_ramp_deg_c_s": "degC/s",
}


class OfficialManufacturerSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    manufacturer: str
    title: str
    url: str
    source_type: SourceType
    accessed_date: date
    document_date: str | None = None
    local_archive_path: str | None = None
    official: bool = True

    @model_validator(mode="after")
    def validate_source(self) -> OfficialManufacturerSource:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.url.startswith("https://"):
            raise ValueError("official source URL must use https")
        if not self.official:
            raise ValueError("manufacturer catalog only accepts official sources")
        if self.local_archive_path is not None:
            path = Path(self.local_archive_path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("local_archive_path must stay relative to the catalog directory")
        return self


class ManufacturerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str
    value: ObservationValue
    unit: str | None = None
    raw_value: str
    qualifier: ObservationQualifier = "exact"
    source_id: str
    source_locator: str
    confidence: ObservationConfidence = "high"
    notes_zh: str | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> ManufacturerObservation:
        if not _PROPERTY_ID_RE.fullmatch(self.property_id):
            raise ValueError("property_id must be lower-case snake/dotted notation")
        if not self.raw_value.strip() or not self.source_locator.strip():
            raise ValueError("raw_value and source_locator are required")

        value = self.value
        if isinstance(value, list):
            if not value:
                raise ValueError("observation list values must not be empty")
            for item in value:
                if isinstance(item, float) and not math.isfinite(item):
                    raise ValueError("numeric observation values must be finite")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric observation values must be finite")

        expected_unit = _COMMON_UNITS.get(self.property_id)
        if expected_unit is not None and self.unit != expected_unit:
            raise ValueError(f"{self.property_id} must use canonical unit {expected_unit!r}")
        if self.property_id == "dimensions_wdh_m":
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError("dimensions_wdh_m must be a three-value W,D,H vector")
            if any(not isinstance(item, float) or item <= 0.0 for item in value):
                raise ValueError("dimensions_wdh_m values must be positive numbers")
        if self.property_id == "mass_kg":
            mass_values = value if isinstance(value, list) else [value]
            if len(mass_values) not in {1, 2}:
                raise ValueError("mass_kg must be a scalar or a two-value range")
            if any(not isinstance(item, float) or item <= 0.0 for item in mass_values):
                raise ValueError("mass_kg values must be positive numbers")
            if len(mass_values) == 2:
                if self.qualifier != "range" or mass_values[0] > mass_values[1]:
                    raise ValueError("two-value mass_kg must be an ordered range")
        return self


class ManufacturerProductSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str
    category: DeviceCategory
    manufacturer: str
    model: str
    variant: str | None = None
    observations: list[ManufacturerObservation]
    not_found_in_cited_sources: list[str] = Field(default_factory=list)
    notes_zh: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_product(self) -> ManufacturerProductSpec:
        if not self.product_id.strip() or not self.observations:
            raise ValueError("product_id and observations are required")
        keys = [(item.property_id, item.source_id, item.raw_value) for item in self.observations]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate observation in product")
        properties = {item.property_id for item in self.observations}
        for required in ("dimensions_wdh_m", "mass_kg"):
            if required not in properties:
                raise ValueError(f"product {self.product_id!r} is missing {required}")
        if len(self.not_found_in_cited_sources) != len(set(self.not_found_in_cited_sources)):
            raise ValueError("not_found_in_cited_sources must be unique")
        return self


class ManufacturerSpecCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    catalog_id: str
    generated_date: date
    scope_zh: str
    sources: list[OfficialManufacturerSource]
    products: list[ManufacturerProductSpec]

    @model_validator(mode="after")
    def validate_catalog(self) -> ManufacturerSpecCatalog:
        source_ids = [source.source_id for source in self.sources]
        product_ids = [product.product_id for product in self.products]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("product_id values must be unique")
        source_by_id = {source.source_id: source for source in self.sources}
        known_sources = set(source_by_id)
        for product in self.products:
            unknown = sorted(
                {observation.source_id for observation in product.observations} - known_sources
            )
            if unknown:
                raise ValueError(
                    f"product {product.product_id!r} references unknown sources: {unknown}"
                )
            mismatched = sorted(
                {
                    observation.source_id
                    for observation in product.observations
                    if source_by_id[observation.source_id].manufacturer != product.manufacturer
                }
            )
            if mismatched:
                raise ValueError(
                    f"product {product.product_id!r} uses sources from another manufacturer: "
                    f"{mismatched}"
                )
        return self


def load_manufacturer_spec_catalog(path: Path) -> ManufacturerSpecCatalog:
    return ManufacturerSpecCatalog.model_validate_json(path.read_text(encoding="utf-8"))


def missing_source_archives(catalog: ManufacturerSpecCatalog, catalog_path: Path) -> list[str]:
    root = catalog_path.resolve().parent
    return [
        source.source_id
        for source in catalog.sources
        if source.local_archive_path is not None
        and not (root / source.local_archive_path).is_file()
    ]


def source_archive_issues(
    catalog: ManufacturerSpecCatalog,
    catalog_path: Path,
) -> list[dict[str, str]]:
    root = catalog_path.resolve().parent
    issues: list[dict[str, str]] = []
    for source in catalog.sources:
        if source.local_archive_path is None:
            continue
        path = root / source.local_archive_path
        if not path.is_file():
            issues.append({"source_id": source.source_id, "code": "missing"})
            continue
        if path.stat().st_size < 1024:
            issues.append({"source_id": source.source_id, "code": "too_small"})
            continue
        if path.suffix.lower() == ".pdf" and path.read_bytes()[:5] != b"%PDF-":
            issues.append({"source_id": source.source_id, "code": "invalid_pdf_signature"})
    return issues


def build_source_archive_manifest(
    catalog: ManufacturerSpecCatalog,
    catalog_path: Path,
) -> dict[str, object]:
    root = catalog_path.resolve().parent
    entries: list[dict[str, object]] = []
    for source in catalog.sources:
        if source.local_archive_path is None:
            continue
        path = root / source.local_archive_path
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(
            {
                "source_id": source.source_id,
                "local_archive_path": source.local_archive_path,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return {
        "schema_version": "1.0",
        "catalog_id": catalog.catalog_id,
        "archive_count": len(entries),
        "entries": entries,
    }


def category_coverage_issues(
    catalog: ManufacturerSpecCatalog,
    *,
    minimum_products_per_category: int,
) -> list[str]:
    if minimum_products_per_category < 1:
        raise ValueError("minimum_products_per_category must be positive")
    counts = Counter(product.category for product in catalog.products)
    return [
        f"{category}: {counts[category]} < {minimum_products_per_category}"
        for category in ("microcentrifuge", "thermal_cycler", "microplate_reader")
        if counts[category] < minimum_products_per_category
    ]


def _observation(product: ManufacturerProductSpec, property_id: str) -> ManufacturerObservation:
    return next(item for item in product.observations if item.property_id == property_id)


def _mass_midpoint(observation: ManufacturerObservation) -> float:
    value = observation.value
    if isinstance(value, list):
        return (float(value[0]) + float(value[1])) / 2.0
    return float(value)


def rank_products_by_dimensions(
    catalog: ManufacturerSpecCatalog,
    *,
    category: DeviceCategory,
    target_dimensions_wdh_m: tuple[float, float, float],
    limit: int = 3,
) -> list[dict[str, object]]:
    if any(value <= 0.0 for value in target_dimensions_wdh_m):
        raise ValueError("target_dimensions_wdh_m values must be positive")
    if limit < 1:
        raise ValueError("limit must be positive")
    matches: list[dict[str, object]] = []
    for product in catalog.products:
        if product.category != category:
            continue
        dimensions = [
            float(value) for value in _observation(product, "dimensions_wdh_m").value
        ]
        log_errors = [
            math.log(target / reference)
            for target, reference in zip(target_dimensions_wdh_m, dimensions, strict=True)
        ]
        relative_delta = [
            (target - reference) / reference
            for target, reference in zip(target_dimensions_wdh_m, dimensions, strict=True)
        ]
        matches.append(
            {
                "product_id": product.product_id,
                "manufacturer": product.manufacturer,
                "model": product.model,
                "variant": product.variant,
                "official_dimensions_wdh_m": dimensions,
                "target_relative_delta_wdh": [round(value, 6) for value in relative_delta],
                "dimension_log_rmse": round(
                    math.sqrt(sum(value * value for value in log_errors) / 3.0),
                    9,
                ),
            }
        )
    matches.sort(key=lambda item: (item["dimension_log_rmse"], item["product_id"]))
    return matches[:limit]


def _axis_summary(vectors: list[list[float]]) -> dict[str, object]:
    axes = list(zip(*vectors, strict=True))
    return {
        "sample_count": len(vectors),
        "minimum_wdh_m": [round(min(axis), 12) for axis in axes],
        "median_wdh_m": [round(median(axis), 12) for axis in axes],
        "maximum_wdh_m": [round(max(axis), 12) for axis in axes],
    }


def build_category_priors(catalog: ManufacturerSpecCatalog) -> dict[str, object]:
    categories: dict[str, object] = {}
    for category in ("microcentrifuge", "thermal_cycler", "microplate_reader"):
        products = [product for product in catalog.products if product.category == category]
        dimensions = [
            [float(value) for value in _observation(product, "dimensions_wdh_m").value]
            for product in products
        ]
        masses = [_mass_midpoint(_observation(product, "mass_kg")) for product in products]
        categories[category] = {
            "product_ids": [product.product_id for product in products],
            "dimensions": _axis_summary(dimensions),
            "mass": {
                "sample_count": len(masses),
                "minimum_kg": min(masses),
                "median_kg": median(masses),
                "maximum_kg": max(masses),
                "range_midpoint_count": sum(
                    _observation(product, "mass_kg").qualifier == "range"
                    for product in products
                ),
            },
        }
    return {
        "schema_version": "1.0",
        "derived_from_catalog_id": catalog.catalog_id,
        "generated_date": catalog.generated_date.isoformat(),
        "method_zh": (
            "尺寸按官方宽深高取逐轴最小值、中位数和最大值；质量区间仅为统计先验取区间中点。"
            "这些先验用于约束生成和发现离群值，不替代具体型号的厂家数据或实测。"
        ),
        "categories": categories,
    }


def catalog_coverage_summary(catalog: ManufacturerSpecCatalog) -> dict[str, object]:
    category_counts = Counter(product.category for product in catalog.products)
    property_counts = Counter(
        observation.property_id
        for product in catalog.products
        for observation in product.observations
    )
    missing_counts = Counter(
        field
        for product in catalog.products
        for field in product.not_found_in_cited_sources
    )
    return {
        "catalog_id": catalog.catalog_id,
        "source_count": len(catalog.sources),
        "product_count": len(catalog.products),
        "category_counts": dict(sorted(category_counts.items())),
        "property_counts": dict(sorted(property_counts.items())),
        "not_found_in_cited_sources_counts": dict(sorted(missing_counts.items())),
    }


def write_catalog_derivatives(
    catalog: ManufacturerSpecCatalog,
    *,
    priors_path: Path,
    coverage_path: Path,
) -> None:
    for path, payload in (
        (priors_path, build_category_priors(catalog)),
        (coverage_path, catalog_coverage_summary(catalog)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def write_source_archive_manifest(
    catalog: ManufacturerSpecCatalog,
    catalog_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            build_source_archive_manifest(catalog, catalog_path),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
