"""What a finished asset has to agree with, in machine-checkable form.

A `GroundingSpec` is derived from a task's input specification and nothing else. It is
deliberately not the rubric: it carries the targets themselves (this beaker is 70 mm
across and holds 250 mL) and not the grading policy — the weights, the gates and the
pass threshold — that will be applied to them.

Everything is optional. An unset target is not checked, which is what lets the same
spec cover a beaker, a pipette and a centrifuge without pretending a centrifuge has a
brim capacity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Interval = tuple[float, float]

MeasurementKind = Literal[
    "extent_x",
    "extent_y",
    "extent_z",
    "extent_z_min",
    "extent_max",
    "interior_extent_z",
    "diameter_outer",
    "diameter_inner",
    "width_inner",
    "wall_thickness",
    "footprint_max",
    "footprint_min",
    "pitch",
    "feature_gap_min",
    "feature_gap_max",
]
"""How to turn geometry into the one number a dimension target compares against.

`diameter_outer` is the median diameter of the body of revolution rather than the
bounding box, because a spout, a handle or a graduation ridge is not the diameter and
a bounding box cannot tell the difference.

`pitch` is centre-to-centre spacing between repeated features, taken as the median
nearest-neighbour distance. It is the one number that decides whether a rack or a plate
can be worked with a multichannel pipette, and until it could be measured a rack was
free to put its wells anywhere.

`feature_gap_min` and `feature_gap_max` are the clear material left between neighbouring
features, the closer way across the grid and the wider one: a rack states 7 mm along a row and
13 mm between rows. Ordered by size rather than by axis, because which way an author lays the
rows out is not something the specification fixes. A square grid reports one number twice.

`width_inner` is the clear width across an opening that is not round, where a diameter is not
the number a datasheet quotes: the side of a square mouth, the narrow way across a slot. A
round bore measures the same either way, so the distinction only ever adds readings.

`interior_extent_z` is how deep the enclosed space is, floor to overflow, rather than how tall
the object is. A chamber states both -- a water bath is 233 mm tall and 150 mm usable inside --
and read as one kind they were two irreconcilable claims about the same extent.

`extent_z` and `extent_z_min` are the tallest and the shortest the upper surface gets. They
differ only for an object whose top is not flat, which is most of a stepped rack and every
sloped instrument, and specifications quote both from the same datum. Sharing one kind made
them two claims about a single number, so at most one could be true of anything built.
"""


class DimensionTarget(BaseModel):
    """One number from a datasheet, and how to measure it on the model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    value_m: float = Field(gt=0.0)
    kind: MeasurementKind
    part: str = ""
    """Which body to measure, by name. Empty means the whole assembly.

    Every measurement used to be of the assembled object, and a datasheet does not work
    that way: BUC-001 states a 57 mm bowl, a 10 mm stem and a 48 mm support disc, all of
    which read as one `diameter_outer`. At most one could ever match, so the other two
    failed no matter what was built, and the generator spent its whole turn budget being
    refused over them. Naming the part is what makes those three separate measurements.

    Matched against the body tree the same way `ComponentTarget` is — normalised
    substrings, either direction — because an author writes `bowl` for a body called
    `bucket_bowl` and neither spelling is wrong.
    """
    part_excluded: str = ""
    """A part to leave out of the measurement, by name. Empty means measure everything.

    Specifications quote envelopes both ways round: a water bath is 233 mm tall "excluding
    cover", a conical tube has an end-to-end length "excluding cap". Both numbers are of the
    assembly, so they cannot be scoped to a part, and both are smaller than the assembly, so
    measuring all of it fails them for having the very part the datasheet set aside.

    Only bounding-box readings honour this. A diameter is a median radius, which already ignores
    a spout or a handle by construction, so "outside diameter excluding spout" needs nothing
    done to it. A part that does not resolve to a body leaves the measurement as it was, because
    the phrase is sometimes prose rather than a part -- "height without treating the surface".
    """
    measurement_location: str = ""
    measurement_state: str = ""
    tolerance_rel: float = 0.05
    """Relative error at or below which the dimension is fully conforming."""
    hard_fail_rel: float = 0.2
    critical: bool = True

    # A body of revolution is only round about one axis, and the profile measurements
    # need to know which. Upright parts are the overwhelming majority, hence the default.
    axis: Literal["x", "y", "z"] = "z"

    height_fraction_override: float | None = None
    """Where to slice, when the caller knows better than the location prose does.

    The compiled rubric carries this as a number rather than hoping the word "mid"
    appears in a datasheet's location field.
    """

    def height_fraction(self) -> float | None:
        """Where up the part to take a profile measurement, if the location says.

        Wall thickness and inner diameter are quoted at a stated height ("mid-height
        wall section"), and taking them at the wrong height on a tapered part is simply
        a different number.
        """
        if self.height_fraction_override is not None:
            return self.height_fraction_override
        text = self.measurement_location.lower()
        if "mid" in text:
            return 0.5
        if "base" in text or "bottom" in text:
            return 0.15
        if "rim" in text or "mouth" in text or "top" in text:
            return 0.85
        return None


class ComponentTarget(BaseModel):
    """A part the asset is required to have, and where it sits in the tree."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    kind: Literal["fixed", "cavity", "visual", "moving", "articulated"] = "fixed"
    quantity: int = 1
    parent: str | None = None
    critical: bool = True


class CavityTarget(BaseModel):
    """A connected open volume the asset has to enclose."""

    model_config = ConfigDict(extra="forbid")

    minimum_volume_ml: float = Field(gt=0.0)
    convergence_tolerance_rel: float = 0.01
    """How closely two successive voxel resolutions must agree for the number to count."""


class StabilityTarget(BaseModel):
    """The asset, left alone on a flat surface, should stay where it was put."""

    model_config = ConfigDict(extra="forbid")

    duration_s: float = 5.0
    max_translation_mm: float = 1.0
    max_tilt_deg: float = 2.0
    max_penetration_mm: float = 0.2


class TiltTarget(BaseModel):
    """The asset has to survive being rotated and put back."""

    model_config = ConfigDict(extra="forbid")

    axis: Literal["x", "y", "z"] = "y"
    angle_deg: float = 90.0
    tolerance_deg: float = 2.0
    speed_deg_s: float = 30.0
    repetitions: int = 1


class ProbeTarget(BaseModel):
    """A reference body that has to be able to get in, and must not fall out the bottom."""

    model_config = ConfigDict(extra="forbid")

    diameter_mm: float = 5.0
    mass_g: float = 0.1
    approach_axis: Literal["x", "y", "z"] = "z"
    approach_clearance_mm: float = 50.0


class VisualTarget(BaseModel):
    """Statements a render of the asset has to support."""

    model_config = ConfigDict(extra="forbid")

    features: list[str] = Field(default_factory=list)
    views: list[str] = Field(default_factory=lambda: ["front", "left", "top", "iso"])
    reference_note: str = ""


class OperationTarget(BaseModel):
    """One real-world affordance that must exist as executable kinematics."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    kind: Literal["press", "hinge", "slide", "rotate", "remove", "adjust", "generic"]
    child_hint: str
    parent_hint: str = ""
    expected_joint_types: list[
        Literal["hinge", "slide", "free"]
    ] = Field(default_factory=list)
    count: int = Field(default=1, ge=1)
    range_min: float | None = None
    range_max: float | None = None
    continuous: bool = False
    return_required: bool = True
    source_text: str = ""


class GroundingSpec(BaseModel):
    """Everything checkable about one asset, derived from its input specification."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    asset_class: str = ""
    summary: str = ""

    dimensions: list[DimensionTarget] = Field(default_factory=list)
    components: list[ComponentTarget] = Field(default_factory=list)
    cavity: CavityTarget | None = None
    stability: StabilityTarget | None = None
    tilt: TiltTarget | None = None
    probe: ProbeTarget | None = None
    visual: VisualTarget = Field(default_factory=VisualTarget)
    operations: list[OperationTarget] = Field(default_factory=list)
    total_mass_kg: Interval | None = None

    articulated: bool = False
    """Whether the object has an internal mechanism at all. False suppresses joint checks."""

    @classmethod
    def read(cls, path: Path) -> "GroundingSpec":
        return cls.model_validate(json.loads(Path(path).read_text()))

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude_none=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return path

    def is_empty(self) -> bool:
        """Whether there is anything here worth running a check for."""
        return not (
            self.dimensions
            or self.components
            or self.cavity
            or self.stability
            or self.tilt
            or self.probe
            or self.visual.features
            or self.operations
            or self.total_mass_kg
        )
