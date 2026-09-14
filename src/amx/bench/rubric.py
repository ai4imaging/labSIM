"""The answer key, written as measurements instead of prose.

A rubric item names one primitive from a fixed list and supplies its parameters as
numbers. Nothing in the scoring path is free text, and nothing is translated at judging
time — the translation happens once, when the rubric is compiled from `input.md`, and it
lands on disk where it can be read and disputed before a single asset is scored.

That constraint is the whole design, and it comes from watching the alternative fail.
The scheme this replaced asked its questions in English and had a language model turn
them into measurements while judging. A clause nobody could measure was dropped from the
list before the rest were averaged, so it cost nothing; a quantity in grams was compared
against a length in millimetres; and "the model has a body called `lid`" was accepted as
proof that the lid opens. An asset that would not load in MuJoCo at all scored 39 out of
100. None of that was a threshold that needed tightening.

So anything the primitive list cannot express is written down as `not_scorable` when the
rubric is compiled, where it is visible and subtracted from the reachable total, rather
than discovered while judging and quietly skipped.

Three axes, because the question this benchmark is asking has three parts:

* `parts` — are the pieces there, and are they made of something?
* `operability` — does what should move actually move, the way the real instrument moves?
* `physics` — is the result an object a simulator can accept?

A failed item costs its own weight, and nothing else's: an axis is never wiped, and neither
is the case.

That leaves the score unable to answer one question, so it no longer tries to. A weighted
average of thirteen dimensions says how much of an asset is right; it cannot say whether the
asset may be used, because the one dimension that decides that is worth the same two points as
the twelve that don't. PCR-001 bored its wells 4.00 mm for 5.20 mm tubes, held none of the tubes
it exists to hold, and scored 92.4 and "passed". So the verdict is now separate from the score:
a critical item missed by more than the benchmark's own hard-failure threshold, or a gate that
does not hold, withholds the pass, while every item keeps exactly the credit it earned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from amx.grounding.spec import MeasurementKind

FORMAT = "amx-rubric"
"""File-format marker for a compiled rubric. Not a version number."""

Axis = Literal["parts", "operability", "physics"]

AXIS_BUDGET: dict[Axis, float] = {"parts": 35.0, "operability": 40.0, "physics": 25.0}
"""How the hundred points divide when a case exercises all three axes.

A beaker has nothing that moves, and an axis with no items gets its budget shared out
across the axes that do have them, so a beaker is still marked out of 100 and a
centrifuge's forty operability points are not quietly awarded to the glassware.
"""

PASS_THRESHOLD = 80.0

Primitive = Literal[
    # ---- parts and properties -------------------------------------------------
    "part_present",
    "part_dimension",
    "part_mass",
    "part_density",
    "part_inertia",
    "cavity_volume",
    "feature_count",
    "visual_feature",
    "property_assert",
    # ---- operability ----------------------------------------------------------
    "operation_exercise",
    "swept_collision",
    # ---- physical validity ----------------------------------------------------
    "load_compiles",
    "rest_stability",
    "rest_interpenetration",
    "assembly_connected",
    "com_support",
    "probe_insert",
    "tilt_restore",
    # ---- the honest escape hatch ----------------------------------------------
    "not_scorable",
]
"""Every measurement the judge can take. There is no eighteenth option.

A rubric naming something outside this list does not load, which is the point: the
corpus this replaced carried thirty-five `evaluation_mode` strings against eight
implemented measurements, and the gap between those two numbers was where the score went.
"""

QuantityClass = Literal[
    "length",
    "mass",
    "volume",
    "count",
    "area",
    "angle",
    "rotational_speed",
    "temperature",
    "rcf",
    "other",
]
"""What kind of thing a stated number is.

Carried explicitly because the corpus states 220 g, 4400 rpm, 24 positions and 500 °C in
the same `dimensions` block as 70 mm. Anything that is not a `length`, `mass`, `volume`
or `count` cannot be grounded in a mesh and compiles to `not_scorable` rather than to a
guess.
"""

GROUNDABLE_QUANTITIES: frozenset[str] = frozenset({"length", "mass", "volume", "count"})


class ItemParams(BaseModel):
    """Every parameter any primitive can take, named rather than described.

    One flat model instead of a union because it is also the schema a language model
    fills in during compilation: a fixed set of typed fields with `extra="forbid"` can be
    wrong about which measurement to take, and that is visible on disk, whereas free-form
    parameters can be subtly wrong in a way that looks like a result.
    """

    model_config = ConfigDict(extra="forbid")

    # subject selection
    part: str = ""
    """Component name or id the item is about. Empty means the asset as a whole."""
    joint: str = ""
    operation_id: str = ""

    # operation_exercise — the whole affordance, carried here so the rubric is
    # self-contained and the judge never has to re-derive it from the specification.
    operation_kind: Literal[
        "press", "hinge", "slide", "rotate", "remove", "adjust", "generic"
    ] = "generic"
    parent_part: str = ""
    expected_joint_types: list[Literal["hinge", "slide", "free"]] = Field(default_factory=list)
    count: int = Field(default=1, ge=1)
    range_min: float | None = None
    range_max: float | None = None
    continuous: bool = False
    return_required: bool = True

    # part_dimension
    dimension_id: str = ""
    value_m: float = 0.0
    kind: MeasurementKind = "extent_z"
    axis: Literal["x", "y", "z"] = "z"
    height_fraction: float | None = None
    part_excluded: str = ""
    """A part to leave off before measuring, for an envelope stated "excluding" something.

    Carries `DimensionTarget.part_excluded` through to the judge so the in-loop check and the
    score are taken on the same geometry.
    """
    local_feature: bool = False
    """Whether the number describes a feature cut into the part rather than the part.

    A well's depth is not the plate's height and its mouth is not the plate's width.
    Reading a feature off the part's bounding box was the single biggest source of false
    failures in the old scheme, so a feature is measured by cross-section instead.
    """
    tolerance_rel: float = 0.05
    partial_rel: float = 0.10
    hard_fail_rel: float = 0.20

    # part_mass / part_density / part_inertia
    min_mass_kg: float = 0.0
    max_mass_kg: float = 0.0
    """Zero means unbounded. A stated total mass fills both."""

    # cavity_volume
    min_volume_ml: float = 0.0

    # feature_count
    feature_axis: Literal["x", "y", "z"] = "z"
    """Which way to slice to bring the repeated features into one section."""

    # rest_stability
    duration_s: float = 5.0
    max_translation_mm: float = 1.0
    max_tilt_deg: float = 2.0
    max_penetration_mm: float = 0.2

    # probe_insert
    probe_diameter_mm: float = 5.0
    probe_mass_g: float = 0.1

    # tilt_restore
    tilt_angle_deg: float = 90.0
    tilt_axis: Literal["x", "y", "z"] = "y"
    tilt_tolerance_deg: float = 2.0

    # swept_collision
    samples: int = 16

    # com_support
    min_margin_mm: float = 0.0

    # property_assert
    quantity: str = ""
    quantity_class: QuantityClass = "other"
    value: float = 0.0
    unit: str = ""

    # visual_feature
    feature: str = ""

    # not_scorable
    reason: str = ""


class RubricItem(BaseModel):
    """One thing the asset has to demonstrate, and exactly how that will be measured."""

    model_config = ConfigDict(extra="forbid")

    id: str
    axis: Axis
    primitive: Primitive
    subject: str = ""
    """What the item is about, for a human reading the scorecard."""
    weight: float = Field(default=1.0, gt=0.0)
    critical: bool = False
    """Whether the asset is unusable without this. Missing it by more than `hard_fail_rel`
    withholds the pass; it still costs only this item's weight."""
    params: ItemParams = Field(default_factory=ItemParams)
    note: str = ""

    @property
    def scorable(self) -> bool:
        return self.primitive != "not_scorable"


class Gate(BaseModel):
    """A named bundle of items that has to hold for the asset to pass.

    A reader can see at a glance whether the asset loaded, whether the critical parts are
    present, and whether the named operations ran. A gate that does not hold withholds the
    pass without changing the score: each referenced item still costs only its own weight.
    """

    model_config = ConfigDict(extra="forbid")

    id: Literal["G0-LOAD", "G1-PARTS", "G2-OPERABILITY", "G3-PHYSICS"]
    item_ids: list[str] = Field(default_factory=list)
    zeroes: list[Axis] = Field(default_factory=list)
    """Unused. Older compiled files still carry it; the judge ignores it."""
    description: str = ""


class Rubric(BaseModel):
    """One case's answer key.

    Derived from `input.md` alone, which is also all the generator sees, so publishing it
    leaks nothing. The wall the benchmark used to maintain between task and answer key
    was protecting prose thresholds that the specification already stated out loud.
    """

    model_config = ConfigDict(extra="forbid")

    format: Literal["amx-rubric"] = FORMAT
    case_id: str
    asset_class: str = ""
    source_sha: str = ""
    """Hash of the `input.md` this was compiled from, so a stale rubric is detectable."""

    items: list[RubricItem] = Field(default_factory=list)
    gates: list[Gate] = Field(default_factory=list)

    # ---- reading ---------------------------------------------------------------

    @classmethod
    def read(cls, path: Path) -> "Rubric":
        return cls.model_validate(json.loads(Path(path).read_text()))

    @staticmethod
    def is_rubric(payload: Any) -> bool:
        """Whether a parsed JSON file is a compiled rubric."""
        return isinstance(payload, dict) and (
            payload.get("format") == FORMAT or ("items" in payload and "gates" in payload)
        )

    def write(self, path: Path) -> Path:
        """Write only what differs from the defaults.

        A rubric is meant to be read by the person disputing a score, and forty parameter
        slots per item — thirty-eight of them at their default — is noise nobody opens.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": FORMAT, **self.model_dump(mode="json", exclude_defaults=True)}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return path

    def by_axis(self, axis: Axis) -> list[RubricItem]:
        return [item for item in self.items if item.axis == axis]

    def item(self, item_id: str) -> RubricItem | None:
        return next((item for item in self.items if item.id == item_id), None)

    # ---- budgets ---------------------------------------------------------------

    def budgets(self) -> dict[Axis, float]:
        """Points per axis, with empty axes shared out over the populated ones.

        Shared proportionally rather than equally, so dropping operability from a beaker
        keeps the parts-to-physics ratio the corpus was designed around.
        """
        populated = [axis for axis in AXIS_BUDGET if any(self.by_axis(axis))]
        if not populated:
            return dict.fromkeys(AXIS_BUDGET, 0.0)
        total = sum(AXIS_BUDGET[axis] for axis in populated)
        return {
            axis: (AXIS_BUDGET[axis] / total * 100.0 if axis in populated else 0.0)
            for axis in AXIS_BUDGET
        }

    def achievable(self) -> float:
        """The most any asset could score, given what this case declares unmeasurable.

        A `not_scorable` item is removed from its axis entirely and its share of the
        budget vanishes from the total, so the reachable column is honest.
        """
        budgets = self.budgets()
        total = 0.0
        for axis, budget in budgets.items():
            items = self.by_axis(axis)
            if not items:
                continue
            weight = sum(item.weight for item in items)
            scorable = sum(item.weight for item in items if item.scorable)
            if weight <= 0.0:
                continue
            total += budget * scorable / weight
        return round(total, 2)


__all__ = [
    "AXIS_BUDGET",
    "FORMAT",
    "GROUNDABLE_QUANTITIES",
    "PASS_THRESHOLD",
    "Axis",
    "Gate",
    "ItemParams",
    "Primitive",
    "QuantityClass",
    "Rubric",
    "RubricItem",
]
