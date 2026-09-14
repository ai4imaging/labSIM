"""Compiling `input.md` into a rubric.

Deterministic by default, and that is the whole point. Measurements used to be decided
at judging time by a language model reading prose, which meant the same
asset could be graded differently on two runs and nobody could tell whether a score
moved because the asset changed or because the translation did. Here the translation
happens once, lands on disk as numbers, and can be read, diffed and argued with before a
single asset is scored.

Everything the compiler needs is already structured in the specification: components
carry a kind and a criticality, joints carry a type and a range, dimensions carry a unit.
The previous pipeline had all of this too and threw most of it away — it read the unit
field and then multiplied by 0.001 regardless, which is how a balance readability of
0.0001 g came to be compared against a 315 mm height.

The one genuinely ambiguous decision is which part a named dimension belongs to, and
which cross-section of it to take. That is resolved here by rule, never at judging time,
and never by inventing a target the specification did not state.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from amx.bench import case as case_module
from amx.bench.case import BenchCase
from amx.bench.rubric import (
    GROUNDABLE_QUANTITIES,
    Gate,
    ItemParams,
    QuantityClass,
    Rubric,
    RubricItem,
)
from amx.grounding.spec import GroundingSpec, MeasurementKind

RUBRIC_NAME = "rubric.json"
PROVENANCE_NAME = "provenance.json"


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #

_UNITS: dict[str, tuple[QuantityClass, float]] = {
    # length, to metres
    "mm": ("length", 1e-3),
    "cm": ("length", 1e-2),
    "m": ("length", 1.0),
    "um": ("length", 1e-6),
    "µm": ("length", 1e-6),
    "in": ("length", 0.0254),
    # mass, to kilograms
    "g": ("mass", 1e-3),
    "kg": ("mass", 1.0),
    "mg": ("mass", 1e-6),
    # volume, to millilitres
    "ml": ("volume", 1.0),
    "l": ("volume", 1000.0),
    "ul": ("volume", 1e-3),
    "µl": ("volume", 1e-3),
    # things a mesh has no opinion about
    "count": ("count", 1.0),
    "rpm": ("rotational_speed", 1.0),
    "c": ("temperature", 1.0),
    "°c": ("temperature", 1.0),
    "cm2": ("area", 1.0),
    "mm2": ("area", 1.0),
    "deg": ("angle", 1.0),
    "x_g": ("rcf", 1.0),
}


def classify_unit(unit: Any) -> tuple[QuantityClass, float]:
    """What kind of quantity a unit denotes, and how to get it into SI.

    An unrecognised unit is `other`, never a length. Guessing millimetres is what turned
    seventeen percent of this corpus's stated targets into impossible geometry checks.
    """
    text = str(unit or "").strip().lower()
    return _UNITS.get(text, ("other", 1.0))


# --------------------------------------------------------------------------- #
# measurement kinds
# --------------------------------------------------------------------------- #

_LONGEST_SIDE: MeasurementKind = "extent_max"
"""Alias for `extent_max`. Same name as in `case.py`, so a length with no axis is
labelled consistently even when the spec itself declines to make it a target."""

_WALL = re.compile(r"wall|thickness|gauge", re.I)
_INNER = re.compile(r"inner|internal|clear|bore|lumen|aperture|hole|well|opening|mouth", re.I)
_DIAMETER = re.compile(r"diameter|\bdia\b|\bod\b|\bid\b|bore", re.I)
_HEIGHT = re.compile(r"height|depth|\btall\b", re.I)
_LENGTH = re.compile(r"length|\blong\b|reach|span", re.I)
_WIDTH = re.compile(r"width|breadth", re.I)
_DEPTH_AXIS = re.compile(r"\bdepth\b", re.I)
_FOOTPRINT = re.compile(r"footprint|bench area|base area", re.I)
_PITCH = re.compile(r"pitch|spacing|centre[- ]to[- ]centre|center[- ]to[- ]center|interval", re.I)

_LOCAL_FEATURE = re.compile(
    r"well|hole|bore|slot|tip|cone|tooth|teeth|port|nozzle|socket|pocket|aperture|seat",
    re.I,
)

_HORIZONTAL = re.compile(r"width|depth|length|breadth|footprint", re.I)

_UNREACHABLE_STATES = frozenset(
    {"open", "opened", "extended", "deployed", "raised", "unfolded", "swung", "tilted"}
)
"""States the judge cannot put a static asset into before measuring it.

"Open height" is a real number on a real centrifuge and it is not the height of the
thing the author exported, which sits closed. Measuring it against the closed asset is a
guaranteed false failure, so it is declared unmeasurable instead of guessed at.
"""


def measurement_kind(name: str, location: str) -> MeasurementKind:
    """Which of the eight measurements a named dimension is asking for.

    Order matters: "wall thickness at mid height" is a thickness, not a height, and
    "clear inner diameter" is a bore, not an outside diameter. Anything with no signal at
    all becomes `extent_max`, which at least cannot be confused with a different axis the
    way the old `extent_z` fallback silently was.
    """
    text = f"{name} {location}"
    if _WALL.search(text):
        return "wall_thickness"
    if _DIAMETER.search(text):
        return "diameter_inner" if _INNER.search(text) else "diameter_outer"
    if _FOOTPRINT.search(text):
        return "footprint_max"
    if _HEIGHT.search(name):
        return "extent_y" if _DEPTH_AXIS.search(name) and "well" not in name.lower() else "extent_z"
    if _WIDTH.search(name):
        return "extent_x"
    if _LENGTH.search(name):
        return _LONGEST_SIDE
    return _LONGEST_SIDE


def slice_height(location: str) -> float | None:
    """Where up the part a profile measurement is quoted, compiled to a number now."""
    text = str(location or "").lower()
    if "mid" in text:
        return 0.5
    if "base" in text or "bottom" in text or "foot" in text:
        return 0.15
    if "rim" in text or "mouth" in text or "top" in text or "neck" in text:
        return 0.85
    return None


# --------------------------------------------------------------------------- #
# compilation
# --------------------------------------------------------------------------- #


def build_rubric(case: BenchCase) -> Rubric:
    """The whole answer key for one case, from its specification alone."""
    spec = case.to_grounding_spec()
    raw = case.spec
    items: list[RubricItem] = []

    items.extend(_physics_items(spec, raw))
    items.extend(_parts_items(case, spec, raw))
    items.extend(_operability_items(spec))

    rubric = Rubric(
        case_id=case.case_id,
        asset_class=case.asset_class,
        source_sha=_compiled_from(case),
        items=items,
    )
    rubric.gates = _gates(rubric)
    return rubric


# ---- physics ---------------------------------------------------------------- #


def _physics_items(spec: GroundingSpec, raw: dict[str, Any]) -> list[RubricItem]:
    conditions = raw.get("test_conditions") or {}
    penetration = float(conditions.get("abnormal_penetration_max_mm") or 0.2)

    items = [
        RubricItem(
            id="PHY-LOAD",
            axis="physics",
            primitive="load_compiles",
            subject="the exported MJCF",
            weight=3.0,
            critical=True,
            note="Nothing else is worth measuring on an asset a simulator will not accept.",
        ),
        RubricItem(
            id="PHY-INERTIA",
            axis="physics",
            primitive="part_inertia",
            subject="every part's mass and inertia",
            weight=3.0,
            critical=True,
        ),
        RubricItem(
            id="PHY-CONNECTED",
            axis="physics",
            primitive="assembly_connected",
            subject="the assembly holding together",
            weight=3.0,
            critical=True,
        ),
        RubricItem(
            id="PHY-INTERPENETRATION",
            axis="physics",
            primitive="rest_interpenetration",
            subject="parts sharing space at rest",
            weight=3.0,
            critical=True,
            params=ItemParams(max_penetration_mm=penetration),
        ),
        RubricItem(
            id="PHY-DENSITY",
            axis="physics",
            primitive="part_density",
            subject="every part being made of something",
            weight=2.0,
        ),
        RubricItem(
            id="PHY-COM",
            axis="physics",
            primitive="com_support",
            subject="the centre of mass over the footprint",
            weight=2.0,
        ),
    ]

    if spec.stability is not None:
        items.append(
            RubricItem(
                id="PHY-STABILITY",
                axis="physics",
                primitive="rest_stability",
                subject="standing on a bench unaided",
                weight=3.0,
                params=ItemParams(
                    duration_s=spec.stability.duration_s,
                    max_translation_mm=spec.stability.max_translation_mm,
                    max_tilt_deg=spec.stability.max_tilt_deg,
                    max_penetration_mm=spec.stability.max_penetration_mm,
                ),
            )
        )
    if spec.probe is not None:
        items.append(
            RubricItem(
                id="PHY-PROBE",
                axis="physics",
                primitive="probe_insert",
                subject="the reference body going in and staying in",
                weight=2.0,
                params=ItemParams(
                    probe_diameter_mm=spec.probe.diameter_mm,
                    probe_mass_g=spec.probe.mass_g,
                ),
            )
        )
    if spec.tilt is not None:
        items.append(
            RubricItem(
                id="PHY-TILT",
                axis="physics",
                primitive="tilt_restore",
                subject="surviving a pour or an inversion",
                weight=2.0,
                params=ItemParams(
                    tilt_angle_deg=spec.tilt.angle_deg,
                    tilt_axis=spec.tilt.axis,
                    tilt_tolerance_deg=spec.tilt.tolerance_deg,
                ),
            )
        )
    return items


# ---- parts ------------------------------------------------------------------ #


def _parts_items(
    case: BenchCase, spec: GroundingSpec, raw: dict[str, Any]
) -> list[RubricItem]:
    items: list[RubricItem] = []

    for component in spec.components:
        if component.kind == "cavity" and component.quantity > 1:
            # A cavity component used to produce nothing at all, and for the ones stated
            # in quantity that lost the requirement the object exists for. PCR-001 asks for
            # 96 tube wells; the topology check skipped the component because a cavity is
            # not a body, the volume integration reported the shallow tray between the
            # wells instead, and nothing ever counted a well. Ninety-six holes is a fact
            # about the geometry, and a section through them counts them.
            items.append(
                RubricItem(
                    id=f"PART-{component.id}-COUNT",
                    axis="parts",
                    primitive="feature_count",
                    subject=f"{component.quantity} × {component.name or component.id}",
                    weight=4.0,
                    critical=component.critical,
                    params=ItemParams(
                        part=component.name or component.id, count=component.quantity
                    ),
                )
            )
            continue
        if component.kind in {"cavity", "visual"}:
            continue
        items.append(
            RubricItem(
                id=f"PART-{component.id}",
                axis="parts",
                primitive="part_present",
                subject=component.name or component.id,
                weight=3.0,
                # Only a part that has to move has to be its own rigid body. A beaker's
                # wall and its pouring spout are one piece of glass, and a monolithic
                # beaker is a fair beaker — it still loses this item's points for not
                # naming them, but it does not lose the whole axis to a gate.
                critical=component.critical and component.kind in {"moving", "articulated"},
                params=ItemParams(part=component.name or component.id, count=component.quantity),
            )
        )

    items.append(
        RubricItem(
            id="PART-MASS",
            axis="parts",
            primitive="part_mass",
            subject="every part weighing something",
            weight=2.0,
            note="A welded part may legally weigh nothing in MuJoCo. It may not here.",
        )
    )

    single_cavity = any(c.kind == "cavity" and c.quantity == 1 for c in spec.components)
    if spec.cavity is not None or single_cavity:
        # A stated capacity gives the item a floor to grade against. Without one it still
        # asks the question the object exists to answer — is this hollow — which twenty-six
        # cases were never asked. A beaker whose interior was never subtracted is a glass
        # cylinder, and it used to lose no points for it.
        items.append(
            RubricItem(
                id="PART-CAVITY",
                axis="parts",
                primitive="cavity_volume",
                subject=(
                    "the internal volume" if spec.cavity is not None else "an enclosed interior"
                ),
                weight=3.0,
                params=ItemParams(
                    min_volume_ml=spec.cavity.minimum_volume_ml if spec.cavity else 0.0
                ),
            )
        )

    items.extend(_dimension_items(spec, raw))

    for index, feature in enumerate(spec.visual.features):
        items.append(
            RubricItem(
                id=f"VIS-{index:02d}",
                axis="parts",
                primitive="visual_feature",
                subject=feature,
                weight=1.0,
                params=ItemParams(feature=feature),
            )
        )
    return items


def _dimension_items(spec: GroundingSpec, raw: dict[str, Any]) -> list[RubricItem]:
    """One item per stated number, routed by what kind of number it is."""
    items: list[RubricItem] = []
    components = [component.name for component in spec.components if component.name]
    dimensions = raw.get("dimensions") or {}
    horizontal = _horizontal_ranking(dimensions)
    stated = {target.name: target for target in spec.dimensions}

    for name, entry in dimensions.items():
        if not isinstance(entry, dict):
            continue
        identifier = str(entry.get("id") or name)
        value = entry.get("value")
        critical = bool(entry.get("critical", True))
        quantity_class, scale = classify_unit(entry.get("unit"))
        location = str(entry.get("measurement_location") or "")

        if not isinstance(value, (int, float)) or value <= 0:
            items.append(
                _unscorable(
                    identifier,
                    "parts",
                    name,
                    "the benchmark itself records no target for this dimension",
                )
            )
            continue

        if quantity_class not in GROUNDABLE_QUANTITIES:
            items.append(
                _unscorable(
                    identifier,
                    "parts",
                    name,
                    f"{quantity_class} is a device property, not geometry; "
                    f"no mesh measurement can confirm {value} {entry.get('unit')}",
                )
            )
            continue

        local = bool(_LOCAL_FEATURE.search(name))

        if quantity_class != "length":
            if quantity_class == "volume" and local:
                items.append(
                    _unscorable(
                        identifier,
                        "parts",
                        name,
                        "a per-feature volume: the cavity integration reports the whole "
                        "part's capacity, so comparing it to one well's would pass any "
                        "plate with wells at all",
                    )
                )
                continue
            items.append(
                RubricItem(
                    id=f"PROP-{identifier}",
                    axis="parts",
                    primitive="property_assert",
                    subject=name,
                    weight=1.0,
                    params=ItemParams(
                        quantity=name,
                        quantity_class=quantity_class,
                        value=float(value) * scale,
                        unit=str(entry.get("unit") or ""),
                        part=_bind_part(name, components),
                        tolerance_rel=_tolerance(entry)[0],
                    ),
                )
            )
            continue


        if _tokens(name) & _UNREACHABLE_STATES:
            items.append(
                _unscorable(
                    identifier,
                    "parts",
                    name,
                    "quoted in a state the asset is not exported in; the judge would be "
                    "measuring the closed instrument against an open dimension",
                )
            )
            continue

        # `to_grounding_spec` already decided which lengths are measurable. Re-deriving
        # the kind here used to put `overall_length` back as `extent_max` after the spec
        # had declined it — two sources of truth, and the generator was held to one of
        # them it never saw as a check.
        target = stated.get(name)
        if target is None:
            items.append(
                _unscorable(
                    identifier,
                    "parts",
                    name,
                    "the name does not say how to measure this on the assembled object",
                )
            )
            continue

        full, partial, hard = _tolerance(entry)
        kind = horizontal.get(name) or target.kind
        # A well's "depth" is the cavity, not the plate's bench-plane Y. The spec names
        # it `extent_y` because the word is "depth"; the measurement that can actually
        # be taken on a hole cut into a part is the cavity's Z.
        if local and kind == "extent_y":
            kind = "extent_z"
        # The spec's own part wins where it has one. `to_grounding_spec` sets it exactly
        # where two same-kind dimensions had to be told apart, and `check_dimensions`
        # measures against it — so binding the rubric item to a different part by matching
        # component names would have the loop and the grader measuring two things again.
        part = target.part or _bind_part(name, components)
        # `pitch` is here because a repeated feature's spacing is the one local measurement
        # that gets *easier* the more of them there are: the section yields every centre,
        # so the spacing between neighbours needs no way to isolate a single well.
        if local and kind not in {"diameter_inner", "extent_z", "wall_thickness", "pitch"}:
            items.append(
                _unscorable(
                    identifier,
                    "parts",
                    name,
                    f"a {kind} of a local feature is not separable from the part it is "
                    "cut into with the cross-section machinery available",
                )
            )
            continue

        items.append(
            RubricItem(
                id=f"DIM-{identifier}",
                axis="parts",
                primitive="part_dimension",
                subject=name,
                weight=2.0,
                critical=critical,
                params=ItemParams(
                    part=part,
                    part_excluded=target.part_excluded,
                    dimension_id=identifier,
                    value_m=target.value_m,
                    kind=kind,
                    axis=target.axis,
                    height_fraction=slice_height(location),
                    local_feature=local,
                    tolerance_rel=full,
                    partial_rel=partial,
                    hard_fail_rel=hard,
                ),
                note=location,
            )
        )
    return items


def _horizontal_ranking(dimensions: dict[str, Any]) -> dict[str, MeasurementKind]:
    """Resolve a width/depth/length pair by size rather than by axis name.

    Whether a datasheet's "width" runs along x or along y in the exported model is the
    author's choice and not a defect, so the two bench-plane extents are ranked against
    each other: the larger stated number is the larger measured extent. With anything
    other than a clean pair this declines to guess and the keyword rule stands.
    """
    candidates = {
        name: float(entry["value"])
        for name, entry in dimensions.items()
        if isinstance(entry, dict)
        and _HORIZONTAL.search(name)
        and not _LOCAL_FEATURE.search(name)
        and not _tokens(name) & _UNREACHABLE_STATES
        and classify_unit(entry.get("unit"))[0] == "length"
        and isinstance(entry.get("value"), (int, float))
        and entry["value"] > 0
    }
    if len(candidates) != 2:
        return {}
    larger, smaller = sorted(candidates, key=lambda name: -candidates[name])
    return {larger: "footprint_max", smaller: "footprint_min"}


def _tolerance(entry: dict[str, Any]) -> tuple[float, float, float]:
    tolerance = entry.get("tolerance") or {}
    hard_raw = entry.get("hard_fail_relative_error_max") or {}
    hard = hard_raw.get("value") if isinstance(hard_raw, dict) else hard_raw
    return (
        float(tolerance.get("full_relative_error_max") or 0.05),
        float(tolerance.get("partial_relative_error_max") or 0.10),
        float(hard or 0.20),
    )


def _axis_for(name: str) -> str:
    lowered = name.lower()
    if "width" in lowered:
        return "x"
    if "depth" in lowered and "well" not in lowered:
        return "y"
    return "z"


def _bind_part(name: str, components: list[str]) -> str:
    """Which component a named dimension is about, if any.

    A dimension called `well_depth` belongs to the well, not to the plate the wells are
    in, and measuring it on the plate is how the old judge read 85 mm for an 8.2 mm
    mouth. Matching is on shared distinctive tokens, and no match means the whole asset,
    which is the right answer for "overall width".
    """
    tokens = {token for token in _tokens(name) if len(token) >= 4 and token not in _STOPWORDS}
    if not tokens:
        return ""
    best: tuple[int, str] = (0, "")
    for component in components:
        overlap = len(tokens & _tokens(component))
        if overlap > best[0]:
            best = (overlap, component)
    return best[1]


_STOPWORDS = frozenset(
    {
        # measurement words: every dimension has one, so they identify nothing
        "overall",
        "outer",
        "inner",
        "total",
        "nominal",
        "clear",
        "external",
        "internal",
        "width",
        "height",
        "depth",
        "length",
        "breadth",
        "diameter",
        "thickness",
        "footprint",
        # state words: "open height" is not a dimension of the OPEN button
        "open",
        "closed",
        "folded",
        "extended",
        "retracted",
        "assembled",
        "working",
    }
)


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(text).lower()) if token}


def _unscorable(identifier: str, axis: str, subject: str, reason: str) -> RubricItem:
    return RubricItem(
        id=f"NA-{identifier}",
        axis=axis,  # type: ignore[arg-type]
        primitive="not_scorable",
        subject=subject,
        weight=1.0,
        params=ItemParams(reason=reason),
    )


# ---- operability ------------------------------------------------------------ #


def _operability_items(spec: GroundingSpec) -> list[RubricItem]:
    """Every affordance the specification names, as something that has to actually work.

    The derivation of what counts as an affordance already lives in `BenchCase` and is
    shared with the authoring prompt, so the generator is told exactly what it will be
    measured on. That is deliberate: this axis is not a trap, it is a requirement.
    """
    items: list[RubricItem] = []
    for target in spec.operations:
        items.append(
            RubricItem(
                id=f"OP-{target.id}",
                axis="operability",
                primitive="operation_exercise",
                subject=target.name,
                weight=4.0,
                critical=True,
                params=ItemParams(
                    part=target.child_hint,
                    parent_part=target.parent_hint,
                    operation_id=target.id,
                    operation_kind=target.kind,
                    expected_joint_types=list(target.expected_joint_types),
                    count=target.count,
                    range_min=target.range_min,
                    range_max=target.range_max,
                    continuous=target.continuous,
                    return_required=target.return_required,
                ),
                note=target.source_text,
            )
        )
        items.append(
            RubricItem(
                id=f"SWEEP-{target.id}",
                axis="operability",
                primitive="swept_collision",
                subject=f"{target.name} clearing its whole travel",
                weight=2.0,
                params=ItemParams(part=target.child_hint, samples=16),
            )
        )
    return items


# ---- gates ------------------------------------------------------------------ #


def _gates(rubric: Rubric) -> list[Gate]:
    critical_parts = [
        item.id
        for item in rubric.items
        if item.primitive == "part_present" and item.critical
    ]
    critical_ops = [
        item.id
        for item in rubric.items
        if item.primitive == "operation_exercise" and item.critical
    ]
    gates = [
        Gate(
            id="G0-LOAD",
            item_ids=["PHY-LOAD"],
            description="Whether the compiled MJCF loads and stays finite.",
        ),
        Gate(
            id="G3-PHYSICS",
            item_ids=["PHY-INERTIA", "PHY-CONNECTED", "PHY-INTERPENETRATION"],
            description=(
                "Whether the assembly carries mass, stays in one piece, and does not "
                "occupy the same space twice. Each miss costs its own item."
            ),
        ),
    ]
    if critical_parts:
        gates.append(
            Gate(
                id="G1-PARTS",
                item_ids=critical_parts,
                description="Whether every named critical component is present as a body.",
            )
        )
    if critical_ops:
        gates.append(
            Gate(
                id="G2-OPERABILITY",
                item_ids=critical_ops,
                description="Whether every named affordance executes. Each miss costs that operation.",
            )
        )
    return gates


def write_case_artifacts(case: BenchCase) -> dict[str, Path]:
    """Compile one case and write `rubric.json` next to its specification."""
    path = build_rubric(case).write(case.directory / RUBRIC_NAME)
    return {"rubric": path}


def _compiled_from(case: BenchCase) -> str:
    """Everything the compiled rubric depends on: the specification and the derivation.

    Fingerprinting `input.md` alone was not enough, and the gap was not theoretical. A
    committed `rubric.json` for CRY-001 held a screw cap's travel as 6.28319 — an angle in
    a linear joint's limits — and the fix went into how the range is derived, not into the
    case. `input.md` had not changed, so the stale key was reused and the run was still
    graded against the old requirement. Any repair to the derivation would have been
    silently discarded on every case whose rubric was already on disk.
    """
    digest = hashlib.sha256(case.input_path.read_bytes())
    for module in (Path(__file__), Path(case_module.__file__)):
        digest.update(module.read_bytes())
    return digest.hexdigest()[:16]


def load_rubric(case: BenchCase) -> Rubric:
    """The compiled rubric for a case, built on demand when it is missing or stale."""
    path = case.directory / RUBRIC_NAME
    digest = _compiled_from(case)
    if path.is_file():
        try:
            existing = Rubric.read(path)
            if existing.source_sha == digest:
                return existing
        except ValueError:
            pass
    return build_rubric(case)


__all__ = [
    "PROVENANCE_NAME",
    "RUBRIC_NAME",
    "build_rubric",
    "classify_unit",
    "load_rubric",
    "measurement_kind",
    "write_case_artifacts",
]
