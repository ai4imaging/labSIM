"""A benchmark case: the task, and everything derived from it.

`BenchCase` reads `input.md` and nothing else. `to_asset_request()` gives the generator
the specification; `to_grounding_spec()` turns the same text into the targets the
authoring loop checks itself against; and the rubric the judge scores against is compiled
from it too, by `amx.bench.compiler`.

That there is only one source is the point of the file. The benchmark used to keep a
separate hand-written answer key beside each specification, which created a wall to
police — it is very easy to make a benchmark score well by leaking pass conditions into
the generation prompt, and impossible to notice afterwards from the numbers. Now there is
nothing on the answer-key side that the generator was not already shown, so the rubric can
be published, read and disputed without any of it leaking an advantage.
"""

from __future__ import annotations

import math
import re
from functools import cached_property
from pathlib import Path
from typing import Any

from amx.bench.spec_md import parse_spec
from amx.grounding.spec import (
    CavityTarget,
    ComponentTarget,
    DimensionTarget,
    GroundingSpec,
    OperationTarget,
    ProbeTarget,
    StabilityTarget,
    TiltTarget,
    VisualTarget,
)

MM = 0.001


class BenchCase:
    """One directory under `3D_asset_cases/`."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.input_path = self.directory / "input.md"
        if not self.input_path.is_file():
            raise FileNotFoundError(
                f"{self.directory} is not a benchmark case; it needs an input.md"
            )

    # ---- identity -------------------------------------------------------------

    @cached_property
    def spec(self) -> dict[str, Any]:
        return parse_spec(self.input_path)

    @property
    def case_id(self) -> str:
        return self.directory.name.split("_")[0]

    @property
    def asset_class(self) -> str:
        """The class name, in the underscored form every artefact is keyed by."""
        identity = self.spec.get("asset_identity") or {}
        name = str(identity.get("name") or identity.get("name_en") or "")
        return re.sub(r"\s+", "_", name.strip())

    @property
    def slug(self) -> str:
        return self.directory.name

    def __repr__(self) -> str:
        return f"BenchCase({self.case_id} {self.asset_class})"

    # ---- the task half --------------------------------------------------------

    def to_asset_request(self) -> str:
        """What the generator is asked to build.

        The whole of `input.md`, with a short instruction on top. Not a summary: the
        specification is the task, summarising it would be the benchmark quietly doing
        part of the work, and the parts that look like boilerplate — measurement states,
        provenance classes — are exactly what stops a model from measuring the wrong
        feature.
        """
        operations = self._operation_targets()
        operation_contract = "\n".join(
            (
                f"- {target.id}: {target.kind} {target.name!r}; create {target.count} "
                f"independent movable Part(s), expected MuJoCo joint type(s) "
                f"{', '.join(target.expected_joint_types) or 'any non-fixed joint'}"
                + (
                    f", limits [{target.range_min:g}, {target.range_max:g}] SI"
                    if target.range_min is not None and target.range_max is not None
                    else _operation_limit_text(target)
                )
            )
            for target in operations
        )
        operation_section = (
            "\n\nPhysical-operation contract (derived only from the specification):\n"
            "Every listed affordance must be a separate physical Part with positive mass and "
            "inertia, collision geometry, and executable kinematics. A painted, fused, or "
            "fixed visual proxy does not satisfy an operation. Verify both endpoint motion "
            "and return to the initial state before finishing.\n"
            f"{operation_contract}"
            if operation_contract
            else ""
        )
        return (
            f"Build a simulation-ready 3D asset of a {self.asset_class} to the "
            f"specification below.\n\n"
            f"Reproduce every dimension whose value is given, to within its stated "
            f"tolerance, measured at the stated location and in the stated state. Where a "
            f"value is null the specification does not know it; choose a defensible value "
            f"for the class and stay consistent with the dimensions that are known.\n\n"
            f"Build every required component. Components of kind `cavity` are open "
            f"internal regions and must be genuinely hollow, not a visual impression of "
            f"one. Components of kind `visual` need only be visible.\n\n"
            f"{_MOULDED_FORM}"
            f"{self._measurement_contract()}"
            f"{operation_section}\n\n"
            f"---\n\n{self.input_path.read_text()}"
        )

    def _measurement_contract(self) -> str:
        """How each stated dimension will be read off the finished geometry.

        The specification says what the numbers are; it does not say where a caliper goes,
        and the difference is not academic. PCR-001 states a 5.2 mm `hole_diameter` and the
        asset built 5.2 mm sockets with 4.0 mm bores — a defensible reading of the words,
        and the wrong one, because the number is the opening a tube has to pass through.
        Saying which measurement will be taken is the only part of the task the input file
        cannot supply, so it is the only part worth adding to it.

        Dimensions the checks decline are named too. Silence used to be indistinguishable
        from approval, and a generator that cannot tell the two apart will read four
        unmeasured requirements as four satisfied ones.
        """
        spec = self.to_grounding_spec()
        lines = [_MEASUREMENT_PROSE[target.kind].format(
            name=target.name,
            value=target.value_m * 1000.0,
            part=f"the {target.part}" if target.part else "the object",
        ) for target in spec.dimensions if target.kind in _MEASUREMENT_PROSE]

        counted = [
            f"- {component.name!r}: a section across the object must find "
            f"{component.quantity} of them. They are openings, so each one is counted by "
            f"its own closed ring — {component.quantity} solid studs would count as zero."
            for component in spec.components
            if component.kind == "cavity" and component.quantity > 1
        ]

        declined = sorted(
            name
            for name, entry in (self.spec.get("dimensions") or {}).items()
            if isinstance(entry, dict)
            and isinstance(entry.get("value"), (int, float))
            and (entry.get("value") or 0) > 0
            and name not in {target.name for target in spec.dimensions}
        )

        if not (lines or counted or declined):
            return ""

        parts = ["\n\nHow these will be measured (the input file states the numbers, not "
                 "the datum):\n"]
        parts.extend(f"{line}\n" for line in [*lines, *counted])
        if declined:
            parts.append(
                f"\nNot checked, so build them to your own judgement and do not read "
                f"silence as approval: {', '.join(declined)}.\n"
            )
        return "".join(parts)

    def to_grounding_spec(self) -> GroundingSpec:
        """The in-loop checks, derived from the specification alone.

        Every threshold here is one the task states, so an asset that satisfies this is one
        that satisfies what it was asked for — which is what makes the benchmark score
        afterwards worth anything.
        """
        spec = self.spec
        return GroundingSpec(
            asset_id=self.case_id,
            asset_class=self.asset_class,
            summary=self._summary(),
            dimensions=self._dimension_targets(),
            components=self._component_requirements(),
            cavity=self._cavity_requirement(),
            stability=self._stability_requirement(),
            tilt=self._tilt_requirement(),
            probe=self._probe_requirement(),
            visual=self._visual_requirement() or VisualTarget(),
            operations=self._operation_targets(),
            articulated=bool((spec.get("articulation_requirements") or {}).get("applicable")),
        )

    # ---- derivation -----------------------------------------------------------

    def _summary(self) -> str:
        identity = self.spec.get("asset_identity") or {}
        parts = [str(identity.get("name") or self.asset_class)]
        if identity.get("manufacturer"):
            parts.append(str(identity["manufacturer"]))
        if identity.get("model"):
            parts.append(str(identity["model"]))
        if identity.get("nominal_capacity_ml"):
            parts.append(f"{identity['nominal_capacity_ml']} mL nominal")
        return ", ".join(parts)

    def _dimension_targets(self) -> list[DimensionTarget]:
        """Every dimension the specification actually knows.

        A null value is not a target. The specification says so in as many words —
        `unknown_reason` explains why nobody measured it — and inventing one here would
        hold the generator to a number the benchmark itself refuses to assert.
        """
        targets = []
        for name, entry in (self.spec.get("dimensions") or {}).items():
            if not isinstance(entry, dict):
                continue
            value = entry.get("value")
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            location = str(entry.get("measurement_location") or "")
            scale = _unit_scale(entry.get("unit"))
            kind = _measurement_kind(name, location)
            if scale is None or kind is None:
                # Not a length, or a length of something nothing here can find. Either
                # way there is no measurement to hold the generator to.
                continue
            tolerance = entry.get("tolerance") or {}
            hard = entry.get("hard_fail_relative_error_max") or {}
            targets.append(
                DimensionTarget(
                    id=str(entry.get("id") or name),
                    name=name,
                    value_m=float(value) * scale,
                    kind=kind,
                    measurement_location=location,
                    measurement_state=str(entry.get("measurement_state") or ""),
                    tolerance_rel=float(tolerance.get("full_relative_error_max") or 0.05),
                    hard_fail_rel=float(
                        (hard.get("value") if isinstance(hard, dict) else hard) or 0.2
                    ),
                    critical=bool(entry.get("critical", True)),
                    axis=_axis_for(name),
                    part_excluded=_excluded_part(name, location, kind),
                )
            )
        return _without_contradictions(targets)

    def _component_requirements(self) -> list[ComponentTarget]:
        return [
            ComponentTarget(
                id=str(entry.get("id") or ""),
                name=str(entry.get("name") or ""),
                kind=_component_kind(entry.get("kind")),
                quantity=int(entry.get("quantity") or 1),
                parent=str(entry["parent"]) if entry.get("parent") else None,
                critical=bool(entry.get("critical", False)),
            )
            for entry in (self.spec.get("required_components") or [])
            if isinstance(entry, dict) and entry.get("id")
        ]

    def _operation_targets(self) -> list[OperationTarget]:
        """Derive executable affordances from joints, components and named controls."""
        components = {
            str(entry.get("id")): entry
            for entry in (self.spec.get("required_components") or [])
            if isinstance(entry, dict) and entry.get("id")
        }
        targets: list[OperationTarget] = []
        covered: set[str] = set()
        joints = (self.spec.get("articulation_requirements") or {}).get("joints") or []
        for index, joint in enumerate(joints):
            if not isinstance(joint, dict):
                continue
            raw_child = str(joint.get("child_component") or joint.get("child") or "")
            child_entry = components.get(raw_child, {})
            child = str(child_entry.get("name") or raw_child).strip()
            raw_parent = str(joint.get("parent_component") or joint.get("parent") or "")
            parent = str(components.get(raw_parent, {}).get("name") or raw_parent).strip()
            raw_type = str(joint.get("type") or "").lower()
            # A grouped contact-trigger requirement is expanded into the visible named
            # controls below, otherwise one invisible "controls" body could satisfy it.
            if "contact_trigger" in raw_type:
                continue
            kind, joint_types, continuous = _operation_shape(raw_type, child)
            range_min, range_max, angular = _joint_range_si(joint.get("range") or {})
            if not _range_fits_joint_types(angular, joint_types):
                range_min = range_max = None
            count = _operation_count(raw_type, int(child_entry.get("quantity") or 1))
            targets.append(
                OperationTarget(
                    id=str(joint.get("id") or f"OP-J{index + 1}"),
                    name=child or raw_type or f"mechanism {index + 1}",
                    kind=kind,
                    child_hint=child or raw_type,
                    parent_hint=parent,
                    expected_joint_types=joint_types,
                    count=count,
                    range_min=range_min,
                    range_max=range_max,
                    continuous=continuous,
                    source_text="; ".join(
                        str(value)
                        for value in (joint.get("zero"), joint.get("limits"))
                        if value
                    ),
                )
            )
            covered.add(_normalise(child))

        for entry in components.values():
            name = str(entry.get("name") or "")
            kind = str(entry.get("kind") or "").lower()
            if kind == "removable" and not _covered_operation(name, covered):
                targets.append(
                    OperationTarget(
                        id=f"OP-{entry['id']}-REMOVE",
                        name=name,
                        kind="remove",
                        child_hint=name,
                        parent_hint=str(entry.get("parent") or ""),
                        expected_joint_types=["free", "slide"],
                        count=int(entry.get("quantity") or 1),
                        source_text="required component is explicitly removable",
                    )
                )
                covered.add(_normalise(name))
            elif kind in {"moving", "movable", "independent_moving", "articulated"} and not _covered_operation(name, covered):
                op_kind, joint_types, continuous = _operation_shape("", name)
                targets.append(
                    OperationTarget(
                        id=f"OP-{entry['id']}-MOVE",
                        name=name,
                        kind=op_kind,
                        child_hint=name,
                        parent_hint=str(entry.get("parent") or ""),
                        expected_joint_types=joint_types,
                        count=int(entry.get("quantity") or 1),
                        continuous=continuous,
                        source_text="required component is explicitly movable",
                    )
                )
                covered.add(_normalise(name))

            for control in _named_controls(name):
                targets.append(
                    OperationTarget(
                        id=f"OP-{entry['id']}-{_normalise(control).upper()}",
                        name=control,
                        kind="press",
                        child_hint=control,
                        parent_hint=name,
                        expected_joint_types=["slide"],
                        source_text=f"named physical control in required component {entry['id']}",
                    )
                )
        return targets

    def _cavity_requirement(self) -> CavityTarget | None:
        """The volume the cavity has to hold, when the specification names one.

        `geometry_proxy_min_ml` is the field to read, and reading it rather than
        `nominal_capacity_ml` matters: the specification is explicit that nominal capacity
        is a product label and not a brim-volume target, and the proxy is the number it is
        willing to be measured against.
        """
        for entry in self.spec.get("functional_requirements") or []:
            if not isinstance(entry, dict):
                continue
            minimum = entry.get("geometry_proxy_min_ml")
            if isinstance(minimum, (int, float)) and minimum > 0:
                return CavityTarget(minimum_volume_ml=float(minimum))

        # Only two cases in the corpus state a proxy, so reading nothing else left the
        # cavity ungraded in thirty-five of the thirty-nine that require one. A nominal
        # capacity is a product label rather than a brim volume, which rules it out as an
        # equality and not as a floor: a vessel sold as 250 mL holds at least 250 mL or it
        # is mislabelled. `CavityTarget` is a floor, so the objection does not apply.
        for entry in self.spec.get("functional_requirements") or []:
            if not isinstance(entry, dict):
                continue
            for field in ("working_capacity_ml", "nominal_capacity_ml"):
                stated = entry.get(field)
                if isinstance(stated, (int, float)) and stated > 0:
                    return CavityTarget(minimum_volume_ml=float(stated))
        return None

    def _stability_requirement(self) -> StabilityTarget | None:
        conditions = self.spec.get("test_conditions") or {}
        translation = conditions.get("stable_translation_max_mm")
        tilt = conditions.get("stable_tilt_max_deg")
        if not isinstance(translation, (int, float)) or not isinstance(tilt, (int, float)):
            return None
        return StabilityTarget(
            duration_s=float(_support_duration(self.spec) or 5.0),
            max_translation_mm=float(translation),
            max_tilt_deg=float(tilt),
            max_penetration_mm=float(conditions.get("abnormal_penetration_max_mm") or 0.2),
        )

    def _tilt_requirement(self) -> TiltTarget | None:
        """A pour or an inversion, when a protocol asks for one."""
        for entry in self.spec.get("protocol_conditioned_requirements") or []:
            if not isinstance(entry, dict):
                continue
            parameters = entry.get("parameters") or {}
            angle = parameters.get("tilt_deg")
            if isinstance(angle, (int, float)) and angle > 0:
                return TiltTarget(
                    angle_deg=float(angle),
                    speed_deg_s=float(parameters.get("speed_deg_s") or 30.0),
                    repetitions=int(parameters.get("repetitions") or 1),
                )
        return None

    def _probe_requirement(self) -> ProbeTarget | None:
        """The reference sphere, where the case supplies one.

        Its diameter and mass are stated per case, and using the stated ones rather than a
        default is what makes "can something be put in and got out" mean the same thing
        for a microtube as for a water bath.
        """
        for entry in self.spec.get("reference_consumables") or []:
            if not isinstance(entry, dict):
                continue
            geometry = entry.get("geometry") or {}
            diameter = geometry.get("diameter_mm")
            if not isinstance(diameter, (int, float)) or diameter <= 0:
                continue
            return ProbeTarget(
                diameter_mm=float(diameter),
                mass_g=float(geometry.get("mass_g") or 0.1),
                approach_clearance_mm=_approach_clearance(self.spec),
            )
        return None

    def _visual_requirement(self) -> VisualTarget | None:
        visual = self.spec.get("visual_requirements") or {}
        features = [str(item) for item in (visual.get("features") or []) if item]
        if not features:
            return None
        reference = visual.get("reference") or {}
        return VisualTarget(
            features=features,
            views=[str(item) for item in (visual.get("views") or []) if item]
            or ["front", "left", "top", "iso"],
            reference_note=str(reference.get("locator") or visual.get("state") or ""),
        )


def discover(root: Path) -> list[BenchCase]:
    """Every case under `root`, in a stable order."""
    cases = []
    for directory in sorted(Path(root).iterdir()):
        if directory.is_dir() and (directory / "rubric.json").is_file():
            cases.append(BenchCase(directory))
    return cases


def load_case(root: Path, case_id: str) -> BenchCase:
    """One case by id or by directory name."""
    wanted = case_id.strip().lower()
    for case in discover(root):
        if wanted in {case.case_id.lower(), case.slug.lower()}:
            return case
    raise KeyError(f"no case matching {case_id!r} under {root}")


_UNITS = {"mm": MM, "cm": 0.01, "m": 1.0, "um": 1e-6, "µm": 1e-6, "in": 0.0254}


def _unit_scale(unit: Any) -> float | None:
    """Metres per unit, or None when this is not a length at all.

    A specification's `dimensions` block holds things a caliper cannot read: a maximum
    volume in µL, a channel count. Scaling those as though they were millimetres turned
    "300 µL" into a demand for a 300 mm object, and the generator had no way to see that
    the number it was being held to was not a length.
    """
    return _UNITS.get(str(unit or "mm").strip().lower())


_DIAMETER_INNER = re.compile(r"inner|internal|clear|bore|mouth i|\bi\.?d\b", re.I)
_DIAMETER_OUTER = re.compile(r"outer|outside|external|\bo\.?d\b", re.I)
_APERTURE = re.compile(r"hole|well|socket|aperture|opening|seat|mouth|slot", re.I)
"""Words that name a clear opening, read from the dimension's own name only.

`hole` was the expensive omission. PCR-001 states `hole_diameter: 5.2 mm`, located at
"Published well diameter" — the opening a tube goes into. Reading it as an outside
diameter sent it to `diameter_outer`, which the compiler then declined as a local feature
it could not separate, so the number was never measured. The asset built 5.2 mm sockets
with 4.0 mm bores and scored 91.8/100 while being unable to accept a tube.

Read from the name and not from `measurement_location`, unlike the older signals above.
Location prose is written for a person and mentions the opening while quoting the material
around it — "Neck opening, outside" — so matching it there turned five outside diameters
into bores, including `neck_outer_diameter`, which says what it is in the name.
"""
_DIAMETER = re.compile(r"diameter|dia\b|bore|\b[oi]\.?d\b", re.I)
_HEIGHT = re.compile(r"height|tall", re.I)
_DEPTH = re.compile(r"depth", re.I)
_ANY_EXTENT = re.compile(r"height|tall|depth|length|width|breadth|side", re.I)
_WALL = re.compile(r"wall|thickness", re.I)
_FOOTPRINT = re.compile(r"footprint|base|bench", re.I)
_WIDTH = re.compile(r"width|breadth", re.I)

_MOULDED_FORM = (
    "This is a moulded or machined laboratory article, so its form carries the marks of "
    "being made: no external edge on injection-moulded labware is sharp, walls hold a "
    "roughly even thickness, and a draft angle opens a bore towards its mouth. Aim for rounded "
    "external edges and transitions that blend, by whatever means the modelling API gives you: "
    "a swept or revolved profile that is already round beats a fillet operation you have to "
    "talk into selecting the right edges. An assembly of square-cut prisms reads as a diagram "
    "of the object rather than the object, however correct its dimensions are.\n\n"
    "Do not add a solid part that imitates a functional one. A well is an opening; a stud "
    "of the same diameter beside it looks the same in a render and is the opposite thing.\n\n"
)
"""What separates the object from a diagram of it.

The upstream system prompt already asks for realistic geometry and plausible materials, so
this says only what that does not: the specific tells of a moulded article. Every asset in
this corpus is moulded or machined, and the ones that looked wrong looked wrong for the same
reason each time — every edge square, every transition abrupt.

The second paragraph is PCR-001's hinge posts, which were two solid cylinders of exactly the
well diameter standing among ninety-six wells. Indistinguishable in a render, and the reason
the bore measurement read a post instead of a well.

Naming the result rather than the operation is deliberate. "Radius the edges" reads as an
instruction to call a fillet, and a fillet needs edges selected: one run spent a turn on
`ValueError: Fillets requires that edges be selected` before abandoning the idea. The round
form is the requirement; which operation produces it is the generator's to choose.
"""

_MEASUREMENT_PROSE = {
    "diameter_inner": (
        "- `{name}` = {value:.2f} mm is the clear opening through {part}: the narrowest "
        "closed ring a cross-section finds, not the outside of the sleeve around it. A "
        "socket of this outside diameter with a narrower bore fails this."
    ),
    "diameter_outer": (
        "- `{name}` = {value:.2f} mm is the outside of {part}, taken as the median radius "
        "about the section's centroid — a spout, handle or graduation ridge does not widen it."
    ),
    "wall_thickness": (
        "- `{name}` = {value:.2f} mm is the material between the outer surface of {part} "
        "and its bore, at mid-height."
    ),
    "pitch": (
        "- `{name}` = {value:.2f} mm is centre-to-centre between neighbouring repeated "
        "features on {part}, measured as the distance from each one to its nearest "
        "neighbour."
    ),
    "extent_max": (
        "- `{name}` = {value:.2f} mm is the longest bounding-box edge of {part}, whichever "
        "axis you lay it out along."
    ),
    "extent_x": "- `{name}` = {value:.2f} mm is one bench-plane extent of {part}.",
    "extent_y": "- `{name}` = {value:.2f} mm is the other bench-plane extent of {part}.",
    "extent_z": "- `{name}` = {value:.2f} mm is the height of {part}, as exported and at rest.",
    "footprint_max": "- `{name}` = {value:.2f} mm is the longer bench-plane extent of {part}.",
    "footprint_min": "- `{name}` = {value:.2f} mm is the shorter bench-plane extent of {part}.",
}
"""What each measurement kind actually reads, in the words the generator needs.

Written per kind rather than per case because the datum is a property of the measurement,
not of the object: every inner diameter in the corpus is the narrowest closed ring, and
saying so once is what stops the next asset from building a 5.2 mm socket around a 4 mm hole.
"""

_PITCH = re.compile(r"pitch|spacing", re.I)
_LENGTH = re.compile(r"length|\bl\b|\bsl\b|\blong(est)? side\b", re.I)
_SHORT_SIDE = re.compile(r"\bshort(est)? side\b", re.I)
_SUBFEATURE = re.compile(
    r"stroke|travel|\btip\b|cone|nozzle|thread|channel|edge|gap|radius|rating",
    re.I,
)
"""Names for a feature's own size rather than the object's.

Nothing in this module can measure a named feature in isolation — every measurement it
takes is of the assembled object — so these are dimensions it has to decline.

`pitch` and `spacing` used to be here and are not any more: a section through a grid of
repeated features yields their centres, so the spacing between them is measurable after
all. It was the one number deciding whether a rack takes a multichannel pipette, and
declining it let a rack put its wells wherever it liked.
"""


_LONGEST_SIDE = "extent_max"
"""The longest bounding-box edge, whichever way the part was laid out.

What an unqualified `length` means. A tube's overall length runs up Z and a plate's runs
across the bench, and the name never says which — but it does not have to, because the
longest edge is the same number either way. Declining these cost the corpus about twenty
measurable dimensions, `overall_length` among them, for an ambiguity that only ever
existed if the measurement had to name an axis.
"""


_NEGATED = re.compile(r"\bnot\b.*", re.I | re.S)


def _asserted(location: str) -> str:
    """The part of a stated location that says what the measurement *is*.

    These fields routinely rule something out, and they do it with the word the ruled-out
    reading would have been recognised by: PAS-001's stem diameter is located at
    "Published stem diameter, not tip bore" and PHM-001's holder at "nominal size, not
    exact bore tolerance". Matching `bore` inside that clause read both as inner diameters
    — the one thing the specification had gone out of its way to deny. A negated clause
    can never supply a positive signal, so it is dropped before any word is read.
    """
    return _NEGATED.sub("", location)


def _excluded_part(name: str, location: str, kind: str) -> str:
    """Which part a stated envelope leaves out, for the readings that would otherwise include it.

    Bounding-box readings only. A diameter is a median radius about the section's centroid, so a
    spout is already a minority of the perimeter and already ignored; subtracting it as well
    would change a number that was right.
    """
    if kind not in _WHOLE_OBJECT:
        return ""
    match = _EXCLUDED_PART.search(name.replace("_", " ")) or _EXCLUDED_PART.search(location)
    return match.group(1).lower() if match else ""


def _measurement_kind(name: str, location: str) -> str | None:
    """How a named dimension should be measured, or None if the name does not say.

    Read from the dimension's own name and stated location, because that is all the
    specification gives and it is consistent across the corpus: the field that says
    "Clear inner diameter" is asking for a bore, not a bounding box.

    Anything unrecognised used to fall back to a Z extent, and so did depth and length.
    That is a guess, and the measurement it produces is of the whole object — so an
    instrument's 390 mm depth and its 243 mm closed height became the same measurement
    with two different required values, and no geometry could satisfy both. The axes a
    name implies are the ones `_axis_for` already reads off it; the two have to agree.
    A dimension whose meaning cannot be read is not a target, for the same reason a null
    one is not.
    """
    # Underscores are word characters, so `body_od` hides its own last word from a
    # boundary match. The names in this corpus are written both ways.
    text = f"{name} {_asserted(location)}".replace("_", " ").replace("-", " ")
    spaced_name = name.replace("_", " ").replace("-", " ")
    if _WALL.search(text):
        return "wall_thickness"
    if _DIAMETER.search(text):
        if _DIAMETER_OUTER.search(spaced_name):
            # A name that says "outer" has settled the question; nothing in the location
            # prose gets to overrule it.
            return "diameter_outer"
        if _DIAMETER_INNER.search(text) or _APERTURE.search(spaced_name):
            return "diameter_inner"
        return "diameter_outer"
    if _PITCH.search(spaced_name):
        return "pitch"
    # The clear material between neighbouring features. Only where the wording puts it on a grid
    # of them: a gel cassette's 1 mm "spacer-defined gap" is between two plates, and neither a
    # pitch nor a population has anything to say about it.
    if _GAP.search(text) and _GRID.search(text):
        return "feature_gap_max" if _ACROSS_ROWS.search(text) else "feature_gap_min"
    # An opening quoted by its width rather than a diameter is not round, so a median radius
    # is the wrong reading and a bounding box of the whole part is not the opening at all.
    # DWP-001's "square internal mouth width" is the corpus's one example; the reading is the
    # same one any non-round aperture needs.
    if _APERTURE.search(spaced_name) and _CLEAR_WIDTH.search(spaced_name):
        return "width_inner"
    # Read the feature words off the name alone. A stated location often mentions the
    # landmarks a measurement runs between — "lowest conical tip to the top face" is the
    # object's overall length, described by way of its tip.
    if _SUBFEATURE.search(spaced_name):
        return None
    if _FOOTPRINT.search(text) and not _ANY_EXTENT.search(spaced_name):
        return "footprint_max"
    # A height stated from the bench up to the top is a height whether or not the name uses
    # the word: CTR-001 calls 56.7 mm and 41.5 mm its `high_side` and `low_side`, located
    # "Base to rear top" and "Base to front top".
    # A chamber's usable depth is not the object's height, and a specification that states both
    # means two numbers. Only the height is read this way: the same block quotes a work-area
    # length and width whose ordering contradicts the one it uses for the outer envelope, so
    # which interior axis is which cannot be read off the names.
    if _INTERIOR.search(spaced_name):
        if _HEIGHT.search(spaced_name):
            return "interior_extent_z"
        # Anything else about the enclosed space goes unread rather than being taken off the
        # outside of the object, which is a different number and a much larger one: a bath's
        # 138 mm work area measured as its 230 mm envelope fails whatever is built.
        return None
    if _HEIGHT.search(spaced_name) or _BASE_TO_TOP.search(_asserted(location)):
        # Two heights from one datum are the two ends of a top that is not flat, not two
        # readings of one extent. Told apart, both are measurable; sharing `extent_z` meant
        # whichever the asset matched, the other failed.
        return "extent_z_min" if _LOWEST.search(spaced_name) else "extent_z"
    if _DEPTH.search(spaced_name):
        return "extent_y"
    if _WIDTH.search(spaced_name):
        return "extent_x"
    if _SHORT_SIDE.search(spaced_name):
        return "footprint_min"
    if _LENGTH.search(spaced_name):
        # A length has no axis in this corpus — a centrifuge tube's runs up its standing
        # axis, a microplate's across the bench — and it does not need one. The longest
        # edge is the same number whichever way the author laid the part out.
        return _LONGEST_SIDE
    return None


_WHOLE_OBJECT = {
    "extent_x",
    "extent_y",
    "extent_z",
    "extent_z_min",
    "extent_max",
    "footprint_max",
    "footprint_min",
}
_COLLIDING = _WHOLE_OBJECT | {"diameter_outer", "diameter_inner", "pitch"}
"""Kinds where two targets of the same kind are two claims about one number.

Diameters were left out, and they are the worst case in the corpus: BUC-001 hangs a 57 mm
bowl, a 10 mm stem and a 48 mm support disc off `diameter_outer`, so two of the three could
never pass. Whatever a whole-object measurement is, there is only one of it.

Every kind that names a single number belongs here, not just the ones a bug was found in.
`extent_max` was added with the reading of `length` that produces it, and leaving it out
had the same effect as leaving diameters out once did: SPO-001's 375 mm overall length, its
300 mm handle and its 68 mm bowl all became the object's longest edge, and two of the three
could never pass. Being in this set is what routes them through `_part_prefix` instead.
"""

_GAP = re.compile(r"\bgap\b", re.I)
_GRID = re.compile(
    r"\b(rows?|columns?|openings?|wells?|holes?|positions?|neighbouring|neighboring)\b", re.I
)
"""Wording that puts a gap between repeated features rather than between two faces."""

_ACROSS_ROWS = re.compile(r"\bbetween\s+(rows|columns)\b|\bacross\b", re.I)
"""Which way across the grid a gap is measured. Neighbours within a row are the closer pair."""

_INTERIOR = re.compile(r"\b(work|workarea|usable|useable|interior|inside|chamber)\b", re.I)
"""Words saying the number is of the enclosed space rather than of the object."""

_EXCLUDED_PART = re.compile(r"\b(?:without|excluding|less|minus)\b[ _-]+(?:the[ _-]+)?([a-z]+)", re.I)
"""The part a stated envelope leaves out, if it names one.

Sometimes the phrase is prose rather than a part — "height without treating the surface" — so
whatever this captures has to be allowed not to resolve to a body.
"""

_CLEAR_WIDTH = re.compile(r"\b(width|across)\b", re.I)
"""How a width is asked for. Paired with an aperture word it is the opening's clear width."""

_LOWEST = re.compile(r"\b(low|lower|lowest|min|minimum|shortest)\b", re.I)
"""Which end of a range a dimension names, read off the name only.

`min` and `low` appear on plenty of things that are not lengths — a temperature setting, a
supported reaction volume — and those never reach a length measurement anyway.
"""

_BASE_TO_TOP = re.compile(r"\b(base|bench|bottom|floor)\b.*\bto\b.*\btop\b", re.I)
"""A location that measures from the standing surface up to the top: a height, by any name."""

_UNQUALIFIED = re.compile(r"(overall[_ ])?(height|width|depth|length|breadth)", re.I)
"""A dimension named for the object itself, with no part qualifying it."""

_MEASUREMENT_WORDS = frozenset(
    {
        "overall", "total", "max", "maximum", "min", "minimum", "nominal", "approx",
        "height", "tall", "width", "breadth", "depth", "length", "size", "span",
        "dimension", "dimensions", "diameter", "dia", "od", "id", "outer", "inner",
        "internal", "external", "outside", "inside", "clear", "bore", "mouth",
        "wall", "thickness", "footprint", "base", "bench",
    }
)
"""Words that say how something was measured rather than what was measured."""

_STATE_WORDS = frozenset(
    {
        "closed", "open", "shut", "bare", "installed", "absent", "fitted", "removed",
        "unloaded", "loaded", "empty", "full", "folded", "unfolded", "extended",
        "retracted", "collapsed", "without", "with", "excluding", "including",
        "low", "high", "work", "usable", "effective",
    }
)
"""Words naming a configuration the object can be in, rather than a part of it.

MCT-001 states a 41 mm `closed_height` and a 59 mm `open90_height`. Those are one tube in
two poses, and the checker stages one pose, so neither is measurable — scoping them to
parts called `closed` and `open90` would only replace an impossible measurement with a
missing one. A dimension a state qualifies is declined for the same reason a null one is.
"""


def _part_words(name: str) -> list[str]:
    """A dimension name with its measurement vocabulary stripped off the end.

    `bowl_od` is the bowl, `support_disc_diameter` is the support disc, `overall_height`
    leaves nothing, which is how these names are written throughout the corpus.
    """
    words = [word for word in re.split(r"[^a-z0-9]+", name.lower()) if word]
    while words and words[-1] in _MEASUREMENT_WORDS:
        words.pop()
    return words


def _part_prefix(name: str) -> str:
    """The part a dimension names, or "" when it does not name one.

    A state qualifier is not a part, and neither is a bare digit left over from a name
    like `open90`, so both come back empty rather than as something `check_dimensions`
    would go looking for in the body tree.
    """
    words = _part_words(name)
    if not words or any(word.strip("0123456789") in _STATE_WORDS for word in words):
        return ""
    return "_".join(words)


def _without_contradictions(targets: list[DimensionTarget]) -> list[DimensionTarget]:
    """Separate same-kind targets by the part they name, and drop what is left over.

    There is exactly one height of an object. MOR-001 states a 47 mm mortar and a 114 mm
    pestle, and read against the assembly both are the Z extent, so whichever value the
    geometry matched the other failed — and the generator spent its whole turn budget
    alternating between them.

    The specification was never contradicting itself; it was describing two parts, and the
    measurement had no way to say which. Now it has one, so a collision is resolved by
    scoping each target to its own part rather than by discarding both. Only targets a part
    name cannot separate are dropped, and the old rule decides which of those survives.

    Scoping is applied to collided groups only. A kind claimed by a single target is
    already unambiguous, and narrowing it to a part whose name has to match a body would
    trade a measurement that works for one that might not resolve.
    """
    by_kind: dict[tuple[str, str], list[DimensionTarget]] = {}
    for target in targets:
        if target.kind in _COLLIDING:
            # Two envelopes of one object are not the same number when one of them sets a part
            # aside: a water bath's 233 mm "excluding cover" and its overall height are both
            # heights, and both measurable, once the reading knows to leave the cover off.
            by_kind.setdefault((target.kind, target.part_excluded), []).append(target)

    scoped: dict[int, str] = {}
    for group in by_kind.values():
        if len(group) < 2:
            continue
        for target in group:
            prefix = _part_prefix(target.name)
            if prefix:
                scoped[id(target)] = prefix

    resolved = [
        target.model_copy(update={"part": scoped[id(target)]})
        if id(target) in scoped
        else target
        for target in targets
    ]

    # Group again now that the parts are known: two targets of one kind on two different
    # parts are two measurements, and no longer a contradiction.
    by_scope: dict[tuple[str, str, str], list[DimensionTarget]] = {}
    for target in resolved:
        if target.kind in _COLLIDING:
            by_scope.setdefault((target.part, target.kind, target.part_excluded), []).append(target)

    contradictory: set[int] = set()
    for group in by_scope.values():
        disagreeing = [
            target
            for target in group
            if any(
                abs(other.value_m - target.value_m) > target.tolerance_rel * target.value_m
                for other in group
            )
        ]
        if not disagreeing:
            continue
        # If exactly one of them is named for the object rather than for a part of it —
        # `width` beside `plate_width`, `overall_length` beside `handle_length` — then
        # the specification is not contradicting itself and that one is the target.
        whole = [target for target in disagreeing if _UNQUALIFIED.fullmatch(target.name)]
        contradictory.update(
            id(target) for target in disagreeing if len(whole) != 1 or target is not whole[0]
        )
    return [target for target in resolved if id(target) not in contradictory]


def _axis_for(name: str) -> str:
    lowered = name.lower()
    if "width" in lowered:
        return "x"
    if "depth" in lowered and "well" not in lowered:
        return "y"
    return "z"


def _support_duration(spec: dict[str, Any]) -> float | None:
    for entry in spec.get("functional_requirements") or []:
        if isinstance(entry, dict):
            match = re.search(r"(\d+(?:\.\d+)?)\s*s\b", str(entry.get("description") or ""))
            if match:
                return float(match.group(1))
    return None


def _approach_clearance(spec: dict[str, Any]) -> float:
    for entry in spec.get("interfaces") or []:
        if isinstance(entry, dict):
            clearance = entry.get("approach_clearance_mm")
            if isinstance(clearance, (int, float)):
                return float(clearance)
    return 50.0


_COMPONENT_KINDS = {"fixed", "cavity", "visual", "moving", "articulated"}


def _component_kind(raw: Any) -> str:
    """Map the specification's component kinds onto the ones the checks understand.

    The corpus mostly uses the same five words, but a handful of cases say `movable` or
    `rotating` where they mean `moving`. Falling back to `fixed` for anything unrecognised
    is the conservative choice: it still requires the component to exist, it just does not
    additionally demand that it move.
    """
    text = str(raw or "fixed").strip().lower()
    if text in _COMPONENT_KINDS:
        return text
    if text in {"movable", "independent_moving", "rotating", "sliding", "hinged", "removable"}:
        return "moving"
    return "fixed"


def _normalise(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def _covered_operation(name: str, covered: set[str]) -> bool:
    wanted = _normalise(name)
    return any(wanted and (wanted in item or item in wanted) for item in covered)


def _operation_shape(raw_type: str, child: str) -> tuple[str, list[str], bool]:
    text = f"{raw_type} {child}".lower()
    if any(word in text for word in ("knob", "dial", "wheel", "valve", "stopcock", "ring")):
        return "rotate", ["hinge"], False
    # A removable part comes off, which is a translation or six free degrees of freedom.
    # Accepting a hinge here is what let trays, adapters, gaskets and caps be authored as
    # things that swing 45° in place: the motion is not the one the object has, and
    # swinging a part seated in a recess drives it through the wall it sits in.
    if "screw" in text and any(word in text for word in ("cap", "closure", "lid", "stopper")):
        return "remove", ["free", "slide"], False
    if any(word in text for word in ("detachable", "removable", "loose", "remove")):
        return "remove", ["free", "slide"], False
    if any(word in text for word in ("screw", "adjust", "leveling foot", "levelling foot")):
        return "adjust", ["slide", "hinge"], False
    if any(word in text for word in ("continuous", "rotor", "shaft", "turntable", "spin")):
        return "rotate", ["hinge"], True
    if any(word in text for word in ("button", "control", "key", "trigger", "plunger", "ejector")):
        return "press", ["slide"], False
    if any(word in text for word in ("slide", "translation", "orbit", "drawer", "tray", "panel")):
        return "slide", ["slide", "hinge"], False
    if any(word in text for word in ("lid", "door", "cap", "hinge", "lever", "revolute")):
        return "hinge", ["hinge"], False
    return "generic", [], False


def _joint_range_si(raw_range: Any) -> tuple[float | None, float | None, bool]:
    """The stated travel in SI, and whether it was stated as an angle.

    The third element is what keeps radians out of a joint measured in metres. See
    `_range_fits_joint_types`.
    """
    if not isinstance(raw_range, dict):
        return None, None, False
    lower, upper = raw_range.get("min"), raw_range.get("max")
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        return None, None, False
    unit = str(raw_range.get("unit") or "").lower()
    angular = "deg" in unit or "rad" in unit
    scale = math.pi / 180.0 if "deg" in unit else MM if "mm" in unit else 1.0
    return float(lower) * scale, float(upper) * scale, angular


def _range_fits_joint_types(angular: bool, joint_types: list[str]) -> bool:
    """Whether a travel of this dimensionality can be expressed on these joints at all.

    CRY-001 states its screw cap opens over `0..360 deg`, and `_operation_shape` rightly
    calls a screw cap a part that comes off rather than one that swings, so it expects a
    `free` or `slide` joint. Carrying the angle onto that linear joint made the graded
    requirement "travel 6.28319", and MuJoCo reads a slide joint's range in metres: the
    only asset that could pass was a cryovial whose cap lifts 6.3 m off the tube, which is
    what one was authored to do. An angle says nothing about how far a part slides, so it
    is dropped rather than reinterpreted, and the operation falls back to needing
    conservative finite travel.
    """
    if not joint_types:
        return True
    rotational = {"hinge", "ball", "revolute", "continuous"}
    linear = {"slide", "free", "prismatic", "floating"}
    wanted = rotational if angular else linear
    return any(joint_type in wanted for joint_type in joint_types)


def _operation_count(raw_type: str, component_quantity: int) -> int:
    words = {"two": 2, "three": 3, "four": 4, "six": 6, "eight": 8}
    lowered = raw_type.lower()
    for word, count in words.items():
        if word in lowered:
            return count
    return max(component_quantity, 1)


_CONTROL_WORDS = re.compile(
    r"\b(start[_ /-]?stop|start|stop|open|close|zero|tare|menu|power|eject|reset)\b",
    re.I,
)


def _named_controls(component_name: str) -> list[str]:
    lowered = component_name.lower()
    if not any(word in lowered for word in ("control", "button", "key", "switch", "trigger")):
        return []
    found: list[str] = []
    for match in _CONTROL_WORDS.finditer(component_name):
        label = match.group(1).replace(" ", "_").replace("-", "_").replace("/", "_").upper()
        if label not in found:
            found.append(label)
    return found


def _operation_limit_text(target: OperationTarget) -> str:
    if target.continuous:
        return ", with unlimited continuous travel"
    if target.kind == "remove":
        return ", with an executable removal and replacement path"
    return ", with conservative finite limits when the source leaves travel unknown"
