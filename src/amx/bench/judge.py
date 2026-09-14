"""Scoring an asset against its rubric. Fail-closed, on purpose.

The rule this file exists to enforce is one sentence: a measurement that did not happen
is never worth anything. The judge this replaced broke it in two places — a recipe step that
raised was dropped from the list before the remaining steps were averaged, and a report
containing only an "I could not measure this" finding was filtered down to no findings
and read as "nothing failed". Between them, thirty assets that MuJoCo refuses to load
averaged 39 out of 100.

So every path out of `_run_item` is one of five explicit statuses, `blocked` earns zero
while staying in the denominator, and the only status that does not cost the submission
points is `not_scorable`, which is declared when the rubric is compiled and subtracted
from the reachable total where anyone can see it.

A failed item costs its own weight. It does not wipe the axis, and it does not wipe the
case. Continuous measurements decay with how far they missed; a binary miss (the lid is
not there, the MJCF will not load) is still a zero on that item, and only that item.

The measurements themselves are not implemented here. They are the same functions the
authoring loop optimises against — `check_protocol`, `check_operations`, `check_topology`
from `amx.grounding.checks`, plus `amx.grounding.physics` — because a judge with its own
copy of a measurement is a judge that can disagree with the loop about what passing
means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amx.bench.rubric import (
    AXIS_BUDGET,
    PASS_THRESHOLD,
    Axis,
    Rubric,
    RubricItem,
)
from amx.grounding import mesh as mesh_module
from amx.grounding import physics
from amx.grounding.build import GroundedAsset, GroundingBuildError
from amx.grounding.checks import check_operations, check_protocol, check_visual
from amx.grounding.mesh import GeometryUnavailable, cavity_volume
from amx.grounding.physics import AssetContext, PhysicsUnavailable
from amx.grounding.scene import SceneError
from amx.grounding.spec import (
    DimensionTarget,
    GroundingSpec,
    OperationTarget,
    ProbeTarget,
    StabilityTarget,
    TiltTarget,
    VisualTarget,
)
from amx.llm import LlmClient
from amx.report import Report, Severity

PASSED = "passed"
FAILED = "failed"
BLOCKED = "blocked"
NOT_SCORABLE = "not_scorable"
INVALID = "invalid"


@dataclass
class ItemResult:
    """One item's verdict, and enough of the measurement to argue with it."""

    item_id: str
    axis: Axis
    primitive: str
    subject: str
    weight: float
    critical: bool
    status: str
    credit: float
    observed: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def scorable(self) -> bool:
        return self.status != NOT_SCORABLE

    @property
    def conforming(self) -> bool:
        return self.status == PASSED

    @property
    def hard_failed(self) -> bool:
        """Whether the miss was large enough that the benchmark calls the requirement unmet.

        This is not the same as failing. A dimension a few percent outside its tolerance loses
        part of its weight and says so; one past the hard-failure threshold is a different claim,
        that the thing does not meet the requirement at all.
        """
        return bool(self.detail.get("hard_failed"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "axis": self.axis,
            "primitive": self.primitive,
            "subject": self.subject,
            "weight": self.weight,
            "critical": self.critical,
            "status": self.status,
            "credit": round(self.credit, 3),
            "observed": self.observed,
            "detail": self.detail,
        }


@dataclass
class GateResult:
    gate_id: str
    status: str
    reason: str = ""
    zeroed: list[Axis] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status,
            "reason": self.reason,
            "zeroed": list(self.zeroed),
        }


@dataclass
class Scorecard:
    """One asset's result against one rubric."""

    case_id: str
    asset_class: str
    score: float
    achievable: float
    status: str
    axes: dict[str, dict[str, float]] = field(default_factory=dict)
    items: list[ItemResult] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    total_score: float = 100.0

    @property
    def certified(self) -> bool:
        return self.status == PASSED

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "asset_class": self.asset_class,
            "score": round(self.score, 2),
            "total_score": self.total_score,
            "achievable": round(self.achievable, 2),
            "status": self.status,
            "certified": self.certified,
            "reasons": self.reasons,
            "categories": self.axes,
            "items": [item.as_dict() for item in self.items],
            "hard_gates": [gate.as_dict() for gate in self.gates],
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False) + "\n")
        return path

    def to_text(self) -> str:
        lines = [
            f"{self.case_id} ({self.asset_class}): {self.score:.1f}/100 "
            f"(of {self.achievable:.0f} reachable) — {self.status}"
        ]
        for axis in AXIS_BUDGET:
            bucket = self.axes.get(axis)
            if not bucket:
                continue
            lines.append(
                f"  {axis:<12} {bucket['earned']:5.1f}/{bucket['available']:<5.1f}"
                f"  (budget {bucket['budget']:.1f})"
            )
        for item in sorted(self.items, key=lambda item: (item.axis, item.item_id)):
            lines.append(
                f"    {item.axis[:4]:<5}{item.item_id:<22} {item.primitive:<22} "
                f"{item.status:<13} {item.credit * item.weight:5.1f}w  {item.observed[:70]}"
            )
        for gate in self.gates:
            lines.append(f"  gate {gate.gate_id:<16} {gate.status:<9} {gate.reason}")
        lines.extend(f"  ! {reason}" for reason in self.reasons)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def judge_asset(
    rubric: Rubric,
    asset: GroundedAsset | None,
    *,
    work_dir: Path,
    visual_client: LlmClient | None = None,
    build_note: str = "",
) -> Scorecard:
    """Measure every item and add it up. Gates are diagnostics, not score wipes."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    unreviewed = visual_client is None

    if asset is None:
        results = [
            _declared(item, unreviewed)
            or _result(item, BLOCKED, 0.0, build_note or "no asset was produced")
            for item in rubric.items
        ]
        return _assemble(rubric, results, no_asset=True)

    context = AssetContext(asset)
    visual = _visual_verdicts(rubric, asset, work_dir, visual_client)

    results: list[ItemResult] = []
    for item in rubric.items:
        declared = _declared(item, unreviewed)
        results.append(declared or _run_item(item, context, visual=visual))
    _write_evidence(work_dir, results)
    return _assemble(rubric, results)


def _declared(item: RubricItem, unreviewed: bool) -> ItemResult | None:
    """Items that are settled before any measurement is attempted.

    Two of them. One is the rubric's own `not_scorable`, declared when it was compiled.
    The other is a visual feature in a run with no reviewer configured: nobody was
    asked, so nobody could answer, and charging the asset for that would make a
    `--no-vision` sweep look like a corpus of worse assets.
    """
    if not item.scorable:
        return _result(item, NOT_SCORABLE, 0.0, item.params.reason or "declared unmeasurable")
    if item.primitive == "visual_feature" and unreviewed:
        return _result(item, NOT_SCORABLE, 0.0, "no visual reviewer was configured for this run")
    return None


def _run_item(
    item: RubricItem, context: AssetContext, *, visual: dict[str, bool] | None
) -> ItemResult:
    """One item. Never raises, and never turns "could not measure" into credit."""
    try:
        status, credit, observed, detail = _dispatch(item, context, visual=visual)
    except (PhysicsUnavailable, GeometryUnavailable, SceneError, GroundingBuildError) as error:
        # The asset could not be put in a state where the question makes sense. That is
        # the asset's problem, it earns nothing, and it stays in the denominator.
        return _result(item, BLOCKED, 0.0, str(error))
    except Exception as error:  # noqa: BLE001 - a checker fault is not an asset fault
        return _result(item, INVALID, 0.0, f"{type(error).__name__}: {error}")
    return _result(item, status, credit, observed, detail)


def _result(
    item: RubricItem,
    status: str,
    credit: float,
    observed: str = "",
    detail: dict[str, Any] | None = None,
) -> ItemResult:
    return ItemResult(
        item_id=item.id,
        axis=item.axis,
        primitive=item.primitive,
        subject=item.subject,
        weight=item.weight,
        critical=item.critical,
        status=status,
        credit=max(0.0, min(1.0, credit)),
        observed=observed,
        detail=detail or {},
    )


# --------------------------------------------------------------------------- #
# credit
# --------------------------------------------------------------------------- #


def _relative_credit(error: float, full: float, hard: float) -> float:
    """1 inside `full`, 0 at `hard`, linear between. Past `hard` stays 0 on this item."""
    if error <= full:
        return 1.0
    if hard <= full:
        return 0.0
    return max(0.0, min(1.0, (hard - error) / (hard - full)))


def _ratio_credit(measured: float, target: float) -> float:
    """How much of a minimum was reached. Above the target is full credit, not extra."""
    if target <= 0.0:
        return 1.0 if measured > 0.0 else 0.0
    return max(0.0, min(1.0, measured / target))


def _over_limit_credit(value: float, limit: float, *, scale: float | None = None) -> float:
    """1 at the limit, 0.5 at one `scale` past it, approaching 0 without a cliff."""
    excess = max(0.0, value - limit)
    if excess <= 0.0:
        return 1.0
    span = scale if scale is not None and scale > 0.0 else (limit if limit > 0.0 else 1.0)
    return 1.0 / (1.0 + excess / span)


def _fraction_credit(good: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, good / total))


def _verdict(credit: float) -> str:
    return PASSED if credit >= 1.0 - 1e-12 else FAILED


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def _dispatch(
    item: RubricItem, context: AssetContext, *, visual: dict[str, bool] | None
) -> tuple[str, float, str, dict[str, Any]]:
    handler = _HANDLERS.get(item.primitive)
    if handler is None:
        raise RuntimeError(f"no handler for primitive {item.primitive!r}")
    if item.primitive == "visual_feature":
        return _visual_feature(item, visual)
    return handler(item, context)


def _load_compiles(item: RubricItem, context: AssetContext):
    import mujoco
    import numpy as np

    staged = context.static()
    for _ in range(50):
        mujoco.mj_step(staged.model, staged.data)
    finite = bool(
        np.all(np.isfinite(staged.data.qpos)) and np.all(np.isfinite(staged.data.qvel))
    )
    detail = {
        "bodies": int(staged.model.nbody) - 1,
        "geoms": int(staged.model.ngeom),
        "joints": int(staged.model.njnt),
        "states_finite": finite,
    }
    mujoco.mj_resetData(staged.model, staged.data)
    mujoco.mj_forward(staged.model, staged.data)
    if not finite:
        return FAILED, 0.0, "the state went non-finite within 50 steps", detail
    return (
        PASSED,
        1.0,
        f"loads: {detail['bodies']} bodies, {detail['joints']} joints, {detail['geoms']} geoms",
        detail,
    )


def _part_present(item: RubricItem, context: AssetContext):
    matches = context.match_bodies(item.params.part or item.subject)
    if not matches:
        bodies = ", ".join(context.topology().bodies[:12]) or "(none)"
        return FAILED, 0.0, f"no body matches {item.subject!r}; present: {bodies}", {}
    # Presence only. How many of the part there are is a separate question with a
    # separate answer — a rack's twenty-four positions are usually holes in one body,
    # and refusing the rack for that would be measuring the modelling style.
    return PASSED, 1.0, f"present as {matches[0]!r}", {"matched": matches[:4]}


def _assembly_without(context: AssetContext, excluded: str):
    """The assembly with one part left off, for an envelope a datasheet quotes that way.

    Falls back to all of it when the name does not resolve to a body, which is what the phrase
    being prose rather than a part looks like. `check_dimensions` reads these the same way; a
    number that changes depending on which of the two takes it is not a number to build to.
    """
    matches = context.match_bodies(excluded)
    if not matches:
        return context.world(), "whole asset"
    keep = [
        mesh
        for body in context.topology().bodies
        if body != matches[0] and (mesh := context.body_mesh(body)) is not None and not mesh.is_empty
    ]
    if not keep:
        return context.world(), "whole asset"
    import trimesh  # noqa: PLC0415

    return trimesh.util.concatenate(keep), f"whole asset without {matches[0]!r}"


def _part_dimension(item: RubricItem, context: AssetContext):
    from amx.grounding.checks import measure_dimension

    params = item.params
    if params.value_m <= 0.0:
        raise PhysicsUnavailable("the item carries no target value")

    if params.part:
        matches = context.match_bodies(params.part)
        if not matches:
            # Not reaching the geometry is a fact about this checker, not about the asset, and
            # the two used to come out as the same verdict: a `failed` item reading "no body
            # matches 'Tube seats' to measure on", which docks credit and now also withholds the
            # pass, for a measurement that was never taken. Whether the part is genuinely absent
            # is a question `part_present` already asks and answers on its own weight; saying it
            # again here charges one absence to every dimension of the part.
            raise PhysicsUnavailable(f"no body matches {params.part!r} to measure on")
        mesh = context.body_mesh(matches[0])
        where = matches[0]
    elif params.part_excluded:
        mesh, where = _assembly_without(context, params.part_excluded)
    else:
        mesh = context.world()
        where = "whole asset"

    target = DimensionTarget(
        id=params.dimension_id or item.id,
        name=item.subject or params.dimension_id or item.id,
        value_m=params.value_m,
        kind=params.kind,
        axis=params.axis,
        tolerance_rel=params.tolerance_rel,
        hard_fail_rel=params.hard_fail_rel,
        height_fraction_override=params.height_fraction,
    )
    if params.local_feature:
        measured = _feature_dimension(mesh, params.kind, target)
    else:
        measured = measure_dimension(mesh, target)
    if measured is None or measured <= 0.0:
        raise PhysicsUnavailable(
            f"{params.kind} is not measurable on {where!r}"
        )

    error = abs(measured - params.value_m) / params.value_m
    credit = _relative_credit(error, params.tolerance_rel, params.hard_fail_rel)
    detail = {
        "measured_mm": measured * 1000.0,
        "target_mm": params.value_m * 1000.0,
        "relative_error": error,
        "measured_on": where,
        "kind": params.kind,
        "hard_failed": error > params.hard_fail_rel,
    }
    observed = (
        f"{measured * 1000:.2f} mm vs {params.value_m * 1000:.2f} mm "
        f"({error * 100:.1f}% on {where})"
    )
    return _verdict(credit), credit, observed, detail


def _feature_dimension(mesh, kind: str, target: DimensionTarget) -> float | None:
    """Read a feature cut into a part, rather than the part's envelope.

    A depth goes through the cavity integration, which finds the floor and the overflow edge
    of whatever is hollowed out of the part; the distance between them is the depth, and
    nothing else here can find either landmark.

    Everything else defers to `measure_dimension`, the same function the in-loop check calls.
    That matters more than which of the two readings is better: a bore taken one way here and
    another way there gave PCR-001 a 4.00 mm well from `check_dimensions` and a 66.46 mm tray
    from the judge, for one requirement, in one run. A generator cannot build to a number that
    changes depending on who is asking.
    """
    from amx.grounding.checks import measure_dimension  # noqa: PLC0415

    if kind == "extent_z":
        measurement = cavity_volume(mesh)
        if measurement is None:
            return None
        depth = measurement.overflow_height - measurement.floor_height
        return depth if depth > 0.0 else None
    return measure_dimension(mesh, target)


def _part_mass(item: RubricItem, context: AssetContext):
    masses = physics.part_masses(context)
    if item.params.part:
        matches = context.match_bodies(item.params.part)
        if not matches:
            # Same reasoning as `_part_dimension`, and the line below already agreed: a part this
            # checker cannot reach is an unweighed mass, not a wrong one.
            raise PhysicsUnavailable(f"no body matches {item.params.part!r} to weigh")
        wanted = set(matches[: max(1, item.params.count)])
        masses = [entry for entry in masses if entry.body in wanted]
        if not masses:
            raise PhysicsUnavailable(f"{item.params.part!r} has no body carrying geometry")

    total = sum(entry.mass_kg for entry in masses)
    weightless = [entry.body for entry in masses if entry.mass_kg <= 0.0]
    detail = {
        "total_mass_kg": total,
        "bodies": [
            {"body": entry.body, "mass_kg": entry.mass_kg, "movable": entry.movable}
            for entry in masses
        ],
    }
    present = _fraction_credit(len(masses) - len(weightless), len(masses))
    low, high = item.params.min_mass_kg, item.params.max_mass_kg
    range_credit = 1.0
    if low > 0.0 and total < low:
        range_credit = _ratio_credit(total, low)
    elif high > 0.0 and total > high:
        range_credit = _ratio_credit(high, total)
    credit = present * range_credit
    if weightless:
        return (
            _verdict(credit),
            credit,
            f"{len(weightless)}/{len(masses)} part(s) weigh nothing: {', '.join(weightless[:4])}",
            detail,
        )
    if low > 0.0 and total < low:
        return (
            FAILED,
            credit,
            f"{total * 1000:.1f} g against a {low * 1000:.1f} g minimum",
            detail,
        )
    if high > 0.0 and total > high:
        return (
            FAILED,
            credit,
            f"{total * 1000:.1f} g against a {high * 1000:.1f} g maximum",
            detail,
        )
    return PASSED, 1.0, f"{total * 1000:.1f} g across {len(masses)} part(s)", detail


def _part_density(item: RubricItem, context: AssetContext):
    masses = physics.part_masses(context)
    measured = [entry for entry in masses if entry.density_kg_m3 is not None]
    if not measured:
        raise PhysicsUnavailable("no part has both a mass and a measurable volume")
    offenders = [
        entry
        for entry in measured
        if not (
            physics.PLAUSIBLE_DENSITY_KG_M3[0]
            <= (entry.density_kg_m3 or 0.0)
            <= physics.PLAUSIBLE_DENSITY_KG_M3[1]
        )
    ]
    detail = {
        "densities": {entry.body: round(entry.density_kg_m3 or 0.0, 1) for entry in measured}
    }
    credit = 1.0 - len(offenders) / len(measured)
    if offenders:
        worst = offenders[0]
        return (
            FAILED,
            credit,
            f"{len(offenders)}/{len(measured)} parts have impossible density, "
            f"worst {worst.body} at {worst.density_kg_m3:.0f} kg/m^3",
            detail,
        )
    return PASSED, 1.0, f"all {len(measured)} parts sit in a real material band", detail


def _part_inertia(item: RubricItem, context: AssetContext):
    masses = physics.part_masses(context)
    offenders = [entry for entry in masses if not entry.sound]
    detail = {
        "problems": {entry.body: entry.problems for entry in offenders},
        "inspected": len(masses),
    }
    credit = 1.0 - len(offenders) / len(masses) if masses else 0.0
    if offenders:
        return (
            FAILED,
            credit,
            f"{len(offenders)}/{len(masses)} parts have unusable mass or inertia: "
            f"{offenders[0].body} — {offenders[0].problems[0]}",
            detail,
        )
    return PASSED, 1.0, f"all {len(masses)} parts carry usable mass and inertia", detail


def _cavity_volume(item: RubricItem, context: AssetContext):
    minimum = item.params.min_volume_ml
    measurement = cavity_volume(context.world())
    if measurement is None:
        return FAILED, 0.0, "no enclosed cavity: every cross-section is solid", {}
    if minimum <= 0.0:
        # No capacity is stated, so there is no volume to grade — but "is this hollow at
        # all" is still a question with an answer, and it is the one that separates a
        # beaker from a glass cylinder.
        return (
            PASSED,
            1.0,
            f"hollow, holding {measurement.volume_ml:.1f} mL to the overflow edge "
            f"(no capacity stated to grade against)",
            {"volume_ml": measurement.volume_ml, "graded_against": None},
        )
    millilitres = measurement.volume_ml
    detail = {
        "volume_ml": millilitres,
        "minimum_ml": minimum,
        "convergence_rel": measurement.convergence_rel,
        "overflow_height_mm": measurement.overflow_height * 1000.0,
    }
    if not measurement.converged:
        raise PhysicsUnavailable(
            f"the cavity volume did not converge ({measurement.convergence_rel:.3f})"
        )
    credit = _ratio_credit(millilitres, minimum)
    if millilitres < minimum:
        return FAILED, credit, f"holds {millilitres:.1f} mL against {minimum:.1f} mL", detail
    return PASSED, 1.0, f"holds {millilitres:.1f} mL to the overflow edge", detail


def _feature_count(item: RubricItem, context: AssetContext):
    """How many alike features the object repeats, against how many it is meant to.

    The count is taken from the section that finds the most of them rather than from one
    fixed height, because the features do not all begin and end together: PCR-001's wells
    run from the base plate to the top face and its two hinge posts overlap only part of
    that, and a slice taken where the lid ribs cross would count ribs.

    A shortfall is graded on the ratio, so a rack with 48 wells scores half and not zero.
    A surplus is not credit — a plate with 192 wells is not two plates — and is graded the
    same distance the other way.
    """
    wanted = item.params.count
    if wanted <= 1:
        raise PhysicsUnavailable("no repeated count is stated")

    axis = item.params.feature_axis
    best, at_height = mesh_module.repeated_feature_count(context.world(), axis=axis)
    if best == 0:
        raise PhysicsUnavailable(f"no section across {axis} yielded a countable feature")

    detail = {"counted": best, "wanted": wanted, "at_height_fraction": at_height}
    credit = min(best, wanted) / max(best, wanted)
    observed = f"{best} × {item.params.part or 'feature'} against {wanted}"
    return _verdict(credit), credit, observed, detail


def _property_assert(item: RubricItem, context: AssetContext):
    """A stated non-geometric number, grounded in the model where that is possible.

    Mass, volume and counts have model-side counterparts. Speeds, temperatures and
    relative centrifugal fields do not, and the compiler turns those into `not_scorable`
    before they ever reach here — reaching here with one is a compiler bug, not a
    measurement to approximate.
    """
    params = item.params
    if params.quantity_class == "mass":
        masses = physics.part_masses(context)
        total = sum(entry.mass_kg for entry in masses)
        error = abs(total - params.value) / params.value if params.value > 0 else 1.0
        full = max(params.tolerance_rel, 0.15)
        hard = max(params.hard_fail_rel, full * 2.0, 0.50)
        credit = _relative_credit(error, full, hard)
        detail = {"measured_kg": total, "target_kg": params.value, "relative_error": error}
        return _verdict(credit), credit, (
            f"{total * 1000:.1f} g against {params.value * 1000:.1f} g"
        ), detail

    if params.quantity_class == "volume":
        measurement = cavity_volume(context.world())
        if measurement is None:
            return FAILED, 0.0, "no cavity to compare a stated capacity against", {}
        error = abs(measurement.volume_ml - params.value) / params.value
        detail = {
            "measured_ml": measurement.volume_ml,
            "target_ml": params.value,
            "relative_error": error,
        }
        # A nominal capacity is a product label, not a brim volume: at or above 90% of
        # the label is full credit; short of that is the fraction of that floor.
        floor = params.value * 0.9
        credit = _ratio_credit(measurement.volume_ml, floor) if floor > 0 else 0.0
        return (
            _verdict(credit),
            credit,
            f"{measurement.volume_ml:.1f} mL against a stated {params.value:.1f} mL",
            detail,
        )

    if params.quantity_class == "count":
        matches = context.match_bodies(params.part or params.quantity)
        if not matches:
            # Nothing by that name exists as a body, so the count is not observable.
            # Calling that a failure would penalise an asset for cutting its wells into
            # one solid piece, which is how a well plate is actually made.
            raise PhysicsUnavailable(
                f"nothing named like {params.part or params.quantity!r} is a separate body, "
                "so there is nothing to count"
            )
        found = len(matches)
        target = int(params.value)
        detail = {"matched": matches[:24], "target": target}
        credit = _ratio_credit(float(found), float(target))
        return (
            _verdict(credit),
            credit,
            f"{found} matching part(s) against {target} stated",
            detail,
        )

    raise PhysicsUnavailable(
        f"{params.quantity_class} is not groundable in geometry; the rubric should have "
        "compiled this to not_scorable"
    )


def _operation_exercise(item: RubricItem, context: AssetContext):
    """Does the thing that should move, move — the way the real instrument moves.

    Delegated whole to `check_operations`, which already tests the joint type, the travel
    limits, the drive to an endpoint, the return, the endpoint clearance and the button
    stroke. The judge this replaced had all of this available and asked instead whether a joint
    with a matching name existed.
    """
    params = item.params
    target = OperationTarget(
        id=params.operation_id or item.id,
        name=item.subject or params.part,
        kind=params.operation_kind,
        child_hint=params.part or item.subject,
        parent_hint=params.parent_part,
        expected_joint_types=list(params.expected_joint_types),
        count=max(1, params.count),
        range_min=params.range_min,
        range_max=params.range_max,
        continuous=params.continuous,
        return_required=params.return_required,
    )
    fragment = GroundingSpec(
        asset_id=context.asset.asset_id, asset_class="", operations=[target]
    )
    report = check_operations(context.asset, fragment)
    return _interpret(report, ("G-OP",))


def _swept_collision(item: RubricItem, context: AssetContext):
    params = item.params
    joint = params.joint
    if not joint:
        joint = _joint_for_part(context, params.part or item.subject)
    sweep = physics.swept_collision(context, joint, samples=max(4, params.samples))
    detail = {
        "joint": sweep.joint,
        "body": sweep.body,
        "samples": sweep.samples,
        "worst_penetration_mm": sweep.worst_penetration_m * 1000.0,
        "worst_at": sweep.worst_at,
        "blocked_fraction": sweep.blocked_fraction,
    }
    depth_credit = _over_limit_credit(
        sweep.worst_penetration_m, physics.PENETRATION_TOLERANCE_M
    )
    credit = min(depth_credit, 1.0 - sweep.blocked_fraction)
    if credit < 1.0:
        return (
            FAILED,
            credit,
            f"{sweep.body!r} ploughs {sweep.worst_penetration_m * 1000:.2f} mm through "
            f"geometry over {sweep.blocked_fraction * 100:.0f}% of its travel",
            detail,
        )
    return (
        PASSED,
        1.0,
        f"{sweep.body!r} clears its whole travel ({sweep.samples} samples)",
        detail,
    )


def _joint_for_part(context: AssetContext, part: str) -> str:
    matches = context.match_bodies(part)
    if not matches:
        raise PhysicsUnavailable(f"no body matches {part!r} to sweep")
    topology = context.topology()
    for body in matches:
        for joint, driven in topology.joints.items():
            if driven == body:
                return joint
    raise PhysicsUnavailable(f"{matches[0]!r} carries no joint to sweep")


def _rest_stability(item: RubricItem, context: AssetContext):
    params = item.params
    fragment = GroundingSpec(
        asset_id=context.asset.asset_id,
        asset_class="",
        stability=StabilityTarget(
            duration_s=params.duration_s,
            max_translation_mm=params.max_translation_mm,
            max_tilt_deg=params.max_tilt_deg,
            max_penetration_mm=params.max_penetration_mm,
        ),
    )
    return _interpret(check_protocol(context.asset, fragment), ("G-STABILITY",))


def _probe_insert(item: RubricItem, context: AssetContext):
    params = item.params
    fragment = GroundingSpec(
        asset_id=context.asset.asset_id,
        asset_class="",
        probe=ProbeTarget(
            diameter_mm=params.probe_diameter_mm,
            mass_g=params.probe_mass_g,
        ),
    )
    return _interpret(check_protocol(context.asset, fragment), ("G-PROBE",))


def _tilt_restore(item: RubricItem, context: AssetContext):
    params = item.params
    fragment = GroundingSpec(
        asset_id=context.asset.asset_id,
        asset_class="",
        tilt=TiltTarget(
            axis=params.tilt_axis,
            angle_deg=params.tilt_angle_deg,
            tolerance_deg=params.tilt_tolerance_deg,
        ),
    )
    return _interpret(check_protocol(context.asset, fragment), ("G-TILT",))


def _rest_interpenetration(item: RubricItem, context: AssetContext):
    depth, where = physics.rest_interpenetration(context)
    limit = item.params.max_penetration_mm / 1000.0
    detail = {"worst_penetration_mm": depth * 1000.0, "between": where, "limit_mm": limit * 1000}
    credit = _over_limit_credit(depth, limit)
    if credit < 1.0:
        return FAILED, credit, f"{where} by {depth * 1000:.2f} mm at rest", detail
    return PASSED, 1.0, f"no part overlaps another by more than {limit * 1000:.2f} mm", detail


def _assembly_connected(item: RubricItem, context: AssetContext):
    result = physics.assembly_connectivity(context)
    detail = {
        "groups": result.groups,
        "orphans": result.orphans,
        "inspected": result.inspected,
    }
    credit = _fraction_credit(result.inspected - len(result.orphans), result.inspected)
    if not result.connected:
        return (
            FAILED,
            credit,
            f"{len(result.orphans)} part(s) float free of the assembly: "
            f"{', '.join(result.orphans[:4])}",
            detail,
        )
    return PASSED, 1.0, f"all {result.inspected} parts touch the assembly", detail


def _com_support(item: RubricItem, context: AssetContext):
    margin, detail = physics.com_support_margin(context)
    limit = item.params.min_margin_mm / 1000.0
    detail = {**detail, "margin_mm": margin * 1000.0}
    shortfall = max(0.0, limit - margin)
    credit = _over_limit_credit(shortfall, 0.0, scale=0.005)
    if credit < 1.0:
        return (
            FAILED,
            credit,
            f"the centre of mass sits {margin * 1000:.1f} mm inside the footprint",
            detail,
        )
    return PASSED, 1.0, f"centre of mass {margin * 1000:.1f} mm inside the footprint", detail


def _visual_feature(item: RubricItem, visual: dict[str, bool] | None):
    if visual is None:
        raise PhysicsUnavailable("no pinned vision model is configured to review the renders")
    feature = item.params.feature or item.subject
    if feature not in visual:
        raise PhysicsUnavailable(f"the reviewer returned no verdict for {feature!r}")
    visible = visual[feature]
    return (
        PASSED if visible else FAILED,
        1.0 if visible else 0.0,
        f"{feature!r} {'is' if visible else 'is not'} visible in the renders",
        {},
    )


_HANDLERS = {
    "load_compiles": _load_compiles,
    "part_present": _part_present,
    "part_dimension": _part_dimension,
    "part_mass": _part_mass,
    "part_density": _part_density,
    "part_inertia": _part_inertia,
    "cavity_volume": _cavity_volume,
    "feature_count": _feature_count,
    "property_assert": _property_assert,
    "visual_feature": _visual_feature,
    "operation_exercise": _operation_exercise,
    "swept_collision": _swept_collision,
    "rest_stability": _rest_stability,
    "rest_interpenetration": _rest_interpenetration,
    "assembly_connected": _assembly_connected,
    "com_support": _com_support,
    "probe_insert": _probe_insert,
    "tilt_restore": _tilt_restore,
}


# --------------------------------------------------------------------------- #
# reading a grounding report without lying about it
# --------------------------------------------------------------------------- #


def _interpret(report: Report, prefixes: tuple[str, ...]):
    """Turn a grounding report into a verdict, with "unmeasurable" surviving the trip.

    The old judge filtered findings by code prefix and then asked whether any failures
    remained. `G-UNMEASURABLE` starts with none of the prefixes, so an asset that could
    not be staged produced an empty list, no failures, and a pass. Both of the holes that
    opened up — the unmeasurable finding and the empty list — are closed here.
    """
    unmeasurable = [f for f in report.findings if f.code == "G-UNMEASURABLE"]
    if unmeasurable:
        return BLOCKED, 0.0, unmeasurable[0].summary, {}

    relevant = [
        finding
        for finding in report.findings
        if finding.code.startswith(prefixes) and finding.severity is not Severity.INFO
    ]
    informative = [
        finding
        for finding in report.findings
        if finding.code.startswith(prefixes) and finding.severity is Severity.INFO
    ]
    if not relevant and not informative:
        return BLOCKED, 0.0, "the checker produced no verdict for this item", {}

    failures = [f for f in relevant if f.severity is Severity.FAILURE]
    detail = {
        "findings": [
            {"code": f.code, "severity": f.severity.value, "summary": f.summary}
            for f in relevant + informative
        ][:8]
    }
    if not relevant:
        summary = informative[0].summary
        return PASSED, 1.0, summary, detail

    credits = [_finding_credit(finding) for finding in relevant]
    credit = sum(credits) / len(credits)
    summary = (
        failures[0].summary
        if failures
        else (informative or relevant)[0].summary
    )
    return _verdict(credit), credit, summary, detail


def _finding_credit(finding) -> float:
    """One check's share of an item. A miss costs this check, not the rest of the report."""
    if finding.severity is Severity.INFO:
        return 1.0
    if finding.severity is Severity.WARNING:
        return 0.5
    metrics, thresholds = finding.metrics, finding.thresholds
    error = metrics.get("relative_error")
    tolerance = thresholds.get("tolerance_rel")
    hard = thresholds.get("hard_fail_rel")
    if error is not None and tolerance is not None and hard is not None:
        return _relative_credit(error, tolerance, hard)
    for measured_key, minimum_key in (("volume_ml", "minimum_ml"), ("measured", "minimum")):
        measured = metrics.get(measured_key)
        minimum = thresholds.get(minimum_key)
        if measured is not None and minimum:
            return _ratio_credit(measured, minimum)
    return 0.0


def _visual_verdicts(
    rubric: Rubric,
    asset: GroundedAsset,
    work_dir: Path,
    client: LlmClient | None,
) -> dict[str, bool] | None:
    """Run the visual reviewer once for the whole case rather than once per feature."""
    features = [
        item.params.feature or item.subject
        for item in rubric.items
        if item.primitive == "visual_feature"
    ]
    if not features:
        return {}
    if client is None or not client.available:
        return None
    spec = GroundingSpec(
        asset_id=asset.asset_id,
        asset_class=rubric.asset_class,
        visual=VisualTarget(features=features),
    )
    try:
        report = check_visual(
            asset, spec, client=client, blocking=True, image_dir=work_dir / "renders"
        )
    except Exception:  # noqa: BLE001 - a reviewer outage blocks, it does not pass
        return None
    verdicts: dict[str, bool] = {}
    for finding in report.findings:
        if finding.code != "G-VISUAL-CLAIM":
            continue
        # "visible: <feature> — <observation>". Parsed rather than substring-matched,
        # because one feature's name turning up inside another's prose observation
        # would otherwise award it a verdict nobody gave.
        _, _, tail = finding.summary.partition(": ")
        named = tail.split("—")[0].strip()
        if named:
            verdicts[named] = finding.severity is Severity.INFO
    # Features the reviewer skipped are left out, so they block rather than fail.
    return verdicts or None


# --------------------------------------------------------------------------- #
# gates and arithmetic
# --------------------------------------------------------------------------- #


def _assemble(
    rubric: Rubric, results: list[ItemResult], *, no_asset: bool = False
) -> Scorecard:
    by_id = {result.item_id: result for result in results}
    gates: list[GateResult] = []
    reasons: list[str] = []

    for gate in rubric.gates:
        referenced = [by_id[item_id] for item_id in gate.item_ids if item_id in by_id]
        offenders = [item for item in referenced if not item.conforming]
        if not referenced:
            gates.append(GateResult(gate.id, PASSED, "nothing to check"))
            continue
        if offenders:
            reason = "; ".join(
                f"{item.item_id} {item.status}: {item.observed[:80]}" for item in offenders[:3]
            )
            gates.append(GateResult(gate.id, FAILED, reason))
        else:
            gates.append(GateResult(gate.id, PASSED))

    budgets = rubric.budgets()
    axes: dict[str, dict[str, float]] = {}
    score = 0.0
    achievable = 0.0
    for axis, budget in budgets.items():
        items = [result for result in results if result.axis == axis]
        if not items:
            continue
        weight = sum(item.weight for item in items)
        scorable = [item for item in items if item.scorable]
        available = budget * sum(item.weight for item in scorable) / weight if weight else 0.0
        earned = (
            budget * sum(item.weight * item.credit for item in scorable) / weight
            if weight
            else 0.0
        )
        axes[axis] = {
            "earned": round(earned, 2),
            "available": round(available, 2),
            "budget": round(budget, 2),
        }
        score += earned
        achievable += available

    invalid = [item for item in results if item.status == INVALID]
    blocked = [item for item in results if item.status == BLOCKED]
    if blocked:
        reasons.append(
            f"{len(blocked)} item(s) could not be measured and earned nothing: "
            + ", ".join(item.item_id for item in blocked[:5])
        )

    # A weighted average is the right way to say how much of an asset is correct, and the wrong
    # way to say whether it may be used. PCR-001 built 4.00 mm bores for 5.20 mm tubes -- it
    # cannot hold a single tube it exists to hold -- and lost 2 of 100 points for it, because one
    # dimension out of thirteen is worth about that much. So the score stays as it is, an honest
    # partial mark, and it no longer doubles as the verdict: a critical requirement missed by more
    # than its hard-failure threshold, or a failed gate, withholds the pass without touching a
    # single item's credit.
    breaches = [item for item in results if item.critical and item.hard_failed]
    failed_gates = [gate for gate in gates if gate.status == FAILED]

    if invalid:
        status = "invalid_test"
        reasons.insert(0, f"checker fault on {invalid[0].item_id}: {invalid[0].observed}")
    elif no_asset:
        status = FAILED
    elif score < PASS_THRESHOLD:
        status = FAILED
        reasons.append(f"scored {score:.1f} against a threshold of {PASS_THRESHOLD:.0f}")
    elif breaches:
        status = FAILED
        reasons.insert(
            0,
            f"{len(breaches)} critical requirement(s) missed by more than the benchmark's "
            "hard-failure threshold: "
            + "; ".join(f"{item.item_id} ({item.observed[:60]})" for item in breaches[:3]),
        )
    elif failed_gates:
        status = FAILED
        reasons.insert(
            0,
            f"{len(failed_gates)} gate(s) did not hold: "
            + "; ".join(f"{gate.gate_id} ({gate.reason[:60]})" for gate in failed_gates[:3]),
        )
    else:
        status = PASSED

    return Scorecard(
        case_id=rubric.case_id,
        asset_class=rubric.asset_class,
        score=round(score, 2),
        achievable=achievable,
        status=status,
        axes=axes,
        items=results,
        gates=gates,
        reasons=reasons,
    )


def _write_evidence(work_dir: Path, results: list[ItemResult]) -> None:
    """One directory per item, so a disputed verdict has something to dispute."""
    for result in results:
        directory = work_dir / result.axis / result.item_id
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "result.json").write_text(
                json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n"
            )
        except OSError:
            continue


__all__ = ["GateResult", "ItemResult", "Scorecard", "judge_asset"]
