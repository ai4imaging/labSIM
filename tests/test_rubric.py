"""The rubric: the compiler, the primitives, and the promise that nothing passes by default.

Every test here corresponds to a way the scheme this replaced awarded points it should
not have. They are written against the failure rather than against the fix, so that a
future rewrite of the internals still has to keep the guarantee: a measurement that did
not happen is worth nothing, and a quantity that is not a length is never compared to one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from amx.bench.compiler import (
    RUBRIC_NAME,
    build_rubric,
    classify_unit,
    load_rubric,
    measurement_kind,
)
from amx.bench.judge import (
    BLOCKED,
    FAILED,
    NOT_SCORABLE,
    PASSED,
    _assemble,
    _interpret,
    _relative_credit,
    _result,
    _run_item,
    judge_asset,
)
from amx.bench.rubric import AXIS_BUDGET, Gate, ItemParams, Rubric, RubricItem
from amx.grounding.build import GroundedAsset
from amx.grounding.physics import (
    AssetContext,
    PhysicsUnavailable,
    assembly_connectivity,
    part_masses,
    swept_collision,
)
from amx.report import Finding, Report, Severity

CASES = Path(__file__).resolve().parents[1] / "3D_asset_cases"


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("mm", "length"),
        ("cm", "length"),
        ("g", "mass"),
        ("mL", "volume"),
        ("count", "count"),
        ("rpm", "rotational_speed"),
        ("C", "temperature"),
        ("x_g", "rcf"),
        ("cm2", "area"),
        ("mixed_properties", "other"),
        (None, "other"),
    ],
)
def test_unit_is_classified_rather_than_assumed_to_be_millimetres(unit, expected):
    """The bug: `float(value) * 0.001` on everything, whatever the unit field said.

    A balance readability of 0.0001 g became a 0.1 µm length and reported a relative
    error of three million against the instrument's height.
    """
    assert classify_unit(unit)[0] == expected


def test_no_non_length_quantity_ever_compiles_to_a_geometry_comparison():
    for directory in sorted(CASES.iterdir()):
        if not (directory / "rubric.json").is_file():
            continue
        rubric = Rubric.read(directory / "rubric.json")
        for item in rubric.items:
            if item.primitive != "part_dimension":
                continue
            assert item.params.value_m > 0.0, f"{rubric.case_id}/{item.id} has no target"
            assert item.params.value_m < 5.0, (
                f"{rubric.case_id}/{item.id} wants {item.params.value_m} m — a quantity "
                "in another unit has been read as a length"
            )


# --------------------------------------------------------------------------- #
# measurement kinds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "location", "expected"),
    [
        ("wall_thickness", "mid-height wall section", "wall_thickness"),
        ("mouth_inner_diameter", "at the rim", "diameter_inner"),
        ("body_outer_diameter", "widest point", "diameter_outer"),
        ("clear_bore_diameter", "", "diameter_inner"),
        ("height", "", "extent_z"),
        ("overall_length", "", "extent_max"),
    ],
)
def test_measurement_kind_reads_the_feature_not_just_the_axis(name, location, expected):
    assert measurement_kind(name, location) == expected


def test_a_local_feature_is_never_measured_as_the_whole_asset():
    """The bug: `DIM-HOLE_DIAMETER` measured 158 mm on a 5.2 mm hole.

    A feature cut into a part is either measured by cross-section on the part it belongs
    to, or declared unmeasurable. It is never allowed to fall back to a bounding box.
    """
    rubric = build_rubric(_load("MWP-001"))
    features = [
        item
        for item in rubric.items
        if item.subject.startswith("well_") and item.primitive == "part_dimension"
    ]
    assert features, "the microplate case states well dimensions"
    assert all(item.params.local_feature for item in features)


def test_centre_to_centre_spacing_is_measured_between_the_features():
    """A 9 mm well pitch is what decides whether a multichannel pipette can work a plate.

    It used to be declared unmeasurable, which left a plate free to put its wells
    anywhere. A section through them yields every centre, so the spacing between
    neighbours needs no way to isolate a single well — being repeated is what makes it
    measurable.
    """
    rubric = build_rubric(_load("MWP-001"))
    pitch = next(item for item in rubric.items if item.subject == "well_pitch")

    assert pitch.primitive == "part_dimension"
    assert pitch.params.kind == "pitch"
    assert pitch.params.value_m == pytest.approx(0.009)


def test_a_dimension_quoted_in_an_unreachable_state_is_not_scored():
    """`open_height` is a real number about a centrifuge with its lid up.

    The asset is exported closed, so measuring it would fail every correct submission.
    """
    rubric = build_rubric(_load("CEN-001"))
    open_height = next(item for item in rubric.items if item.subject == "open_height")
    assert open_height.primitive == "not_scorable"


# --------------------------------------------------------------------------- #
# fail-closed
# --------------------------------------------------------------------------- #


def test_an_unmeasurable_report_blocks_instead_of_passing():
    """The bug: `_from_report` filtered by code prefix, and `G-UNMEASURABLE` matched none.

    The filtered list came out empty, no failures were found, and the check passed.
    """
    report = Report(
        kind="grounding-protocol",
        subject="x",
        findings=[
            Finding(
                code="G-UNMEASURABLE",
                severity=Severity.FAILURE,
                summary="the asset will not load",
            )
        ],
    )
    status, credit, _, _ = _interpret(report, ("G-STABILITY",))
    assert (status, credit) == (BLOCKED, 0.0)


def test_a_report_with_no_relevant_finding_blocks_instead_of_passing():
    status, credit, _, _ = _interpret(Report(kind="k", subject="x"), ("G-OP",))
    assert (status, credit) == (BLOCKED, 0.0)


def test_a_report_that_measured_and_is_happy_passes():
    report = Report(
        kind="grounding-protocol",
        subject="x",
        findings=[Finding(code="G-STABILITY", severity=Severity.INFO, summary="rests stably")],
    )
    status, credit, _, _ = _interpret(report, ("G-STABILITY",))
    assert (status, credit) == (PASSED, 1.0)


def test_a_blocked_item_costs_points_and_a_not_scorable_one_lowers_the_ceiling():
    """The two must not be confused. One is the asset's problem, the other is the corpus's."""
    rubric = _toy_rubric()
    passing = _assemble(rubric, [_result(item, PASSED, 1.0) for item in rubric.items])
    blocked = _assemble(
        rubric,
        [
            _result(item, BLOCKED if item.id == "A2" else PASSED, 0.0 if item.id == "A2" else 1.0)
            for item in rubric.items
        ],
    )
    unscorable = _assemble(
        rubric,
        [
            _result(
                item,
                NOT_SCORABLE if item.id == "A2" else PASSED,
                0.0 if item.id == "A2" else 1.0,
            )
            for item in rubric.items
        ],
    )
    assert passing.score == pytest.approx(passing.achievable)
    assert blocked.score < passing.score
    assert blocked.achievable == pytest.approx(passing.achievable)
    assert unscorable.score == pytest.approx(unscorable.achievable)
    assert unscorable.achievable < passing.achievable


def test_a_failed_item_costs_only_its_own_weight():
    """A load failure used to wipe every axis. It now costs the load item."""
    rubric = _toy_rubric()
    results = [
        _result(item, FAILED if item.id == "P1" else PASSED, 0.0 if item.id == "P1" else 1.0)
        for item in rubric.items
    ]
    card = _assemble(rubric, results)
    assert card.axes["physics"]["earned"] == 0.0
    assert card.axes["parts"]["earned"] == pytest.approx(card.axes["parts"]["available"])
    assert card.axes["operability"]["earned"] == pytest.approx(
        card.axes["operability"]["available"]
    )
    assert card.score == pytest.approx(
        card.axes["parts"]["earned"] + card.axes["operability"]["earned"]
    )
    assert card.score > 0.0


def test_an_envelope_without_a_part_measures_the_same_for_the_judge_and_the_check(tmp_path):
    """Both sides take the reading with the part off, and must agree on the number.

    A 40 mm body under a 14 mm cover is 54 mm tall assembled and 40 mm "excluding cover". If one
    side subtracts the cover and the other does not, the generator is being held to two heights.
    """
    xml = """
    <mujoco>
      <worldbody>
        <body name="housing">
          <geom type="box" pos="0 0 0.02" size="0.05 0.04 0.02" mass="1.0"/>
        </body>
        <body name="gable_cover">
          <geom type="box" pos="0 0 0.047" size="0.05 0.04 0.007" mass="0.2"/>
        </body>
      </worldbody>
    </mujoco>
    """
    item = RubricItem(
        id="DIM-H",
        axis="parts",
        primitive="part_dimension",
        subject="height_without_cover",
        weight=2.0,
        params=ItemParams(value_m=0.040, kind="extent_z", part_excluded="cover"),
    )
    result = _run_item(item, _context(tmp_path, xml), visual=None)

    assert result.status == PASSED
    assert result.detail["measured_mm"] == pytest.approx(40.0, abs=0.5)
    assert "without" in result.detail["measured_on"]


def test_a_part_the_judge_cannot_find_is_unmeasured_rather_than_wrong(tmp_path):
    """"I could not measure this" and "the asset got this wrong" are different claims.

    They used to be the same verdict. A dimension scoped to a part no body matched returned
    `failed`, which docks credit and now also withholds the pass, over a measurement that was
    never taken -- so a specification naming an object the asset is right not to contain, or
    simply naming a part in words the asset spelled differently, read as a conformance failure.
    Whether the part is there at all is what `part_present` asks, on its own weight.
    """
    xml = """
    <mujoco>
      <worldbody>
        <body name="rack"><geom type="box" size="0.03 0.02 0.01" mass="0.05"/></body>
      </worldbody>
    </mujoco>
    """
    item = RubricItem(
        id="DIM-ABSENT",
        axis="parts",
        primitive="part_dimension",
        subject="reference_tube_outer_diameter",
        weight=2.0,
        critical=True,
        params=ItemParams(value_m=0.0108, kind="diameter_outer", part="Tube seats"),
    )
    result = _run_item(item, _context(tmp_path, xml), visual=None)

    assert result.status == BLOCKED
    assert "no body matches" in result.observed
    # It earns nothing either way. What it must not do is stand as evidence that the asset
    # breached a critical requirement, which is what withholds the pass.
    assert result.credit == 0.0
    assert not result.hard_failed


def test_a_critical_requirement_missed_outright_withholds_the_pass_without_moving_the_score():
    """The score says how much is right; the verdict says whether it may be used.

    PCR-001 bored 4.00 mm wells for 5.20 mm tubes, so it holds none of the tubes it exists to
    hold, and reported 92.4/100 and "passed" -- correctly, arithmetically, because one dimension
    of thirteen is worth about two points. Docking the axis instead would go back to letting one
    item wipe out unrelated work. So the two answers come apart here.
    """
    rubric = _toy_rubric()
    bore = next(item for item in rubric.items if item.id == "A2")
    bore.critical = True
    results = []
    for item in rubric.items:
        result = _result(
            item, PASSED if item.id != "A2" else FAILED, 1.0 if item.id != "A2" else 0.0
        )
        if item.id == "A2":
            result.detail = {"hard_failed": True}
        results.append(result)
    card = _assemble(rubric, results)

    assert card.status == FAILED
    assert not card.certified
    assert "hard-failure threshold" in card.reasons[0]
    # The graded score is untouched: everything but the one item still earned its weight.
    assert card.score == pytest.approx(
        card.axes["physics"]["earned"]
        + card.axes["operability"]["earned"]
        + card.axes["parts"]["earned"]
    )
    assert card.axes["parts"]["earned"] > 0.0


def test_a_near_miss_is_still_a_pass():
    """The hard-failure threshold is the line, not the tolerance. A 6% miss is not a breach."""
    rubric = _toy_rubric()
    bore = next(item for item in rubric.items if item.id == "A2")
    bore.critical = True
    results = []
    for item in rubric.items:
        result = _result(
            item, PASSED if item.id != "A2" else FAILED, 1.0 if item.id != "A2" else 0.7
        )
        if item.id == "A2":
            result.detail = {"hard_failed": False}
        results.append(result)
    card = _assemble(rubric, results)

    assert card.status == PASSED
    assert card.score < card.achievable


def test_a_dimension_error_decays_instead_of_jumping_to_zero():
    """5% is full credit, 20% is none, and 10% is halfway — not a 0/1 cliff at 5%."""
    assert _relative_credit(0.04, 0.05, 0.20) == 1.0
    assert _relative_credit(0.20, 0.05, 0.20) == 0.0
    assert _relative_credit(0.125, 0.05, 0.20) == pytest.approx(0.5)


def test_a_report_with_one_failed_check_keeps_the_checks_that_passed():
    report = Report(
        kind="grounding-operations",
        subject="x",
        findings=[
            Finding(code="G-OP-JOINT-TYPE", severity=Severity.FAILURE, summary="wrong joint"),
            Finding(code="G-OP-RANGE", severity=Severity.INFO, summary="range is enough"),
            Finding(code="G-OP-EXECUTION", severity=Severity.FAILURE, summary="did not drive"),
        ],
    )
    status, credit, _, _ = _interpret(report, ("G-OP",))
    assert status == FAILED
    assert credit == pytest.approx(0.0)
    # INFO findings are commentary; the two actual checks both failed.
    report = Report(
        kind="grounding-operations",
        subject="x",
        findings=[
            Finding(code="G-OP-JOINT-TYPE", severity=Severity.FAILURE, summary="wrong joint"),
            Finding(code="G-OP-RANGE", severity=Severity.WARNING, summary="range is short"),
            Finding(code="G-OP-EXECUTION", severity=Severity.INFO, summary="drove to the stop"),
        ],
    )
    status, credit, _, _ = _interpret(report, ("G-OP",))
    assert status == FAILED
    assert 0.0 < credit < 1.0
    assert credit == pytest.approx(0.25)


def test_an_empty_axis_hands_its_budget_to_the_axes_that_have_items():
    """A beaker has nothing that moves and is still marked out of a hundred."""
    rubric = build_rubric(_load("BEA-001"))
    budgets = rubric.budgets()
    assert budgets["operability"] == 0.0
    assert sum(budgets.values()) == pytest.approx(100.0)
    assert budgets["parts"] > AXIS_BUDGET["parts"]


def test_every_compiled_case_is_reachable_and_gated():
    for directory in sorted(CASES.iterdir()):
        path = directory / "rubric.json"
        if not path.is_file():
            continue
        rubric = Rubric.read(path)
        assert 0.0 < rubric.achievable() <= 100.0, rubric.case_id
        assert {"G0-LOAD", "G3-PHYSICS"} <= {gate.id for gate in rubric.gates}, rubric.case_id
        assert all(
            rubric.item(item_id) is not None for gate in rubric.gates for item_id in gate.item_ids
        ), rubric.case_id


def test_a_case_with_no_asset_scores_nothing_and_says_why(tmp_path):
    rubric = build_rubric(_load("BEA-001"))
    card = judge_asset(rubric, None, work_dir=tmp_path, build_note="nothing was submitted")
    assert card.score == 0.0
    assert card.status == FAILED
    assert any("nothing was submitted" in item.observed for item in card.items)


# --------------------------------------------------------------------------- #
# physical primitives
# --------------------------------------------------------------------------- #


def test_a_weightless_part_is_found(tmp_path):
    context = _context(tmp_path, _MJCF_WEIGHTLESS)
    masses = part_masses(context)
    weightless = [entry for entry in masses if entry.mass_kg <= 0.0]
    assert [entry.body for entry in weightless] == ["floater"]
    assert not weightless[0].sound


def test_parts_that_never_touch_are_reported_as_a_disconnected_assembly(tmp_path):
    context = _context(tmp_path, _MJCF_DISCONNECTED)
    result = assembly_connectivity(context)
    assert not result.connected
    assert result.orphans == ["adrift"]


def test_parts_that_touch_are_one_assembly(tmp_path):
    context = _context(tmp_path, _MJCF_TOUCHING)
    assert assembly_connectivity(context).connected


def test_a_lid_that_clears_both_endpoints_and_ploughs_through_the_middle_is_caught(tmp_path):
    """The failure the endpoint test cannot see, and the reason sweeping exists."""
    context = _context(tmp_path, _MJCF_SWEEP)
    sweep = swept_collision(context, "hinge", samples=24)
    assert sweep.worst_penetration_m > 0.001
    assert 0.0 < sweep.blocked_fraction < 1.0


def test_sweeping_a_joint_that_does_not_exist_raises_rather_than_passing(tmp_path):
    context = _context(tmp_path, _MJCF_TOUCHING)
    with pytest.raises(PhysicsUnavailable):
        swept_collision(context, "nonexistent")


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #


def test_judging_the_same_asset_twice_gives_the_same_answer(tmp_path):
    """The precondition for reading anything into a score change.

    The first scoring scheme put a language model in the path, so the same asset could
        be graded differently twice and nobody could tell whether a number moved because
        the asset changed or because the translation did. With the prose compiled away,
        scoring is arithmetic over simulation, and simulation from a fixed pose is
        deterministic — so this has to hold exactly, not approximately.
    """
    path = tmp_path / "asset.xml"
    path.write_text(_MJCF_SWEEP)
    asset = GroundedAsset(
        asset_id="toy",
        root=tmp_path,
        model_path=tmp_path / "model.py",
        urdf_path=tmp_path / "model.urdf",
        mjcf_path=path,
        source_sha="0" * 16,
    )
    rubric = _physical_rubric()
    first = judge_asset(rubric, asset, work_dir=tmp_path / "a")
    second = judge_asset(rubric, asset, work_dir=tmp_path / "b")

    assert first.score == second.score
    assert [(item.item_id, item.status, item.credit) for item in first.items] == [
        (item.item_id, item.status, item.credit) for item in second.items
    ]


def test_a_rubric_on_disk_is_rebuilt_when_the_derivation_changes(tmp_path):
    """A cached answer key has to expire when the code that compiles it is repaired.

    CRY-001's committed rubric required a screw cap to travel 6.28319 along a linear
    joint. The repair went into the derivation, `input.md` never moved, and the old key —
    fingerprinted against the specification alone — was handed straight back, so the run
    was graded against the requirement that had just been removed.
    """
    from amx.bench.case import BenchCase

    source = CASES / "CRY-001_cryovial"
    directory = tmp_path / source.name
    shutil.copytree(source, directory)
    case = BenchCase(directory)

    stale = build_rubric(case)
    stale.source_sha = "0" * 16
    stale.write(directory / RUBRIC_NAME)

    rebuilt = load_rubric(case)
    assert rebuilt.source_sha == build_rubric(case).source_sha
    assert rebuilt.source_sha != "0" * 16


def test_a_rubric_on_disk_is_reused_when_nothing_moved(tmp_path):
    """The other half: the fingerprint must not churn, or every run recompiles."""
    from amx.bench.case import BenchCase

    source = CASES / "CRY-001_cryovial"
    directory = tmp_path / source.name
    shutil.copytree(source, directory)
    case = BenchCase(directory)
    build_rubric(case).write(directory / RUBRIC_NAME)

    assert load_rubric(case).source_sha == build_rubric(case).source_sha


def _physical_rubric() -> Rubric:
    """Every primitive that needs no specification, so the test needs no case."""
    return Rubric(
        case_id="TOY-002",
        asset_class="toy",
        items=[
            RubricItem(id="PHY-LOAD", axis="physics", primitive="load_compiles", weight=3.0),
            RubricItem(id="PHY-INERTIA", axis="physics", primitive="part_inertia", weight=3.0),
            RubricItem(
                id="PHY-CONNECTED", axis="physics", primitive="assembly_connected", weight=3.0
            ),
            RubricItem(
                id="PHY-INTERPENETRATION",
                axis="physics",
                primitive="rest_interpenetration",
                weight=3.0,
            ),
            RubricItem(id="PHY-DENSITY", axis="physics", primitive="part_density", weight=2.0),
            RubricItem(id="PHY-COM", axis="physics", primitive="com_support", weight=2.0),
            RubricItem(id="PART-MASS", axis="parts", primitive="part_mass", weight=2.0),
            RubricItem(
                id="PART-LID",
                axis="parts",
                primitive="part_present",
                subject="lid",
                weight=3.0,
                params=ItemParams(part="lid"),
            ),
            RubricItem(
                id="SWEEP-LID",
                axis="operability",
                primitive="swept_collision",
                subject="lid",
                weight=2.0,
                params=ItemParams(part="lid", samples=12),
            ),
        ],
        gates=[Gate(id="G0-LOAD", item_ids=["PHY-LOAD"])],
    )


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _load(case_id: str):
    from amx.bench.case import load_case

    return load_case(CASES, case_id)


def test_the_judge_and_the_in_loop_check_read_a_bore_the_same_way(tmp_path):
    """One requirement must not have two answers depending on who asks.

    PCR-001 was measured twice in one run: `check_dimensions` sliced the well population and
    reported a 4.00 mm bore, and the judge sent the same requirement through the cavity
    integration, which found the shallow tray and reported 66.46 mm. Both numbers were of the
    same geometry against the same target. A generator cannot build to that.
    """
    from amx.bench.judge import _feature_dimension
    from amx.grounding.checks import measure_dimension
    from amx.grounding.mesh import world_mesh
    from amx.grounding.scene import stage
    from amx.grounding.spec import DimensionTarget

    # Two bored tubes in a shallow tray: the tray is the larger opening, the tubes are the
    # feature, and the two readings used to disagree about which one the target names.
    xml = """
    <mujoco>
      <worldbody>
        <body name="rack">
          <geom type="box" pos="0 0 0.001" size="0.03 0.02 0.001" mass="0.05"/>
          <geom type="box" pos="0.029 0 0.004" size="0.001 0.02 0.004" mass="0.01"/>
          <geom type="box" pos="-0.029 0 0.004" size="0.001 0.02 0.004" mass="0.01"/>
          <geom type="cylinder" pos="0.01 0 0.02" size="0.0026 0.012" mass="0.01"/>
          <geom type="cylinder" pos="-0.01 0 0.02" size="0.0026 0.012" mass="0.01"/>
        </body>
      </worldbody>
    </mujoco>
    """
    context = _context(tmp_path, xml)
    staged = stage(context.asset.mjcf_path, ground=False, free=False)
    mesh = world_mesh(staged.model, staged.data)
    target = DimensionTarget(
        id="DIM-HOLE", name="hole_diameter", value_m=0.004, kind="diameter_inner"
    )

    assert _feature_dimension(mesh, "diameter_inner", target) == measure_dimension(mesh, target)


def _toy_rubric() -> Rubric:
    rubric = Rubric(
        case_id="TOY-001",
        asset_class="toy",
        items=[
            RubricItem(id="P1", axis="physics", primitive="load_compiles", weight=3.0),
            RubricItem(id="A1", axis="parts", primitive="part_present", weight=3.0),
            RubricItem(
                id="A2",
                axis="parts",
                primitive="part_dimension",
                weight=2.0,
                params=ItemParams(value_m=0.05),
            ),
            RubricItem(id="B1", axis="operability", primitive="operation_exercise", weight=4.0),
        ],
        gates=[
            Gate(id="G0-LOAD", item_ids=["P1"]),
        ],
    )
    return rubric


def _context(tmp_path: Path, xml: str) -> AssetContext:
    path = tmp_path / "asset.xml"
    path.write_text(xml)
    return AssetContext(
        GroundedAsset(
            asset_id="toy",
            root=tmp_path,
            model_path=tmp_path / "model.py",
            urdf_path=tmp_path / "model.urdf",
            mjcf_path=path,
            source_sha="0" * 16,
        )
    )


_MJCF_WEIGHTLESS = """
<mujoco model="toy">
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <geom name="base_g" type="box" size="0.05 0.05 0.05" mass="0.5"/>
      <body name="floater" pos="0 0 0.06">
        <geom name="floater_g" type="box" size="0.01 0.01 0.01" mass="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_MJCF_DISCONNECTED = """
<mujoco model="toy">
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <geom name="base_g" type="box" size="0.05 0.05 0.05" mass="0.5"/>
      <body name="adrift" pos="0 0 0.3">
        <geom name="adrift_g" type="box" size="0.01 0.01 0.01" mass="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_MJCF_TOUCHING = """
<mujoco model="toy">
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <geom name="base_g" type="box" size="0.05 0.05 0.05" mass="0.5"/>
      <body name="stacked" pos="0 0 0.06">
        <geom name="stacked_g" type="box" size="0.01 0.01 0.01" mass="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_MJCF_SWEEP = """
<mujoco model="toy">
  <compiler angle="radian"/>
  <worldbody>
    <body name="housing" pos="0 0 0.1">
      <geom name="housing_g" type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="lid" pos="0.115 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0" range="0 3.1416"/>
        <geom name="lid_g" type="box" pos="0.1 0 0" size="0.1 0.09 0.005" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
"""A lid hinged just outside a housing, which clears at closed and sweeps through at open.

At the near endpoint the lid lies flat beside the housing and at right angles it hangs
clear of the face; carried all the way over, it passes straight through the box. An
endpoint-only check calls this fine, which is why `swept_collision` exists.
"""
