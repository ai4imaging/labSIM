"""Reading a benchmark case: what the generator is shown, and what is derived from it.

Everything here is about the one file a case actually has. `input.md` is the task, the
in-loop grounding spec is derived from it, and so is the rubric — so these tests are
mostly about the derivation refusing to guess. A dimension whose meaning cannot be read
off its name must not become a measurement of the whole object, and a quantity in grams
must not become a length, because both of those produce a target no geometry can satisfy
and a generator that spends its whole budget failing to.

Scoring is tested in `test_rubric.py`, against the primitives rather than against prose.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from amx.bench.case import BenchCase, discover, load_case
from amx.bench.compiler import build_rubric

CASES = Path(__file__).resolve().parents[1] / "3D_asset_cases"


@pytest.fixture(scope="module")
def beaker_case() -> BenchCase:
    return load_case(CASES, "BEA-001")


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #


def test_every_case_in_the_corpus_parses():
    cases = discover(CASES)
    assert len(cases) >= 60
    for case in cases:
        assert case.case_id and case.asset_class
        assert case.to_grounding_spec().asset_id == case.case_id


def test_a_case_needs_nothing_but_its_specification(tmp_path):
    """One file, because a second one is a second source of truth to keep in step.

    The corpus used to carry a hand-written answer key beside each `input.md`, and the
    two disagreed: the key stated thresholds for dimensions the specification records as
    unknown. Now the key is compiled, so there is nothing to disagree with.
    """
    directory = tmp_path / "BEA-001_beaker"
    directory.mkdir()
    (directory / "input.md").write_text(load_case(CASES, "BEA-001").input_path.read_text())

    case = BenchCase(directory)
    assert case.case_id == "BEA-001"
    assert case.asset_class == "beaker"
    assert build_rubric(case).items


def test_operation_contract_is_derived_from_the_specification():
    centrifuge = load_case(CASES, "CEN-001").to_grounding_spec()
    by_name = {operation.name: operation for operation in centrifuge.operations}
    assert by_name["external_lid"].expected_joint_types == ["hinge"]
    assert by_name["external_lid"].range_max == pytest.approx(math.pi / 2)
    assert by_name["rotor"].continuous
    assert by_name["OPEN"].kind == "press"
    assert by_name["START_STOP"].expected_joint_types == ["slide"]

    plate = load_case(CASES, "CCP-001").to_grounding_spec()
    assert any(operation.kind == "remove" and "lid" in operation.name.lower() for operation in plate.operations)


def test_an_angle_is_not_carried_onto_a_joint_measured_in_metres():
    """CRY-001 states its screw cap opens over 0..360 deg, and a screw cap comes off.

    Those are both right, and putting them together produced a graded requirement to
    travel 6.28319 along a `slide` joint, which MuJoCo reads as metres. The asset that
    passed had a cap that lifted 6.3 m off a 12.5 mm tube. An angle says nothing about how
    far a part slides, so the range is dropped rather than reinterpreted, leaving the
    operation held to the general requirement of conservative finite travel.
    """
    cap = {
        operation.name: operation
        for operation in load_case(CASES, "CRY-001").to_grounding_spec().operations
    }["Screw cap"]
    assert cap.kind == "remove"
    assert cap.expected_joint_types == ["free", "slide"]
    assert cap.range_min is None and cap.range_max is None


def test_a_distance_is_not_carried_onto_a_joint_measured_in_radians():
    """The same confusion the other way round, which PET-001 has.

    A 30 mm travel on a lid the contract expects to hinge would have asked for a 0.03 rad
    joint: a lid that opens 1.7° and is graded as correct for it.
    """
    lid = {
        operation.name: operation
        for operation in load_case(CASES, "PET-001").to_grounding_spec().operations
    }["lid"]
    assert lid.expected_joint_types == ["hinge"]
    assert lid.range_min is None and lid.range_max is None


def test_an_angle_on_a_hinge_is_left_alone():
    """The guard drops mismatches, not ranges: CEN-001's lid still has its 90°."""
    lid = {
        operation.name: operation
        for operation in load_case(CASES, "CEN-001").to_grounding_spec().operations
    }["external_lid"]
    assert lid.range_max == pytest.approx(math.pi / 2)


def test_no_case_asks_for_a_travel_longer_than_a_bench():
    """A linear range over half a metre is a unit error, not a laboratory instrument."""
    absurd = [
        (case.case_id, operation.id, operation.range_max)
        for case in discover(CASES)
        for operation in case.to_grounding_spec().operations
        if operation.range_max is not None
        and any(kind in ("slide", "free") for kind in operation.expected_joint_types)
        and abs(operation.range_max - (operation.range_min or 0.0)) > 0.5
    ]
    assert absurd == []


def test_the_generation_prompt_is_the_task_file(beaker_case):
    prompt = beaker_case.to_asset_request()
    assert beaker_case.input_path.read_text() in prompt


def test_the_rubric_holds_the_asset_to_nothing_the_prompt_withheld(beaker_case):
    """The whole benchmark rests on this, and it is now true by construction.

    There used to be a wall to police: a separate answer key, with weights and gates and
    a pass threshold that had to be kept out of the generation prompt, because leaking
    them turns a good asset into a tuned one — and it is very hard to notice afterwards
    from the numbers. The rubric is compiled from `input.md` now, so the test is no
    longer "did anything leak" but "is there anything left that could".

    Checked on the dimension targets specifically, because they are the only items that
    carry a number the asset has to hit, and a target the specification does not state is
    exactly what nobody could have built to.
    """
    rubric = build_rubric(beaker_case)
    stated = {
        round(target.value_m, 6) for target in beaker_case.to_grounding_spec().dimensions
    }
    assert stated

    for item in rubric.items:
        if item.primitive != "part_dimension":
            continue
        assert round(item.params.value_m, 6) in stated, item.id


def test_the_grounding_spec_carries_only_dimensions_the_task_states(beaker_case):
    """Two of the beaker's four dimensions are null in the specification."""
    spec = beaker_case.to_grounding_spec()
    names = {target.name for target in spec.dimensions}
    assert names == {"body_outer_diameter", "height"}
    assert {t.value_m for t in spec.dimensions} == {0.070, 0.095}


def test_measurement_kinds_follow_the_stated_location(beaker_case):
    spec = beaker_case.to_grounding_spec()
    kinds = {target.name: target.kind for target in spec.dimensions}
    assert kinds["body_outer_diameter"] == "diameter_outer"
    assert kinds["height"] == "extent_z"


def test_a_quantity_that_is_not_a_length_is_not_a_dimension(beaker_case):
    """`dimensions` also holds capacities, speeds and counts.

    Scaling those as though they were millimetres turned MCP-001's "300 µL maximum
    volume" into a demand for a 300 mm object.
    """
    case = load_case(CASES, "BAL-001")
    names = {target.name for target in case.to_grounding_spec().dimensions}

    assert "capacity" not in names, "a capacity in grams is not a caliper measurement"
    assert "readability" not in names


def test_a_feature_s_own_size_is_not_measured_as_the_whole_object(beaker_case):
    """Unreadable names used to fall back to the object's Z extent, and contradict.

    MCP-001 states a 9 mm cone pitch, a 55 mm tip length and a 300 mm body. All three
    became the same measurement — the whole object's height — with three different
    required values, so no geometry could satisfy the spec and the generator spent its
    whole budget failing to.
    """
    spec = load_case(CASES, "MCP-001").to_grounding_spec()
    kinds = [target.kind for target in spec.dimensions]
    by_name = {target.name: target for target in spec.dimensions}

    assert kinds.count("extent_z") <= 1
    # A pitch is not the object's height, and it is not unmeasurable either: a section
    # through the cones yields their centres, so the spacing between them is a number.
    assert by_name["cone_pitch"].kind == "pitch"
    assert "tip_length" not in by_name
    # The components it does state are still required; dropping a dimension does not
    # drop the object.
    assert {component.name for component in spec.components}


def test_an_outer_diameter_written_as_od_is_still_an_outer_diameter():
    spec = load_case(CASES, "CTU-001").to_grounding_spec()
    kinds = {target.name: target.kind for target in spec.dimensions}

    assert kinds["body_od"] == "diameter_outer"
    assert kinds["cap_od"] == "diameter_outer"


def test_a_hole_diameter_is_the_clear_bore_and_not_the_socket_around_it():
    """The defect that let PCR-001 pass at 91.8 while unable to accept a tube.

    Its rack states a 5.2 mm `hole_diameter` — the opening a tube goes into. Read as an
    outside diameter it became a local feature the compiler could not separate, so the
    number was never measured at all, and the asset built 5.2 mm sockets with 4.0 mm bores.
    """
    for case_id in ("PCR-001", "RAC-001"):
        kinds = {t.name: t.kind for t in load_case(CASES, case_id).to_grounding_spec().dimensions}
        assert kinds["hole_diameter"] == "diameter_inner", case_id


def test_the_request_says_where_the_caliper_goes_for_a_bore():
    """The words alone admit the reading that cost PCR-001 its function.

    "Hole diameter 5.2 mm" is satisfied by a 5.2 mm socket to a reader who is not told the
    number is the opening something passes through. That datum is the one thing `input.md`
    does not carry, so it is the one thing worth adding to it.
    """
    request = load_case(CASES, "PCR-001").to_asset_request()
    contract = request[: request.index("---")]

    assert "narrowest closed ring" in contract
    assert "96 solid studs would count as zero" in contract


def test_the_request_names_what_it_will_not_check():
    """Silence used to be indistinguishable from approval.

    WBA-001 states seven dimensions and every one is quoted in a state or against a feature
    the checks decline, so a generator reading no complaint would read seven satisfied
    requirements.
    """
    request = load_case(CASES, "WBA-001").to_asset_request()
    contract = request[: request.index("---")]

    assert "do not read silence as approval" in contract
    assert "work_length" in contract


def test_a_ruled_out_bore_does_not_make_a_dimension_a_bore():
    """These fields rule a reading out using the word that reading is recognised by.

    PAS-001's stem is located at "Published stem diameter, not tip bore" and PHM-001's
    holder at "nominal size, not exact bore tolerance". Matching `bore` inside those
    clauses read both as inner diameters — the one thing the specification had gone out of
    its way to deny.
    """
    stem = {t.name: t for t in load_case(CASES, "PAS-001").to_grounding_spec().dimensions}

    assert stem["stem_diameter"].kind == "diameter_outer"


def test_three_diameters_on_three_parts_are_three_measurements():
    """BUC-001 states a 57 mm bowl, a 10 mm stem and a 48 mm support disc.

    All three are `diameter_outer`, and measured against the assembled object all three
    read the same number, so at most one could ever pass. The run was refused over the
    other two until its turn budget ran out.
    """
    targets = {t.name: t for t in load_case(CASES, "BUC-001").to_grounding_spec().dimensions}

    assert targets["bowl_od"].part == "bowl"
    assert targets["stem_od"].part == "stem"
    assert targets["support_disc_diameter"].part == "support_disc"


def test_no_case_states_two_values_for_one_measurement():
    """The property the part field exists to establish, over the whole corpus.

    Two targets of the same kind on the same part with values further apart than the
    tolerance is a specification no geometry can satisfy. Ten of the sixty-four cases had
    one; a regression here puts those runs back into a refusal loop.
    """
    offenders = []
    for case in discover(CASES):
        groups: dict[tuple[str, str], list] = {}
        for target in case.to_grounding_spec().dimensions:
            groups.setdefault((target.part, target.kind), []).append(target)
        for (part, kind), group in groups.items():
            spread = max(t.value_m for t in group) - min(t.value_m for t in group)
            if len(group) > 1 and spread > min(t.tolerance_rel * t.value_m for t in group):
                offenders.append(f"{case.case_id} {part or '<whole>'}/{kind}")

    assert offenders == []


def test_a_dimension_quoted_in_a_pose_is_not_scoped_to_a_part_called_that():
    """MCT-001's 41 mm `closed_height` and 59 mm `open90_height` are one tube, two poses.

    The checker stages one pose, so neither is measurable, and inventing parts named
    `closed` and `open90` would only swap an impossible measurement for a missing one.
    """
    targets = load_case(CASES, "MCT-001").to_grounding_spec().dimensions
    names = {t.name for t in targets}

    assert "closed_height" not in names and "open90_height" not in names
    assert all(t.part not in {"closed", "open90", "open"} for t in targets)


def test_a_length_with_no_axis_behind_it_is_the_longest_edge():
    """A tube's overall length runs up its standing axis; a plate's runs across the bench.

    The name never says which — and it does not have to. The longest bounding-box edge is
    the same number either way, so the ambiguity that used to disqualify these only
    existed while the measurement had to name an axis. Declining them cost the corpus
    about twenty stated dimensions.
    """
    targets = {t.name: t for t in load_case(CASES, "CTU-001").to_grounding_spec().dimensions}

    assert targets["overall_length"].kind == "extent_max"


def test_one_length_each_for_three_parts_is_three_measurements():
    """SPO-001 states a 375 mm spoon, a 300 mm handle and a 68 mm bowl.

    Read against the assembly all three are the longest edge, so two could never pass —
    the same trap the diameters were in before `part` existed. Reading `length` at all is
    only safe because a name that qualifies it scopes the measurement to that part.
    """
    targets = {
        t.name: t
        for t in load_case(CASES, "SPO-001").to_grounding_spec().dimensions
        if t.kind == "extent_max"
    }

    assert targets["overall_length"].part == ""
    assert targets["handle_length"].part == "handle"
    assert targets["bowl_length"].part == "bowl"


def test_the_gap_between_openings_is_read_the_two_ways_a_grid_has():
    """A rack states 7 mm of material along a row and 13 mm between rows."""
    rack = {t.name: t.kind for t in load_case(CASES, "RAC-001").to_grounding_spec().dimensions}

    assert rack["column_edge_gap"] == "feature_gap_min"
    assert rack["row_edge_gap"] == "feature_gap_max"


def test_a_gap_between_two_plates_is_not_a_gap_between_features():
    """A gel cassette's 1 mm spacer gap is between two faces; no grid, no pitch, no population."""
    for case_id in ("GEL-001", "VEL-001"):
        stated = {t.name for t in load_case(CASES, case_id).to_grounding_spec().dimensions}
        assert "gel_gap" not in stated


def test_an_envelope_stated_without_a_part_leaves_that_part_off():
    """A water bath quotes three envelope dimensions "excluding cover" and three work-area ones.

    Both trios are of the whole object, so neither can be scoped to a part, and every pair shared
    a kind -- so all six read as contradictions and none was measured. The bath could be any size.
    """
    bath = {t.name: t for t in load_case(CASES, "WBA-001").to_grounding_spec().dimensions}

    assert bath["height_without_cover"].kind == "extent_z"
    assert bath["height_without_cover"].part_excluded == "cover"
    assert bath["work_height"].kind == "interior_extent_z"
    assert bath["work_height"].part_excluded == ""


def test_the_horizontal_size_of_an_enclosed_space_is_left_unread():
    """The bath's work area is 138 × 155 mm while its envelope is 230 × 199 mm.

    So the specification's own ordering of length and width is reversed between the two, and
    which interior axis is which cannot be read off the names. Measuring them off the outside of
    the object would fail a correct bath by 90 mm, which is worse than declining to measure.
    """
    bath = {t.name for t in load_case(CASES, "WBA-001").to_grounding_spec().dimensions}

    assert "work_length" not in bath
    assert "work_width" not in bath


def test_a_diameter_stated_without_a_spout_is_left_alone():
    """A median radius already ignores a spout, so nothing needs subtracting for one."""
    beaker = {t.name: t for t in load_case(CASES, "BEA-001").to_grounding_spec().dimensions}

    assert beaker["body_outer_diameter"].kind == "diameter_outer"
    assert beaker["body_outer_diameter"].part_excluded == ""


def test_a_square_opening_is_measured_across_it_and_not_as_a_diameter():
    """A deep-well plate states an 8.2 mm "square internal mouth width".

    A square has no diameter, so the reading a round bore gets is the wrong one, and the plate's
    own 85 mm width is not the mouth. Left unrecognised the number went unmeasured, and a plate
    could put any size of well in and still pass.
    """
    plate = {t.name: t for t in load_case(CASES, "DWP-001").to_grounding_spec().dimensions}

    assert plate["mouth_width"].kind == "width_inner"
    assert plate["width"].kind == "extent_x"


def test_calling_a_bore_a_mouth_does_not_make_it_an_outside_diameter():
    """The opening words gained `mouth`, and the diameters they qualify must not move."""
    tube = {t.name: t.kind for t in load_case(CASES, "MCT-001").to_grounding_spec().dimensions}
    flask = {t.name: t.kind for t in load_case(CASES, "ERL-001").to_grounding_spec().dimensions}

    assert tube["mouth_inner_diameter"] == "diameter_inner"
    assert flask["neck_outer_diameter"] == "diameter_outer"


def test_an_object_with_a_stepped_top_has_two_heights_and_both_are_measured():
    """A rack whose rows sit at two levels is quoted twice from one datum, and means it.

    Both readings were `extent_z`, so they were two claims about one number: whichever the
    asset matched, the other failed no matter what was built, and the pair was declined rather
    than scored. They are the two ends of a top that is not flat.
    """
    rack = {t.name: t.kind for t in load_case(CASES, "RAC-001").to_grounding_spec().dimensions}
    assert rack["max_height"] == "extent_z"
    assert rack["low_height"] == "extent_z_min"


def test_a_height_stated_from_the_bench_to_the_top_is_a_height_whatever_it_is_called():
    """CTR-001 quotes 56.7 mm and 41.5 mm as its `high_side` and `low_side`.

    Neither name contains "height", so neither was measurable, though the stated locations say
    "Base to rear top" and "Base to front top" -- which is what a height is.
    """
    tray = {t.name: t.kind for t in load_case(CASES, "CTR-001").to_grounding_spec().dimensions}
    assert tray["high_side"] == "extent_z"
    assert tray["low_side"] == "extent_z_min"


def test_a_depth_is_measured_across_the_bench_not_up_the_object():
    """`depth` used to be measured as height, which contradicted the stated height.

    An instrument that states a 335 mm depth and a 105 mm height was being held to both
    numbers as its Z extent, so satisfying one failed the other.
    """
    targets = load_case(CASES, "HOT-001").to_grounding_spec().dimensions
    whole = {t.name: t.kind for t in targets if not t.part}

    assert whole == {"width": "extent_x", "depth": "extent_y", "height": "extent_z"}


def test_a_part_keeps_its_own_width_beside_the_instrument_s():
    """A hotplate's top plate is 200 mm across and the instrument is 335 mm.

    Both are `extent_x`, so one of them used to be discarded for contradicting the other —
    the specification was not contradicting itself, it was describing two things. Scoping
    the qualified one to its part is what lets the benchmark hold the generator to both.
    """
    targets = {t.name: t for t in load_case(CASES, "HOT-001").to_grounding_spec().dimensions}

    assert targets["plate_width"].part == "plate"
    assert targets["width"].part == ""
    assert targets["plate_width"].value_m < targets["width"].value_m


def test_the_probe_comes_from_the_case_not_from_a_default(beaker_case):
    spec = beaker_case.to_grounding_spec()
    assert spec.probe is not None
    assert spec.probe.diameter_mm == 5.0
    assert spec.probe.mass_g == pytest.approx(0.1)
    assert spec.probe.approach_clearance_mm == 50.0


def test_nominal_capacity_is_not_used_as_the_cavity_target(beaker_case):
    """The specification says in as many words that nominal capacity is not a target.

    Here both numbers happen to be 250, so the test checks the field that was read rather
    than the value — otherwise it would pass for the wrong reason.
    """
    spec = beaker_case.to_grounding_spec()
    assert spec.cavity is not None
    functional = beaker_case.spec["functional_requirements"][0]
    assert spec.cavity.minimum_volume_ml == functional["geometry_proxy_min_ml"]
