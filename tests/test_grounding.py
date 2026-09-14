"""Measurement tests, run against geometry whose answers are known in closed form.

A synthetic beaker is used rather than a generated asset on purpose. The point is to
pin the measurement code down: a cylinder with a bore has an outer diameter, an inner
diameter, a wall thickness and a capacity that can be written down, so a discrepancy is
unambiguously the measurement's fault. Testing against a generated asset would only
establish that two pieces of unverified code agree with each other.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import trimesh

from amx.grounding import mesh as gmesh
from amx.grounding.build import GroundedAsset
from amx.grounding.checks import check_dimensions, check_operations, check_protocol, check_topology
from amx.grounding.scene import stage
from amx.grounding.signals import render_grounding_signals
from amx.grounding.spec import (
    CavityTarget,
    ComponentTarget,
    DimensionTarget,
    GroundingSpec,
    OperationTarget,
    ProbeTarget,
    StabilityTarget,
)
from amx.report import Severity

OUTER_RADIUS = 0.035
INNER_RADIUS = 0.032
HEIGHT = 0.095
FLOOR = 0.003
MASS = 0.115

BRIM_VOLUME_ML = math.pi * INNER_RADIUS**2 * (HEIGHT - FLOOR) * 1e6


def _beaker(notch_at: float | None = None) -> trimesh.Trimesh:
    """A plain cylindrical vessel, optionally with a notch cut into its rim.

    The notch stands in for a pouring spout: it lowers the height at which the contents
    would run out without changing the wall or the overall height.
    """
    outer = trimesh.creation.cylinder(radius=OUTER_RADIUS, height=HEIGHT, sections=128)
    outer.apply_translation([0, 0, HEIGHT / 2])
    bore = trimesh.creation.cylinder(radius=INNER_RADIUS, height=HEIGHT, sections=128)
    bore.apply_translation([0, 0, FLOOR + HEIGHT / 2])
    solids = [outer, bore]
    if notch_at is not None:
        notch = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
        notch.apply_translation([OUTER_RADIUS - 0.005, 0, notch_at + 0.01])
        solids.append(notch)
    return trimesh.boolean.difference(solids)


@pytest.fixture(scope="module")
def beaker_asset(tmp_path_factory) -> GroundedAsset:
    root = tmp_path_factory.mktemp("beaker")
    mjcf_dir = root / "mjcf"
    (mjcf_dir / "meshes").mkdir(parents=True)
    _beaker().export(mjcf_dir / "meshes" / "body.stl")
    (mjcf_dir / "asset.xml").write_text(
        '<mujoco model="beaker">'
        '<compiler meshdir="meshes"/>'
        '<asset><mesh name="body" file="body.stl"/></asset>'
        '<worldbody><body name="beaker_body">'
        f'<geom type="mesh" mesh="body" mass="{MASS}"/>'
        "</body></worldbody></mujoco>"
    )
    return GroundedAsset(
        asset_id="beaker",
        root=root,
        model_path=root / "model.py",
        urdf_path=root / "model.urdf",
        mjcf_path=mjcf_dir / "asset.xml",
        source_sha="test",
    )


@pytest.fixture(scope="module")
def beaker_world(beaker_asset):
    staged = stage(beaker_asset.mjcf_path, ground=False, free=False)
    return gmesh.world_mesh(staged.model, staged.data)


def _spec(**overrides) -> GroundingSpec:
    base = {
        "asset_id": "beaker",
        "asset_class": "beaker",
        "summary": "250 mL low-form beaker",
        "dimensions": [
            DimensionTarget(
                id="DIM-OD", name="outer_diameter", value_m=2 * OUTER_RADIUS, kind="diameter_outer"
            ),
            DimensionTarget(id="DIM-H", name="overall_height", value_m=HEIGHT, kind="extent_z"),
            DimensionTarget(
                id="DIM-W",
                name="wall_thickness",
                value_m=OUTER_RADIUS - INNER_RADIUS,
                kind="wall_thickness",
                measurement_location="mid-height wall section",
            ),
        ],
        "components": [ComponentTarget(id="CMP-1", name="beaker body")],
        "total_mass_kg": (0.10, 0.13),
    }
    base.update(overrides)
    return GroundingSpec(**base)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def test_outer_diameter_ignores_the_bounding_box(beaker_world):
    measured = gmesh.outer_diameter(beaker_world)
    assert measured == pytest.approx(2 * OUTER_RADIUS, rel=1e-3)


def test_profile_reads_the_bore_and_the_wall(beaker_world):
    profile = gmesh.representative_profile(beaker_world, fraction=0.5)
    assert profile is not None
    assert profile.inner_diameter == pytest.approx(2 * INNER_RADIUS, rel=1e-3)
    assert profile.wall_thickness == pytest.approx(OUTER_RADIUS - INNER_RADIUS, rel=2e-3)


def test_cavity_volume_matches_the_analytic_capacity(beaker_world):
    cavity = gmesh.cavity_volume(beaker_world)
    assert cavity is not None
    assert cavity.volume_ml == pytest.approx(BRIM_VOLUME_ML, rel=5e-3)
    assert cavity.converged
    assert cavity.floor_height == pytest.approx(FLOOR, abs=0.001)
    assert cavity.overflow_height == pytest.approx(HEIGHT, abs=0.001)


def test_a_solid_block_has_no_cavity():
    block = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
    assert gmesh.cavity_volume(block) is None


def _well_grid(*, wells: int, pitch: float, bore: float, posts: int = 0) -> trimesh.Trimesh:
    """A row of bored tubes, optionally beside a couple of solid ones.

    A stand-in for PCR-001's rack: 96 tubes of 5.2 mm bored out to 4 mm, and two solid
    5.2 mm hinge posts the author added to hang the lid off.
    """
    parts = []
    for index in range(wells):
        tube = trimesh.creation.annulus(r_min=bore / 2, r_max=0.0026, height=0.026)
        tube.apply_translation([index * pitch, 0.0, 0.0])
        parts.append(tube)
    for index in range(posts):
        post = trimesh.creation.cylinder(radius=0.0026, height=0.026)
        post.apply_translation([-(index + 1) * pitch, 0.0, 0.0])
        parts.append(post)
    return trimesh.util.concatenate(parts)


def test_a_bore_is_read_from_the_wells_and_not_from_a_solid_post():
    """PCR-001 scored 91.8 with wells 23% too narrow to accept a tube, because of this.

    Its rack sections into 96 wells and two solid hinge posts of the same diameter. A solid
    disc has more area than a ring, so taking the largest polygon always took a post — and
    a post has no interior, so the bore came back unmeasurable rather than wrong. Ninety-six
    wells outvote two posts.
    """
    profile = gmesh.profile_at(_well_grid(wells=12, pitch=0.009, bore=0.004, posts=2), 0.0)

    assert profile is not None
    assert profile.feature_count == 12
    assert profile.inner_diameter == pytest.approx(0.004, rel=0.02)


def test_a_lone_wall_beside_a_handle_is_still_the_wall(beaker_world):
    """Grouping by population must not change the answer where there was no population.

    A beaker sections into one wall and perhaps a handle: every group has one member, so
    the tie falls to the larger area, which is the wall it always was.
    """
    profile = gmesh.profile_at(beaker_world, 0.05)

    assert profile is not None
    assert profile.feature_count == 1
    assert profile.inner_diameter == pytest.approx(2 * INNER_RADIUS, rel=1e-3)


def test_a_pitch_is_the_gap_to_the_next_well_not_the_width_of_the_plate():
    """A mean over every pair would report the size of the plate instead."""
    profile = gmesh.profile_at(_well_grid(wells=8, pitch=0.009, bore=0.004), 0.0)

    assert profile is not None
    assert gmesh.feature_pitch(profile) == pytest.approx(0.009, rel=0.01)


def test_a_body_named_in_a_different_word_order_is_the_same_body():
    """PCR-001 asks for an "8x12 rack"; a body called `rack_8x12` is that rack.

    The in-loop check matched normalised substrings, and "8x12rack" is not a substring of
    "rack8x12" either way round, so it reported the component missing. The judge used a matcher
    with a token pass and found it without difficulty. The generator renamed the body four times
    trying to satisfy the one that could not be satisfied, and ran out of turns.
    """
    from amx.grounding.checks import match_body
    from amx.grounding.physics import match_bodies

    bodies = ["rack_8x12", "removable_hinged_lid"]

    assert match_body("8x12 rack", bodies) == "rack_8x12"
    assert match_body("Removable hinged lid", bodies) == "removable_hinged_lid"
    # The two readers must not disagree about any name, whichever way it is written.
    for hint in ("8x12 rack", "rack 8x12", "Removable hinged lid", "hinged lid", "absent part"):
        ranked = match_bodies(hint, bodies)
        assert match_body(hint, bodies) == (ranked[0] if ranked else None)


def _drilled_rack(bore: float, along: float, across: float) -> trimesh.Trimesh:
    """A solid slab with a 4x3 grid of bores through it, at known spacings."""
    slab = trimesh.creation.box(extents=[0.10, 0.08, 0.03])
    slab.apply_translation([0.0, 0.0, 0.015])
    holes = []
    for i in range(4):
        for j in range(3):
            hole = trimesh.creation.cylinder(radius=bore / 2.0, height=0.05)
            hole.apply_translation(
                [-1.5 * along + i * along, -1.0 * across + j * across, 0.02]
            )
            holes.append(hole)
    return slab.difference(trimesh.util.concatenate(holes))


def test_bores_drilled_through_a_slab_are_counted_as_the_repeated_feature():
    """A rack made this way is one outline with holes in it, not many outlines.

    The population logic looked for separate shapes, which is how PCR-001's standing tubes reach
    a section, and reported one feature for every plate and rack drilled out of a solid slab --
    so no count and no pitch, on exactly the objects whose pitch is the point of them.
    """
    from amx.grounding.mesh import feature_pitch, profile_at

    profile = profile_at(_drilled_rack(0.0109, 0.0179, 0.0239), 0.02)

    assert profile is not None
    assert profile.feature_count == 12
    assert profile.inner_diameter == pytest.approx(0.0109, abs=3e-4)
    assert feature_pitch(profile) == pytest.approx(0.0179, abs=3e-4)


def test_a_rectangular_grid_has_two_spacings_and_reports_both():
    """A rack quotes a gap along a row and a wider one between rows: one grid, two numbers."""
    from amx.grounding.mesh import feature_pitch_axes, profile_at

    profile = profile_at(_drilled_rack(0.0109, 0.0179, 0.0239), 0.02)
    closer, wider = feature_pitch_axes(profile)

    assert closer == pytest.approx(0.0179, abs=3e-4)
    assert wider == pytest.approx(0.0239, abs=3e-4)
    # What the datasheet states is the material left between the openings.
    assert closer - profile.inner_span == pytest.approx(0.007, abs=4e-4)
    assert wider - profile.inner_span == pytest.approx(0.013, abs=4e-4)


def test_a_square_grid_reports_its_one_spacing_for_both():
    """Nothing about the pair requires the grid to be rectangular."""
    from amx.grounding.mesh import feature_pitch_axes, profile_at

    closer, wider = feature_pitch_axes(profile_at(_drilled_rack(0.006, 0.009, 0.009), 0.02))

    assert closer == pytest.approx(wider, abs=3e-4)
    assert closer == pytest.approx(0.009, abs=3e-4)


def test_a_square_bore_is_read_across_its_side_not_around_a_circle():
    """A median radius doubled overstates a square opening; its side is what is quoted."""
    from amx.grounding.mesh import profile_at

    block = trimesh.creation.box(extents=[0.02, 0.02, 0.04])
    bore = trimesh.creation.box(extents=[0.0082, 0.0082, 0.05])
    walled = block.difference(bore)

    profile = profile_at(walled, float(walled.bounds[:, 2].mean()))

    assert profile is not None
    assert profile.inner_span == pytest.approx(0.0082, abs=2e-4)


def test_a_round_bore_reads_the_same_across_as_around():
    """The clear width and the diameter are one number for a circle, and must agree."""
    from amx.grounding.mesh import profile_at

    tube = trimesh.creation.annulus(r_min=0.004, r_max=0.006, height=0.03)

    profile = profile_at(tube, float(tube.bounds[:, 2].mean()))

    assert profile is not None
    assert profile.inner_span == pytest.approx(0.008, abs=3e-4)
    assert profile.inner_span == pytest.approx(profile.inner_diameter, abs=3e-4)


def test_the_top_of_a_stepped_object_is_read_at_both_of_its_levels():
    """The two heights such an object is quoted with, taken from the geometry."""
    from amx.grounding.mesh import top_surface_range

    low = trimesh.creation.box(extents=[0.04, 0.02, 0.028])
    low.apply_translation([0.02, 0.0, 0.014])
    high = trimesh.creation.box(extents=[0.04, 0.02, 0.044])
    high.apply_translation([-0.02, 0.0, 0.022])

    lowest, highest = top_surface_range(trimesh.util.concatenate([low, high]))

    assert lowest == pytest.approx(0.028, abs=5e-4)
    assert highest == pytest.approx(0.044, abs=5e-4)


def test_a_flat_topped_object_has_one_height_reported_twice():
    """Nothing about the pair says the top has to be stepped, and a box must not confuse it."""
    from amx.grounding.mesh import top_surface_range

    lowest, highest = top_surface_range(trimesh.creation.box(extents=[0.05, 0.03, 0.02]))

    assert lowest == pytest.approx(highest, abs=5e-4)
    assert highest == pytest.approx(0.02, abs=5e-4)


def test_a_single_feature_has_no_pitch():
    profile = gmesh.profile_at(_well_grid(wells=1, pitch=0.009, bore=0.004), 0.0)

    assert profile is not None
    assert gmesh.feature_pitch(profile) is None


def test_ribs_around_a_vessel_do_not_outvote_its_interior(beaker_world):
    """The population rule reads one well's bore; it must not decide what a cavity is.

    Four moulded grip ribs section into four solid shapes beside one walled one, so the
    ribs are the larger population — and integrating them would report a beaker with a
    perfectly good interior as solid.
    """
    ribbed = [beaker_world]
    for index in range(4):
        rib = trimesh.creation.box(extents=[0.004, 0.004, 0.06])
        rib.apply_translation([0.037, 0.01 * index - 0.015, 0.03])
        ribbed.append(rib)
    world = trimesh.util.concatenate(ribbed)

    cavity = gmesh.cavity_volume(world)

    assert cavity is not None
    assert cavity.volume_ml == pytest.approx(BRIM_VOLUME_ML, rel=5e-2)


def test_a_spout_lowers_the_overflow_edge(tmp_path):
    """The usable volume is capped by the lowest opening, not by the height of the wall."""
    notched = _beaker(notch_at=0.075)
    plain = gmesh.cavity_volume(_beaker())
    spouted = gmesh.cavity_volume(notched)
    assert plain is not None and spouted is not None
    assert spouted.overflow_height < plain.overflow_height - 0.005
    assert spouted.volume_ml < plain.volume_ml


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def test_a_conforming_asset_produces_no_failures(beaker_asset):
    report = check_dimensions(beaker_asset, _spec())
    assert report.passed, report.to_text()
    assert all(f.severity is Severity.INFO for f in report.findings)


def test_a_wrong_dimension_is_reported_with_both_numbers(beaker_asset):
    spec = _spec(
        dimensions=[
            DimensionTarget(id="DIM-OD", name="outer_diameter", value_m=0.050, kind="diameter_outer")
        ]
    )
    report = check_dimensions(beaker_asset, spec)
    assert not report.passed
    finding = report.failures[0]
    assert finding.code == "G-DIM-HARD"
    assert finding.metrics["measured_m"] == pytest.approx(0.070, rel=1e-2)
    assert finding.metrics["target_m"] == 0.050


@pytest.fixture(scope="module")
def bucket_asset(tmp_path_factory) -> GroundedAsset:
    """Three concentric parts of three different diameters, as BUC-001 states them.

    A bowl 57 mm across on a 10 mm stem over a 48 mm support disc. Nothing about it is
    unusual, and it was unbuildable: all three are `diameter_outer`, and measured on the
    assembled mesh all three read the same number.
    """
    root = tmp_path_factory.mktemp("bucket")
    mjcf_dir = root / "mjcf"
    mjcf_dir.mkdir()
    (mjcf_dir / "asset.xml").write_text(
        '<mujoco model="bucket"><worldbody>'
        '<body name="bowl" pos="0 0 0.06">'
        '<geom type="cylinder" size="0.0285 0.02" mass="0.3"/></body>'
        '<body name="support_disc" pos="0 0 0.038">'
        '<geom type="cylinder" size="0.024 0.002" mass="0.05"/></body>'
        '<body name="stem" pos="0 0 0.015">'
        '<geom type="cylinder" size="0.005 0.021" mass="0.02"/></body>'
        "</worldbody></mujoco>"
    )
    return GroundedAsset(
        asset_id="bucket",
        root=root,
        model_path=root / "model.py",
        urdf_path=root / "model.urdf",
        mjcf_path=mjcf_dir / "asset.xml",
        source_sha="test",
    )


def _bucket_dimensions(scoped: bool) -> list[DimensionTarget]:
    parts = {"bowl_od": "bowl", "stem_od": "stem", "support_disc_diameter": "support disc"}
    values = {"bowl_od": 0.057, "stem_od": 0.010, "support_disc_diameter": 0.048}
    return [
        DimensionTarget(
            id=f"DIM-{index}",
            name=name,
            value_m=values[name],
            kind="diameter_outer",
            part=parts[name] if scoped else "",
        )
        for index, name in enumerate(values)
    ]


def test_three_parts_of_three_diameters_can_all_be_right(bucket_asset):
    report = check_dimensions(
        bucket_asset,
        GroundingSpec(asset_id="bucket", dimensions=_bucket_dimensions(scoped=True)),
    )

    assert report.failures == []
    measured = {f.subject.split("/")[-1]: f.metrics["measured_m"] for f in report.findings}
    assert measured["bowl_od"] == pytest.approx(0.057, abs=1e-4)
    assert measured["stem_od"] == pytest.approx(0.010, abs=1e-4)
    assert measured["support_disc_diameter"] == pytest.approx(0.048, abs=1e-4)


def test_without_a_part_they_all_read_the_same_number(bucket_asset):
    """Why the field had to exist, kept as a test so the reason stays visible.

    Two of these three cannot pass whatever the generator builds, and the run was refused
    over them until its turn budget ran out.
    """
    report = check_dimensions(
        bucket_asset,
        GroundingSpec(asset_id="bucket", dimensions=_bucket_dimensions(scoped=False)),
    )

    measured = {f.metrics["measured_m"] for f in report.findings}
    assert len(measured) == 1, "one measurement is being shared by three different parts"
    assert len(report.failures) == 2


def test_a_part_that_is_not_there_is_not_evidence_of_a_wrong_size(bucket_asset):
    """A warning, not a failure: the measurement was never taken.

    Calling an unrecognised part name a wrong dimension would recreate the unfixable
    refusal the part field exists to remove. The missing part is `check_topology`'s to
    report, against the component list.
    """
    report = check_dimensions(
        bucket_asset,
        GroundingSpec(
            asset_id="bucket",
            dimensions=[
                DimensionTarget(
                    id="DIM-X", name="lid_od", value_m=0.06, kind="diameter_outer", part="lid"
                )
            ],
        ),
    )

    assert report.failures == []
    finding = report.findings[0]
    assert finding.code == "G-DIM-UNSCOPED"
    assert "bowl" in finding.summary, "say what is there, so the name can be fixed"


def test_a_part_name_need_not_match_the_body_name_exactly(bucket_asset):
    """`support disc` is the body `support_disc`, the same way components are matched."""
    report = check_dimensions(
        bucket_asset,
        GroundingSpec(
            asset_id="bucket",
            dimensions=[
                DimensionTarget(
                    id="DIM-D",
                    name="support_disc_diameter",
                    value_m=0.048,
                    kind="diameter_outer",
                    part="Support Disc",
                )
            ],
        ),
    )

    assert report.findings[0].code == "G-DIM"
    assert report.findings[0].severity is Severity.INFO


def test_a_short_cavity_is_a_failure_and_a_sufficient_one_is_not(beaker_asset):
    generous = check_topology(beaker_asset, _spec(cavity=CavityTarget(minimum_volume_ml=250.0)))
    assert generous.passed, generous.to_text()

    demanding = check_topology(beaker_asset, _spec(cavity=CavityTarget(minimum_volume_ml=500.0)))
    assert not demanding.passed
    assert demanding.failures[0].code == "G-CAVITY-VOLUME"


def test_a_missing_component_names_what_is_present(beaker_asset):
    spec = _spec(
        components=[
            ComponentTarget(id="CMP-1", name="beaker body"),
            ComponentTarget(id="CMP-9", name="hinged lid"),
        ]
    )
    report = check_topology(beaker_asset, spec)
    failure = next(f for f in report.failures if f.code == "G-COMPONENT-MISSING")
    assert "beaker_body" in failure.summary


def test_it_stands_up_on_a_bench(beaker_asset):
    report = check_protocol(beaker_asset, _spec(stability=StabilityTarget()))
    assert report.passed, report.to_text()


def test_a_probe_falls_into_the_cavity_rather_than_onto_the_rim(beaker_asset):
    """Only true against a concave decomposition; MuJoCo collides a mesh as its hull."""
    report = check_protocol(beaker_asset, _spec(probe=ProbeTarget(diameter_mm=5.0, mass_g=0.1)))
    entered = next(f for f in report.findings if f.code == "G-PROBE")
    assert entered.severity is Severity.INFO
    assert entered.metrics["probe_final_z_mm"] < entered.metrics["cavity_overflow_mm"]


def test_a_probe_too_wide_for_the_mouth_is_rejected(beaker_asset):
    report = check_protocol(beaker_asset, _spec(probe=ProbeTarget(diameter_mm=90.0, mass_g=0.1)))
    failure = report.failures[0]
    assert failure.code == "G-PROBE-CLEARANCE"


def _operation_asset(tmp_path: Path, joint: str) -> GroundedAsset:
    mjcf = tmp_path / "asset.xml"
    mjcf.write_text(
        '<mujoco model="controls"><worldbody><body name="control_panel">'
        '<geom type="box" size=".05 .05 .005" mass=".1"/>'
        '<body name="start_button" pos="0 0 .01">'
        f"{joint}"
        '<geom type="box" size=".01 .01 .002" mass=".005"/>'
        "</body></body></worldbody></mujoco>"
    )
    return GroundedAsset(
        asset_id="controls",
        root=tmp_path,
        model_path=tmp_path / "model.py",
        urdf_path=tmp_path / "model.urdf",
        mjcf_path=mjcf,
        source_sha="test",
    )


def test_operation_check_physically_presses_and_returns_button(tmp_path):
    asset = _operation_asset(
        tmp_path,
        '<joint name="press" type="slide" axis="0 0 -1" limited="true" range="0 .002"/>',
    )
    spec = GroundingSpec(
        asset_id="controls",
        operations=[
            OperationTarget(
                id="OP-START",
                name="START",
                kind="press",
                child_hint="start button",
                expected_joint_types=["slide"],
            )
        ],
    )
    report = check_operations(asset, spec)
    assert report.passed, report.to_text()
    assert report.findings[0].metrics["translation_mm"] == pytest.approx(2.0)


def test_operation_check_rejects_a_fixed_visual_button(tmp_path):
    asset = _operation_asset(tmp_path, "")
    spec = GroundingSpec(
        asset_id="controls",
        operations=[
            OperationTarget(
                id="OP-START",
                name="START",
                kind="press",
                child_hint="start button",
                expected_joint_types=["slide"],
            )
        ],
    )
    report = check_operations(asset, spec)
    assert not report.passed
    assert report.failures[0].code == "G-OP-FIXED"


def test_a_button_that_turns_instead_of_pressing_is_not_a_button(tmp_path):
    asset = _operation_asset(
        tmp_path,
        '<joint name="press" type="hinge" axis="0 0 1" limited="true" range="0 .01"/>',
    )
    spec = GroundingSpec(
        asset_id="controls",
        operations=[
            OperationTarget(
                id="OP-START", name="START", kind="press", child_hint="start button"
            )
        ],
    )
    report = check_operations(asset, spec)
    assert not report.passed
    failure = report.failures[0]
    assert failure.code == "G-OP-JOINT-TYPE"
    assert "does not rotate" in failure.summary


def _lever_asset(tmp_path: Path, *, obstruct: bool) -> GroundedAsset:
    """A lever that swings over a post, or straight through it.

    The post stands where the lever passes at roughly 45°, and nowhere near either end of
    the travel — so the interference exists only partway along the path and an
    endpoint-only check cannot see it.
    """
    post = (
        '<body name="post" pos=".03 0 .03">'
        '<geom type="box" size=".004 .01 .02" mass=".05"/></body>'
    ) if obstruct else ""
    mjcf = tmp_path / "asset.xml"
    mjcf.write_text(
        '<mujoco model="lever"><worldbody><body name="base">'
        '<geom type="box" size=".05 .05 .005" mass=".2"/>'
        f"{post}"
        '<body name="release_lever" pos="0 0 .005">'
        '<joint name="swing" type="hinge" axis="0 1 0" limited="true" range="0 1.5708"/>'
        '<geom type="box" pos=".025 0 .045" size=".025 .008 .004" mass=".02"/>'
        "</body></body></worldbody></mujoco>"
    )
    return GroundedAsset(
        asset_id="lever",
        root=tmp_path,
        model_path=tmp_path / "model.py",
        urdf_path=tmp_path / "model.urdf",
        mjcf_path=mjcf,
        source_sha="test",
    )


def _lever_spec() -> GroundingSpec:
    return GroundingSpec(
        asset_id="lever",
        operations=[
            OperationTarget(
                id="OP-LEVER",
                name="release lever",
                kind="hinge",
                child_hint="release lever",
                expected_joint_types=["hinge"],
            )
        ],
    )


def test_a_clear_swing_passes_along_its_whole_path(tmp_path):
    report = check_operations(_lever_asset(tmp_path, obstruct=False), _lever_spec())
    assert report.passed, report.to_text()
    assert report.findings[0].metrics["path_samples"] > 1


def test_interference_partway_through_the_travel_is_a_failure(tmp_path):
    """Endpoint-only testing passed lids that plough through their housing on the way."""
    report = check_operations(_lever_asset(tmp_path, obstruct=True), _lever_spec())
    assert not report.passed, report.to_text()
    failure = report.failures[0]
    assert failure.code == "G-OP-EXECUTION"
    assert "partway along the travel" in failure.summary


def test_a_cap_that_slides_off_by_metres_is_not_being_operated(tmp_path):
    """CRY-001 authored a 12.5 mm cryovial whose cap unscrews by sliding 6.28 m.

    The number is 2π: an angle in radians written into a prismatic joint, whose limits
    MuJoCo reads as metres. Every collision test passed, because after the first
    millimetre of travel there is nothing left for the cap to hit, and the case was
    certified at 94/100. Distance travelled has to be held against the size of the object
    that is supposed to contain the motion.
    """
    mjcf = tmp_path / "asset.xml"
    mjcf.write_text(
        '<mujoco model="cryovial"><worldbody><body name="vial_body">'
        '<geom type="cylinder" size=".00625 .022" mass=".005"/>'
        '<body name="screw_cap" pos="0 0 .024">'
        '<joint name="body_to_cap" type="slide" axis="0 0 1" limited="true" range="0 6.28319"/>'
        '<geom type="cylinder" size=".00675 .004" mass=".001"/>'
        "</body></body></worldbody></mujoco>"
    )
    asset = GroundedAsset(
        asset_id="cryovial",
        root=tmp_path,
        model_path=tmp_path / "model.py",
        urdf_path=tmp_path / "model.urdf",
        mjcf_path=mjcf,
        source_sha="test",
    )
    spec = GroundingSpec(
        asset_id="cryovial",
        operations=[
            OperationTarget(
                id="OP-CAP",
                name="screw cap",
                kind="remove",
                child_hint="screw cap",
                expected_joint_types=["free", "slide"],
            )
        ],
    )
    report = check_operations(asset, spec)
    assert not report.passed, report.to_text()
    failure = report.failures[0]
    assert failure.code == "G-OP-EXECUTION"
    assert "the part leaving, not being operated" in failure.summary


def test_a_cap_that_lifts_clear_of_its_own_neck_still_passes(tmp_path):
    """The same asset with the travel a cap actually has, so the bound is not a blanket no."""
    mjcf = tmp_path / "asset.xml"
    mjcf.write_text(
        '<mujoco model="cryovial"><worldbody><body name="vial_body">'
        '<geom type="cylinder" size=".00625 .022" mass=".005"/>'
        '<body name="screw_cap" pos="0 0 .028">'
        '<joint name="body_to_cap" type="slide" axis="0 0 1" limited="true" range="0 .012"/>'
        '<geom type="cylinder" size=".00675 .004" mass=".001"/>'
        "</body></body></worldbody></mujoco>"
    )
    asset = GroundedAsset(
        asset_id="cryovial",
        root=tmp_path,
        model_path=tmp_path / "model.py",
        urdf_path=tmp_path / "model.urdf",
        mjcf_path=mjcf,
        source_sha="test",
    )
    spec = GroundingSpec(
        asset_id="cryovial",
        operations=[
            OperationTarget(
                id="OP-CAP",
                name="screw cap",
                kind="remove",
                child_hint="screw cap",
                expected_joint_types=["free", "slide"],
            )
        ],
    )
    report = check_operations(asset, spec)
    assert report.passed, report.to_text()


def test_a_control_that_carries_the_housing_is_parented_backwards(tmp_path):
    """The knob is the parent of the case, so turning the knob turns the whole object."""
    mjcf = tmp_path / "asset.xml"
    mjcf.write_text(
        '<mujoco model="inverted"><worldbody>'
        '<body name="adjustment_knob">'
        '<joint name="turn" type="hinge" axis="0 0 1" limited="true" range="0 1"/>'
        '<geom type="cylinder" size=".01 .005" mass=".01"/>'
        '<body name="instrument_housing" pos="0 0 .02">'
        '<geom type="box" size=".05 .05 .02" mass="1"/>'
        "</body></body></worldbody></mujoco>"
    )
    asset = GroundedAsset(
        asset_id="inverted",
        root=tmp_path,
        model_path=tmp_path / "model.py",
        urdf_path=tmp_path / "model.urdf",
        mjcf_path=mjcf,
        source_sha="test",
    )
    spec = GroundingSpec(
        asset_id="inverted",
        components=[
            ComponentTarget(id="CMP-1", name="instrument housing", kind="fixed"),
        ],
        operations=[
            OperationTarget(
                id="OP-KNOB",
                name="adjustment knob",
                kind="rotate",
                child_hint="adjustment knob",
                expected_joint_types=["hinge"],
            )
        ],
    )
    report = check_operations(asset, spec)
    assert not report.passed
    failure = report.failures[0]
    assert failure.code == "G-OP-DISTURBANCE"
    assert "instrument_housing" in failure.detail["dragged_bodies"]


def test_a_removable_part_is_lifted_out_rather_than_swung(tmp_path):
    """A tray with six free degrees of freedom comes out upwards; nothing rotates it 45°.

    Rotating removable parts is what produced the penetration failures on trays, gaskets
    and adapters: the motion was never the one the object has, and swinging a part seated
    in a recess drives it through the wall it sits in.

    The three slides and a ball are how a floating joint reaches MuJoCo on a nested body,
    which is the shape the exporter produces — a `freejoint` is only legal at top level.
    """
    mjcf = tmp_path / "asset.xml"
    mjcf.write_text(
        '<mujoco model="rack"><worldbody><body name="rack_base">'
        '<geom type="box" size=".05 .05 .005" mass=".3"/>'
        '<body name="sample_tray" pos="0 0 .012">'
        '<joint name="tray__x" type="slide" axis="1 0 0"/>'
        '<joint name="tray__y" type="slide" axis="0 1 0"/>'
        '<joint name="tray__z" type="slide" axis="0 0 1"/>'
        '<joint name="tray__rot" type="ball"/>'
        '<geom type="box" size=".04 .04 .006" mass=".05"/>'
        "</body></body></worldbody></mujoco>"
    )
    asset = GroundedAsset(
        asset_id="rack",
        root=tmp_path,
        model_path=tmp_path / "model.py",
        urdf_path=tmp_path / "model.urdf",
        mjcf_path=mjcf,
        source_sha="test",
    )
    spec = GroundingSpec(
        asset_id="rack",
        operations=[
            OperationTarget(
                id="OP-TRAY",
                name="removable sample tray",
                kind="remove",
                child_hint="sample tray",
                expected_joint_types=["free", "slide"],
                return_required=False,
            )
        ],
    )
    report = check_operations(asset, spec)
    assert report.passed, report.to_text()
    finding = report.findings[0]
    assert finding.metrics["rotation_deg"] == pytest.approx(0.0, abs=1e-6)
    assert finding.metrics["translation_mm"] == pytest.approx(20.0, abs=0.1)


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #


def test_signals_mirror_the_compile_block_the_model_already_reads(beaker_asset):
    spec = _spec(
        dimensions=[
            DimensionTarget(id="DIM-OD", name="outer_diameter", value_m=0.050, kind="diameter_outer")
        ]
    )
    text = render_grounding_signals(check_dimensions(beaker_asset, spec), check_name="physical")
    assert text.startswith("<grounding_signals>")
    assert "<summary>" in text and "<failures>" in text and "<response_rules>" in text
    assert "status=failure" in text


def test_a_clean_check_says_so_without_a_failures_section(beaker_asset):
    text = render_grounding_signals(check_dimensions(beaker_asset, _spec()))
    assert "status=success" in text
    assert "<failures>" not in text


# --------------------------------------------------------------------------- #
# meshes that are not clean solids
# --------------------------------------------------------------------------- #


def _beaker_with_graduations() -> trimesh.Trimesh:
    """A beaker with tick marks stuck on the outside, left unmerged.

    This is what a generated asset actually looks like: several primitives concatenated
    rather than booleaned, so the whole thing is non-watertight and a cross-section comes
    back as a mix of closed loops and open arcs where the coincident surfaces met the
    plane. Sectioning used to give up on that, and the caller could not tell a mesh it
    failed to read from a vessel with no cavity.
    """
    band = trimesh.creation.annulus(
        r_min=OUTER_RADIUS, r_max=OUTER_RADIUS + 0.0006, height=0.02, sections=128
    )
    band.apply_translation([0, 0, 0.03])
    return trimesh.util.concatenate([_beaker(), band])


def test_a_section_survives_surfaces_that_only_touch():
    """The tick marks sit on the wall; the wall and the bore must still be readable."""
    marked = _beaker_with_graduations()

    # The precondition: the plane cuts coincident surfaces, so the section is a mix of
    # closed loops and open arcs rather than the clean pair of circles it would be on a
    # merged solid. Assert it, or a future change to the fixture could make this test
    # pass for the wrong reason.
    section = marked.section(plane_origin=[0, 0, 0.03], plane_normal=[0, 0, 1])
    planar, _ = section.to_2D()
    assert any(not entity.closed for entity in planar.entities)

    profile = gmesh.profile_at(marked, 0.03)
    assert profile is not None
    # The band really is 0.6 mm proud, so the outer diameter is legitimately larger here.
    # What must survive is the bore, which the decoration does not touch.
    assert profile.outer_diameter == pytest.approx(2 * OUTER_RADIUS + 0.0012, abs=0.001)
    assert profile.inner_diameter == pytest.approx(2 * INNER_RADIUS, abs=0.001)


def test_decoration_does_not_shorten_the_cavity():
    """The bug this guards cost a correct asset 260 of its 296 mL.

    The sections through the decorated band came back empty, the overflow scan read the
    first empty section as the rim, and a full-height beaker measured as a 51 mL dish —
    in the judge and, worse, in the tool the agent was using to correct itself.
    """
    plain = gmesh.cavity_volume(_beaker())
    marked = gmesh.cavity_volume(_beaker_with_graduations())
    assert marked is not None
    assert marked.volume_m3 * 1e6 == pytest.approx(plain.volume_m3 * 1e6, rel=0.02)
    assert marked.overflow_height == pytest.approx(HEIGHT, abs=0.002)
