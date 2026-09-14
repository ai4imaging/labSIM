"""The parametric part library: every template builds, and what it builds is printable.

These are the tests that make the codesign agent trustworthy. The agent only ever emits
parameters, so if every parameter set inside the declared bounds produces a manifold,
loadable, DFM-clean part, then the agent cannot emit something unbuildable — which is the
whole reason for choosing parameters over generated geometry.
"""

from __future__ import annotations

import mujoco
import pytest
from pydantic import ValidationError

from amx.codesign import TEMPLATES, export_part
from amx.report import Severity


def _defaults(template):
    """A template's default parameters, which every template must define completely."""
    return template(part_id="probe")


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_defaults_build_and_are_manufacturable(name, tmp_path):
    part = _defaults(TEMPLATES[name])
    geometry = part.build()

    assert geometry.mesh.is_watertight, f"{name} built a mesh with holes in it"
    assert geometry.mesh.body_count == 1, f"{name} built {geometry.mesh.body_count} bodies"
    assert geometry.mesh.volume > 0.0
    assert geometry.collision, f"{name} has no collision volumes, so it cannot be simulated"

    _, report = export_part(part, tmp_path / name)
    failures = [f for f in report.findings if f.severity is Severity.FAILURE]
    assert not failures, f"{name} is not manufacturable by default: {failures}"


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_exported_part_loads_in_mujoco(name, tmp_path):
    """A part is only useful if MuJoCo will take it, so that is what the test asserts."""
    export_part(_defaults(TEMPLATES[name]), tmp_path / name)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / name / "part.xml"))
    assert model.ngeom > 0
    # The display mesh must not collide, or a socket's own walls fight whatever is seated.
    for geom in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) == "visual":
            assert int(model.geom_contype[geom]) == 0


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_collision_volumes_are_named(name):
    """Every collision volume carries a role, so rules and reports can refer to it.

    Without this the collision set is an unlabelled pile of boxes: a plan cannot say "the
    tube must reach the well floor" and the judge cannot measure how far in it went.
    """
    geometry = _defaults(TEMPLATES[name]).build()
    roles = [p.role for p in geometry.collision]
    assert all(roles), f"{name} has unnamed collision volumes"
    assert len(set(roles)) == len(roles), f"{name} reuses a collision role: {roles}"


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_each_parameter_swept_alone_stays_buildable(name):
    """The property the agent depends on: in-bounds parameters never fail to build.

    One parameter is moved at a time, from the defaults, across its declared range. Moving
    all of them together instead tests almost nothing on these templates — the parameters
    are coupled (a rack's pitch has to admit its own wells, a clamp's throat has to admit
    its jaws), so a joint random draw violates some validator essentially always, and a
    test that only ever sees rejections would pass while building nothing.

    A `ValidationError` is a pass. It is the schema refusing a combination before any
    geometry is attempted, which is the mechanism being relied on: the agent emits
    parameters, the schema is what stops bad ones, and the point of the test is that
    anything getting past the schema builds.
    """
    template = TEMPLATES[name]
    built = 0
    for field, info in template.model_fields.items():
        low, high = _bounds(info)
        if low is None or high is None:
            continue
        for fraction in (0.0, 0.5, 1.0):
            value = low + (high - low) * fraction
            try:
                geometry = template(part_id="probe", **{field: value}).build()
            except ValidationError:
                continue
            built += 1
            where = f"{name} with {field}={value:.5g}"
            assert geometry.mesh.is_watertight, f"{where} produced a leaky mesh"
            assert geometry.mesh.body_count == 1, f"{where} produced separate bodies"
            assert geometry.collision, f"{where} produced no collision volumes"
    assert built >= 3, f"almost every parameter of {name} was rejected; the bounds are wrong"


def _bounds(info):
    low = high = None
    for item in info.metadata:
        low = getattr(item, "gt", None) or getattr(item, "ge", None) or low
        high = getattr(item, "lt", None) or getattr(item, "le", None) or high
    return low, high


def test_material_limits_are_enforced_not_advisory(tmp_path):
    """A wall under the process minimum has to be reported, or the DFM check is decorative."""
    thin = TEMPLATES["tube_rack"](part_id="thin", wall_m=0.001, material_key="fdm_pla")
    _, report = export_part(thin, tmp_path / "thin")
    codes = {f.code for f in report.findings if f.severity is Severity.FAILURE}
    assert "D-WALL" in codes, f"a 1 mm wall in FDM PLA passed DFM: {report.findings}"
