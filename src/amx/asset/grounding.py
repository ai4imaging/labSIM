"""Checking that a generated asset is what the request asked for.

Three checks, in order of how much they cost to run and how much they can be trusted:

* physical — measure the compiled model and compare against the datasheet intervals.
* functional — drive each articulation and drop the asset on a plane, in simulation.
* vision — render it and ask a model whether the stated claims hold.

The first two are deterministic and are the ones that gate. Vision produces findings but
is advisory by default, because a rendering critic is the least reliable of the three and
should not be able to block a run on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from amx.asset.generate import AssetBundle
from amx.asset.spec import AssetRequest
from amx.llm import LlmClient, LlmUnavailable
from amx.report import Finding, RepairTarget, Report, Severity, interval_finding

SETTLE_SECONDS = 1.5
DRIVE_SECONDS = 1.0
MIN_ARTICULATION_TRAVEL_RAD = 0.05
DRIVE_GRAVITY_MULTIPLE = 4.0
"""How hard to drive an articulation, as a multiple of what gravity already applies to it.

A fixed torque cannot work across the range of things this pipeline builds: 5 N·m is far
too little to shift an instrument's lid and enough to fire a 0.2 g cap through its own stops
at several hundred radians a second, which measures nothing except how the solver handles
being abused. Scaling to the child's own gravitational torque asks the same question of
every joint — can something a few times heavier than the part itself move it — and gets an
answer in the range where the limits still hold.
"""

FRICTION_MARGIN = 1.5
"""How far past a joint's declared friction to drive it, as a multiple.

Enough to break stiction and no more. A detent's whole job is to resist, so it has to be
overcome to test the hinge behind it — but a snap-cap's friction is a hundred times the
weight of the cap, and driving at several times that is a torque nothing in the cell could
apply and the integrator cannot represent.
"""

MIN_DRIVE_TORQUE_N_M = 1e-4
LIMIT_OVERSHOOT_TOLERANCE_RAD = 0.05
MAX_SETTLE_DRIFT_M = 0.01
MAX_SETTLE_TILT_RAD = math.radians(5.0)


@dataclass(frozen=True)
class _Loaded:
    model: mujoco.MjModel
    data: mujoco.MjData


def _load(bundle: AssetBundle) -> _Loaded:
    model = mujoco.MjModel.from_xml_path(str(bundle.mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return _Loaded(model, data)


def check_physical(bundle: AssetBundle, request: AssetRequest) -> Report:
    """Measure mass, extents and friction on the compiled model."""
    report = Report(kind="asset-physical", subject=bundle.asset_id)
    loaded = _load(bundle)
    model, data = loaded.model, loaded.data
    sheet = request.datasheet

    lower, upper = _world_bounds(model, data)
    extents = upper - lower
    for axis, interval, label in (
        (0, sheet.bbox_x_m, "X extent"),
        (1, sheet.bbox_y_m, "Y extent"),
        (2, sheet.bbox_z_m, "Z extent"),
    ):
        finding = interval_finding(
            code="A-PHY-EXTENT",
            subject=f"{bundle.asset_id}/{'xyz'[axis]}",
            measured=float(extents[axis]),
            interval=interval,
            unit="m",
            repair_target=RepairTarget.ASSET,
            what=label,
        )
        if finding:
            report.findings.append(finding)

    total_mass = float(model.body_mass.sum())
    finding = interval_finding(
        code="A-PHY-MASS",
        subject=bundle.asset_id,
        measured=total_mass,
        interval=sheet.total_mass_kg,
        unit="kg",
        repair_target=RepairTarget.ASSET,
        what="total mass",
    )
    if finding:
        report.findings.append(finding)

    if sheet.sliding_friction is not None and model.ngeom:
        friction = float(np.median(model.geom_friction[:, 0]))
        finding = interval_finding(
            code="A-PHY-FRICTION",
            subject=bundle.asset_id,
            measured=friction,
            interval=sheet.sliding_friction,
            unit="",
            repair_target=RepairTarget.ASSET,
            what="median sliding friction",
        )
        if finding:
            report.findings.append(finding)

    report.findings.extend(_massless_movable_bodies(model, bundle.asset_id))
    report.findings.extend(_joint_travel(model, sheet.joint_ranges_rad, bundle.asset_id))
    return report


def check_functional(bundle: AssetBundle, request: AssetRequest) -> Report:
    """Drive the articulations, then drop the asset on a plane and see if it settles."""
    report = Report(kind="asset-functional", subject=bundle.asset_id)
    wanted = request.grounding.functional

    for name in wanted.movable_articulations:
        report.findings.append(_articulation_moves(bundle, name))

    if wanted.must_rest_stably:
        report.findings.append(_settles_on_plane(bundle))

    if wanted.reachable_sites:
        loaded = _load(bundle)
        present = {
            mujoco.mj_id2name(loaded.model, mujoco.mjtObj.mjOBJ_SITE, i)
            for i in range(loaded.model.nsite)
        }
        for site in wanted.reachable_sites:
            if site in present:
                continue
            report.findings.append(
                Finding(
                    code="A-FUN-SITE-MISSING",
                    severity=Severity.FAILURE,
                    subject=f"{bundle.asset_id}/{site}",
                    summary=(
                        f"the request requires an arm to reach site {site!r}, but the asset "
                        "does not define it. Reachability is checked once the asset is placed "
                        "in a workcell; the site has to exist first."
                    ),
                    repair_target=RepairTarget.ASSET,
                )
            )
    return report


class VisionVerdict(BaseModel):
    """What the vision critic returns, one entry per claim."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    supported: bool
    observation: str = Field(description="What is actually visible, in one sentence.")


class VisionVerdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[VisionVerdict]


def check_vision(
    bundle: AssetBundle,
    request: AssetRequest,
    *,
    client: LlmClient | None = None,
    blocking: bool = False,
    image_dir: Path | None = None,
) -> Report:
    """Render the asset and ask a model whether the request's visual claims hold."""
    report = Report(kind="asset-vision", subject=bundle.asset_id)
    claims = request.grounding.vision.claims
    if not claims:
        report.notes.append("no visual claims were declared, so nothing was checked")
        return report

    try:
        images = render_views(bundle, request.grounding.vision.views, output_dir=image_dir)
    except RuntimeError as error:
        report.notes.append(f"rendering unavailable, vision check skipped: {error}")
        return report

    client = client or LlmClient()
    try:
        answer = client.structured(
            purpose="asset-vision-grounding",
            system=(
                "You inspect renders of a 3D laboratory object and report only what the "
                "images show. Judge each claim independently. If a claim concerns something "
                "the given views cannot show, mark it unsupported and say so."
            ),
            user=(
                f"Object: {request.prompt}\n\n"
                f"Views supplied, in order: {', '.join(request.grounding.vision.views)}\n\n"
                "Claims to judge:\n"
                + "\n".join(f"{i + 1}. {claim}" for i, claim in enumerate(claims))
            ),
            schema=VisionVerdicts,
            images=[blob for _, blob in images],
        )
    except LlmUnavailable as error:
        report.notes.append(f"vision check skipped: {error}")
        return report

    severity = Severity.FAILURE if blocking else Severity.WARNING
    seen = {v.claim for v in answer.verdicts}
    for verdict in answer.verdicts:
        report.findings.append(
            Finding(
                code="A-VIS-CLAIM",
                severity=Severity.INFO if verdict.supported else severity,
                subject=bundle.asset_id,
                summary=(
                    f"{'supported' if verdict.supported else 'not supported'}: "
                    f"{verdict.claim} — {verdict.observation}"
                ),
                repair_target=RepairTarget.NONE if verdict.supported else RepairTarget.ASSET,
            )
        )
    for missing in [c for c in claims if c not in seen]:
        report.notes.append(f"the critic did not return a verdict for: {missing}")
    return report


def ground_asset(
    bundle: AssetBundle,
    request: AssetRequest,
    *,
    client: LlmClient | None = None,
    vision_blocks: bool = False,
    image_dir: Path | None = None,
) -> Report:
    """Run all three checks and merge them into one report."""
    report = Report(kind="asset-grounding", subject=bundle.asset_id)
    report.extend(check_physical(bundle, request))
    report.extend(check_functional(bundle, request))
    report.extend(
        check_vision(bundle, request, client=client, blocking=vision_blocks, image_dir=image_dir)
    )
    return report


def render_views(
    bundle: AssetBundle,
    views: list[str],
    *,
    output_dir: Path | None = None,
    width: int = 640,
    height: int = 480,
) -> list[tuple[str, bytes]]:
    """Render the asset from named directions. Requires a working offscreen GL backend."""
    from PIL import Image  # noqa: PLC0415

    loaded = _load(bundle)
    model, data = loaded.model, loaded.data
    lower, upper = _world_bounds(model, data)
    centre = (lower + upper) / 2.0
    radius = max(float(np.linalg.norm(upper - lower)), 0.05)

    angles = {
        "front": (90.0, -10.0),
        "side": (0.0, -10.0),
        "top": (90.0, -89.0),
        "iso": (45.0, -25.0),
    }
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as error:  # noqa: BLE001 - any GL failure means no renders
        raise RuntimeError(f"MuJoCo offscreen rendering failed ({error}); set MUJOCO_GL=egl or osmesa")

    camera = mujoco.MjvCamera()
    camera.lookat[:] = centre
    camera.distance = radius * 2.2

    out: list[tuple[str, bytes]] = []
    with renderer:
        for view in views:
            azimuth, elevation = angles.get(view, angles["iso"])
            camera.azimuth, camera.elevation = azimuth, elevation
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            image = Image.fromarray(frame)
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                image.save(output_dir / f"{view}.png")
            import io  # noqa: PLC0415

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            out.append((view, buffer.getvalue()))
    return out


_AABB_SIGNS = np.array(
    [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float
)


def _world_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned world bounds over all geoms in the current configuration.

    Uses each geom's own local AABB rather than its bounding sphere; a bounding sphere
    overstates a flat box by up to its diagonal, which is enough to fail a datasheet check
    on an object that is actually in tolerance.
    """
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for geom in range(model.ngeom):
        centre_local = model.geom_aabb[geom, :3]
        half = model.geom_aabb[geom, 3:]
        if not np.any(half > 0.0):
            continue
        corners = centre_local + _AABB_SIGNS * half
        rotation = data.geom_xmat[geom].reshape(3, 3)
        world = corners @ rotation.T + data.geom_xpos[geom]
        lower = np.minimum(lower, world.min(axis=0))
        upper = np.maximum(upper, world.max(axis=0))
    if not np.isfinite(lower).all():
        return np.zeros(3), np.zeros(3)
    return lower, upper


def _massless_movable_bodies(model: mujoco.MjModel, asset_id: str) -> list[Finding]:
    """A jointed body with no mass makes the solver produce nonsense forces."""
    findings: list[Finding] = []
    for body in range(1, model.nbody):
        if model.body_jntnum[body] == 0:
            continue
        if model.body_mass[body] > 1e-9:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body#{body}"
        findings.append(
            Finding(
                code="A-PHY-MASSLESS",
                severity=Severity.FAILURE,
                subject=name,
                summary=(
                    f"{name} carries a joint but has zero mass, so contact forces on it are "
                    "unphysical and any episode involving it cannot be judged."
                ),
                repair_target=RepairTarget.ASSET,
                metrics={"mass_kg": float(model.body_mass[body])},
            )
        )
    return findings


def _joint_travel(
    model: mujoco.MjModel, wanted: dict[str, tuple[float, float]], asset_id: str
) -> list[Finding]:
    findings: list[Finding] = []
    for name, (low, high) in wanted.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            findings.append(
                Finding(
                    code="A-PHY-JOINT-MISSING",
                    severity=Severity.FAILURE,
                    subject=f"{asset_id}/{name}",
                    summary=f"the datasheet specifies travel for joint {name!r}, which the model does not define.",
                    repair_target=RepairTarget.ASSET,
                )
            )
            continue
        actual_low, actual_high = (float(v) for v in model.jnt_range[joint_id])
        if actual_low <= low + 1e-6 and actual_high >= high - 1e-6:
            continue
        findings.append(
            Finding(
                code="A-PHY-JOINT-RANGE",
                severity=Severity.FAILURE,
                subject=f"{asset_id}/{name}",
                summary=(
                    f"joint {name!r} travels {actual_low:.4g} to {actual_high:.4g} rad, which "
                    f"does not cover the required {low:.4g} to {high:.4g} rad."
                ),
                repair_target=RepairTarget.ASSET,
                metrics={"actual_lower": actual_low, "actual_upper": actual_high},
                thresholds={"required_lower": low, "required_upper": high},
            )
        )
    return findings


def _articulation_moves(bundle: AssetBundle, joint_name: str) -> Finding:
    """Drive a joint with a torque and confirm the configuration actually changes."""
    loaded = _load(bundle)
    model, data = loaded.model, loaded.data
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return Finding(
            code="A-FUN-JOINT-MISSING",
            severity=Severity.FAILURE,
            subject=f"{bundle.asset_id}/{joint_name}",
            summary=f"the request requires articulation {joint_name!r}, which the model does not define.",
            repair_target=RepairTarget.ASSET,
        )

    address = int(model.jnt_qposadr[joint_id])
    dof = int(model.jnt_dofadr[joint_id])
    start = float(data.qpos[address])
    torque = _drive_torque(model, joint_id)
    # Drive towards whichever side of the range has room. A closed lid sits on one of its
    # own stops, and pushing further into it measures the stop rather than the hinge.
    direction = _open_direction(model, joint_id, start)

    travel = _drive(model, data, dof, address, torque * direction, start)
    metrics = {"travel_rad": travel, "drive_torque_n_m": torque}

    overshoot = _stop_holds(model, joint_id, bundle, torque)
    if overshoot > LIMIT_OVERSHOOT_TOLERANCE_RAD:
        return Finding(
            code="A-FUN-ARTICULATION",
            severity=Severity.FAILURE,
            subject=f"{bundle.asset_id}/{joint_name}",
            summary=(
                f"{joint_name} settles {overshoot:.4g} rad past its own limit while a "
                f"{torque:.3g} N·m drive holds it there, so the stop does not hold. Whatever "
                "the joint is supposed to bear against will be pushed through in simulation."
            ),
            repair_target=RepairTarget.ASSET,
            metrics={**metrics, "limit_overshoot_rad": overshoot},
            thresholds={"tolerated_overshoot_rad": LIMIT_OVERSHOOT_TOLERANCE_RAD},
        )
    if travel >= MIN_ARTICULATION_TRAVEL_RAD:
        return Finding(
            code="A-FUN-ARTICULATION",
            severity=Severity.INFO,
            subject=f"{bundle.asset_id}/{joint_name}",
            summary=f"{joint_name} moved {travel:.4g} rad under a {torque:.3g} N·m drive.",
            metrics=metrics,
        )
    return Finding(
        code="A-FUN-ARTICULATION",
        severity=Severity.FAILURE,
        subject=f"{bundle.asset_id}/{joint_name}",
        summary=(
            f"{joint_name} moved only {travel:.4g} rad under a {torque:.3g} N·m drive, so it "
            "is fused in practice: check its range, damping, and whether neighbouring "
            "geometry blocks it."
        ),
        repair_target=RepairTarget.ASSET,
        metrics=metrics,
        thresholds={"minimum_travel_rad": MIN_ARTICULATION_TRAVEL_RAD},
    )


def _drive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dof: int,
    address: int,
    torque: float,
    start: float,
) -> float:
    """Push on the joint until it has clearly moved, and report how far it got.

    Stopping as soon as the threshold is passed is the point. Whether a joint is fused is
    answered by the first fraction of a radian; running the full second afterwards only
    accelerates a light part into its far stop and turns a question about the hinge into a
    question about the integrator.
    """
    steps = int(DRIVE_SECONDS / model.opt.timestep)
    travel = 0.0
    for _ in range(steps):
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof] = torque
        mujoco.mj_step(model, data)
        travel = abs(float(data.qpos[address]) - start)
        if travel >= MIN_ARTICULATION_TRAVEL_RAD:
            break
    return travel


def _stop_holds(
    model: mujoco.MjModel, joint_id: int, bundle: AssetBundle, torque: float
) -> float:
    """How far past its limit the joint sits once it has stopped moving.

    A separate run from the mobility probe, and a static measurement rather than a peak
    one: a limit is a spring, so what matters is where it comes to rest under load, not how
    far a light part's momentum carries it on first contact. Measuring the peak instead
    reports the timestep.
    """
    if not bool(model.jnt_limited[joint_id]):
        return 0.0
    loaded = _load(bundle)
    model, data = loaded.model, loaded.data
    address = int(model.jnt_qposadr[joint_id])
    dof = int(model.jnt_dofadr[joint_id])
    low, high = (float(v) for v in model.jnt_range[joint_id])
    # Towards the nearer stop, since that is the one reachable within the probe's budget.
    into = -1.0 if (float(data.qpos[address]) - low) < (high - float(data.qpos[address])) else 1.0

    for _ in range(int((DRIVE_SECONDS + SETTLE_SECONDS) / model.opt.timestep)):
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof] = torque * into
        mujoco.mj_step(model, data)
    return _limit_overshoot(model, joint_id, float(data.qpos[address]))


def _drive_torque(model: mujoco.MjModel, joint_id: int) -> float:
    """A drive scaled to what the joint already carries, not to a number picked in advance.

    Both things it carries count: its own weight, and whatever friction it declares. A
    hinge with a detent in it is meant to resist gravity — that is the whole point of a
    snap-cap — so testing it with gravity's torque alone would report every well-designed
    catch as a seized joint.
    """
    body_id = int(model.jnt_bodyid[joint_id])
    mass = float(model.body_subtreemass[body_id])
    anchor = np.asarray(model.jnt_pos[joint_id], dtype=float)
    lever = float(np.linalg.norm(np.asarray(model.body_ipos[body_id], dtype=float) - anchor))
    gravity = abs(float(model.opt.gravity[2])) or 9.81
    dof = int(model.jnt_dofadr[joint_id])
    # Friction is a threshold to clear rather than a load to carry, so it gets a margin
    # instead of the multiple. Multiplying it too would drive a snap-cap at hundreds of
    # times the torque anything in the cell can apply to it.
    friction = float(model.dof_frictionloss[dof]) * FRICTION_MARGIN
    weight = DRIVE_GRAVITY_MULTIPLE * mass * gravity * lever
    return max(weight + friction, MIN_DRIVE_TORQUE_N_M)


def _open_direction(model: mujoco.MjModel, joint_id: int, start: float) -> float:
    """+1 or -1, whichever way the joint has further to travel from where it sits."""
    if not bool(model.jnt_limited[joint_id]):
        return 1.0
    low, high = (float(v) for v in model.jnt_range[joint_id])
    return 1.0 if (high - start) >= (start - low) else -1.0


def _limit_overshoot(model: mujoco.MjModel, joint_id: int, value: float) -> float:
    if not bool(model.jnt_limited[joint_id]):
        return 0.0
    low, high = (float(v) for v in model.jnt_range[joint_id])
    return max(0.0, low - value, value - high)


def _settles_on_plane(bundle: AssetBundle) -> Finding:
    """Drop the asset on a ground plane and check it neither drifts nor tips."""
    xml = bundle.mjcf_path.read_text()
    injected = xml.replace(
        "<worldbody>",
        '<worldbody>\n    <geom name="__ground" type="plane" size="2 2 0.05" pos="0 0 0"/>',
        1,
    )
    if "__ground" not in injected:
        return Finding(
            code="A-FUN-STABILITY",
            severity=Severity.WARNING,
            subject=bundle.asset_id,
            summary="could not inject a ground plane into the asset MJCF, so stability was not checked.",
        )

    model = mujoco.MjModel.from_xml_string(injected, _mesh_assets(bundle))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = data.xpos.copy()
    start_mat = data.xmat.copy()

    for _ in range(int(SETTLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qacc).all():
            return Finding(
                code="A-FUN-STABILITY",
                severity=Severity.FAILURE,
                subject=bundle.asset_id,
                summary=(
                    f"the simulation diverged {data.time:.3g} s after the asset was placed on a "
                    "plane. Usually a massless or inertia-free body, or interpenetrating geoms."
                ),
                repair_target=RepairTarget.ASSET,
            )

    drift = float(np.linalg.norm(data.xpos - start, axis=1).max())
    tilt = _max_tilt(start_mat, data.xmat)
    if drift <= MAX_SETTLE_DRIFT_M and tilt <= MAX_SETTLE_TILT_RAD:
        return Finding(
            code="A-FUN-STABILITY",
            severity=Severity.INFO,
            subject=bundle.asset_id,
            summary=f"settled on a plane: {drift * 1000:.2f} mm drift, {math.degrees(tilt):.2f}° tilt.",
            metrics={"drift_m": drift, "tilt_rad": tilt},
        )
    return Finding(
        code="A-FUN-STABILITY",
        severity=Severity.FAILURE,
        subject=bundle.asset_id,
        summary=(
            f"the asset does not rest stably: it drifted {drift * 1000:.2f} mm and tipped "
            f"{math.degrees(tilt):.2f}° over {SETTLE_SECONDS:.1f} s on a flat plane."
        ),
        repair_target=RepairTarget.ASSET,
        metrics={"drift_m": drift, "tilt_rad": tilt},
        thresholds={"maximum_drift_m": MAX_SETTLE_DRIFT_M, "maximum_tilt_rad": MAX_SETTLE_TILT_RAD},
    )


def _mesh_assets(bundle: AssetBundle) -> dict[str, bytes]:
    """MJCF loaded from a string cannot resolve relative mesh paths, so supply them directly."""
    assets: dict[str, bytes] = {}
    if not bundle.mesh_dir.is_dir():
        return assets
    for path in bundle.mesh_dir.rglob("*"):
        if path.is_file():
            assets[f"meshes/{path.relative_to(bundle.mesh_dir).as_posix()}"] = path.read_bytes()
    return assets


def _max_tilt(start_mat: np.ndarray, end_mat: np.ndarray) -> float:
    worst = 0.0
    for before, after in zip(start_mat, end_mat, strict=True):
        relative = after.reshape(3, 3) @ before.reshape(3, 3).T
        cosine = (np.trace(relative) - 1.0) / 2.0
        worst = max(worst, math.acos(min(1.0, max(-1.0, cosine))))
    return worst
