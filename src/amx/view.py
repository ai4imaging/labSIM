"""Open a generated asset in MuJoCo and move its joints.

Everything else in this package looks at an asset through a measurement: a diameter, a
volume, a probe that either fits or does not. None of that answers the question you ask
first about an articulated object, which is whether the lid actually opens and the rotor
actually spins. This module is that answer, and it is deliberately the only place in the
project that opens a window.

Two things make it more than a one-line call to `mujoco.viewer`.

The asset the benchmark scores is a `model.py`, not an MJCF. `resolve` walks the same path
the judge does — find the newest `model.py` under the case directory, compile it through
`build_grounded_asset`, reuse the cache — so what you look at is what was scored, not a
separate export that might differ.

And the MJCF Articraft exports gives both geometry classes `density="0"`, on the
assumption that explicit inertials arrive from a `PhysicalSpec`. In this pipeline they do
not, so every body behind a joint has zero mass and MuJoCo refuses the model outright with
`mjMINVAL`. That is a real defect in the asset and the judge is right to fail it, but
refusing to *show* it is unhelpful: the kinematics are intact and looking at them is how
you find out what the agent built. `viewable` therefore writes a patched copy alongside
the original and says so. The patch is never fed back into scoring.
"""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

VIEW_DENSITY = 1000.0
"""kg/m3 for collision geometry when the asset states no mass at all.

Water, near enough to moulded plastic for a model that is only being looked at. It is not
a guess at the real part: an asset that cared about its mass would have said so.
"""

NOMINAL_MASS_KG = 0.05
NOMINAL_INERTIA = 1e-5
"""Last-resort inertia for a moving body with no collision geometry to derive one from."""


class ViewError(RuntimeError):
    """The asset could not be resolved or could not be made loadable."""


@dataclass(frozen=True)
class Viewable:
    """An MJCF that MuJoCo will load, and what had to be done to it."""

    path: Path
    source: Path
    patched: bool
    note: str = ""


# --------------------------------------------------------------------------- #
# finding the asset
# --------------------------------------------------------------------------- #


def resolve(target: str | Path, *, run_dir: Path | None = None) -> Path:
    """Return the MJCF for `target`, compiling the case's `model.py` if needed.

    `target` may be an MJCF, a `model.py`, a case directory, or a bare case id such as
    `CEN-001` to be looked up under `run_dir`.
    """
    path = Path(target)
    if path.is_file():
        if path.suffix == ".xml":
            return path
        if path.name == "model.py":
            return _compile(path, asset_id=_asset_id_for(path))
        raise ViewError(f"{path} is neither an MJCF nor a model.py")

    case_dir = path if path.is_dir() else _case_dir(str(target), run_dir)
    for direct in (
        case_dir / "asset" / "mjcf" / "asset.xml",
        case_dir / "visualization" / "mjcf" / "asset.xml",
        case_dir / "mjcf" / "asset.xml",
    ):
        if direct.is_file():
            return direct
    models = _find_models(case_dir)
    if not models:
        raise ViewError(
            f"no model.py under {case_dir} — the agent never produced an asset for this case"
        )
    failures: list[str] = []
    for model in models:
        try:
            return _compile(model, asset_id=_asset_id_for(case_dir))
        except ViewError as error:
            failures.append(f"{model.relative_to(case_dir)}: {error}")
    preview = "\n".join(failures[:8])
    raise ViewError(
        f"no visualizable model was produced for {_asset_id_for(case_dir)}; "
        f"tried {len(models)} candidate revision(s):\n{preview}"
    )


def case_dir_for(target: str | Path, *, run_dir: Path | None = None) -> Path | None:
    """The benchmark case directory `target` names, if it names one.

    A bare MJCF or `model.py` has no case behind it and therefore no operation contract;
    the viewer treats that as "nothing is claimed to move" rather than as an error.
    """
    path = Path(target)
    if path.is_file():
        for parent in path.parents:
            if (parent / "grounding-spec.json").is_file():
                return parent
        return None
    if path.is_dir():
        return path
    try:
        return _case_dir(str(target), run_dir)
    except ViewError:
        return None


def _case_dir(case_id: str, run_dir: Path | None) -> Path:
    from amx.paths import RUNS_DIR  # noqa: PLC0415

    if run_dir is not None:
        candidate = Path(run_dir) / case_id
        if candidate.is_dir():
            return candidate
        raise ViewError(f"{case_id} is not in {run_dir}")

    sweeps = sorted(
        (RUNS_DIR / "bench").glob(f"*/{case_id}"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not sweeps:
        raise ViewError(
            f"no run of {case_id} found under {RUNS_DIR / 'bench'}; pass --run-dir or a path"
        )
    return sweeps[0]


def _find_models(case_dir: Path) -> list[Path]:
    """Candidate revisions, preferring submitted and explicit last-good models."""
    root = case_dir / "asset" if (case_dir / "asset").is_dir() else case_dir
    direct = root / "model.py"
    preferred = [
        direct,
        case_dir / "visualization" / "model.py",
        *sorted(root.rglob("model.last-good.py"), key=lambda item: item.stat().st_mtime, reverse=True),
    ]
    revisions = sorted(
        case_dir.rglob("model.py"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    found: list[Path] = []
    seen: set[Path] = set()
    for candidate in [*preferred, *revisions]:
        resolved = candidate.resolve()
        if candidate.is_file() and resolved not in seen:
            found.append(candidate)
            seen.add(resolved)
    return found


def _asset_id_for(path: Path) -> str:
    for part in (path if path.is_dir() else path.parent).parts[::-1]:
        if part[:3].isalpha() and "-" in part:
            return part
    return "asset"


def _compile(model_path: Path, *, asset_id: str) -> Path:
    from amx.grounding.build import build_grounded_asset  # noqa: PLC0415

    try:
        return build_grounded_asset(model_path, asset_id=asset_id).mjcf_path
    except Exception:  # noqa: BLE001 — a rejected draft is exactly what you want to see
        pass

    # The judge compiles with the quality checks on, and a draft that overlaps itself or
    # has a massless body is rightly rejected there. Refusing to show it as well would
    # leave nothing to look at in the one situation where looking is the whole point.
    try:
        return build_grounded_asset(model_path, asset_id=asset_id, run_checks=False).mjcf_path
    except Exception as error:  # noqa: BLE001 — a build failure is the answer, not a crash
        raise ViewError(f"{asset_id} does not compile: {error}") from error


# --------------------------------------------------------------------------- #
# making it loadable
# --------------------------------------------------------------------------- #


def viewable(mjcf_path: Path) -> Viewable:
    """Load the MJCF, giving massless moving bodies a nominal mass if that is what it takes."""
    import mujoco  # noqa: PLC0415

    mjcf_path = Path(mjcf_path)
    try:
        mujoco.MjModel.from_xml_path(str(mjcf_path))
    except ValueError as error:
        if "mjMINVAL" not in str(error):
            raise ViewError(f"{mjcf_path} will not load: {error}") from error
    else:
        return Viewable(path=mjcf_path, source=mjcf_path, patched=False)

    patched = mjcf_path.with_name(f"{mjcf_path.stem}-viewable.xml")
    note = _write_massful_copy(mjcf_path, patched)
    try:
        mujoco.MjModel.from_xml_path(str(patched))
    except ValueError as error:
        raise ViewError(f"{mjcf_path} will not load even with masses added: {error}") from error
    return Viewable(path=patched, source=mjcf_path, patched=True, note=note)


def _write_massful_copy(source: Path, destination: Path) -> str:
    """Give the collision class a density, and any jointed body still without one a mass."""
    tree = ET.parse(source)
    root = tree.getroot()

    changed: list[str] = []
    for geom in root.findall("./default/default[@class='contact']/geom"):
        if geom.attrib.get("density") in {"0", "0.0", None}:
            geom.attrib["density"] = f"{VIEW_DENSITY:g}"
            changed.append(f"collision density set to {VIEW_DENSITY:g} kg/m3")

    bodies = _jointed_bodies_without_mass(root)
    if bodies:
        changed.append(f"nominal inertia on {', '.join(bodies)}")

    destination.write_text(ET.tostring(root, encoding="unicode"))
    return "; ".join(changed) or "no change was needed"


def _jointed_bodies_without_mass(root: ET.Element) -> list[str]:
    """Add an inertial to every moving body that has no collision geometry to weigh."""
    named: list[str] = []
    for body in root.iter("body"):
        moves = any(child.tag in {"joint", "freejoint"} for child in body)
        if not moves or body.find("inertial") is not None:
            continue
        if any(geom.attrib.get("class") == "contact" for geom in body.findall("geom")):
            continue
        body.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": f"{NOMINAL_MASS_KG:g}",
                    "diaginertia": " ".join([f"{NOMINAL_INERTIA:g}"] * 3),
                },
            ),
        )
        named.append(str(body.attrib.get("name") or "?"))
    return named


# --------------------------------------------------------------------------- #
# moving it
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Articulation:
    """One joint the viewer knows how to drive."""

    name: str
    address: int
    lower: float
    upper: float
    spins: bool
    """True for a joint the specification says turns continuously — a rotor or a stirrer.

    Not simply "unlimited hinge". Reading it off the model that way is how a lid whose
    author forgot its limits, and a removable part exported as an unbounded rotation,
    came out of the viewer spinning on the spot: the asset is wrong, and the viewer was
    inventing a motion to illustrate it with.
    """

    def at(self, phase: float) -> float:
        if self.spins:
            return phase * 2.0 * math.pi
        centre = 0.5 * (self.lower + self.upper)
        half = 0.5 * (self.upper - self.lower)
        return centre - half * math.cos(phase * 2.0 * math.pi)


@dataclass(frozen=True)
class AnimationPlan:
    """Which joints `--animate` will drive, and why those."""

    driven: list[Articulation]
    note: str


def articulations(model) -> list[Articulation]:  # noqa: ANN001 — MjModel, imported lazily
    """Every scalar joint in the model, with the travel it declares.

    `spins` is left False here. Nothing in an MJCF distinguishes a rotor from a hinge
    whose limits were never written down, so that call belongs to the specification and
    is made in `animation_plan`.
    """
    import mujoco  # noqa: PLC0415

    found: list[Articulation] = []
    for index in range(model.njnt):
        kind = int(model.jnt_type[index])
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
        if kind not in {hinge, slide}:
            continue
        limited = bool(model.jnt_limited[index])
        lower, upper = (float(v) for v in model.jnt_range[index])
        found.append(
            Articulation(
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or f"joint{index}",
                address=int(model.jnt_qposadr[index]),
                lower=lower if limited else 0.0,
                upper=upper if limited else 0.0,
                spins=False,
            )
        )
    return found


def grounding_spec_for(case_dir: Path | None):
    """The case's `grounding-spec.json`, or None when the target is not a scored case."""
    if case_dir is None:
        return None
    path = Path(case_dir) / "grounding-spec.json"
    if not path.is_file():
        return None
    from amx.grounding.spec import GroundingSpec  # noqa: PLC0415

    try:
        return GroundingSpec.read(path)
    except (OSError, ValueError):
        return None


def animation_plan(
    model,  # noqa: ANN001 — MjModel, imported lazily
    joints: list[Articulation],
    *,
    spec=None,
    requested: list[str] | None = None,
) -> AnimationPlan:
    """Decide what to move, from what the task says moves — not from what can move.

    Driving every joint in the model is how a still object came to be shown in motion.
    Some of those joints are wrong (a removable tray on a hinge), and some are right but
    are not what the object does when it is sitting on a bench. So the default is the set
    of joints belonging to a mechanism the specification actually states, and everything
    else holds its pose. `--joint` is the way to look at something outside that set.
    """
    by_name = {joint.name: joint for joint in joints}
    if requested:
        wanted = {name.lower() for name in requested}
        chosen = [joint for joint in joints if joint.name.lower() in wanted]
        return AnimationPlan(chosen, "driving the joint(s) you named")

    if spec is None or not getattr(spec, "operations", None):
        return AnimationPlan(
            [],
            "no physical-operation contract was found for this asset, so nothing is "
            "animated; name a joint with --joint to drive one anyway",
        )

    from amx.grounding.checks import operable_joints  # noqa: PLC0415

    matched = operable_joints(model, spec)
    driven: list[Articulation] = []
    for name, target in matched.items():
        joint = by_name.get(name)
        if joint is None:
            continue
        driven.append(replace(joint, spins=bool(target.continuous)))

    if not driven:
        return AnimationPlan(
            [],
            "none of this model's joints could be matched to a stated operation, so "
            "nothing is animated; name a joint with --joint to drive one anyway",
        )
    unstated = [joint.name for joint in joints if joint.name not in matched]
    note = f"driving {len(driven)} joint(s) named by the operation contract"
    if unstated:
        note += f"; holding {len(unstated)} joint(s) the contract does not mention"
    return AnimationPlan(driven, note)


def launch(
    mjcf_path: Path,
    *,
    animate: bool = False,
    period_s: float = 6.0,
    driven: list[Articulation] | None = None,
) -> None:
    """Open the viewer, driving `driven` through its travel on a loop if `animate`.

    Animation writes `qpos` and calls `mj_forward` rather than stepping the simulation.
    The question here is what the mechanism does, and an unactuated lid under gravity just
    falls shut — which tells you about gravity, not about the asset.
    """
    import mujoco  # noqa: PLC0415
    import mujoco.viewer  # noqa: PLC0415

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    if not animate or not driven:
        mujoco.viewer.launch(model, data)
        return

    started = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            phase = ((time.monotonic() - started) / period_s) % 1.0
            for joint in driven:
                data.qpos[joint.address] = joint.at(phase)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)
