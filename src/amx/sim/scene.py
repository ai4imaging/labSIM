"""Declaring a workcell, and composing it into one MJCF scene.

A `Workcell` says what is in the cell and where: an arm, a bench, the assets from part 1,
and the co-designed parts from `amx.codesign`. `build_scene` turns that into a directory
containing `scene.xml` and a flat `meshes/`, which is loadable by MuJoCo with no further
setup and is also exactly what `sim_judge` expects to find as a case's closed scene.

Every identifier in the composed scene lives under one of the namespaces in `amx.naming`.
That is what lets the judge policy classify a contact and the repair router decide what to
change, without either of them inspecting geometry.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
from pydantic import BaseModel, ConfigDict, Field, field_validator

from amx import naming
from amx.geometry import Pose
from amx.paths import ROBOTS_DIR
from amx.sim import mjcf

DEFAULT_ROBOT = "ur5e_robotiq85"


class RobotRef(BaseModel):
    """Which vendored arm to use, and where its base is bolted down."""

    model_config = ConfigDict(extra="forbid")

    model: str = DEFAULT_ROBOT
    base: Pose = Field(default_factory=lambda: Pose(pos=(0.0, 0.0, 0.0)))
    home_qpos: list[float] | None = Field(
        default=None,
        description="Joint configuration the episode starts from. A safe elbow-up pose if unset.",
    )


class Bench(BaseModel):
    """The work surface. A fixed box; the arm's base sits on top of it by convention."""

    model_config = ConfigDict(extra="forbid")

    size_xy: tuple[float, float] = (1.2, 0.8)
    top_z: float = 0.0
    thickness: float = 0.04
    friction: float = 1.0

    def surface_z(self) -> float:
        return self.top_z


class Placement(BaseModel):
    """One asset from part 1, placed in the cell."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    source: Path = Field(description="The asset bundle directory, or its asset.xml directly.")
    pose: Pose = Field(default_factory=Pose)
    attachment: Literal["fixed", "free"] = Field(
        default="fixed",
        description="`fixed` welds it to the bench; `free` gives it a freejoint so it can be picked up.",
    )

    @property
    def namespace(self) -> str:
        return naming.asset_namespace(self.asset_id)

    def resolve_xml(self) -> Path:
        source = Path(self.source)
        if source.is_file():
            return source
        for candidate in (source / "mjcf" / "asset.xml", source / "asset.xml"):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no asset.xml for {self.asset_id!r} under {source}")


class FixturePlacement(BaseModel):
    """One co-designed part, either bolted to the bench or carried by the arm."""

    model_config = ConfigDict(extra="forbid")

    part_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    source: Path = Field(description="The part directory written by `amx.codesign.export`.")
    mount: Literal["bench", "tool"]
    pose: Pose = Field(
        default_factory=Pose,
        description="For `bench`, relative to the bench surface. For `tool`, relative to the flange's tool_mount site.",
    )

    @property
    def namespace(self) -> str:
        if self.mount == "tool":
            return naming.tool_namespace(self.part_id)
        return naming.fixture_namespace(self.part_id)

    def resolve_xml(self) -> Path:
        source = Path(self.source)
        if source.is_file():
            return source
        candidate = source / "part.xml"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"no part.xml for {self.part_id!r} under {source}")


class Workcell(BaseModel):
    """The complete declaration of a cell. Serialise it and a scene rebuilds byte-for-byte."""

    model_config = ConfigDict(extra="forbid")

    workcell_id: str = Field(default="cell", pattern=r"^[a-z0-9][a-z0-9_-]*$")
    robot: RobotRef = Field(default_factory=RobotRef)
    bench: Bench = Field(default_factory=Bench)
    assets: list[Placement] = Field(default_factory=list)
    fixtures: list[FixturePlacement] = Field(default_factory=list)
    timestep: float = 0.002

    @field_validator("assets")
    @classmethod
    def unique_assets(cls, value: list[Placement]) -> list[Placement]:
        _reject_duplicates([p.asset_id for p in value], "asset_id")
        return value

    @field_validator("fixtures")
    @classmethod
    def unique_fixtures(cls, value: list[FixturePlacement]) -> list[FixturePlacement]:
        _reject_duplicates([p.part_id for p in value], "part_id")
        return value

    def asset(self, asset_id: str) -> Placement:
        for placement in self.assets:
            if placement.asset_id == asset_id:
                return placement
        raise KeyError(f"no asset {asset_id!r} in workcell {self.workcell_id!r}")

    def fixture(self, part_id: str) -> FixturePlacement:
        for placement in self.fixtures:
            if placement.part_id == part_id:
                return placement
        raise KeyError(f"no fixture {part_id!r} in workcell {self.workcell_id!r}")


def _reject_duplicates(values: list[str], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label} {value!r}; namespaces must be unique in a cell")
        seen.add(value)


class RobotSpec(BaseModel):
    """The vendored robot's contract, read from its `spec.json`."""

    model_config = ConfigDict(extra="ignore")

    robot_id: str
    namespace: str = naming.ROBOT
    arm_joints: list[str]
    arm_actuators: list[str]
    gripper_actuators: list[dict[str, Any]]
    gripper_rest_qpos: dict[str, float] = Field(
        default_factory=dict,
        description="Where the gripper's driven joints park when open, just off their hard stop.",
    )
    tcp_site: str
    finger_reach_m: float = Field(
        default=0.0,
        description="How far the fingers extend past the tool centre. A plan reaching into "
        "a socket has to clear its rim by at least this much.",
    )
    pad_reach_m: float = Field(
        default=0.0,
        description="How far the gripping faces extend past the tool centre. A part must "
        "stand at least `finger_reach_m - pad_reach_m` proud of its socket's rim for the "
        "pads to touch it while the fingertips stay clear.",
    )
    tool_mount_site: str
    flange_body: str
    root_body: str
    bodies: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, model: str) -> "RobotSpec":
        directory = ROBOTS_DIR / model
        spec_path = directory / "spec.json"
        if not spec_path.is_file():
            raise FileNotFoundError(
                f"no vendored robot at {directory}. Run `python scripts/vendor_robots.py`."
            )
        return cls.model_validate(json.loads(spec_path.read_text()))

    def gripper_range(self) -> tuple[float, float]:
        """Control values for fully open and fully closed, shared by both finger actuators."""
        if not self.gripper_actuators:
            return (0.0, 0.0)
        first = self.gripper_actuators[0]
        return (float(first["open"]), float(first["closed"]))


@dataclass(frozen=True)
class BuiltScene:
    """A composed, on-disk scene plus the handles needed to drive it."""

    workcell: Workcell
    robot: RobotSpec
    scene_path: Path
    mesh_dir: Path
    asset_namespaces: dict[str, str]
    fixture_namespaces: dict[str, str]

    def load_model(self) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_path(str(self.scene_path))

    def bench_surface_z(self) -> float:
        return self.workcell.bench.surface_z()


def build_scene(workcell: Workcell, output_dir: Path) -> BuiltScene:
    """Compose the cell into `output_dir/scene.xml` and verify MuJoCo accepts it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)

    robot_spec = RobotSpec.load(workcell.robot.model)
    scene = _scene_shell(workcell)
    world = mjcf.section(scene, "worldbody")
    _add_bench(world, workcell.bench)

    robot_dir = ROBOTS_DIR / workcell.robot.model
    arm = mjcf.load(robot_dir / "arm.xml")
    mjcf.strip_globals(arm)
    mjcf.absorb_meshes(arm, robot_dir, mesh_dir)
    arm_root = mjcf.find_body(arm, robot_spec.root_body)
    _set_pose(arm_root, workcell.robot.base)
    world.append(arm_root)
    mjcf.merge_sections(scene, arm)

    asset_namespaces: dict[str, str] = {}
    for placement in workcell.assets:
        namespace = placement.namespace
        asset_namespaces[placement.asset_id] = namespace
        source_xml = placement.resolve_xml()
        tree = mjcf.load(source_xml)
        mjcf.strip_globals(tree)
        mjcf.apply_prefix(tree, namespace)
        mjcf.absorb_meshes(tree, source_xml.parent, mesh_dir)
        roots = mjcf.root_bodies(tree)
        if len(roots) != 1:
            raise mjcf.MjcfError(
                f"asset {placement.asset_id!r} has {len(roots)} root bodies; exactly one is required"
            )
        root = roots[0]
        _set_pose(root, _on_bench(placement.pose, workcell.bench))
        if placement.attachment == "free":
            root.insert(0, ET.Element("freejoint", {"name": f"{namespace}root"}))
        world.append(root)
        mjcf.merge_sections(scene, tree)

    fixture_namespaces: dict[str, str] = {}
    excluded: list[tuple[str, str]] = []
    for placement in workcell.fixtures:
        namespace = placement.namespace
        fixture_namespaces[placement.part_id] = namespace
        source_xml = placement.resolve_xml()
        tree = mjcf.load(source_xml)
        mjcf.strip_globals(tree)
        mjcf.apply_prefix(tree, namespace)
        mjcf.absorb_meshes(tree, source_xml.parent, mesh_dir)
        roots = mjcf.root_bodies(tree)
        if len(roots) != 1:
            raise mjcf.MjcfError(
                f"part {placement.part_id!r} has {len(roots)} root bodies; exactly one is required"
            )
        root = roots[0]
        if placement.mount == "tool":
            _set_pose(root, placement.pose)
            mjcf.find_body(scene, robot_spec.flange_body).append(root)
            excluded.extend(
                (body, robot_body)
                for body in _body_names(root)
                for robot_body in robot_spec.bodies
            )
        else:
            _set_pose(root, _on_bench(placement.pose, workcell.bench))
            world.append(root)
        mjcf.merge_sections(scene, tree)

    if excluded:
        _add_exclusions(scene, excluded)

    # Compile once before the keyframe exists: a malformed cell fails here with MuJoCo's
    # own message rather than halfway through an episode, and compiling is also the only
    # reliable way to learn the qpos layout, which free assets and gripper linkages both
    # contribute to.
    scene_path = mjcf.write(scene, output_dir / "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    _add_home_keyframe(scene, model, workcell, robot_spec)
    scene_path = mjcf.write(scene, output_dir / "scene.xml")
    mujoco.MjModel.from_xml_path(str(scene_path))

    return BuiltScene(
        workcell=workcell,
        robot=robot_spec,
        scene_path=scene_path,
        mesh_dir=mesh_dir,
        asset_namespaces=asset_namespaces,
        fixture_namespaces=fixture_namespaces,
    )


def _scene_shell(workcell: Workcell) -> ET.Element:
    scene = ET.Element("mujoco", {"model": workcell.workcell_id})
    ET.SubElement(
        scene,
        "compiler",
        {"angle": "radian", "autolimits": "true", "meshdir": "meshes", "balanceinertia": "true"},
    )
    ET.SubElement(
        scene,
        "option",
        {
            "timestep": f"{workcell.timestep:g}",
            "integrator": "implicitfast",
            "cone": "elliptic",
            "impratio": "10",
            "gravity": "0 0 -9.81",
        },
    )
    # Mesh-heavy scenes with several assets overflow the default contact and constraint
    # arenas, and MuJoCo's failure mode there is a hard error mid-episode.
    ET.SubElement(scene, "size", {"memory": "64M"})
    defaults = ET.SubElement(scene, "default")
    bench_class = ET.SubElement(defaults, "default", {"class": f"{naming.BENCH}surface"})
    ET.SubElement(
        bench_class,
        "geom",
        {
            "type": "box",
            "condim": "3",
            "friction": f"{workcell.bench.friction:g} 0.005 0.0001",
            "rgba": "0.82 0.80 0.76 1",
        },
    )
    return scene


def _add_bench(world: ET.Element, bench: Bench) -> None:
    ET.SubElement(
        world,
        "light",
        {"name": f"{naming.BENCH}light", "pos": "0 0 2.5", "dir": "0 0 -1", "directional": "true"},
    )
    ET.SubElement(
        world,
        "geom",
        {
            "name": f"{naming.BENCH}floor",
            "type": "plane",
            "size": "4 4 0.05",
            "pos": f"0 0 {bench.surface_z() - bench.thickness - 0.7:g}",
            "rgba": "0.35 0.36 0.38 1",
        },
    )
    top = ET.SubElement(
        world,
        "body",
        {"name": f"{naming.BENCH}table", "pos": f"0 0 {bench.surface_z() - bench.thickness / 2:g}"},
    )
    ET.SubElement(
        top,
        "geom",
        {
            "name": f"{naming.BENCH}top",
            "class": f"{naming.BENCH}surface",
            "size": f"{bench.size_xy[0] / 2:g} {bench.size_xy[1] / 2:g} {bench.thickness / 2:g}",
        },
    )
    ET.SubElement(
        top,
        "site",
        {
            "name": f"{naming.BENCH}origin",
            "pos": f"0 0 {bench.thickness / 2:g}",
            "size": "0.005",
            "group": "4",
        },
    )


def _on_bench(pose: Pose, bench: Bench) -> Pose:
    """Interpret a placement's Z as height above the bench surface."""
    return pose.translated((0.0, 0.0, bench.surface_z()))


def _set_pose(body: ET.Element, pose: Pose) -> None:
    body.attrib.pop("quat", None)
    body.attrib.pop("euler", None)
    for key, value in pose.mjcf().items():
        body.set(key, value)


def _body_names(root: ET.Element) -> list[str]:
    return [body.get("name", "") for body in root.iter("body") if body.get("name")]


def _add_exclusions(scene: ET.Element, pairs: list[tuple[str, str]]) -> None:
    """Suppress contact between arm-mounted tooling and the arm itself.

    A tool adapter is welded to the flange, so any contact MuJoCo reports between the two
    is an artefact of them overlapping by design, and it would otherwise show up in the
    judge's contact stream as an unexplained collision.
    """
    contact = mjcf.section(scene, "contact")
    for index, (first, second) in enumerate(pairs):
        ET.SubElement(
            contact,
            "exclude",
            {"name": f"tool_self_{index}", "body1": first, "body2": second},
        )


def _add_home_keyframe(
    scene: ET.Element, model: mujoco.MjModel, workcell: Workcell, robot: RobotSpec
) -> None:
    """Record the arm's start configuration as a named keyframe.

    Storing it in the scene rather than applying it at run time means anyone opening
    `scene.xml` sees the same starting pose the episode used.
    """
    home = workcell.robot.home_qpos or default_home_qpos(robot)
    if len(home) != len(robot.arm_joints):
        raise ValueError(
            f"home_qpos has {len(home)} entries but {robot.robot_id} has "
            f"{len(robot.arm_joints)} arm joints"
        )
    data = mujoco.MjData(model)
    placements = dict(zip(robot.arm_joints, home, strict=True))
    # Start the gripper parked where its `open` command holds it, so the first step does not
    # begin with the linkage being dragged off a hard stop.
    placements.update(robot.gripper_rest_qpos)
    for name, value in placements.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        data.qpos[model.jnt_qposadr[joint_id]] = value

    keyframe = mjcf.section(scene, "keyframe")
    for existing in list(keyframe):
        keyframe.remove(existing)
    ET.SubElement(
        keyframe,
        "key",
        {"name": "home", "qpos": " ".join(f"{v:g}" for v in data.qpos)},
    )


def default_home_qpos(robot: RobotSpec) -> list[float]:
    """An elbow-up pose with the tool pointing down, well clear of the bench.

    Starting from a singular configuration — all joints at zero, arm straight out — makes
    the first IK solve of every episode ill-conditioned, so the default is a real pose.
    """
    if len(robot.arm_joints) == 6:
        return [0.0, -1.2, 1.6, -1.95, -1.57, 0.0]
    return [0.0] * len(robot.arm_joints)
