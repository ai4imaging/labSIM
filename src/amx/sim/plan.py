"""The operation plan: four action primitives and nothing else.

Every wetlab motion this pipeline has to express — pick a tube out of a rack, seat it in a
socket, press a button, swing a lid shut, start a centrifuge — is a sequence of these four:

    Move     drive a site to a pose
    Grip     open or close the gripper to a fraction of its travel
    Hold     command nothing and let the physics settle
    Actuate  drive one named actuator to a value

There are deliberately no device-specific actions. A lid closes because a `Move` pushes it
or an `Actuate` drives its hinge; the executor does not need to know that the thing being
moved is a lid. That is what keeps the executor small enough to trust, and it means adding
a new instrument to the cell requires no new code here at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from amx.geometry import PoseTarget

STEP_ID_PATTERN = r"^step\.[a-z0-9_.-]+$"


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(
        default="",
        description="Unique within a plan. Filled in from the action's index if left empty.",
    )
    step_id: str = Field(
        pattern=STEP_ID_PATTERN,
        description="The protocol step this action belongs to, e.g. `step.p01.pick_tube`. "
        "Recorded on every simulation step and used by the judge to scope its rules.",
    )
    duration_s: float = Field(gt=0.0, le=120.0)
    note: str = ""


class Move(_Action):
    """Drive a site to a pose by solving IK, then hold the resulting configuration."""

    action: Literal["move"] = "move"
    site: str = Field(
        default="",
        description="The site to drive. The robot's TCP if left empty.",
    )
    target: PoseTarget


class Grip(_Action):
    """Hold the fingers a given distance apart.

    A width, not a fraction of travel, because the width is what the caller knows: the
    barrel is 10.8 mm across. The executor measures the gripper it was given and works out
    the joint angle, so the same 9 mm means the same opening on any gripper. To grasp,
    command a width under the part's own and let the fingers load it.
    """

    action: Literal["grip"] = "grip"
    width_m: float = Field(
        ge=0.0,
        le=0.5,
        description="Clear distance between the fingerpads. Clamped to what the gripper "
        "can reach, so 0 always means fully closed.",
    )


class Hold(_Action):
    """Command nothing new and let the state settle. Used after a release or an impact."""

    action: Literal["hold"] = "hold"


class Actuate(_Action):
    """Drive one named actuator to a value.

    This is how anything that is not the arm gets moved: an instrument's lid hinge, a
    rotor, a plunger. The actuator has to exist in the composed scene, which means it came
    from a generated asset or a co-designed part.
    """

    action: Literal["actuate"] = "actuate"
    actuator: str
    value: float


Action = Annotated[Move | Grip | Hold | Actuate, Field(discriminator="action")]


class Destination(BaseModel):
    """Where the manipulated part is supposed to end up.

    This is what turns "the arm moved" into "the tube is seated in the well", and it is the
    only place the pipeline states a task objective in measurable terms. `amx.sim.policy`
    translates it into the judge's labware and receiver criteria.
    """

    model_config = ConfigDict(extra="forbid")

    site: str = Field(description="Site marking the centre of the destination socket or slot.")
    tolerance_xy_m: float = Field(default=0.004, gt=0.0)
    seated_z_range_m: tuple[float, float] = Field(
        description="Allowed height band for the part's centre once seated, in world Z."
    )
    maximum_tilt_rad: float = Field(default=0.15, gt=0.0)
    minimum_transfer_distance_m: float = Field(
        default=0.05,
        description="How far the part must actually have travelled, so a plan that never "
        "picked it up cannot pass by leaving it where it started.",
    )

    socket_floor_geom: str = Field(
        default="",
        description="If set, the judge additionally checks insertion depth and side clearance "
        "against this socket floor.",
    )
    socket_wall_geoms: list[str] = Field(default_factory=list)
    inner_radius_m: float = Field(default=0.0, ge=0.0)
    minimum_insertion_depth_m: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def socket_needs_radius(self) -> "Destination":
        if self.socket_floor_geom and self.inner_radius_m <= 0.0:
            raise ValueError("socket_floor_geom was given without inner_radius_m")
        low, high = self.seated_z_range_m
        if low > high:
            raise ValueError(f"seated_z_range_m is inverted: {self.seated_z_range_m}")
        return self


class OperationPlan(BaseModel):
    """A complete, ordered episode."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(default="plan", pattern=r"^[a-z0-9][a-z0-9_-]*$")
    protocol_id: str = Field(default="protocol", description="Which protocol this realises.")
    actions: list[Action] = Field(min_length=1)

    manipulated_asset: str = Field(
        default="",
        description="The asset the arm is transporting, if any. Its namespace scopes the "
        "judge's grasp and seating rules.",
    )
    manipulated_body: str = Field(
        default="",
        description="The specific body being transported. Defaults to the manipulated "
        "asset's root body.",
    )
    destination: Destination | None = None

    settle_s: float = Field(
        default=0.5,
        ge=0.0,
        description="A trailing hold appended to every plan, so the final state the judge "
        "grades is a settled one rather than whatever the last command left behind.",
    )

    @field_validator("actions")
    @classmethod
    def assign_ids(cls, value: list[Action]) -> list[Action]:
        seen: set[str] = set()
        for index, action in enumerate(value):
            if not action.action_id:
                action.action_id = f"a{index:02d}.{action.action}"
            if action.action_id in seen:
                raise ValueError(f"duplicate action_id {action.action_id!r}")
            seen.add(action.action_id)
        return value

    @model_validator(mode="after")
    def destination_needs_a_subject(self) -> "OperationPlan":
        if self.destination is not None and not self.manipulated_asset:
            raise ValueError("a destination was given but no manipulated_asset to put there")
        return self

    def step_ids(self) -> list[str]:
        """Distinct protocol steps, in the order they first appear."""
        ordered: list[str] = []
        for action in self.actions:
            if action.step_id not in ordered:
                ordered.append(action.step_id)
        return ordered

    def total_duration_s(self) -> float:
        return sum(a.duration_s for a in self.actions) + self.settle_s

    def grip_step_ids(self) -> list[str]:
        """Steps during which the gripper is commanded closed, plus everything up to the
        next time it is commanded open.

        Those are the steps in which finger-to-part contact is expected rather than a
        collision, so the judge's rules have to know them.
        """
        steps: list[str] = []
        for action, holding in self._grip_phases():
            if holding and action.step_id not in steps:
                steps.append(action.step_id)
        return steps

    def release_step_ids(self) -> list[str]:
        """Steps in which the gripper is commanded open after having been closed."""
        steps: list[str] = []
        was_holding = False
        for action, holding in self._grip_phases():
            if isinstance(action, Grip) and was_holding and not holding:
                steps.append(action.step_id)
            was_holding = holding
        return steps

    def _grip_phases(self) -> list[tuple[Action, bool]]:
        """Each action, paired with whether the gripper is holding something during it.

        Read off the direction the width changes rather than from a fixed threshold: a
        `Grip` that narrows the opening is taking hold of something and one that widens it
        is letting go, whatever the absolute numbers are. A threshold cannot do this,
        because the width that grasps a microcentrifuge tube would be wide open around a
        50 mL conical and the plan is the same plan either way.

        The starting width is taken to be the plan's own widest command, on the grounds
        that a gripper parks open and no plan opens wider than it ever needs to. Without
        that seed the first `Grip` has nothing to be compared against, so a plan that goes
        straight to a grasp — rather than opening first, as the plans here happen to — is
        read as never having taken hold of anything, and the judge is then told the episode
        contains no release and no retention. It reports a clean pass on the strength of it.
        """
        phases: list[tuple[Action, bool]] = []
        holding = False
        widths = [a.width_m for a in self.actions if isinstance(a, Grip)]
        commanded = max(widths) if widths else None
        for action in self.actions:
            if isinstance(action, Grip):
                if commanded is not None and abs(action.width_m - commanded) > 1e-6:
                    holding = action.width_m < commanded
                commanded = action.width_m
            phases.append((action, holding))
        return phases

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def read(cls, path: Path) -> "OperationPlan":
        return cls.model_validate(json.loads(Path(path).read_text()))
