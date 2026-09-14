"""Turning a verdict into a change to the design.

A repair is a small, typed patch — never a rewritten plan and never a rewritten part. The
model sees what failed and which of four things the failure points at, and answers with
deltas: move this waypoint 8 mm up, open that well 0.4 mm, shift the rack 15 mm towards the
arm. Everything else stays exactly as it was.

Deltas rather than absolutes, for two reasons. A model asked to restate a twelve-action
plan will quietly change things nobody asked it to, and the diff between two rounds stops
being readable. And a delta is bounded: `MAX_*` below cap how far one round may move
anything, so a loop that is not converging wanders slowly instead of teleporting the arm
across the bench and producing a completely different failure each round.

What may be patched at all is decided by the routing in `judge.py`, not here and not by the
model: if nothing failed against the fixtures, the fixtures are not in the prompt.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from amx.codesign.parts import TEMPLATES, Part
from amx.geometry import Pose, Vec3
from amx.llm import LlmClient
from amx.loop.judge import Judgement
from amx.report import Finding, RepairTarget
from amx.sim.plan import Actuate, Grip, Move, OperationPlan
from amx.sim.scene import Workcell

MAX_WAYPOINT_DELTA_M = 0.05
MAX_LAYOUT_DELTA_M = 0.08
MAX_DURATION_SCALE = 3.0
MIN_DURATION_SCALE = 0.34

SYSTEM = """\
You are correcting a robot-arm simulation of a wet-lab procedure that a physics judge has
rejected.

You will be given the judge's findings, each tagged with what it points at: the trajectory,
an arm-mounted tool, a bench fixture, or the layout. You answer with a patch containing
only the smallest changes that address those findings.

How to read the findings:

- A collision or a clearance violation means the path goes through something. Move the
  waypoint that was active when it happened, along the axis that gets it clear.
- A part that is not seated, is tilted, or is still moving at the end means the release
  happened in the wrong place or too early. Adjust the last approach waypoint, or lengthen
  the settling action.
- A socket that is too tight or too loose is a dimension on the part, not a motion. Change
  the part's clearance parameter.
- A body resting inside the bench or floating above it was placed wrong. Move it.

Rules:

- Change as little as possible. One finding usually needs one delta.
- Do not fix a symptom in the wrong place. If the arm hits the rack, move the arm or the
  rack, but do not shrink the rack.
- If two findings have the same cause, address the cause once.
- Say in the diagnosis what you think actually went wrong, in one or two sentences. Not a
  restatement of the findings.
"""


class WaypointPatch(BaseModel):
    """A nudge to one action in the plan."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(description="Which action to change. Must already exist in the plan.")
    position_delta_m: Vec3 = Field(
        default=(0.0, 0.0, 0.0),
        description=f"Offset applied to a Move's target, in its own frame. "
        f"At most {MAX_WAYPOINT_DELTA_M * 1000:.0f} mm per axis per round.",
    )
    width_m: float | None = Field(
        default=None,
        ge=0.0,
        le=0.5,
        description="New opening between the fingerpads, for a Grip action. Narrow it to "
        "hold something more firmly; widen it if the fingers are crushing the part.",
    )
    value: float | None = Field(default=None, description="New target, for an Actuate action.")
    duration_scale: float = Field(
        default=1.0,
        ge=MIN_DURATION_SCALE,
        le=MAX_DURATION_SCALE,
        description="Multiplier on the action's duration. Slow a motion down that is hitting "
        "things, or lengthen a settle.",
    )
    reason: str = ""


class LayoutPatch(BaseModel):
    """A nudge to where something sits in the cell."""

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(description="An asset_id or a fixture part_id already in the cell.")
    position_delta_m: Vec3 = Field(
        default=(0.0, 0.0, 0.0),
        description=f"At most {MAX_LAYOUT_DELTA_M * 1000:.0f} mm per axis per round.",
    )
    yaw_delta_rad: float = Field(default=0.0, ge=-1.5708, le=1.5708)
    reason: str = ""


class ParameterPatch(BaseModel):
    """New values for some of a co-designed part's dimensions.

    Only the named parameters change; anything left out keeps its current value, and the
    template never changes. A finding about a fixture means a dimension was wrong, not that
    the mechanism was the wrong choice.
    """

    model_config = ConfigDict(extra="forbid")

    part_id: str
    params: dict[str, float] = Field(
        default_factory=dict, description="Parameter name to new value, in metres or radians."
    )
    material_key: str | None = None
    reason: str = ""


class RepairPatch(BaseModel):
    """Everything one round is allowed to change."""

    model_config = ConfigDict(extra="forbid")

    diagnosis: str = ""
    waypoints: list[WaypointPatch] = Field(default_factory=list)
    layout: list[LayoutPatch] = Field(default_factory=list)
    parameters: list[ParameterPatch] = Field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.waypoints or self.layout or self.parameters)

    def describe(self) -> str:
        lines = [self.diagnosis] if self.diagnosis else []
        for item in self.waypoints:
            change = []
            if any(item.position_delta_m):
                change.append(
                    "move by (" + ", ".join(f"{v * 1000:+.1f}" for v in item.position_delta_m) + ") mm"
                )
            if item.width_m is not None:
                change.append(f"grip width to {item.width_m * 1000:.1f} mm")
            if item.value is not None:
                change.append(f"actuator to {item.value:.4g}")
            if item.duration_scale != 1.0:
                change.append(f"duration x{item.duration_scale:.2f}")
            lines.append(f"  {item.action_id}: {', '.join(change) or 'no change'} — {item.reason}")
        for item in self.layout:
            lines.append(
                f"  {item.target_id}: shift ("
                + ", ".join(f"{v * 1000:+.1f}" for v in item.position_delta_m)
                + f") mm, yaw {item.yaw_delta_rad:+.3f} rad — {item.reason}"
            )
        for item in self.parameters:
            changed = ", ".join(f"{k}={v:.4g}" for k, v in item.params.items())
            lines.append(f"  {item.part_id}: {changed or '(material only)'} — {item.reason}")
        return "\n".join(lines) or "(empty patch)"


class PatchRejected(ValueError):
    """A patch that does not apply to this design, with the reason a model can act on."""


def propose_repair(
    *,
    judgement: Judgement,
    workcell: Workcell,
    plan: OperationPlan,
    parts: dict[str, Part],
    client: LlmClient | None = None,
    context: str = "",
) -> RepairPatch:
    """Ask a model for a patch addressing the failures in `judgement`."""
    client = client or LlmClient()
    targets = set(judgement.targets())
    user = "\n\n".join(
        block
        for block in (
            f"Verdict: {judgement.verdict}. {judgement.reason}",
            f"Context:\n{context}" if context else "",
            "Findings:\n" + _findings_text(judgement.report.failures + judgement.report.warnings),
            "The plan:\n" + _plan_text(plan),
            ("The layout:\n" + _layout_text(workcell))
            if targets & {RepairTarget.LAYOUT} or not targets
            else "",
            ("Co-designed parts:\n" + _parts_text(parts))
            if targets & {RepairTarget.FIXTURE, RepairTarget.TOOL} or not targets
            else "",
            "Return a patch. Leave a list empty if nothing there needs to change.",
        )
        if block
    )
    return client.structured(
        purpose="loop.propose_repair", system=SYSTEM, user=user, schema=RepairPatch
    )


ALTERNATIVE_HINTS = (
    "",
    (
        "\n\nA previous attempt at this failure has already been tried and is being "
        "evaluated separately. Propose a materially different repair: if the obvious "
        "reading is that the path is wrong, consider instead that the thing being "
        "approached is in the wrong place, or that a part is the wrong size. Do not "
        "restate the same change with different numbers."
    ),
    (
        "\n\nTwo repairs have already been proposed for this failure. Take the least "
        "obvious reading of the findings that the measurements still support, and say "
        "plainly in the diagnosis what that reading is."
    ),
)
"""Nudges appended to successive samples of the same failure.

Sampling one prompt three times mostly yields one idea three times, which gives a search
nothing to choose between. Naming the earlier attempt and asking for a different reading
is what makes the candidates actually diverge — and asking for the *least obvious* reading
last is deliberate, because the obvious reading is already covered twice by then.
"""


def propose_repairs(
    *,
    judgement: Judgement,
    workcell: Workcell,
    plan: OperationPlan,
    parts: dict[str, Part],
    count: int = 2,
    client: LlmClient | None = None,
    context: str = "",
    include_heuristic: bool = True,
) -> list[tuple[RepairPatch, str]]:
    """Several distinct candidate repairs, each with a label saying where it came from.

    The heuristic goes first when it has anything to say. It costs nothing, it is often
    right, and having it in the frontier means a model that talks itself into a bad repair
    does not take the round down with it.

    A sample that fails is skipped rather than raised: one unusable candidate out of three
    is a smaller problem than no candidates at all.
    """
    candidates: list[tuple[RepairPatch, str]] = []

    if include_heuristic:
        from amx.loop.heuristics import heuristic_patch  # noqa: PLC0415

        rule_based = heuristic_patch(
            judgement.report.failures, plan=plan, part_ids=set(parts)
        )
        if not rule_based.empty:
            candidates.append((rule_based, "heuristic"))

    wanted = max(0, count - len(candidates))
    for index in range(wanted):
        hint = ALTERNATIVE_HINTS[min(index, len(ALTERNATIVE_HINTS) - 1)]
        try:
            patch = propose_repair(
                judgement=judgement,
                workcell=workcell,
                plan=plan,
                parts=parts,
                client=client,
                context=context + hint,
            )
        except Exception:  # noqa: BLE001 - one bad sample must not cost the whole round
            continue
        if not patch.empty:
            candidates.append((patch, f"model-{index}"))

    return candidates


def apply_patch(
    *,
    workcell: Workcell,
    plan: OperationPlan,
    parts: dict[str, Part],
    patch: RepairPatch,
) -> tuple[Workcell, OperationPlan, dict[str, Part]]:
    """Apply a patch, returning new objects. Raises `PatchRejected` if it does not fit.

    Nothing is mutated in place: a round keeps its own design so a run directory is a
    readable history rather than a sequence of overwrites.
    """
    return (
        _patch_layout(workcell, patch),
        _patch_plan(plan, patch),
        _patch_parts(parts, patch),
    )


def _patch_plan(plan: OperationPlan, patch: RepairPatch) -> OperationPlan:
    if not patch.waypoints:
        return plan.model_copy(deep=True)
    by_id = {item.action_id: item for item in patch.waypoints}
    updated = plan.model_copy(deep=True)
    known = {action.action_id for action in updated.actions}
    unknown = set(by_id) - known
    if unknown:
        raise PatchRejected(
            f"the patch names actions that are not in the plan: {', '.join(sorted(unknown))}. "
            f"The plan has: {', '.join(sorted(known))}"
        )

    for action in updated.actions:
        item = by_id.get(action.action_id)
        if item is None:
            continue
        action.duration_s = _clamp(
            action.duration_s * item.duration_scale, 0.05, 120.0
        )
        if isinstance(action, Move) and any(item.position_delta_m):
            delta = _clamp_vector(item.position_delta_m, MAX_WAYPOINT_DELTA_M)
            action.target.pos = tuple(  # type: ignore[assignment]
                p + d for p, d in zip(action.target.pos, delta, strict=True)
            )
        elif isinstance(action, Grip) and item.width_m is not None:
            action.width_m = item.width_m
        elif isinstance(action, Actuate) and item.value is not None:
            action.value = item.value
    return OperationPlan.model_validate(updated.model_dump())


def _patch_layout(workcell: Workcell, patch: RepairPatch) -> Workcell:
    if not patch.layout:
        return workcell.model_copy(deep=True)
    updated = workcell.model_copy(deep=True)
    placements: dict[str, Any] = {p.asset_id: p for p in updated.assets}
    placements.update({p.part_id: p for p in updated.fixtures})

    for item in patch.layout:
        placement = placements.get(item.target_id)
        if placement is None:
            raise PatchRejected(
                f"the patch moves {item.target_id!r}, which is not in the cell. It holds: "
                f"{', '.join(sorted(placements)) or '(nothing)'}"
            )
        delta = _clamp_vector(item.position_delta_m, MAX_LAYOUT_DELTA_M)
        pose = placement.pose
        placement.pose = Pose(
            pos=tuple(p + d for p, d in zip(pose.pos, delta, strict=True)),  # type: ignore[arg-type]
            euler=(pose.euler[0], pose.euler[1], pose.euler[2] + item.yaw_delta_rad),
        )
    return Workcell.model_validate(updated.model_dump())


def _patch_parts(parts: dict[str, Part], patch: RepairPatch) -> dict[str, Part]:
    if not patch.parameters:
        return dict(parts)
    updated = dict(parts)
    for item in patch.parameters:
        current = updated.get(item.part_id)
        if current is None:
            raise PatchRejected(
                f"the patch resizes {item.part_id!r}, which is not a co-designed part here. "
                f"There are: {', '.join(sorted(parts)) or '(none)'}"
            )
        payload = current.model_dump()
        cls = TEMPLATES[payload["template"]]
        unknown = set(item.params) - set(cls.model_fields)
        if unknown:
            raise PatchRejected(
                f"{payload['template']} has no parameter(s) {', '.join(sorted(unknown))}. "
                f"It has: {', '.join(sorted(set(cls.model_fields) - {'template'}))}"
            )
        for name, value in item.params.items():
            payload[name] = (
                int(round(value)) if cls.model_fields[name].annotation is int else float(value)
            )
        if item.material_key:
            payload["material_key"] = item.material_key
        updated[item.part_id] = cls.model_validate(payload)
    return updated


def _findings_text(findings: list[Finding]) -> str:
    if not findings:
        return "  (none)"
    lines = []
    for finding in findings:
        detail = finding.detail
        where = detail.get("peak_action_id") or detail.get("peak_step_id") or ""
        measured = ", ".join(f"{k}={v:.4g}" for k, v in finding.metrics.items())
        against = ", ".join(f"{k}={v:.4g}" for k, v in finding.thresholds.items())
        lines.append(
            f"- [{finding.repair_target.value}] {finding.code} on {finding.subject}"
            + (f" during {where}" if where else "")
            + f"\n    {finding.summary}"
            + (f"\n    measured: {measured}" if measured else "")
            + (f"\n    limit: {against}" if against else "")
        )
    return "\n".join(lines)


def _plan_text(plan: OperationPlan) -> str:
    lines = []
    for action in plan.actions:
        head = f"- {action.action_id} ({action.step_id}, {action.duration_s:.2f}s)"
        if isinstance(action, Move):
            frame = action.target.frame or "world"
            position = ", ".join(f"{v:.4f}" for v in action.target.pos)
            lines.append(f"{head} move {action.site or 'tcp'} to ({position}) in {frame}")
        elif isinstance(action, Grip):
            lines.append(f"{head} grip to {action.width_m * 1000:.1f} mm")
        elif isinstance(action, Actuate):
            lines.append(f"{head} actuate {action.actuator} to {action.value:.4g}")
        else:
            lines.append(f"{head} hold")
    lines.append(f"- settle {plan.settle_s:.2f}s")
    return "\n".join(lines)


def _layout_text(workcell: Workcell) -> str:
    lines = []
    for placement in workcell.assets:
        position = ", ".join(f"{v:.4f}" for v in placement.pose.pos)
        lines.append(f"- asset {placement.asset_id} at ({position}), {placement.attachment}")
    for fixture in workcell.fixtures:
        position = ", ".join(f"{v:.4f}" for v in fixture.pose.pos)
        lines.append(f"- part {fixture.part_id} on the {fixture.mount} at ({position})")
    return "\n".join(lines) or "  (empty cell)"


def _parts_text(parts: dict[str, Part]) -> str:
    lines = []
    for part_id, part in sorted(parts.items()):
        payload = part.model_dump()
        template = payload.pop("template")
        payload.pop("part_id", None)
        material = payload.pop("material_key", "")
        numbers = ", ".join(
            f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in payload.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        lines.append(f"- {part_id}: {template} in {material}\n    {numbers}")
    return "\n".join(lines) or "  (none)"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp_vector(delta: Vec3, limit: float) -> Vec3:
    return tuple(_clamp(v, -limit, limit) for v in delta)  # type: ignore[return-value]
