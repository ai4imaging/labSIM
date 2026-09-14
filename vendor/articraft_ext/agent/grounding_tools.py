"""Grounding checks, exposed to the authoring model as tools it can call.

Articraft already gives the model one feedback channel: `compile_model` tells it whether
the geometry it wrote is buildable. That is necessary and nowhere near sufficient — a
solid cylinder of the wrong size compiles perfectly. These three tools add the channels
that decide whether the thing built is the thing that was asked for:

* `check_physical_grounding` — measure it and compare against the stated dimensions.
* `check_protocol_grounding` — stand it up, put something in it, tip it over.
* `check_visual_grounding` — render it and ask whether it looks like the object.

Each answers with a `<grounding_signals>` block, deliberately shaped like the
`<compile_signals>` the model is already reading, so there is one feedback language in
the transcript rather than two.

Like `compile_model`, these are intercepted by the harness rather than executed here:
the tool objects carry only the schema. The harness owns the bound file path, the
grounding spec and the run directory, and it needs to record what each check said in
order to gate the finish attempt on it.
"""

from __future__ import annotations

from typing import Any

from agent.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    ToolParamsModel,
    ToolResult,
    make_tool_schema,
    validate_tool_params,
)

PHYSICAL_TOOL = "check_physical_grounding"
PROTOCOL_TOOL = "check_protocol_grounding"
VISUAL_TOOL = "check_visual_grounding"

GROUNDING_TOOL_NAMES = (PHYSICAL_TOOL, PROTOCOL_TOOL, VISUAL_TOOL)


class _HarnessHandledParams(ToolParamsModel):
    """No parameters. Every check reads the bound file and the run's grounding spec."""


class _HarnessHandledInvocation(BaseToolInvocation[_HarnessHandledParams, str]):
    """Placeholder for a tool the harness intercepts before it is ever executed."""

    tool_name = "grounding"

    def get_description(self) -> str:
        return f"Run {self.tool_name} on the current bound file"

    async def execute(self) -> ToolResult:
        return ToolResult(error=f"{self.tool_name} must be handled by the harness")


class _PhysicalInvocation(_HarnessHandledInvocation):
    tool_name = PHYSICAL_TOOL


class _ProtocolInvocation(_HarnessHandledInvocation):
    tool_name = PROTOCOL_TOOL


class _VisualInvocation(_HarnessHandledInvocation):
    tool_name = VISUAL_TOOL


class CheckPhysicalGroundingTool(BaseDeclarativeTool):
    """Measure the compiled asset against the dimensions the task stated."""

    def __init__(self) -> None:
        schema = make_tool_schema(
            name=PHYSICAL_TOOL,
            description=(
                "Measure the current model against the dimensional specification for this "
                "task and report where it disagrees.\n\n"
                "Compiles the bound file, exports it to MJCF and takes real measurements: "
                "overall extents, outer and inner diameters from cross-sections rather than "
                "from a bounding box, wall thickness, total mass, the body tree against the "
                "required component list, and the internal volume integrated up to the lowest "
                "overflow edge.\n\n"
                "A clean `compile_model` says the geometry builds; it says nothing about "
                "whether it is the right size. Call this whenever you have changed a dimension, "
                "and always before you finish.\n\n"
                "Returns a `<grounding_signals>` block in the same form as `<compile_signals>`."
            ),
            parameters={},
            required=[],
        )
        super().__init__(PHYSICAL_TOOL, schema)

    async def build(self, params: dict[str, Any]) -> _PhysicalInvocation:
        return _PhysicalInvocation(validate_tool_params(_HarnessHandledParams, params))


class CheckProtocolGroundingTool(BaseDeclarativeTool):
    """Simulate the handling the task says the asset has to survive."""

    def __init__(self) -> None:
        schema = make_tool_schema(
            name=PROTOCOL_TOOL,
            description=(
                "Simulate the physical handling this task requires and report what failed.\n\n"
                "Drives every required control and mechanism along its whole travel and back, "
                "checking the joint type and range the operation needs, the worst collision "
                "penetration anywhere on the path rather than only at the endpoint, that a "
                "button presses instead of rotating, and that nothing outside the mechanism "
                "being driven moves while it does.\n\n"
                "Then, depending on what the task specifies: places the object on a flat "
                "surface and checks it neither drifts, tips nor sinks into it; lowers a "
                "reference probe through the mouth and checks it goes in and stays in; rotates "
                "the object to a stated angle and back.\n\n"
                "The insertion test runs against a convex decomposition of your geometry, so a "
                "cavity has to be a genuine subtraction from the solid for anything to be able "
                "to enter it.\n\n"
                "Returns a `<grounding_signals>` block in the same form as `<compile_signals>`."
            ),
            parameters={},
            required=[],
        )
        super().__init__(PROTOCOL_TOOL, schema)

    async def build(self, params: dict[str, Any]) -> _ProtocolInvocation:
        return _ProtocolInvocation(validate_tool_params(_HarnessHandledParams, params))


class CheckVisualGroundingTool(BaseDeclarativeTool):
    """Render the asset and judge it against the appearance the task described."""

    def __init__(self) -> None:
        schema = make_tool_schema(
            name=VISUAL_TOOL,
            description=(
                "Render the current model from several directions and judge whether it looks "
                "like the object that was described.\n\n"
                "Reports, per required visual feature, whether it is actually visible. The "
                "renders are also attached to the conversation so you can see them yourself; "
                "look at them, because a feature can be present in the geometry and still be "
                "wrong in silhouette, proportion or placement.\n\n"
                "Returns a `<grounding_signals>` block in the same form as `<compile_signals>`."
            ),
            parameters={},
            required=[],
        )
        super().__init__(VISUAL_TOOL, schema)

    async def build(self, params: dict[str, Any]) -> _VisualInvocation:
        return _VisualInvocation(validate_tool_params(_HarnessHandledParams, params))


def build_grounding_tools() -> list[BaseDeclarativeTool]:
    return [
        CheckPhysicalGroundingTool(),
        CheckProtocolGroundingTool(),
        CheckVisualGroundingTool(),
    ]
