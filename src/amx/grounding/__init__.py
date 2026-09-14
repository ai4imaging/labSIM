"""Measuring a generated asset against what was asked for.

This package is the shared measurement core. Two very different callers use it:

* `agent.grounding_tools` — the Articraft tools, which run it inside the authoring
  loop so the model gets told what it got wrong while it can still fix it.
* `amx.bench` — the benchmark judge, which runs it once at the end and scores it.

They must measure the same way or the loop is optimising something the judge does not
look at. What differs is only how the targets are packaged: the tools read a
`GroundingSpec` and the judge reads a rubric, and both are derived from the task's own
input specification, so the agent has nothing to tune against that it was not told.
"""

from amx.grounding.build import GroundedAsset, build_grounded_asset
from amx.grounding.checks import (
    check_dimensions,
    check_operations,
    check_protocol,
    check_topology,
    check_visual,
    ground_against_spec,
)
from amx.grounding.signals import render_grounding_signals
from amx.grounding.spec import (
    ComponentTarget,
    DimensionTarget,
    GroundingSpec,
    OperationTarget,
    ProbeTarget,
    StabilityTarget,
    TiltTarget,
    VisualTarget,
)

__all__ = [
    "ComponentTarget",
    "DimensionTarget",
    "GroundedAsset",
    "GroundingSpec",
    "OperationTarget",
    "ProbeTarget",
    "StabilityTarget",
    "TiltTarget",
    "VisualTarget",
    "build_grounded_asset",
    "check_dimensions",
    "check_operations",
    "check_protocol",
    "check_topology",
    "check_visual",
    "ground_against_spec",
    "render_grounding_signals",
]
