"""Asking a model for a part, without letting it write one.

The model's entire output is a template name and a dictionary of numbers. It never writes
geometry code, never writes MJCF, and never chooses a topology that does not already exist
in `parts.py`. That restriction is the whole reason this half of the pipeline is stable: a
proposal either validates into a `Part` — in which case the geometry is a deterministic
function of it and is known to build — or it does not, in which case pydantic says exactly
which number was wrong and the model gets one more go.

Compare with letting a model author the part directly: the failure modes there are
unbounded (it does not run, it runs and produces a shell, it produces a solid nobody can
make) and none of them come with a usable error. Here there is one failure mode and it is
a validation message.

The cost is that a need no template covers cannot be met. `amx.asset.generate` is the
escape hatch for that, and whatever it produces still comes back through `dfm.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from amx.codesign.dfm import check_part
from amx.codesign.export import export_part
from amx.codesign.materials import CATALOGUE
from amx.codesign.parts import TEMPLATES, Part, PartGeometry
from amx.geometry import Pose
from amx.llm import LlmClient
from amx.report import Finding, RepairTarget, Report, Severity

SYSTEM = """\
You size mechanical parts for a robot arm working in a wet lab.

You choose one template from a fixed library and fill in its dimensions. You do not write
geometry, code, or XML — those are generated from your numbers. Work in metres.

Design rules, in priority order:

1. Simple. Use the fewest features that do the job. Do not add a boss, rib, or slot that
   nothing mates with. If a plain plate works, specify a plain plate.
2. Stable in use. Something that locates a part in three directions beats something that
   locates it in one and relies on friction. Prefer a seat the object drops into over a
   clamp that has to be tightened.
3. Makeable. Respect the material's minimum wall and minimum feature. Keep the part small
   and its mass low, especially anything the arm carries.
4. Not over-tight. Give a fit the material's own hole allowance or more. A press fit that
   needs force is a fit that ends up crooked.

Every dimension has a stated range; stay inside it. State your reasoning in one or two
sentences that name the constraint that drove the sizing, not a summary of the numbers.
"""


class PartProposal(BaseModel):
    """What the model returns: a template, its numbers, and where the part goes."""

    model_config = ConfigDict(extra="forbid")

    template: str = Field(description=f"One of: {', '.join(sorted(TEMPLATES))}.")
    part_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9_]*$",
        description="Short lower-case identifier, unique within the cell.",
    )
    material_key: str = Field(
        default="fdm_petg", description=f"One of: {', '.join(sorted(CATALOGUE))}."
    )
    params: dict[str, float] = Field(
        default_factory=dict,
        description="Template parameters, in metres or radians. Omit any you want defaulted.",
    )
    mount: str = Field(
        default="bench", description="`bench` for a bolted-down fixture, `tool` for arm-mounted."
    )
    pose: Pose = Field(
        default_factory=Pose,
        description="For `bench`, relative to the bench surface. For `tool`, relative to the "
        "flange's tool_mount site.",
    )
    rationale: str = ""

    def instantiate(self) -> Part:
        """Turn the proposal into a validated `Part`, or raise a readable error."""
        try:
            cls = TEMPLATES[self.template]
        except KeyError:
            raise ValueError(
                f"unknown template {self.template!r}; available: {', '.join(sorted(TEMPLATES))}"
            ) from None
        payload: dict[str, Any] = {
            "part_id": self.part_id,
            "material_key": self.material_key,
            **{k: v for k, v in self.params.items() if k not in {"part_id", "material_key"}},
        }
        integer_fields = {
            name for name, field in cls.model_fields.items() if field.annotation is int
        }
        for name in integer_fields & payload.keys():
            payload[name] = int(round(float(payload[name])))
        return cls(**payload)


def propose_part(
    *,
    need: str,
    template: str | None = None,
    client: LlmClient | None = None,
    context: str = "",
) -> PartProposal:
    """Ask a model to size a part for `need`.

    `template` pins the mechanism when the caller already knows which one it wants, which
    is the usual case for a repair: the part exists and only its numbers are in question.
    """
    client = client or LlmClient()
    choices = [template] if template else sorted(TEMPLATES)
    catalogue = "\n\n".join(_describe(name) for name in choices)
    user = "\n\n".join(
        section
        for section in (
            f"What is needed:\n{need}",
            f"Context:\n{context}" if context else "",
            f"Templates you may choose from:\n\n{catalogue}",
            "Return one proposal. Leave out any parameter you are happy to default.",
        )
        if section
    )
    proposal = client.structured(
        purpose="codesign.propose_part", system=SYSTEM, user=user, schema=PartProposal
    )
    if template:
        proposal.template = template
    return proposal


def revise_part(
    *,
    part: Part,
    findings: list[Finding],
    goal: str,
    client: LlmClient | None = None,
) -> PartProposal:
    """Ask for new numbers for an existing part, given what went wrong with it.

    The template is fixed. A finding about a fixture almost never means the mechanism was
    the wrong choice; it means a dimension was wrong, and holding the topology steady keeps
    the loop from wandering between designs instead of converging on one.
    """
    client = client or LlmClient()
    problems = "\n".join(f"- {f.code}: {f.summary}" for f in findings) or "- (none given)"
    user = (
        f"Goal:\n{goal}\n\n"
        f"The current {part.template} `{part.part_id}` is:\n"
        f"{json.dumps(_numeric_params(part), indent=2)}\n"
        f"in {part.material.label}.\n\n"
        f"What is wrong with it:\n{problems}\n\n"
        f"Template:\n\n{_describe(part.template)}\n\n"
        "Return a proposal with the same template and part_id, changing only what the "
        "problems above require. Say in the rationale which change addresses which finding."
    )
    proposal = client.structured(
        purpose="codesign.revise_part", system=SYSTEM, user=user, schema=PartProposal
    )
    proposal.template = part.template  # type: ignore[attr-defined]
    proposal.part_id = part.part_id
    return proposal


def realise(
    proposal: PartProposal, output_dir: Path
) -> tuple[Part, PartGeometry | None, Report]:
    """Validate, build, check and write a proposal.

    Returns the part, its geometry when it built, and a report. A proposal that fails
    validation comes back as a report with one finding rather than an exception, because
    the loop's next step is the same either way: hand the finding back for another go.
    """
    report = Report(kind="dfm", subject=f"{proposal.part_id} ({proposal.template})")
    try:
        part = proposal.instantiate()
    except (ValidationError, ValueError) as error:
        report.findings.append(
            Finding(
                code="D-PARAMS",
                severity=Severity.FAILURE,
                subject=proposal.part_id,
                summary=f"the proposed dimensions do not describe this mechanism: {error}",
                repair_target=(
                    RepairTarget.TOOL if proposal.mount == "tool" else RepairTarget.FIXTURE
                ),
            )
        )
        raise ProposalRejected(report) from error

    geometry, dfm = export_part(part, output_dir)
    return part, geometry, dfm


class ProposalRejected(RuntimeError):
    """A proposal that did not describe a buildable part, with the report saying why."""

    def __init__(self, report: Report) -> None:
        super().__init__(report.to_text())
        self.report = report


def check_existing(part: Part, *, mounted_on_arm: bool) -> Report:
    """Re-run manufacturability on a part without writing anything."""
    return check_part(part.build(), mounted_on_arm=mounted_on_arm)


def _numeric_params(part: Part) -> dict[str, float]:
    return {
        name: value
        for name, value in part.model_dump().items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _describe(name: str) -> str:
    """A template's purpose and its parameter ranges, as prompt text.

    Generated from the model rather than written out, so the prompt cannot drift away from
    what the code will actually accept.
    """
    cls = TEMPLATES[name]
    summary = (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n    ", " ")
    lines = [f"{name}: {summary}"]
    for field_name, field in cls.model_fields.items():
        if field_name in {"part_id", "material_key", "template"}:
            continue
        bounds = _bounds(field)
        note = (field.description or "").replace("\n", " ")
        lines.append(f"  {field_name}{bounds}{': ' + note if note else ''}")
    return "\n".join(lines)


def _bounds(field: Any) -> str:
    low = high = None
    for item in field.metadata:
        low = getattr(item, "ge", None) or getattr(item, "gt", None) or low
        high = getattr(item, "le", None) or getattr(item, "lt", None) or high
    if low is None and high is None:
        return ""
    return f" [{low if low is not None else '-'} .. {high if high is not None else '-'}]"
