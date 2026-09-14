"""What you hand part 1 to get an asset back.

An `AssetRequest` carries four kinds of input, in decreasing order of how much the model
is allowed to interpret them: a free-text prompt, protocol excerpts, a datasheet of hard
numbers, and grounding requirements. The datasheet and the grounding requirements are
checked afterwards by `amx.asset.grounding`, so they are commitments, not suggestions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Interval = tuple[float, float]


class Datasheet(BaseModel):
    """Measured or specified numbers the finished asset has to agree with.

    Every field is an inclusive interval so that a spec sheet's tolerances survive into
    the check. Anything left unset is simply not checked.
    """

    model_config = ConfigDict(extra="forbid")

    bbox_x_m: Interval | None = None
    bbox_y_m: Interval | None = None
    bbox_z_m: Interval | None = None
    total_mass_kg: Interval | None = None
    material: str | None = None
    sliding_friction: Interval | None = None

    joint_ranges_rad: dict[str, Interval] = Field(
        default_factory=dict,
        description="Per-articulation travel, keyed by the articulation name in the model.",
    )
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_intervals(self) -> "Datasheet":
        named: list[tuple[str, Interval | None]] = [
            ("bbox_x_m", self.bbox_x_m),
            ("bbox_y_m", self.bbox_y_m),
            ("bbox_z_m", self.bbox_z_m),
            ("total_mass_kg", self.total_mass_kg),
            ("sliding_friction", self.sliding_friction),
            *((f"joint_ranges_rad[{k}]", v) for k, v in self.joint_ranges_rad.items()),
        ]
        for label, interval in named:
            if interval is not None and interval[0] > interval[1]:
                raise ValueError(f"{label} interval is inverted: {interval}")
        return self


class VisionGrounding(BaseModel):
    """Statements a rendered view of the asset must support."""

    model_config = ConfigDict(extra="forbid")

    claims: list[str] = Field(
        default_factory=list,
        description="Each is judged separately against renders, e.g. 'the lid opens upward on a hinge at the rear'.",
    )
    views: list[Literal["front", "side", "top", "iso"]] = Field(
        default_factory=lambda: ["front", "side", "iso"]
    )


class FunctionalGrounding(BaseModel):
    """Things the asset has to be able to do, checked by simulation rather than by eye."""

    model_config = ConfigDict(extra="forbid")

    movable_articulations: list[str] = Field(
        default_factory=list,
        description="Articulations that must actually move when driven, not be fused.",
    )
    reachable_sites: list[str] = Field(
        default_factory=list,
        description="Sites an arm has to be able to touch once the asset is placed on the bench.",
    )
    must_rest_stably: bool = Field(
        default=True,
        description="Whether the asset must settle on a flat surface without toppling or sinking.",
    )


class Grounding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vision: VisionGrounding = Field(default_factory=VisionGrounding)
    functional: FunctionalGrounding = Field(default_factory=FunctionalGrounding)


class AssetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    prompt: str = Field(min_length=1)
    protocol_excerpts: list[str] = Field(
        default_factory=list,
        description="Verbatim protocol text that constrains the design, e.g. tube format and volumes.",
    )
    reference_files: list[Path] = Field(
        default_factory=list,
        description="Datasheets, drawings or photographs to pass through to the authoring model.",
    )
    datasheet: Datasheet = Field(default_factory=Datasheet)
    grounding: Grounding = Field(default_factory=Grounding)

    max_turns: int = 100
    """Upper bound on Articraft authoring turns before the run is abandoned."""

    def authoring_prompt(self) -> str:
        """The single text block handed to Articraft's authoring harness."""
        parts = [self.prompt.strip()]
        if self.protocol_excerpts:
            parts.append(
                "Protocol constraints this object must satisfy:\n"
                + "\n".join(f"- {line.strip()}" for line in self.protocol_excerpts)
            )
        sheet = self._datasheet_lines()
        if sheet:
            parts.append("Datasheet targets (these are checked after you finish):\n" + "\n".join(sheet))
        if self.grounding.functional.movable_articulations:
            parts.append(
                "These articulations must be present and genuinely movable: "
                + ", ".join(self.grounding.functional.movable_articulations)
            )
        if self.grounding.vision.claims:
            parts.append(
                "The finished object will be inspected visually against these statements:\n"
                + "\n".join(f"- {claim}" for claim in self.grounding.vision.claims)
            )
        return "\n\n".join(parts)

    def _datasheet_lines(self) -> list[str]:
        sheet = self.datasheet
        lines: list[str] = []
        for label, interval in (
            ("overall X extent", sheet.bbox_x_m),
            ("overall Y extent", sheet.bbox_y_m),
            ("overall Z extent", sheet.bbox_z_m),
        ):
            if interval:
                lines.append(f"- {label}: {interval[0]:.4g} to {interval[1]:.4g} m")
        if sheet.total_mass_kg:
            lines.append(
                f"- total mass: {sheet.total_mass_kg[0]:.4g} to {sheet.total_mass_kg[1]:.4g} kg"
            )
        if sheet.material:
            lines.append(f"- material: {sheet.material}")
        for name, interval in sheet.joint_ranges_rad.items():
            lines.append(f"- {name} travel: {interval[0]:.4g} to {interval[1]:.4g} rad")
        lines.extend(f"- {note}" for note in sheet.notes)
        return lines
