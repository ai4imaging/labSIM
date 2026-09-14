"""Manufacturing processes and what they can actually make.

Every number here is a process limit, not a preference: a wall thinner than the minimum
will not come off the machine intact, an unsupported overhang past the limit will droop,
and a hole cut to its nominal diameter will not accept the shaft it was drawn for. They
live in one table because `amx.codesign.dfm` checks against them and `parts.py` sizes
against them, and the two must not disagree.

Numbers are for a well-tuned machine in a lab, not a production line with a process
engineer, so they are on the conservative side.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Process(StrEnum):
    FDM = "fdm"
    """Fused deposition. The default for lab fixtures: cheap, fast, good enough."""

    SLA = "sla"
    """Resin. For small parts with fine features, at the cost of brittleness."""

    SLS = "sls"
    """Powder-bed nylon. Tough and isotropic; no support structures needed."""

    CNC = "cnc"
    """Milled aluminium. For anything that carries load or has to hold tolerance."""


class Material(BaseModel):
    """A process and material pairing, with the limits it imposes on geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    label: str
    process: Process
    density_kg_m3: float

    minimum_wall_m: float = Field(description="Thinnest section that will survive the process.")
    minimum_feature_m: float = Field(description="Smallest printable positive detail, e.g. a rib or pin.")
    maximum_overhang_rad: float = Field(
        description="Steepest unsupported downward surface, measured from vertical. "
        "Beyond this the surface needs support material, or a different build orientation."
    )
    hole_allowance_m: float = Field(
        description="How much larger than nominal a hole must be cut so the mating part fits. "
        "Absorbs the process's tendency to shrink holes."
    )
    build_volume_m: tuple[float, float, float]
    friction: float = Field(
        default=0.6,
        description="Sliding friction of the finished surface against glass or plastic labware. "
        "Used for the part's contacts in simulation, so a printed rack grips a tube the way "
        "the real one does and a milled face does not.",
    )
    notes: str = ""

    def mass_of(self, volume_m3: float) -> float:
        return volume_m3 * self.density_kg_m3

    def fits_build_volume(self, extents: tuple[float, float, float]) -> bool:
        """Whether the part fits, allowing it to be rotated onto any of the three axes."""
        return all(
            e <= b for e, b in zip(sorted(extents), sorted(self.build_volume_m), strict=True)
        )


FDM_PLA = Material(
    key="fdm_pla",
    label="FDM, PLA",
    process=Process.FDM,
    density_kg_m3=1240.0,
    minimum_wall_m=0.0016,
    minimum_feature_m=0.0008,
    maximum_overhang_rad=0.7854,
    hole_allowance_m=0.0002,
    friction=0.55,
    build_volume_m=(0.25, 0.21, 0.21),
    notes="Loses stiffness above 50 C; do not use where an autoclave or a heat block is nearby.",
)

FDM_PETG = Material(
    key="fdm_petg",
    label="FDM, PETG",
    process=Process.FDM,
    density_kg_m3=1270.0,
    minimum_wall_m=0.0018,
    minimum_feature_m=0.0010,
    maximum_overhang_rad=0.6981,
    hole_allowance_m=0.0003,
    friction=0.6,
    build_volume_m=(0.25, 0.21, 0.21),
    notes="Tolerates solvents and moderate heat better than PLA; slightly worse dimensional accuracy.",
)

SLA_RESIN = Material(
    key="sla_resin",
    label="SLA, standard resin",
    process=Process.SLA,
    density_kg_m3=1180.0,
    minimum_wall_m=0.0010,
    minimum_feature_m=0.0004,
    maximum_overhang_rad=0.5236,
    hole_allowance_m=0.0001,
    friction=0.45,
    build_volume_m=(0.145, 0.145, 0.175),
    notes="Brittle. Suitable for probes and locating features, not for anything that takes a knock.",
)

SLS_PA12 = Material(
    key="sls_pa12",
    label="SLS, PA12 nylon",
    process=Process.SLS,
    density_kg_m3=1010.0,
    minimum_wall_m=0.0012,
    minimum_feature_m=0.0006,
    maximum_overhang_rad=1.5708,
    hole_allowance_m=0.0003,
    friction=0.7,
    build_volume_m=(0.28, 0.28, 0.38),
    notes="Self-supporting, so overhangs are unconstrained. The default choice for a part that "
    "has to be tough and has awkward geometry.",
)

CNC_ALUMINIUM = Material(
    key="cnc_al6061",
    label="CNC, aluminium 6061",
    process=Process.CNC,
    density_kg_m3=2700.0,
    minimum_wall_m=0.0015,
    minimum_feature_m=0.0010,
    maximum_overhang_rad=1.5708,
    hole_allowance_m=0.0001,
    friction=0.4,
    build_volume_m=(0.3, 0.2, 0.1),
    notes="For load-bearing adapters between the arm and anything heavy. Undercuts cost real money; "
    "keep the geometry reachable from three axes.",
)

CATALOGUE: dict[str, Material] = {
    m.key: m
    for m in (FDM_PLA, FDM_PETG, SLA_RESIN, SLS_PA12, CNC_ALUMINIUM)
}

DEFAULT_MATERIAL = FDM_PETG


def material(key: str) -> Material:
    try:
        return CATALOGUE[key]
    except KeyError:
        raise KeyError(
            f"unknown material {key!r}; available: {', '.join(sorted(CATALOGUE))}"
        ) from None
