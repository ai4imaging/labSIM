"""Co-designing the parts that sit between the arm and the experiment.

An arm and a piece of labware rarely mate. What is missing is small: an adapter on the
flange, a rack that holds tubes where the arm can reach them, a hook that catches a lid.
This package designs those, under one constraint that shapes everything else — a model
chooses a mechanism and fills in its dimensions, and nothing else. Geometry is a
deterministic function of those dimensions (`parts.py`), what can be made is decided by
process limits rather than by judgement (`materials.py`, `dfm.py`), and each part comes
out as a simulation model, a printable mesh and a datasheet at once (`export.py`).
"""

from amx.codesign.dfm import check_part, measure_wall_thickness
from amx.codesign.export import export_part, write_geometry
from amx.codesign.materials import CATALOGUE, Material, material
from amx.codesign.parts import (
    TEMPLATES,
    AnyPart,
    BenchClamp,
    Cradle,
    FlangeAdapter,
    LidHook,
    Part,
    PartGeometry,
    PressFinger,
    Primitive,
    TubeRack,
    template_schema,
)
from amx.codesign.propose import PartProposal, ProposalRejected, propose_part, realise, revise_part

__all__ = [
    "TEMPLATES",
    "AnyPart",
    "BenchClamp",
    "CATALOGUE",
    "Cradle",
    "FlangeAdapter",
    "LidHook",
    "Material",
    "Part",
    "PartGeometry",
    "PartProposal",
    "PressFinger",
    "Primitive",
    "ProposalRejected",
    "TubeRack",
    "check_part",
    "export_part",
    "material",
    "measure_wall_thickness",
    "propose_part",
    "realise",
    "revise_part",
    "template_schema",
    "write_geometry",
]
