"""Mapping between the model's internal ids and human-readable names.

A diagnostic report has to state exactly which two parts are involved, yet MuJoCo
only hands out integer geom / body ids. This module translates an id into three
layers of name:

``geom name``  the name inside MuJoCo, e.g. ``centrifuge:rotor__sample_socket_floor_contact``
``body name``  the body that owns the geom, e.g. ``centrifuge:rotor``
``entity_id``  the domain entity, e.g. ``entity.centrifuge``, taken from the geometry
               bindings table in ``task-execution.json``

It also exposes the characteristic geom dimensions the generic penetration criterion
needs to make its numbers dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Iterable, Mapping

import mujoco

from sim_judge.world.geometry import min_half_extent


@dataclass(frozen=True, slots=True)
class GeomInfo:
    """Snapshot of a geom's static properties. Queried over and over during replay,
    so it is computed up front."""

    id: int
    name: str
    body_id: int
    body_name: str
    entity_id: str
    group: int
    geom_type: int
    size: tuple[float, float, float]
    min_half_extent_m: float
    """Minimum half-extent. Serves as the yardstick that makes penetration depth
    dimensionless -- exceeding it means half of this geom's thickness has been
    passed through."""

    collidable: bool
    """Whether the geom takes part in collisions (False for purely visual geoms,
    whose contype and conaffinity are both 0)."""


class NameResolver:
    """Translate ids into names, and answer what a given geom or body actually
    is."""

    def __init__(self, model: mujoco.MjModel, geometry_bindings: Iterable[Mapping[str, Any]] = ()) -> None:
        self._model = model
        self._geom_to_entity = {
            str(b["geom_name"]): str(b["entity_id"])
            for b in geometry_bindings
            if b.get("geom_name") and b.get("entity_id")
        }
        self._geoms: list[GeomInfo] = [self._build_geom(i) for i in range(model.ngeom)]
        self._geom_by_name = {g.name: g for g in self._geoms if g.name}
        self._body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body:{i}>"
            for i in range(model.nbody)
        ]
        self._body_to_entity = self._infer_body_entities()

    # -- geom --------------------------------------------------------------

    def geom(self, geom_id: int) -> GeomInfo:
        return self._geoms[geom_id]

    def geom_by_name(self, name: str) -> GeomInfo | None:
        return self._geom_by_name.get(name)

    def geom_id(self, name: str) -> int | None:
        info = self._geom_by_name.get(name)
        return None if info is None else info.id

    def geoms_of_body(self, body_id: int) -> list[GeomInfo]:
        start = int(self._model.body_geomadr[body_id])
        count = int(self._model.body_geomnum[body_id])
        return self._geoms[start : start + count]

    # -- body --------------------------------------------------------------

    def body_name(self, body_id: int) -> str:
        return self._body_names[body_id]

    def body_id(self, name: str) -> int | None:
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
        return None if bid < 0 else bid

    def body_entity(self, body_id: int) -> str:
        return self._body_to_entity.get(body_id, "")

    # -- construction ------------------------------------------------------

    def _build_geom(self, geom_id: int) -> GeomInfo:
        m = self._model
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"<geom:{geom_id}>"
        body_id = int(m.geom_bodyid[geom_id])
        size = tuple(float(v) for v in m.geom_size[geom_id])
        return GeomInfo(
            id=geom_id,
            name=name,
            body_id=body_id,
            body_name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"<body:{body_id}>",
            entity_id=self._geom_to_entity.get(name, ""),
            group=int(m.geom_group[geom_id]),
            geom_type=int(m.geom_type[geom_id]),
            size=size,  # type: ignore[arg-type]
            min_half_extent_m=min_half_extent(
                int(m.geom_type[geom_id]), size, float(m.geom_rbound[geom_id])
            ),
            collidable=bool(m.geom_contype[geom_id] or m.geom_conaffinity[geom_id]),
        )

    def _infer_body_entities(self) -> dict[int, str]:
        """Decide which entity a body belongs to by taking a vote among the entity
        labels of its own geoms.

        The geometry bindings table is given per geom, but when a report describes
        which part is involved, body granularity reads more naturally. The geoms on
        one body almost always belong to the same entity, and the rare exceptions
        are settled by majority.
        """
        result: dict[int, str] = {}
        for body_id in range(self._model.nbody):
            votes: dict[str, int] = {}
            for geom in self.geoms_of_body(body_id):
                if geom.entity_id:
                    votes[geom.entity_id] = votes.get(geom.entity_id, 0) + 1
            if votes:
                result[body_id] = max(votes.items(), key=lambda kv: kv[1])[0]
        return result


def load_geometry_bindings(
    task_execution: Mapping[str, Any], case_sha256: str | None
) -> tuple[list[Mapping[str, Any]], str | None]:
    """Pull out the geometry bindings table of the case that corresponds to this
    directory's trace, along with a warning (``None`` when there is nothing to warn
    about).

    ``task-execution.json`` may record several cases from the same batch, and only
    the one whose ``case_sha256`` matches corresponds to the recording in this
    directory. With a single case there is no ambiguity, so we just use it. With
    several cases and no match we do **not** guess -- guessing wrong makes the report
    attribute a finding to the wrong part, which is worse than having no entity
    labels at all. In that case we return an empty table and let the report fall
    back to geom / body names.
    """
    cases = task_execution.get("cases") or []
    if not cases:
        return [], None

    def bindings_of(case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return list((case.get("raw_dynamics") or {}).get("geometry_bindings") or [])

    for case in cases:
        if case_sha256 and (case.get("realization") or {}).get("case_sha256") == case_sha256:
            return bindings_of(case), None
    if len(cases) == 1:
        return bindings_of(cases[0]), None
    return [], (
        f"The execution record contains {len(cases)} cases, but none of their case_sha256 "
        "values match this directory's trace, so the geometry bindings table cannot be "
        "determined; parts in the report will be identified by geom / body name only."
    )
