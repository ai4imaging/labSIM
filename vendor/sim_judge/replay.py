"""Replay core: push recorded states back into MuJoCo and rebuild the full physical
state of every step.

How it works: after ``mj_setState`` restores a state we call ``mj_forward``, which
makes MuJoCo redo collision detection and solve the constraints. That gives us
``data.contact[i].dist`` as the contact clearance (negative values being the
penetration depth) and ``mj_contactForce`` as the normal and tangential force of
that contact. None of these quantities are stored directly in the raw recording;
recovering them this way is the only option.

Before replaying we run a **determinism self-check**: for a handful of sampled steps
we execute ``before -> mj_step -> after`` and compare against the recorded
``after``. Only a vanishingly small deviation proves that the model, engine version
and solver options match the recording exactly; otherwise every contact quantity
downstream is untrustworthy, and the report says so explicitly instead of handing
out misleading diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import mujoco
import numpy as np

from sim_judge.loader.trace_reader import StateLayout, TraceReader
from sim_judge.world.naming import GeomInfo, NameResolver
from sim_judge.world.timeline import Timeline

_STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True, slots=True)
class ContactSample:
    """The resolved state of one contact point at one step."""

    geom_a: GeomInfo
    geom_b: GeomInfo
    depth_m: float
    """Penetration depth, equal to ``-contact.dist``. Positive means the two geoms
    already overlap."""

    normal_force_n: float
    tangential_force_n: float

    force_world_on_b: tuple[float, float, float]
    """Contact force in world coordinates acting on the body that owns geom_b
    (normal plus friction).

    The 6-vector returned by ``mj_contactForce`` is expressed in the contact frame
    and, by MuJoCo convention, acts on the geom2 side. Verified against measurement:
    with the tube resting on the rotor socket floor, the force on the floor (geom2)
    is -mg, so the reaction on the tube is +mg, which balances gravity as expected.
    """

    @property
    def body_pair(self) -> tuple[int, int]:
        return (self.geom_a.body_id, self.geom_b.body_id)

    def other_body(self, body_id: int) -> int:
        return self.geom_b.body_id if self.geom_a.body_id == body_id else self.geom_a.body_id

    def involves_body(self, body_id: int) -> bool:
        return body_id in (self.geom_a.body_id, self.geom_b.body_id)

    def force_on_body(self, body_id: int) -> np.ndarray:
        """World-frame force this contact applies to the given body. Newton's third
        law fixes the sign on the A side."""
        force = np.asarray(self.force_world_on_b, dtype=np.float64)
        if body_id == self.geom_b.body_id:
            return force
        if body_id == self.geom_a.body_id:
            return -force
        return np.zeros(3)


@dataclass(frozen=True, slots=True)
class Frame:
    """Read-only physical snapshot of one step, fed to every detector."""

    step_index: int
    time_s: float
    step_id: str
    action_id: str

    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    """Generalised accelerations solved by ``mj_forward``. Used to tell whether a
    body is in free fall or being held up."""

    xpos: np.ndarray
    """World position of each body's centre of mass, shape ``(nbody, 3)``."""

    xquat: np.ndarray
    """World orientation quaternion of each body, shape ``(nbody, 4)``."""

    geom_xpos: np.ndarray
    """World position of each geom's local origin, shape ``(ngeom, 3)``.

    Together with ``model.geom_rbound`` this supports a cheap broad-phase test for
    whether two geoms could possibly be close enough to matter.
    """

    site_xpos: np.ndarray
    """World position of each site, shape ``(nsite, 3)``. Sites are the reference
    points the scene ships with for its own interfaces."""

    contacts: tuple[ContactSample, ...]
    warnings: tuple[str, ...]
    """Names of the MuJoCo engine warnings raised on this step, e.g.
    ``mjWARN_CONTACTFULL``."""

    def contacts_of_body(self, body_id: int) -> list[ContactSample]:
        return [c for c in self.contacts if c.involves_body(body_id)]

    @property
    def phase_labels(self) -> tuple[str, str]:
        """The two labels this step can be matched against a policy's ``phases``:
        the action name and the protocol step id.

        Both namespaces show up in the ``bound-operation.json`` of one task or
        another, so we expose both rather than have a rule silently fail to match
        because the author picked the other spelling.
        """
        return (self.action_id, self.step_id)


@dataclass(frozen=True, slots=True)
class DeterminismCheck:
    """Result of the self-check on how far the replay can be trusted."""

    sample_count: int
    max_state_error: float
    max_qpos_error: float
    tolerance: float
    worst_step: int

    @property
    def trustworthy(self) -> bool:
        return self.max_state_error <= self.tolerance

    def to_dict(self) -> dict[str, object]:
        return {
            "trustworthy": self.trustworthy,
            "sample_count": self.sample_count,
            "max_state_error": self.max_state_error,
            "max_qpos_error": self.max_qpos_error,
            "tolerance": self.tolerance,
            "worst_step": self.worst_step,
        }


class Replayer:
    """Rebuild the physical quantities step by step from the recorded states.

    A single instance reuses one ``MjData``, so the arrays inside a ``Frame`` are
    views into that buffer and are only valid for the current iteration. A detector
    that needs to keep values across steps must copy them itself.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        reader: TraceReader,
        resolver: NameResolver,
        timeline: Timeline,
    ) -> None:
        self._model = model
        self._reader = reader
        self._resolver = resolver
        self._timeline = timeline
        self._data = mujoco.MjData(model)
        self._layout = StateLayout.of(model)
        self._force_buffer = np.zeros(6, dtype=np.float64)
        self._warning_names = _warning_names()

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    def check_determinism(self, sample_count: int, tolerance: float) -> DeterminismCheck:
        """Sample-check that recorded state + this model + one integration step
        reproduces the recorded next state."""
        total = self._reader.step_count
        count = max(1, min(sample_count, total))
        indices = np.linspace(0, total - 1, count, dtype=int).tolist()

        worst_state, worst_qpos, worst_step = 0.0, 0.0, indices[0]
        recovered = np.zeros(self._layout.total)
        qpos = self._layout.qpos

        for raw in self._reader.sample_steps(indices):
            mujoco.mj_setState(self._model, self._data, raw.state_before, _STATE_SPEC)
            mujoco.mj_step(self._model, self._data)
            mujoco.mj_getState(self._model, self._data, recovered, _STATE_SPEC)

            error = float(np.abs(recovered - raw.state_after).max())
            if error > worst_state:
                worst_state, worst_step = error, raw.index
            # Looking at qpos on its own distinguishes "the configuration is
            # genuinely wrong" from "some auxiliary quantity just does not match".
            worst_qpos = max(
                worst_qpos, float(np.abs(recovered[qpos] - raw.state_after[qpos]).max())
            )

        return DeterminismCheck(
            sample_count=len(indices),
            max_state_error=worst_state,
            max_qpos_error=worst_qpos,
            tolerance=tolerance,
            worst_step=worst_step,
        )

    def iter_frames(self, *, stride: int = 1, start: int = 0, stop: int | None = None) -> Iterator[Frame]:
        """Walk the trace, yielding a resolved :class:`Frame` for each step."""
        model, data = self._model, self._data
        for raw in self._reader.iter_steps(stride=stride, start=start, stop=stop):
            data.warning.number[:] = 0
            mujoco.mj_setState(model, data, raw.state_before, _STATE_SPEC)
            mujoco.mj_forward(model, data)

            span = self._timeline.span_of(raw.index)
            yield Frame(
                step_index=raw.index,
                time_s=float(data.time),
                # The phase carried by the recording is first-hand, logged per step,
                # so it wins over anything inferred from the action spans.
                step_id=raw.step_id or (span.step_id if span else ""),
                action_id=span.action_id if span else "",
                qpos=data.qpos,
                qvel=data.qvel,
                qacc=data.qacc,
                xpos=data.xpos,
                xquat=data.xquat,
                geom_xpos=data.geom_xpos,
                site_xpos=data.site_xpos,
                contacts=self._collect_contacts(),
                warnings=self._collect_warnings(),
            )

    def geom_distance(self, geom_a: int, geom_b: int, distmax: float) -> float:
        """Signed distance between two geoms: positive is clearance, negative is
        penetration.

        This reads the configuration currently held in ``MjData``, so it must be
        called immediately after a :class:`Frame` has been yielded. Beyond
        ``distmax`` MuJoCo simply returns ``distmax`` itself.
        """
        return float(mujoco.mj_geomDistance(self._model, self._data, geom_a, geom_b, distmax, None))

    # -- internals ---------------------------------------------------------

    def _collect_contacts(self) -> tuple[ContactSample, ...]:
        model, data = self._model, self._data
        buffer = self._force_buffer
        samples: list[ContactSample] = []

        for index in range(data.ncon):
            contact = data.contact[index]
            mujoco.mj_contactForce(model, data, index, buffer)
            # frame is a row-major 3x3 orthonormal basis whose rows are, in order,
            # the normal, tangent 1 and tangent 2 (all in world coordinates). The
            # transform from contact frame to world frame is therefore its transpose.
            basis = contact.frame.reshape(3, 3)
            force_world = basis.T @ buffer[:3]
            samples.append(
                ContactSample(
                    geom_a=self._resolver.geom(int(contact.geom1)),
                    geom_b=self._resolver.geom(int(contact.geom2)),
                    depth_m=-float(contact.dist),
                    normal_force_n=abs(float(buffer[0])),
                    tangential_force_n=float(np.hypot(buffer[1], buffer[2])),
                    force_world_on_b=(float(force_world[0]), float(force_world[1]), float(force_world[2])),
                )
            )
        return tuple(samples)

    def _collect_warnings(self) -> tuple[str, ...]:
        counts = self._data.warning.number
        return tuple(name for index, name in self._warning_names.items() if counts[index] > 0)


def _warning_names() -> dict[int, str]:
    """Mapping from MuJoCo warning type number to name."""
    names: dict[int, str] = {}
    for index in range(int(mujoco.mjtWarning.mjNWARNING)):
        try:
            names[index] = mujoco.mjtWarning(index).name
        except ValueError:  # skip holes in the enum
            continue
    return names


def free_body_ids(model: mujoco.MjModel) -> list[int]:
    """Find every body driven by a free joint.

    These are the only objects that can possibly be "floating" -- a link constrained
    by an articulated joint is held up by that joint, so leaving a support surface is
    not a physical error for it.
    """
    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    return [
        int(model.jnt_bodyid[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == free_type
    ]


def grounded_body_ids(model: mujoco.MjModel) -> set[int]:
    """Find the bodies pinned to the world through the kinematic chain.

    Walking out from the world body, as long as no free joint is crossed along the
    way the body counts as structurally "grounded": its position is set by joints
    rather than by contacts, so it is supported by construction.
    """
    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    grounded = {0}
    for body_id in range(1, model.nbody):
        parent = int(model.body_parentid[body_id])
        start = int(model.body_jntadr[body_id])
        count = int(model.body_jntnum[body_id])
        has_free = any(int(model.jnt_type[start + k]) == free_type for k in range(count))
        if parent in grounded and not has_free:
            grounded.add(body_id)
    return grounded


def free_joint_of_body(model: mujoco.MjModel, body_id: int) -> tuple[int, int] | None:
    """Return a free body's ``(qpos start index, qvel start index)``, or ``None``
    if the body is not free."""
    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    start = int(model.body_jntadr[body_id])
    for k in range(int(model.body_jntnum[body_id])):
        joint = start + k
        if int(model.jnt_type[joint]) == free_type:
            return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    return None
