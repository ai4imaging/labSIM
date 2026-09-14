"""Streaming reader for the per-step state recording.

A recording is split into several hundred ``steps-*.npz`` files, each holding three arrays of
equal length:

====================  ====================================================
``before[i]``         Full state vector at the start of step i (``mjSTATE_INTEGRATION``)
``after[i]``          State vector after ``mj_step`` for step i, equal to ``before[i+1]``
``phase[i]``          The protocol step id that step i belongs to, e.g. ``step.p03.load``
====================  ====================================================

Reading is strictly on demand: only one chunk is resident at a time, so peak memory for a full
100k-step trace stays on the order of 1 MB. Chunk sha256 digests are verified as a side effect
of reading, without a second pass over the data.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator, Sequence

import numpy as np

from sim_judge.loader.case_bundle import BundleError, CaseBundle

_REQUIRED_FIELDS = ("before", "after", "phase")


@dataclass(frozen=True, slots=True)
class StateLayout:
    """Field slices of an ``mjSTATE_INTEGRATION`` state vector.

    MuJoCo concatenates the fields in a fixed order, and the offsets follow entirely from the
    model's dimensions. We rebuild that layout explicitly so that qpos / qvel can be pulled out
    cheaply for pre-filtering, without having to construct an ``MjData``.
    """

    total: int
    time: slice
    qpos: slice
    qvel: slice
    act: slice
    warmstart: slice
    ctrl: slice
    qfrc_applied: slice
    xfrc_applied: slice
    eq_active: slice
    mocap_pos: slice
    mocap_quat: slice
    userdata: slice

    @classmethod
    def of(cls, model) -> StateLayout:
        widths = {
            "time": 1,
            "qpos": model.nq,
            "qvel": model.nv,
            "act": model.na,
            "warmstart": model.nv,
            "ctrl": model.nu,
            "qfrc_applied": model.nv,
            "xfrc_applied": 6 * model.nbody,
            "eq_active": model.neq,
            "mocap_pos": 3 * model.nmocap,
            "mocap_quat": 4 * model.nmocap,
            "userdata": model.nuserdata,
        }
        slices: dict[str, slice] = {}
        offset = 0
        for name, width in widths.items():
            slices[name] = slice(offset, offset + width)
            offset += width
        return cls(total=offset, **slices)


@dataclass(frozen=True, slots=True)
class RawStep:
    """The recorded content of one step. The arrays are views into the chunk buffer and are only valid for the current iteration."""

    index: int
    step_id: str
    state_before: np.ndarray
    state_after: np.ndarray


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    filename: str
    first_step: int
    step_count: int
    sha256: str | None


class TraceReader:
    """Streams the recorded states in manifest order."""

    def __init__(self, bundle: CaseBundle, *, verify_chunks: bool = True) -> None:
        manifest = bundle.read_trace_manifest()
        self._dir = bundle.raw_dynamics_dir
        self._verify = verify_chunks
        self._chunks = _parse_chunks(manifest, self._dir)

        self.step_count: int = int(manifest.get("step_count", sum(c.step_count for c in self._chunks)))
        self.timestep_s: float = float(manifest.get("timestep_s", 0.0))
        self.state_size: int = int(manifest.get("state_size", 0))
        self._verified: set[str] = set()
        self._mismatched: set[str] = set()

        declared = sum(c.step_count for c in self._chunks)
        if declared != self.step_count:
            raise BundleError(
                f"the manifest contradicts itself: step_count={self.step_count}, but the chunk step counts add up to {declared}"
            )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def chunks_verified(self) -> int:
        """How many chunks have been digested. Covers only the chunks that were actually read."""
        return len(self._verified)

    @property
    def chunk_mismatches(self) -> list[str]:
        """Chunks whose digest disagrees with the manifest, in manifest order."""
        order = {c.filename: i for i, c in enumerate(self._chunks)}
        return sorted(self._mismatched, key=lambda name: order[name])

    def iter_steps(self, *, stride: int = 1, start: int = 0, stop: int | None = None) -> Iterator[RawStep]:
        """Iterate over the steps in ``[start, stop)`` with the given ``stride``.

        ``stride`` only affects how often we yield, not how much we read — a chunk has to be
        decompressed in full before any single step inside it can be reached.
        """
        if stride < 1:
            raise ValueError(f"stride must be a positive integer, got {stride}")
        stop = self.step_count if stop is None else min(stop, self.step_count)

        for chunk in self._chunks:
            chunk_stop = chunk.first_step + chunk.step_count
            if chunk_stop <= start or chunk.first_step >= stop:
                continue

            before, after, phase = self._load_chunk(chunk)
            # Align the global step index to a multiple of stride, so that the stride grid stays continuous across chunk boundaries.
            local_begin = max(start - chunk.first_step, 0)
            first_global = chunk.first_step + local_begin
            if remainder := (first_global - start) % stride:
                local_begin += stride - remainder

            local_stop = min(chunk.step_count, stop - chunk.first_step)
            for local in range(local_begin, local_stop, stride):
                yield RawStep(
                    index=chunk.first_step + local,
                    step_id=str(phase[local]),
                    state_before=before[local],
                    state_after=after[local],
                )

    def sample_steps(self, indices: Sequence[int]) -> list[RawStep]:
        """Pull out a specific set of steps, for one-off sampling such as the determinism self-check.

        The returned arrays are copies, so they are unaffected by later iteration.
        """
        wanted = sorted({i for i in indices if 0 <= i < self.step_count})
        result: list[RawStep] = []
        cursor = 0
        for chunk in self._chunks:
            chunk_stop = chunk.first_step + chunk.step_count
            hits = []
            while cursor < len(wanted) and wanted[cursor] < chunk_stop:
                hits.append(wanted[cursor])
                cursor += 1
            if not hits:
                continue
            before, after, phase = self._load_chunk(chunk)
            for global_index in hits:
                local = global_index - chunk.first_step
                result.append(
                    RawStep(
                        index=global_index,
                        step_id=str(phase[local]),
                        state_before=before[local].copy(),
                        state_after=after[local].copy(),
                    )
                )
        return result

    def _load_chunk(self, chunk: ChunkSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = self._dir / chunk.filename
        payload = path.read_bytes()

        if self._verify and chunk.sha256:
            self._verified.add(chunk.filename)
            if hashlib.sha256(payload).hexdigest() != chunk.sha256:
                self._mismatched.add(chunk.filename)

        with np.load(io.BytesIO(payload)) as archive:
            missing = [f for f in _REQUIRED_FIELDS if f not in archive.files]
            if missing:
                raise BundleError(f"{path} is missing the fields {missing}; the fields actually present are {archive.files}")
            before = archive["before"]
            after = archive["after"]
            phase = archive["phase"]

        if len(before) != chunk.step_count:
            raise BundleError(
                f"{path} actually holds {len(before)} steps, but the manifest declares {chunk.step_count}"
            )
        if self.state_size and before.shape[1] != self.state_size:
            raise BundleError(
                f"{path} has state size {before.shape[1]}, but the manifest declares {self.state_size}"
            )
        return before, after, phase


def _parse_chunks(manifest: dict, directory: Path) -> list[ChunkSpec]:
    raw = manifest.get("chunks")
    if not raw:
        raise BundleError("the recording manifest has no chunks list, so the state files cannot be located")

    chunks = [
        ChunkSpec(
            filename=str(entry["filename"]),
            first_step=int(entry["first_step"]),
            step_count=int(entry["step_count"]),
            sha256=entry.get("sha256"),
        )
        for entry in raw
    ]
    chunks.sort(key=lambda c: c.first_step)

    expected_next = 0
    for chunk in chunks:
        if chunk.first_step != expected_next:
            raise BundleError(
                f"chunk {chunk.filename} starts at step {chunk.first_step}, but the previous chunk ended at {expected_next}; the recording is not contiguous"
            )
        if not (directory / chunk.filename).is_file():
            raise BundleError(f"a chunk referenced by the manifest does not exist: {directory / chunk.filename}")
        expected_next += chunk.step_count
    return chunks
