"""Recording an episode in the format `sim_judge` replays.

One `mjSTATE_INTEGRATION` vector per step, before and after, plus the protocol step the
step belongs to. States are buffered and flushed to `steps-NNN.npz` in fixed-size chunks so
that memory stays flat regardless of episode length, and each chunk's digest goes into the
manifest so a replay can tell whether it is reading the bytes that were written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

CHUNK_STEPS = 2000
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass
class TraceRecorder:
    """Accumulates per-step states and writes a `raw-dynamics/` directory."""

    model: mujoco.MjModel
    output_dir: Path
    chunk_steps: int = CHUNK_STEPS

    _before: list[np.ndarray] = field(default_factory=list, init=False)
    _after: list[np.ndarray] = field(default_factory=list, init=False)
    _phase: list[str] = field(default_factory=list, init=False)
    _chunks: list[dict[str, Any]] = field(default_factory=list, init=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_size = int(mujoco.mj_stateSize(self.model, STATE_SPEC))
        self._scratch = np.zeros(self.state_size)

    def state(self, data: mujoco.MjData) -> np.ndarray:
        mujoco.mj_getState(self.model, data, self._scratch, STATE_SPEC)
        return self._scratch.copy()

    def record(self, before: np.ndarray, after: np.ndarray, step_id: str) -> None:
        self._before.append(before)
        self._after.append(after)
        self._phase.append(step_id)
        self._step_count += 1
        if len(self._before) >= self.chunk_steps:
            self._flush()

    @property
    def step_count(self) -> int:
        return self._step_count

    def _flush(self) -> None:
        if not self._before:
            return
        first_step = self._step_count - len(self._before)
        filename = f"steps-{len(self._chunks):04d}.npz"
        path = self.output_dir / filename
        np.savez_compressed(
            path,
            before=np.asarray(self._before, dtype=np.float64),
            after=np.asarray(self._after, dtype=np.float64),
            phase=np.asarray(self._phase, dtype=np.str_),
        )
        self._chunks.append(
            {
                "filename": filename,
                "first_step": first_step,
                "step_count": len(self._before),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        self._before.clear()
        self._after.clear()
        self._phase.clear()

    def finish(self, *, case_sha256: str, extra: dict[str, Any] | None = None) -> Path:
        """Flush the tail, save a compiled model beside it, and write the manifest."""
        self._flush()
        model_path = self.output_dir / "compiled-model.mjb"
        mujoco.mj_saveModel(self.model, str(model_path), None)

        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "step_count": self._step_count,
            "timestep_s": float(self.model.opt.timestep),
            "state_size": self.state_size,
            "state_spec": "mjSTATE_INTEGRATION",
            "identity": {"evaluation_case_sha256": case_sha256},
            "model_file": {
                "filename": model_path.name,
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            },
            "chunks": self._chunks,
        }
        if extra:
            manifest.update(extra)
        manifest_path = self.output_dir / "raw-trace.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest_path
