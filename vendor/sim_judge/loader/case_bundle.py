"""Layout discovery and integrity verification for an evidence directory.

A case directory is treated as a self-contained evidence bundle. This module only locates the
files and confirms that nothing has been tampered with; it interprets none of the content.
Interpretation is left to :mod:`policy`, :mod:`timeline` and :mod:`trace_reader` respectively.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class BundleError(Exception):
    """The evidence directory is incomplete, or its content disagrees with the manifest."""


@dataclass(frozen=True, slots=True)
class CaseBundle:
    """Every input the judge needs from a single case directory."""

    root: Path

    bound_operation_path: Path
    """The judging policy: contact rules, clearance rules, scene semantics, all thresholds."""

    task_execution_path: Path
    """The execution record: action segmentation, geom bindings, per-case identity digests."""

    trace_manifest_path: Path
    """The recording manifest: chunk list, state size, timestep, model digest."""

    model_path: Path
    """The MuJoCo model. The compiled .mjb is preferred, falling back to closed-scene/scene.xml."""

    model_is_compiled: bool

    @property
    def raw_dynamics_dir(self) -> Path:
        return self.trace_manifest_path.parent

    def read_bound_operation(self) -> dict[str, Any]:
        return _read_json(self.bound_operation_path)

    def read_task_execution(self) -> dict[str, Any]:
        return _read_json(self.task_execution_path)

    def read_trace_manifest(self) -> dict[str, Any]:
        return _read_json(self.trace_manifest_path)


def discover_bundle(case_dir: str | Path) -> CaseBundle:
    """Locate every file the judge needs underneath ``case_dir``.

    We raise :class:`BundleError` rather than letting downstream code trip over a plain
    ``FileNotFoundError``, so that the message can say exactly which kind of evidence is missing.
    """
    root = Path(case_dir).expanduser().resolve()
    if not root.is_dir():
        raise BundleError(f"case directory does not exist or is not a directory: {root}")

    bound_operation = _require(root / "bound-operation.json", "judging policy bound-operation.json")
    task_execution = _require(root / "task-execution.json", "execution record task-execution.json")

    raw_dynamics = root / "raw-dynamics"
    if not raw_dynamics.is_dir():
        raise BundleError(f"missing dynamics recording directory: {raw_dynamics}")
    manifest = _require(raw_dynamics / "raw-trace.json", "recording manifest raw-trace.json")

    scene_xml = root / "closed-scene" / "scene.xml"
    compiled = raw_dynamics / "compiled-model.mjb"
    if compiled.is_file():
        model_path, compiled_flag = compiled, True
    elif scene_xml.is_file():
        model_path, compiled_flag = scene_xml, False
    else:
        raise BundleError(
            f"neither the compiled model {compiled} nor the scene description {scene_xml} is present, so replay is impossible"
        )

    return CaseBundle(
        root=root,
        bound_operation_path=bound_operation,
        task_execution_path=task_execution,
        trace_manifest_path=manifest,
        model_path=model_path,
        model_is_compiled=compiled_flag,
    )


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """The outcome of integrity verification."""

    model_sha256_expected: str | None
    model_sha256_actual: str | None
    model_matches: bool | None
    """``None`` means the manifest carries no model digest, so there is nothing to check against."""

    chunk_count: int
    chunks_verified: int
    """How many chunks we actually digested. Only chunks that were read get verified, so this can be smaller than ``chunk_count``."""

    chunk_mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.model_matches is not False and not self.chunk_mismatches

    def with_chunk_results(self, verified: int, mismatches: Sequence[str]) -> IntegrityReport:
        """Fill in the chunk verification results.

        Chunk digests are computed by :class:`~sim_judge.loader.trace_reader.TraceReader` as a
        side effect of streaming, so they are only known once replay has finished. Hence the
        two-step construction.
        """
        return replace(self, chunks_verified=verified, chunk_mismatches=tuple(mismatches))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model_sha256_expected": self.model_sha256_expected,
            "model_sha256_actual": self.model_sha256_actual,
            "model_matches": self.model_matches,
            "chunk_count": self.chunk_count,
            "chunks_verified": self.chunks_verified,
            "chunk_mismatches": list(self.chunk_mismatches),
        }


def verify_model_integrity(bundle: CaseBundle) -> IntegrityReport:
    """Verify the model file's digest and count how many chunks the manifest declares.

    The chunk digests themselves are not computed here — :class:`~sim_judge.loader.trace_reader.TraceReader`
    verifies them while streaming, which avoids reading 127 MB of data twice. The results are
    folded back in afterwards via :meth:`IntegrityReport.with_chunk_results`.
    """
    manifest = bundle.read_trace_manifest()
    chunks = manifest.get("chunks") or []
    expected = (manifest.get("model_file") or {}).get("sha256")

    if expected is None or not bundle.model_is_compiled:
        return IntegrityReport(expected, None, None, len(chunks), 0, ())

    actual = sha256_of_file(bundle.model_path)
    return IntegrityReport(expected, actual, actual == expected, len(chunks), 0, ())


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _require(path: Path, what: str) -> Path:
    if not path.is_file():
        raise BundleError(f"missing {what}: {path}")
    return path


def _optional(path: Path) -> Path | None:
    return path if path.is_file() else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise BundleError(f"{path} is not valid JSON: {exc}") from exc
