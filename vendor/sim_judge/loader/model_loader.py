"""MuJoCo model loading.

We prefer ``compiled-model.mjb``, the compilation artifact produced at recording time: it
already bakes in meshes, materials and every compile-time option, which guarantees that the
contact set produced during replay is identical to the one seen by the original run. We only
fall back to ``closed-scene/scene.xml`` when the .mjb is missing; recompiling from XML can
introduce subtle differences, so that fallback is flagged in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from sim_judge.loader.case_bundle import BundleError, CaseBundle


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A model together with a note on where it came from."""

    model: mujoco.MjModel
    source: str
    """Either ``"compiled_mjb"`` or ``"scene_xml"``."""

    engine_version: str
    recorded_version: str | None

    @property
    def version_matches(self) -> bool | None:
        if self.recorded_version is None:
            return None
        return self.engine_version == self.recorded_version

    def describe(self) -> dict[str, object]:
        m = self.model
        return {
            "source": self.source,
            "engine_version": self.engine_version,
            "recorded_version": self.recorded_version,
            "version_matches": self.version_matches,
            "timestep_s": float(m.opt.timestep),
            "nq": int(m.nq),
            "nv": int(m.nv),
            "nu": int(m.nu),
            "nbody": int(m.nbody),
            "ngeom": int(m.ngeom),
            "njnt": int(m.njnt),
            "neq": int(m.neq),
        }


def load_model(bundle: CaseBundle) -> LoadedModel:
    manifest = bundle.read_trace_manifest()
    recorded_version = manifest.get("mujoco_version")

    try:
        if bundle.model_is_compiled:
            model = mujoco.MjModel.from_binary_path(str(bundle.model_path))
            source = "compiled_mjb"
        else:
            model = mujoco.MjModel.from_xml_path(str(bundle.model_path))
            source = "scene_xml"
    except (ValueError, RuntimeError) as exc:
        raise BundleError(f"cannot load model {bundle.model_path}: {exc}") from exc

    _check_state_layout(model, manifest)
    return LoadedModel(model, source, mujoco.__version__, recorded_version)


def _check_state_layout(model: mujoco.MjModel, manifest: dict) -> None:
    """Confirm that the model's state vector size matches the recording.

    A size mismatch means the model and the trace simply are not the same artifact pair.
    Replaying anyway would produce plausible-looking nonsense, so we raise instead of warning.
    """
    recorded_size = manifest.get("state_size")
    if recorded_size is None:
        return
    actual_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    if int(recorded_size) != int(actual_size):
        raise BundleError(
            f"state size mismatch: the recording declares {recorded_size}, the current model has {actual_size}. "
            "The model and the trace do not come from the same compilation, so replay is impossible."
        )
