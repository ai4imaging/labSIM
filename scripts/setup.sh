#!/usr/bin/env bash
# One-shot environment bootstrap: uv -> CPython 3.12 -> .venv -> dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "==> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "==> installing CPython 3.12"
uv python install 3.12

echo "==> syncing dependencies"
uv sync --python 3.12 --extra articraft "$@"

# `mujoco.viewer.launch_passive` has to run under `mjpython` on macOS.  uv's
# standalone CPython keeps libpython in the base installation while mjpython
# looks beside the virtual environment, so expose the existing library there.
if [ "$(uname -s)" = "Darwin" ]; then
    PYTHON_DYLIB="$(uv run python -c \
        'import os, sysconfig; print(os.path.join(sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY")))')"
    ln -sf "$PYTHON_DYLIB" ".venv/$(basename "$PYTHON_DYLIB")"
fi

# The robot is extracted from the robosuite wheel and is not in git. Re-extracting
# it costs a download, so only do it when it is actually missing.
if [ -f vendor/robots/ur5e_robotiq85/arm.xml ]; then
    echo "==> robot assets already vendored"
else
    echo "==> vendoring robot assets"
    uv run python scripts/vendor_robots.py
fi

echo
echo "Done. Activate with:  source $ROOT/.venv/bin/activate"
# MUJOCO_GL is platform-specific and getting it wrong fails at render time rather
# than at import: Linux needs an explicit egl/osmesa, macOS has neither and picks
# CGL by itself only when the variable is absent.
case "$(uname -s)" in
    Darwin) echo "note: leave MUJOCO_GL unset on macOS; MuJoCo uses CGL for offscreen renders" ;;
    *)      echo "note: set MUJOCO_GL=egl (or osmesa) for headless renders on this platform" ;;
esac
uv run python -c "
import mujoco, numpy, trimesh, manifold3d, sim_judge
print('mujoco', mujoco.__version__)
print('numpy', numpy.__version__)
print('trimesh', trimesh.__version__)
print('sim_judge', sim_judge.__version__)
"
