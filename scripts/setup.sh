#!/usr/bin/env bash
# One-shot environment bootstrap: uv -> CPython 3.12 -> .venv -> dependencies.
#
# What this does, in order:
#   1. Installs uv if it is not already on PATH
#   2. Pins CPython 3.12 (the only version the project accepts)
#   3. Creates .venv and syncs dependencies, including the Articraft extra
#   4. Copies .env.example to .env when .env is missing (does not overwrite)
#   5. On macOS, exposes libpython next to the venv so mjpython can find it
#   6. Extracts the UR5e + Robotiq 85 meshes if they are not already vendored
#   7. Imports the packages a dry run needs, so a missing wheel fails here
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "==> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv installed but not on PATH. Open a new shell, or add \$HOME/.local/bin to PATH." >&2
    exit 1
fi

echo "==> installing CPython 3.12"
uv python install 3.12

echo "==> syncing dependencies"
uv sync --python 3.12 --extra articraft "$@"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "==> wrote .env from .env.example  (empty; put YOUR own API key in before generating)"
else
    echo "==> .env already present; leaving it alone"
fi

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
    *)      echo "note: set MUJOCO_GL=egl (or osmesa) in .env for headless renders on this platform" ;;
esac
uv run python -c "
import mujoco, numpy, trimesh, manifold3d, sim_judge
print('mujoco', mujoco.__version__)
print('numpy', numpy.__version__)
print('trimesh', trimesh.__version__)
print('sim_judge', sim_judge.__version__)
"
echo
echo "Next:"
echo "  1. Edit .env with YOUR own API key (Anthropic, OpenAI, or your gateway)."
echo "  2. uv run amx llm doctor          # can the gateway be reached"
echo "  3. uv run pytest                  # deterministic tests; no key needed"
echo "  4. uv run amx bench list          # the 64 cases"
