# labSIM

64 labware cases in, a grounded 3D asset and a scorecard out.

An LLM writes CadQuery. The same measurements used in the loop grade the result.
Tests need no API key. Generation needs **your** key — this repo ships none.

## Setup

macOS or Linux. Python 3.12 is installed by the script. Do not use 3.11 or 3.13.

```bash
git clone https://github.com/ai4imaging/labSIM.git
cd labSIM
./setup.sh
```

That installs `uv`, creates `.venv`, copies `.env.example` → `.env` (never overwrites),
and vendors robot meshes. If `uv` is missing after install, add `~/.local/bin` to `PATH`.

Edit `.env` with **your** credentials. Pick one:

```bash
# Anthropic
ANTHROPIC_API_KEY=
AMX_LLM_PROVIDER=anthropic
AMX_LLM_MODEL=claude-sonnet-4-5
ARTICRAFT_MODEL=claude-sonnet-4-5

# OpenAI
OPENAI_API_KEY=
AMX_LLM_PROVIDER=openai
AMX_LLM_MODEL=gpt-4.1
ARTICRAFT_MODEL=gpt-4.1

# Your OpenAI-compatible gateway (no default URL)
GPUGEEK_API_KEY=
GPUGEEK_BASE_URL=https://YOUR-ENDPOINT/v1
AMX_LLM_PROVIDER=gpugeek
AMX_LLM_MODEL=
GPUGEEK_MODEL=
ARTICRAFT_MODEL=
```

Linux: `MUJOCO_GL=egl`. macOS: leave `MUJOCO_GL` unset. Never commit `.env`.

```bash
uv run amx llm doctor    # can your endpoint be reached
uv run pytest            # no key needed
uv run amx bench list    # 64 cases
```

## Run

```bash
uv run amx bench run BEA-001 --run-dir runs/bench/one --max-turns 60
uv run amx view BEA-001 --run-dir runs/bench/one --animate
uv run amx bench sweep --workers 2 --resume --max-turns 60 --run-dir runs/bench/full
```

Empty `checks/` means no asset was submitted. `--resume` skips cases that already have
`scorecard.json`.

```bash
uv run amx bench show BEA-001          # spec + rubric
uv run amx bench show BEA-001 --prompt # what the model is sent
uv run amx bench judge BEA-001 --run-dir runs/bench/one
```

## Layout

```
setup.sh            → scripts/setup.sh
.env.example        every env var; setup copies it
src/amx/            generate, ground, score, CLI
3D_asset_cases/     64 × input.md + rubric.json
vendor/             Articraft + grounding tools + sim_judge
```

The loop will not finish on a compile alone. Three tools run on the current revision —
physical, protocol, visual — and a finish is refused while any required check is stale
or failing. Rubric and GroundingSpec are compiled from the same `input.md`.
