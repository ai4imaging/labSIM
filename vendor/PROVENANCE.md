# Vendored trees

## `articraft/`

Verbatim copy of the official Articraft repository at `/home/yez/articraft`
(`github.com/mattzh72/articraft`, commit `59eb5e0ed`), minus `.git/`, `tests/`,
`viewer/web/`, `data/` and cache directories.

Provides the geometry SDK (`sdk/`), the LLM authoring harness (`agent/harness.py`,
`agent/single_run.py`, `agent/prompts/`, `agent/tools/`, `agent/providers/`), the record
compiler (`agent/compiler.py`) and record storage (`storage/`).

### Local modifications

No longer verbatim. Every change is additive and is listed here so a re-vendoring can be
replayed. `tests/test_vendor_seams.py` asserts each one is still in place, so dropping in
a fresh upstream copy fails loudly rather than silently reverting the integration.

The rule the edits follow: this tree only ever gains a *seam*. Anything with logic in it
lives in `articraft_ext/` or in `src/amx/`, so that a future upstream release can be
dropped in and these seams re-cut by hand in an afternoon.

| File | Change |
| --- | --- |
| `articraft/values.py` | `ProviderName.GPUGEEK`, and a vendor-prefix rule in `infer_provider_from_model_id` ahead of the generic `/` rule, which would otherwise read every GpuGeek model id as OpenRouter |
| `agent/providers/factory.py` | `gpugeek` constructor, default-model and credential branches |
| `agent/harness.py` | `GpuGeekLLM` in the `ProviderConstructors` table; a no-op `_maybe_inject_tool_attachments` hook called at the end of the turn loop, so a subclass can hand the model images that a `role: tool` message cannot carry |
| `agent/tools/__init__.py` | `grounding=` flag on `build_tool_registry` that appends the three grounding tools; GpuGeek's image MIME allow-list; `TURN_BUDGET_GUIDANCE` interpolated into the shared first-turn guidance, which otherwise says "make small focused edits" and reads as advice about edit count |
| `agent/tools/probe_model/description.py` | one line asking for a dict covering every measurement the model currently wants. `probe_model` already answered in that shape; nothing asked it to |
| `agent/workspace_docs.py` | four more paths in `_DEFAULT_PRELOAD_PATHS`. They were each fetched with `read_file` in nearly every run, which spent 190 turns across 58 cases collecting the same four files |
| `agent/tools/edit_code.py` | `replace` accepts an `edits` array as well as a single `old_string`/`new_string` pair, and its result says the geometry is not built yet. Every decision is in `agent/batch_edit.py`; this file gains the field, the schema fragment and two calls. `_validate_python_syntax` was also lifted out of `EditCodeInvocation` into a module-level function so the batch path can reach it |
| `agent/harness.py` (second seam) | the two `replace` pre-checks about an empty `model.py` read the first edit from either call form. Reading `old_string` alone saw `None` for a batch and mistook seeding a blank file for editing one |
| `agent/single_run.py` | `agent_cls` threaded through `run_from_input` and `run_from_input_impl`, and widened from `type[ArticraftAgent]` to a callable so a pre-bound `functools.partial` can carry a subclass's own configuration |
| `agent/runner.py` | `_execute_single_run` now defaults `agent_cls` instead of forcing it. Forcing it made the parameter `single_run` already accepted unreachable from outside, so the row above did nothing until this changed |
| `agent/traces.py` | `write_llm_request` and `write_llm_result` events. The trace already held every message but nothing marked where one turn ended, which model answered, or which tools were on offer — the three things a per-turn export needs |
| `sdk/_profiles.py` | GpuGeek maps to the OpenRouter prompt, which is the one written for the replace/write_file tool surface it shares |
| `agent/prompts/sections/grounding_tools.md` | New. Not compiled into any of the six generated prompts: the grounding tools are conditional on a run having a spec, so `GroundedArticraftAgent` appends this section itself when they are present |

## `articraft_ext/agent/`

Three modules that exist only in the private working tree at
`/home/yez/articraft-mujoco/articraft-work/articraft/agent/`, not in the official
repository. They are the MJCF export path, which Articraft itself does not ship:

| File | Role |
| --- | --- |
| `mujoco_export.py` | `export_record_to_mjcf()`: compiled URDF + model globals -> MJCF, meshes, controller contract |
| `physical_spec.py` | `PhysicalSpec`: overrides URDF inertial and joint dynamics from a datasheet |
| `manufacturer_spec_catalog.py` | Datasheet catalogue that `physical_spec` reads |

They are kept in a separate tree so the official/private boundary stays visible.
`agent/__init__.py` here re-extends the package path, so `import agent.mujoco_export`
resolves from this tree while `import agent.compiler` falls through to `articraft/`.

### bioSIM additions

Written for this project, in the same tree for the same reason: the overlay lets them
live inside the `agent` namespace without forking anything.

| File | Role |
| --- | --- |
| `providers/gpugeek.py` | OpenAI-compatible chat-completions client for the GpuGeek gateway. Modelled on `dashscope.py`, which is the only vendored provider that both allows a base-URL override and speaks chat completions rather than the Responses API. Falls the run down to a weaker model when the one in use has no capacity. Retries — rather than downgrades — a 400 that reports the upstream account cannot serve the model at all: the gateway spreads one model over several accounts, so the next attempt is routed afresh, and treating it as our own bad request cost a sweep three of its first eight cases on turn 1 |
| `grounding_tools.py` | Schemas for `check_physical_grounding`, `check_protocol_grounding` and `check_visual_grounding`. Harness-intercepted, like `compile_model` |
| `batch_edit.py` | All-or-nothing application of several `replace` edits in one call, and the wording that tells the model an edit has not built the geometry. Stock `replace` costs one turn per edit, which was 44.5% of a hundred-turn budget |
| `grounded_harness.py` | `GroundedArticraftAgent`: runs those checks, feeds renders back as images, and refuses a finish attempt while any required check is missing, stale or failing. The measurement itself is `amx.grounding`, which the benchmark judge also uses. Carries the circuit breaker: an identical failure set refused three times running stops blocking, because a specification the geometry cannot satisfy otherwise eats the whole turn budget. A repeat only counts when the revision moved between refusals, so resubmitting untouched code cannot be used to opt out of a check. Because that count restarts whenever the failure set changes, a second guard catches a run that keeps failing differently: past eight refusals of any kind, whatever failure keeps recurring across them stops blocking, which is what one run needed after spending 61% of its turns pressing finish and submitting nothing |

`providers/__init__.py` re-extends the path a second time so `providers/gpugeek.py` can
sit beside the official providers and import from them.

The other 47 private modules in that working tree (the P03 wetlab layer) are deliberately
not vendored; `amx.sim` and `amx.codesign` replace them.

## `sim_judge/`

Copy of `/home/yez/sim_judge`. Refresh with `scripts/sync_sim_judge.sh`.

## `robots/`

Generated, not checked in. `scripts/vendor_robots.py` extracts the arm and gripper MJCF
plus meshes from the `robosuite` wheel on PyPI.

### Known property of the Robotiq 85 as shipped

Each finger's `inner_finger` and `inner_knuckle` collision meshes overlap by 12 mm at rest.
It is in robosuite's meshes, not in the vendoring: the two links share the space where the
real linkage's pivot is, and their convex hulls therefore intersect. The pair is listed in
`contact/exclude`, so MuJoCo never generates a contact from it and the physics is unaffected
— but `sim_judge` measures resting interpenetration geometrically, so every case built on
this arm carries four `S6_RESTING_GEOMETRY_INTERPENETRATION` warnings. `amx.loop.judge`
routes them to `RepairTarget.NONE`, since nothing in the design can act on them.
