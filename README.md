# amx

Generate lab assets, simulate a robot arm working with them, and refine both against a
physics judge.

Three things, in order:

1. **Assets.** A prompt, a protocol excerpt and a datasheet go in; an articulated MJCF part
   comes out, checked against the datasheet it claims to match.
2. **A cell and a motion.** The part, an arm, a bench and some co-designed fixtures are
   composed into a scene, and a plan of four action primitives is executed in it.
3. **Refinement.** [`sim_judge`](vendor/sim_judge) reads the recorded run, and what it
   found is routed back into the waypoints, the layout or the fixture parameters.

## Setup

```bash
scripts/setup.sh
```

Installs `uv`, pins Python 3.12, syncs the dependencies, and extracts the UR5e and
Robotiq 85 from the `robosuite` wheel into `vendor/robots/`. Put API keys in `.env` (see
`.env.example`) if you want the model-driven parts; everything deterministic runs without
them.

```bash
uv run pytest              # 158 tests, about 90 seconds
uv run python examples/wetlab_transfer/run.py --clean
```

The example is the whole pipeline on one task: compile a 1.5 mL microcentrifuge tube from an
Articraft model, ground it against its datasheet, design a staging rack and a destination
rack around the arm's measured gripper, transfer the tube from one to the other, and judge
the result. It passes on the first round, which means it also demonstrates nothing about
the loop, so:

```bash
uv run python examples/wetlab_transfer/run.py --clean --break-the-plan
```

starts from a lift that does not clear the staging well's rim. Round 0 fails — the tube
travels 0.9 mm and never reaches the destination — the repairer reads that from the
findings and raises the lift, and round 1 passes. Add `--llm` to have a model propose the
repair instead of the built-in heuristic.

## Layout

```
src/amx/
├── asset/       part 1: AssetRequest -> Articraft -> MJCF, then grounding checks
├── grounding/   the measurement core: cross-sections, cavities, staging, renders
├── search/      best-first search over candidate designs, with backtracking
├── skills/      lessons learned, written as Articraft examples and retrieved by BM25
├── sim/         part 2: scene assembly, IK, the four primitives, tracing, policy
├── codesign/    part 2c: parametric parts, DFM checks, three-format export
├── loop/        part 3: judging, finding -> repair routing, bounded iteration
├── bench/       the 3D asset benchmark: case parsing, measurement, scoring, sweeps
├── trace_export.py  trajectory.jsonl -> one directory per ReAct turn
└── cli.py       amx llm | asset | codesign | sim | judge | loop | bench | trace
vendor/
├── articraft/       upstream, plus a handful of listed seams (see PROVENANCE.md)
├── articraft_ext/   bioSIM's additions, overlaid onto the `agent` namespace
└── sim_judge/, robots/
examples/        the wetlab transfer, end to end
```

## The generation loop is Articraft's, with three more ways to fail

Stock Articraft finishes as soon as the code compiles. Compiling says the geometry is
*valid*; it says nothing about whether it is the object that was asked for, and a beaker
that is 40 mm across compiles perfectly.

Rather than wrap Articraft in a supervisor that re-runs it, the checks were added inside
its own ReAct loop, as three more tools the model can call and a stricter condition for
being allowed to stop:

| Tool | What it measures |
| --- | --- |
| `check_physical_grounding` | Every stated dimension, on the actual mesh, at the location the specification names. Diameters are the median radius about the section's centroid, so a spout or a handle does not widen them. |
| `check_protocol_grounding` | Stands it on a bench and lets it settle; drops the reference probe in; tips it over and puts it back. |
| `check_visual_grounding` | Renders four views and asks a vision model whether each stated feature is actually visible. The renders come back to the author as images, not as prose. |

The results arrive as a `<grounding_signals>` block that deliberately mirrors the
`<compile_signals>` block the model already reads, and a finish attempt is refused while
any required check is failing *or stale* — stale meaning the code has changed since the
check last ran. Freshness is what stops the obvious exploit of measuring once, then
editing.

Two details that matter more than they look:

- **The checks are derived from the task.** `GroundingSpec` is built from `input.md`
  alone, and so is the rubric it will be scored against, so there is no threshold the
  grader knows that the author was not told.
- **MuJoCo collides meshes as convex hulls.** A beaker's interior does not exist for
  contact, so "can the probe be inserted" silently passed for any solid lump. Every
  contact test runs against a CoACD convex decomposition instead. This one invalidated a
  whole family of checks before it was found.

### Search, not a single chain

`amx.search` runs the repair loop as best-first search rather than a chain: several
different candidates per expansion, a bounded beam, and backtracking when a design's
children all come out worse than it. Scores are graded rather than pass/fail, so a
dimension that is 6% out ranks above one that is 30% out and the score is something a
search can climb.

Candidates come from two places on purpose. `amx.loop.heuristics` derives a repair from
the finding's own measurements — lift the waypoint by twice the penetration depth, open
the socket by half a millimetre — and a model proposes the rest. Sampling one model twice
mostly yields one idea twice; a rule-derived patch is reliably a *different* attempt.

### Lessons are stored where the model already looks

`amx.skills` writes each confirmed repair as an Articraft example document and points
`ARTICRAFT_EXTRA_EXAMPLE_DIRS` at the directory, so lessons are retrieved by Articraft's
own BM25 index alongside its built-in examples. Nothing new had to be built for retrieval,
and both halves of the system draw on the same library — a lesson is indexed by the
failure codes it was learned from, so it reaches whichever half hits that code.

## The 3D asset benchmark

64 cases in `3D_asset_cases/`, each a single task file (`input.md`). The rubric beside it
is compiled from that file — `amx bench rubric --write` — so there is no second source of
truth to keep in step, and publishing the answer key leaks nothing the generator was not
already shown.

```bash
amx bench list                          # every case and what its rubric measures
amx bench show BEA-001                  # the derived spec next to the compiled rubric
amx bench show BEA-001 --prompt         # exactly what the generator will be sent
amx bench rubric BEA-001                # compile and print without writing
amx bench run BEA-001 --run-dir runs/bench/one
amx bench generate BEA-001 --run-dir runs/bench/one \
      --max-turns 40 --beam 3 --samples 2 --max-nodes 24
amx bench judge BEA-001 --run-dir runs/bench/one     # re-score without regenerating
amx bench sweep --workers 4 --resume --run-dir runs/bench/full
amx bench sweep --no-grounding --run-dir runs/bench/baseline   # plain Articraft
```

### Three axes, graded items, and no prose in the scoring path

A rubric item names one primitive from a closed list of eighteen and supplies its
parameters as numbers. The hundred points divide `parts` 35 / `operability` 40 /
`physics` 25, and an axis with no items shares its budget out across the ones that do, so
a beaker with nothing that moves is still marked out of 100.

A failed item costs its own weight. Missing a lid does not zero the beaker's volume;
a load that will not stay finite does not wipe the dimensions that could still be
measured. Gates on the scorecard (`G0-LOAD`, `G1-PARTS`, `G2-OPERABILITY`, `G3-PHYSICS`)
are diagnostics, not score wipes. Continuous measurements (dimensions, cavity, mass,
penetration, connectivity) decay with how far they missed; a binary miss such as "this
part is not there" is still a zero on that item, and only that item.

The measurements are `amx.grounding` and `amx.grounding.physics` — the same code the
generation loop optimises against, so the judge cannot be stricter or laxer than what the
author was held to. Scoring is **fail-closed**: an item the judge could not measure is
`blocked`, which earns nothing and stays in the denominator. The only status that does not
cost the submission points is `not_scorable`, declared when the rubric is compiled and
subtracted from the reachable total where anyone can read it. The leaderboard's `of` column
is that ceiling.

Three things this was written to stop, all of which the previous arrangement did:

- a clause nobody could measure being dropped before the rest were averaged, so it cost
  nothing
- a quantity in grams being compared against a length in millimetres, because the unit
  field was read and then ignored
- "the model has a body called `lid`" being accepted as proof that the lid opens

An asset that would not load in MuJoCo at all used to score 39 out of 100.

`scripts/calibrate_rubric.py` re-judges finished runs and checks the rubric against its
falsification criteria; `scripts/repeat_rubric.py` separates the judge's own score spread
(zero — there is no model in the scoring path) from the authoring loop's.

## The design decisions worth knowing

### The co-design agent proposes parameters, never geometry

`amx.codesign` is a library of six parametric templates — flange adapter, tube rack,
cradle, press finger, lid hook, bench clamp. A model's entire contribution is a template
name and a dictionary of numbers, validated by the template's own schema before any
geometry is attempted:

```python
class PartProposal(BaseModel):
    template: Literal["flange_adapter", "tube_rack", ...]
    params: dict[str, float]
```

The geometry itself is deterministic `trimesh` and `manifold3d`. This is the mechanism
behind "functionally stable": a proposal either fails schema validation and is rejected with
a reason, or it builds. It cannot compile to something unloadable, because it does not
compile — and `tests/test_codesign.py` sweeps every parameter of every template across its
declared range to keep that true.

Every part is checked for manufacturability before it enters a scene — wall thickness
against the process minimum, self-supporting overhang angle, watertightness and single
connectivity, hole allowance, build-volume fit — and exported three ways from one source of
truth: MJCF with convex collision volumes for simulation, STL for printing, and a JSON
datasheet with the parameters, material and tolerances for whoever makes it.

### Measure the gripper; do not describe it

Almost every hard bug in this project came from a number that was assumed instead of
measured, and the fix was always to measure it. `scripts/vendor_robots.py` therefore
calibrates the arm it vendors:

| Quantity | Why it is measured |
| --- | --- |
| Tool centre | The Robotiq's shipped `grip_site` is 4.5 mm off the pads' midpoint — a constant miss on every grasp that reads as a planning error. |
| Opening vs joint angle | So `Grip(width_m=0.0108)` means 10.8 mm on any gripper, rather than a fraction of an unknown travel. |
| Finger reach, pad reach | 31.8 mm and 29.8 mm on this gripper. Their *difference* is how far a part must stand proud of a socket for the pads to reach it while the fingertips stay clear. |
| Servo gains | From each joint's own worst-case inertia at a chosen bandwidth. One gain for the whole arm cannot work: the shoulder and the wrist differ by three orders of magnitude, so a gain that stops the shoulder sagging makes the wrist diverge within fifty steps. |
| Grip gain | From a stated grip force through the linkage's measured mechanical advantage. A gripper spends its life stalled against something, so its gain sets how hard it holds, not how well it tracks. |

### The judge's rules are generated, and scaled to the parts

`amx.sim.policy` writes `bound-operation.json` from the cell and the plan. Two rules there
are worth singling out, because both were wrong in an instructive way first:

- Only the gripping faces may touch a part; every other robot geom is named individually and
  forbidden. A blanket `robot/*` prohibition is a hard constraint that overrides the grasp
  allowance, so it reports every successful pick as a collision with the thing just picked.
- The penetration a grasp may reach is a fraction of the part's own width, not a distance.
  Two millimetres is a fifth of the way through a microcentrifuge tube and a twentieth of
  the way through a 50 mL conical.

### A run that achieved nothing must not pass

`sim_judge` answers "did anything physically implausible happen". It does not answer "did
the task happen", and it says so honestly — it emits a note when a policy declares
something it has no detector for. `amx.sim.outcome` closes that gap deterministically,
comparing the final pose against the plan's `Destination`, and `Judgement.absorb` folds the
result in so a failed objective turns a PASS into a FAIL.

Two things there are load-bearing. The outcome is measured from the part's centre of mass
rather than its root body's origin, because where an author puts their origin is their own
business — this tube's is at its tip. And a diverged run reports its outcome as *unknown*
rather than measuring it: MuJoCo resets the state when it gives up, so the final pose of a
diverged episode is the initial pose, and measuring it reports that nothing ever happened.

### The loop is bounded and keeps the best round

```python
for round in range(max_rounds):
    scene  = build_scene(workcell)
    result = run_episode(scene, plan)
    verdict = judge(result.case_dir)
    verdict.absorb(check_destination(plan, result))
    if verdict.passed:
        break
    workcell, plan, parts = apply_patch(..., propose_repair(verdict, ...))
```

Each round lands in `runs/<id>/round-NN/` as ordinary directories and JSON. Patches are
capped per round, so a confused proposal degrades a round rather than destroying the cell
the next round has to reason about.

## Command line

```bash
amx asset generate request.json --output-root assets      # author with Articraft
amx asset compile model.py --asset-id tube --output-root assets
amx asset check assets/tube request.json                  # grounding
amx codesign templates                                    # parameters and bounds
amx codesign build rack.json --out parts/rack             # rack.json is a PartProposal
amx sim build design.json --out runs/probe
amx sim run design.json --out runs/probe
amx judge runs/probe/case --stride 1
amx loop run design.json --rounds 4 --run-dir runs
amx bench run BEA-001 --run-dir runs/bench/one            # generate and score one case
amx bench sweep --workers 4 --resume                      # all 64
amx trace runs/.../traces --output runs/.../expanded      # per-turn transcript
amx llm doctor                                            # is the endpoint reachable
```

Use `--scan-stride 20` while iterating and `--stride 1` to accept: the judge scans
coarsely, then rescans every suspicious window step by step, and tells you which it did.

### What a run leaves on disk

One directory per case, and nothing you need is inside a database or a log line:

```
runs/bench/<sweep>/BEA-001/
├── request.md              exactly what the model was sent
├── grounding-spec.json     what the loop held itself to, derived from input.md
├── asset/                  model.py, the URDF, the MJCF and its meshes
├── agent/
│   ├── turns.txt           one line per turn: duration, tokens, tools called
│   └── turns/turn-007/
│       ├── request.json        model, message count, tools on offer
│       ├── response.json       duration, tokens, which tools it called
│       ├── thinking.txt        reasoning, when the provider returns any
│       ├── tool-calls.json     the calls, arguments parsed out of their JSON strings
│       ├── tool-results/       one file each, signal blocks left as readable text
│       ├── injected.txt        what the harness said that the model did not ask for
│       └── model.py            the file as it stood after this turn
├── checks/T-VIS/renders/   the images the visual review actually looked at
├── outcomes.json           per-check measurements, with every number that was read
├── scorecard.json          score, per-category split, hard gates, status
└── scorecard.txt           the same, readable
```

`turn-NNN/injected.txt` is worth knowing about: it separates the harness's own
`<compile_required>` and `<grounding_required>` prompts from the task, so a transcript
reads as a conversation rather than as someone repeatedly interrupting.

`amx trace` produces the same expansion from any Articraft `trajectory.jsonl`, including
runs made outside the benchmark.

## Known limits

- Four action primitives, no re-planning and no error recovery. If a `Move` cannot be
  solved the episode stops and says so.
- The vendored Robotiq's `inner_finger` and `inner_knuckle` collision meshes overlap 12 mm
  at rest, in robosuite's own meshes. The pair is excluded from contact so the physics is
  unaffected, but the judge measures resting interpenetration geometrically and every case
  built on this arm carries four such warnings, routed to `RepairTarget.NONE`.
- `sim_judge`'s gravity-support check counts normal force, not friction, so an object held
  in a friction grasp is flagged as insufficiently supported while it is being carried.
- The vision grounding check needs a GL context; without one it is skipped with a note
  rather than silently passing.
- `amx judge` on its own reports only what `sim_judge` detects, which is whether anything
  went wrong, not whether the task succeeded. It will pass a case in which the tube never
  moved. The objective is checked by `amx.sim.outcome` and folded into the verdict by
  `amx loop run`, so a bare `judge` verdict is a safety result and not an outcome.
