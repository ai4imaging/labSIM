"""The command line.

One verb per stage, so each can be run and inspected on its own. That matters more than it
sounds: the failure you are chasing is almost always in one stage, and being able to build
a scene without running it, or judge a case without rebuilding anything, is the difference
between a two-second check and a two-minute one.

Everything the commands read and write is either a JSON document matching a pydantic model
in this package or an ordinary directory. There is no database and no state carried between
invocations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from amx.paths import PROJECT_ROOT, RUNS_DIR


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 2
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 — the CLI reports, it does not traceback
        print(f"error: {error}", file=sys.stderr)
        return 1


def _load_dotenv() -> None:
    """Read `.env` from the project root, without overriding an exported value.

    Credentials in the environment beat the file, so a one-off `GPUGEEK_MODEL=... amx ...`
    does what it looks like it does.
    """
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        return
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amx",
        description="Generate lab assets, simulate an arm working with them, and refine both.",
    )
    sub = parser.add_subparsers(dest="command")

    _llm_commands(sub.add_parser("llm", help="Inspect the configured model endpoint"))
    _asset_commands(sub.add_parser("asset", help="Part 1: generate and ground assets"))
    _codesign_commands(sub.add_parser("codesign", help="Part 2c: parametric fixtures and tools"))
    _sim_commands(sub.add_parser("sim", help="Part 2: compose a cell and run a plan"))
    _judge_command(sub.add_parser("judge", help="Part 3: judge a recorded case"))
    _loop_commands(sub.add_parser("loop", help="Part 3: the bounded refinement loop"))
    _bench_commands(sub.add_parser("bench", help="Score part 1 against the 3D asset benchmark"))
    _trace_command(sub.add_parser("trace", help="Expand an agent trace into per-turn files"))
    _view_command(sub.add_parser("view", help="Open a generated asset in MuJoCo and move it"))
    return parser


# --------------------------------------------------------------------------- #
# amx llm
# --------------------------------------------------------------------------- #


def _llm_commands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="subcommand")

    doctor = sub.add_parser("doctor", help="Can the endpoint be reached, and what does it serve")
    doctor.add_argument("--provider", default=None, help="Defaults to AMX_LLM_PROVIDER")
    doctor.add_argument("--timeout", type=float, default=20.0)
    doctor.set_defaults(handler=_llm_doctor)


def _llm_doctor(args: argparse.Namespace) -> int:
    import os

    from amx.llm import probe

    provider = args.provider or os.environ.get("AMX_LLM_PROVIDER", "anthropic")
    result = probe(provider, timeout=args.timeout)
    print(result.to_text())
    return 0 if result.reachable and result.key_present else 1


# --------------------------------------------------------------------------- #
# amx asset
# --------------------------------------------------------------------------- #


def _asset_commands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="subcommand")

    generate = sub.add_parser("generate", help="Author an asset with Articraft, then compile it")
    generate.add_argument("request", type=Path, help="An AssetRequest as JSON")
    generate.add_argument("--output-root", type=Path, default=None)
    generate.add_argument("--provider", default="anthropic")
    generate.add_argument("--model-id", default=None)
    generate.set_defaults(handler=_asset_generate)

    compile_ = sub.add_parser("compile", help="Compile an existing Articraft model.py to MJCF")
    compile_.add_argument("model", type=Path, help="Path to model.py")
    compile_.add_argument("--asset-id", required=True)
    compile_.add_argument("--output-root", type=Path, default=None)
    compile_.set_defaults(handler=_asset_compile)

    check = sub.add_parser("check", help="Run the grounding checks against a compiled asset")
    check.add_argument("bundle", type=Path, help="The asset directory")
    check.add_argument("request", type=Path, help="The AssetRequest it was built from")
    check.add_argument("--vision", action="store_true", help="Include the LLM visual check")
    check.add_argument("--json", type=Path, default=None)
    check.set_defaults(handler=_asset_check)


def _asset_generate(args: argparse.Namespace) -> int:
    from amx.asset.generate import generate_asset
    from amx.asset.spec import AssetRequest

    request = AssetRequest.model_validate(_read_json(args.request))
    bundle = generate_asset(
        request,
        provider=args.provider,
        model_id=args.model_id,
        output_root=args.output_root,
    )
    print(f"{bundle.asset_id}: {bundle.mjcf_path}")
    return 0


def _asset_compile(args: argparse.Namespace) -> int:
    from amx.asset.generate import bundle_from_model

    bundle = bundle_from_model(
        args.model, asset_id=args.asset_id, output_root=args.output_root
    )
    print(f"{bundle.asset_id}: {bundle.mjcf_path}")
    return 0


def _asset_check(args: argparse.Namespace) -> int:
    from amx.asset.generate import AssetBundle
    from amx.asset.grounding import ground_asset
    from amx.asset.spec import AssetRequest

    bundle = AssetBundle.load(args.bundle)
    request = AssetRequest.model_validate(_read_json(args.request))
    report = ground_asset(bundle, request, vision_blocks=args.vision)
    print(report.to_text())
    if args.json:
        report.write(args.json)
    return 0 if report.passed else 1


# --------------------------------------------------------------------------- #
# amx codesign
# --------------------------------------------------------------------------- #


def _codesign_commands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="subcommand")

    templates = sub.add_parser("templates", help="List the part templates and their parameters")
    templates.add_argument("--template", default=None, help="Show one template's JSON schema")
    templates.set_defaults(handler=_codesign_templates)

    build = sub.add_parser("build", help="Build a part from an explicit parameter set")
    build.add_argument("spec", type=Path, help="A PartProposal as JSON")
    build.add_argument("--out", type=Path, required=True)
    build.set_defaults(handler=_codesign_build)

    propose = sub.add_parser("propose", help="Ask a model to size a part for a stated need")
    propose.add_argument("need", help="What the part has to do")
    propose.add_argument("--template", default=None, help="Pin the mechanism")
    propose.add_argument("--out", type=Path, required=True)
    propose.add_argument("--context", default="")
    propose.set_defaults(handler=_codesign_propose)


def _codesign_templates(args: argparse.Namespace) -> int:
    from amx.codesign.parts import TEMPLATES, template_schema
    from amx.codesign.propose import _describe

    if args.template:
        print(json.dumps(template_schema(args.template), indent=2))
        return 0
    for name in sorted(TEMPLATES):
        print(_describe(name))
        print()
    return 0


def _codesign_build(args: argparse.Namespace) -> int:
    from amx.codesign.propose import PartProposal, realise

    proposal = PartProposal.model_validate(_read_json(args.spec))
    _, geometry, report = realise(proposal, args.out)
    assert geometry is not None
    print(report.to_text())
    print(f"\nwrote {args.out}  ({geometry.mass_kg * 1000:.1f} g)")
    return 0 if report.passed else 1


def _codesign_propose(args: argparse.Namespace) -> int:
    from amx.codesign.propose import propose_part, realise

    proposal = propose_part(need=args.need, template=args.template, context=args.context)
    print(f"{proposal.template} `{proposal.part_id}`: {proposal.rationale}\n")
    _, geometry, report = realise(proposal, args.out)
    assert geometry is not None
    (args.out / "proposal.json").write_text(
        json.dumps(proposal.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    )
    print(report.to_text())
    print(f"\nwrote {args.out}  ({geometry.mass_kg * 1000:.1f} g)")
    return 0 if report.passed else 1


# --------------------------------------------------------------------------- #
# amx sim
# --------------------------------------------------------------------------- #


def _sim_commands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="subcommand")

    build = sub.add_parser("build", help="Compose a design into a loadable scene")
    build.add_argument("design", type=Path, help="A Design as JSON")
    build.add_argument("--out", type=Path, required=True)
    build.set_defaults(handler=_sim_build)

    run = sub.add_parser("run", help="Compose a design and execute its plan")
    run.add_argument("design", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--case-id", default="case")
    run.set_defaults(handler=_sim_run)


def _sim_build(args: argparse.Namespace) -> int:
    from amx.loop.loop import Design, materialise

    design = Design.read(args.design)
    built, dfm = materialise(design, args.out)
    print(dfm.to_text())
    print(f"\nscene: {built.scene_path}")
    return 0 if dfm.passed else 1


def _sim_run(args: argparse.Namespace) -> int:
    from amx.loop.loop import Design, materialise
    from amx.sim.run import run_episode

    design = Design.read(args.design)
    built, dfm = materialise(design, args.out)
    if not dfm.passed:
        print(dfm.to_text())
        return 1
    episode = run_episode(built, design.plan, args.out / "case", case_id=args.case_id)
    print(episode.summary())
    print(f"\ncase: {episode.case_dir}")
    return 0


# --------------------------------------------------------------------------- #
# amx judge
# --------------------------------------------------------------------------- #


def _judge_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("case", type=Path, help="A sim_judge case bundle")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Re-hash the recording")
    parser.add_argument("--json", type=Path, default=None)
    parser.set_defaults(handler=_judge)


def _judge(args: argparse.Namespace) -> int:
    from amx.loop.judge import judge

    result = judge(
        args.case,
        stride=args.stride,
        max_steps=args.max_steps,
        strict=args.strict,
        verify_integrity=args.verify,
    )
    print(f"{result.verdict}: {result.reason}\n")
    print(result.report.to_text())
    targets = result.targets()
    if targets:
        print("\nrepair targets: " + ", ".join(t.value for t in targets))
    if args.json:
        result.write(args.json)
    return 0 if result.passed else 1


# --------------------------------------------------------------------------- #
# amx loop
# --------------------------------------------------------------------------- #


def _loop_commands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="subcommand")

    run = sub.add_parser("run", help="Refine a design until it passes or the budget runs out")
    run.add_argument("design", type=Path)
    run.add_argument("--run-dir", type=Path, default=None)
    run.add_argument("--run-id", default=None)
    run.add_argument("--rounds", type=int, default=4)
    run.add_argument("--scan-stride", type=int, default=20)
    run.add_argument("--max-steps", type=int, default=None)
    run.add_argument("--strict", action="store_true")
    run.add_argument("--context", default="")
    run.set_defaults(handler=_loop_run)


def _loop_run(args: argparse.Namespace) -> int:
    from amx.loop.loop import Design, LoopConfig, run_loop

    design = Design.read(args.design)
    run_dir = args.run_dir or RUNS_DIR / (args.run_id or _timestamp_id())
    result = run_loop(
        design,
        run_dir,
        config=LoopConfig(
            max_rounds=args.rounds,
            scan_stride=args.scan_stride,
            max_steps=args.max_steps,
            strict=args.strict,
        ),
        context=args.context,
    )
    print(result.summary())
    return 0 if result.passed else 1


# --------------------------------------------------------------------------- #
# amx bench
# --------------------------------------------------------------------------- #

CASES_DIR = PROJECT_ROOT / "3D_asset_cases"


def _bench_commands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="subcommand")
    parser.add_argument("--cases", type=Path, default=CASES_DIR, help="The case corpus")

    listing = sub.add_parser("list", help="Every case, with what its rubric checks")
    listing.add_argument("--verbose", action="store_true")
    listing.set_defaults(handler=_bench_list)

    show = sub.add_parser("show", help="One case's generation prompt and derived checks")
    show.add_argument("case")
    show.add_argument("--prompt", action="store_true", help="Print the full authoring prompt")
    show.set_defaults(handler=_bench_show)

    prepare = sub.add_parser("prepare", help="Write the generation inputs without generating")
    prepare.add_argument("case")
    prepare.add_argument("--run-dir", type=Path, default=None)
    prepare.set_defaults(handler=_bench_prepare)

    generate = sub.add_parser("generate", help="Generate one case's asset")
    _generation_arguments(generate)
    generate.add_argument("case")
    generate.set_defaults(handler=_bench_generate)

    judge = sub.add_parser("judge", help="Score an already-generated case")
    judge.add_argument("case")
    judge.add_argument("--run-dir", type=Path, default=None)
    judge.add_argument("--no-vision", action="store_true", help="Skip the visual review")
    judge.set_defaults(handler=_bench_judge)

    run = sub.add_parser("run", help="Generate and score one case")
    _generation_arguments(run)
    run.add_argument("case")
    run.add_argument("--no-vision", action="store_true")
    run.set_defaults(handler=_bench_run)

    sweep = sub.add_parser("sweep", help="Generate and score many cases")
    _generation_arguments(sweep)
    sweep.add_argument("--only", nargs="*", default=None, help="Case ids; default is all")
    sweep.add_argument("--limit", type=int, default=None)
    sweep.add_argument("--workers", type=int, default=1)
    sweep.add_argument("--resume", action="store_true", help="Skip cases already scored")
    sweep.add_argument("--judge-only", action="store_true", help="Score what is on disk")
    sweep.add_argument("--no-vision", action="store_true")
    sweep.set_defaults(handler=_bench_sweep)

    rubric = sub.add_parser("rubric", help="Compile input.md into a machine-checkable rubric")
    rubric.add_argument("case", nargs="?", default=None, help="One case; default is all of them")
    rubric.add_argument("--write", action="store_true", help="Save rubric.json next to the case")
    rubric.set_defaults(handler=_bench_rubric)


def _generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--provider", default="gpugeek")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--thinking-level", default="medium")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="Baseline: plain Articraft, which stops as soon as the code compiles",
    )
    parser.add_argument(
        "--concise-prompt",
        action="store_true",
        help="Use one AI-compressed paragraph from input.md instead of the full specification",
    )
    parser.add_argument("--beam", type=int, default=3, help="Frontier width when searching")
    parser.add_argument("--samples", type=int, default=1, help="Candidates per expansion")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1,
        help="Best-first search budget; 1 (default) is a single Articraft run",
    )


def _bench_settings(args: argparse.Namespace):
    from amx.bench.run import GenerationSettings

    return GenerationSettings(
        provider=args.provider,
        model_id=args.model_id,
        thinking_level=args.thinking_level,
        max_turns=args.max_turns,
        grounding=not args.no_grounding,
        concise_prompt=args.concise_prompt,
        samples=args.samples,
        beam=args.beam,
        max_nodes=args.max_nodes,
    )


def _bench_run_dir(args: argparse.Namespace) -> Path:
    """Where a sweep writes. `results_<local time>` unless the caller names the directory.

    A run is identified by when it happened, not by whatever the framework was called at
    the time: names like `newframework100-opus5` age badly and say nothing a scorecard
    does not already record.
    """
    if args.run_dir:
        return args.run_dir
    return RUNS_DIR / "bench" / (getattr(args, "run_id", None) or f"results_{_timestamp_id()}")


def _bench_list(args: argparse.Namespace) -> int:
    from amx.bench.case import discover
    from amx.bench.compiler import load_rubric

    cases = discover(args.cases)
    for case in cases:
        spec = case.to_grounding_spec()
        rubric = load_rubric(case)
        print(
            f"{case.case_id:<10} {case.asset_class[:30]:<30} "
            f"{len(rubric.items):>3} items  {rubric.achievable():>5.1f} reachable  "
            f"{len(spec.dimensions)} known dims  {len(spec.components)} components"
        )
        if args.verbose:
            for item in rubric.items:
                print(f"    {item.axis[:4]:<5}{item.id:<22} {item.primitive:<22} "
                      f"{item.weight:>4.1f}w{'  critical' if item.critical else ''}")
    print(f"\n{len(cases)} case(s)")
    return 0


def _bench_show(args: argparse.Namespace) -> int:
    from amx.bench.case import load_case

    case = load_case(args.cases, args.case)
    spec = case.to_grounding_spec()
    if args.prompt:
        print(case.to_asset_request())
        return 0
    print(f"{case.case_id} — {case.asset_class}\n")
    print("Derived grounding spec (from input.md only):")
    print(json.dumps(spec.model_dump(mode="json", exclude_none=True), indent=2)[:4000])

    from amx.bench.compiler import load_rubric

    rubric = load_rubric(case)
    print(f"\nRubric ({len(rubric.items)} items, {rubric.achievable():.1f} reachable):")
    for item in rubric.items:
        print(f"  {item.axis[:4]:<5}{item.id:<22} {item.primitive:<22} {item.weight:>4.1f}w"
              f"{'  critical' if item.critical else ''}  {item.subject[:40]}")
    for gate in rubric.gates:
        print(f"  gate {gate.id:<16} {', '.join(gate.item_ids) or '-'}")
    return 0


def _bench_prepare(args: argparse.Namespace) -> int:
    from amx.bench.case import load_case
    from amx.bench.run import prepare

    case = load_case(args.cases, args.case)
    directory = prepare(case, _bench_run_dir(args) / case.case_id)
    print(f"{case.case_id}: {directory}")
    return 0


def _bench_generate(args: argparse.Namespace) -> int:
    from amx.bench.case import load_case
    from amx.bench.run import generate

    case = load_case(args.cases, args.case)
    directory = generate(
        case, _bench_run_dir(args) / case.case_id, settings=_bench_settings(args)
    )
    print(f"{case.case_id}: {directory}")
    return 0


def _bench_judge(args: argparse.Namespace) -> int:
    from amx.bench.case import load_case
    from amx.bench.run import judge

    case = load_case(args.cases, args.case)
    card = judge(
        case,
        _bench_run_dir(args) / case.case_id,
        vision=not args.no_vision,
    )
    print((_bench_run_dir(args) / case.case_id / "scorecard.txt").read_text())
    return 0 if card.certified else 1


def _bench_rubric(args: argparse.Namespace) -> int:
    from amx.bench.case import discover, load_case
    from amx.bench.compiler import build_rubric, write_case_artifacts

    cases = [load_case(args.cases, args.case)] if args.case else discover(args.cases)
    for case in cases:
        if args.write:
            written = write_case_artifacts(case)
            rubric = build_rubric(case)
            print(
                f"{case.case_id:<10} {len(rubric.items):>3} items  "
                f"{rubric.achievable():>5.1f} reachable  -> {written['rubric']}"
            )
            continue
        rubric = build_rubric(case)
        if len(cases) == 1:
            print(json.dumps(rubric.model_dump(mode="json"), indent=2, ensure_ascii=False))
        else:
            unscorable = sum(1 for item in rubric.items if not item.scorable)
            print(
                f"{case.case_id:<10} {len(rubric.items):>3} items "
                f"({unscorable} not scorable)  {rubric.achievable():>5.1f} reachable"
            )
    return 0


def _bench_run(args: argparse.Namespace) -> int:
    from amx.bench.case import load_case
    from amx.bench.run import run_case

    case = load_case(args.cases, args.case)
    card = run_case(
        case,
        _bench_run_dir(args),
        settings=_bench_settings(args),
        vision=not args.no_vision,
    )
    print(card.to_text())
    return 0 if card.certified else 1


def _bench_sweep(args: argparse.Namespace) -> int:
    from amx.bench.case import discover
    from amx.bench.run import sweep

    cases = discover(args.cases)
    if args.only:
        wanted = {item.lower() for item in args.only}
        cases = [c for c in cases if c.case_id.lower() in wanted or c.slug.lower() in wanted]
    if args.limit:
        cases = cases[: args.limit]

    run_dir = _bench_run_dir(args)
    print(f"{len(cases)} case(s) into {run_dir}  workers={args.workers}", flush=True)
    cards = sweep(
        cases,
        run_dir,
        settings=_bench_settings(args),
        workers=args.workers,
        resume=args.resume,
        generate_asset_first=not args.judge_only,
        vision=not args.no_vision,
    )
    print((run_dir / "leaderboard.txt").read_text())
    return 0 if any(card.certified for card in cards) else 1


# --------------------------------------------------------------------------- #
# amx trace
# --------------------------------------------------------------------------- #


def _trace_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("trace", type=Path, help="A directory holding trajectory.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(handler=_trace_export)


def _trace_export(args: argparse.Namespace) -> int:
    from amx.trace_export import export_trace, read_trace

    output = args.output or Path(args.trace) / "expanded"
    turns = export_trace(args.trace, output)
    print(read_trace(args.trace).to_text())
    print(f"\nwritten to {turns}")
    return 0


# --------------------------------------------------------------------------- #
# amx view
# --------------------------------------------------------------------------- #


def _view_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        help="Case id (for example CEN-001), case directory, model.py, or asset.xml",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Sweep directory when target is a case id, for example runs/bench/full",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help=(
            "Drive the mechanisms the task's operation contract names through their "
            "travel; everything else holds its pose"
        ),
    )
    parser.add_argument(
        "--period",
        type=float,
        default=6.0,
        help="Seconds for one open-close cycle (default: 6)",
    )
    parser.add_argument(
        "--joint",
        action="append",
        default=None,
        help="Animate this joint instead of the contract's; repeat to select several",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Resolve and validate the MJCF without opening a window",
    )
    parser.set_defaults(handler=_view)


def _view(args: argparse.Namespace) -> int:
    from amx.view import (
        animation_plan,
        articulations,
        case_dir_for,
        grounding_spec_for,
        launch,
        resolve,
        viewable,
    )

    resolved = viewable(resolve(args.target, run_dir=args.run_dir))
    if resolved.patched:
        print(
            f"view copy: {resolved.path}\n"
            f"source asset was not dynamically loadable; visualization-only fix: {resolved.note}"
        )
    else:
        print(f"asset: {resolved.path}")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(resolved.path))
    joints = articulations(model)
    spec = grounding_spec_for(case_dir_for(args.target, run_dir=args.run_dir))

    if args.joint:
        by_name = {joint.name.lower(): joint for joint in joints}
        missing = [name for name in args.joint if name.lower() not in by_name]
        if missing:
            available = ", ".join(joint.name for joint in joints) or "(none; static asset)"
            raise ValueError(
                f"joint(s) not present in this asset: {', '.join(missing)}; "
                f"available: {available}"
            )

    plan = animation_plan(model, joints, spec=spec, requested=args.joint)
    driving = {joint.name: joint for joint in plan.driven}
    print("joints:")
    for joint in joints:
        chosen = driving.get(joint.name)
        if chosen is None:
            motion = f"{joint.lower:g} .. {joint.upper:g} (held)"
        elif chosen.spins:
            motion = "continuous spin"
        else:
            motion = f"{joint.lower:g} .. {joint.upper:g}"
        print(f"  {joint.name}: {motion}")
    if args.animate:
        print(f"animation: {plan.note}")
    if args.print_only:
        return 0
    selected = plan.driven

    # Static assets use MuJoCo's ordinary blocking viewer. Dynamic animation on
    # macOS uses launch_passive, which must own the main thread through
    # `mjpython`. Resolve and classify first so a beaker is not sent down the
    # articulated-only launch path.
    needs_mjpython = bool(args.animate and selected and sys.platform == "darwin")
    if needs_mjpython:
        import os

        if os.environ.get("_AMX_UNDER_MJPYTHON") != "1":
            mjpython = PROJECT_ROOT / ".venv" / "bin" / "mjpython"
            if not mjpython.is_file():
                raise RuntimeError(
                    "MuJoCo viewer needs .venv/bin/mjpython on macOS; run scripts/setup.sh"
                )
            environment = os.environ.copy()
            environment["_AMX_UNDER_MJPYTHON"] = "1"
            os.execve(
                str(mjpython),
                [str(mjpython), "-m", "amx.cli", *sys.argv[1:]],
                environment,
            )

    launch(
        resolved.path,
        animate=bool(args.animate and selected),
        period_s=max(0.1, args.period),
        driven=selected,
    )
    return 0


def _timestamp_id() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    raise SystemExit(main())
