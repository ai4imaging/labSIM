"""Push the recorded trace from a case directory back into a MuJoCo Simulate window.

On macOS this re-execs itself under ``mjpython`` automatically. Just run:

    python -m sim_judge.view correct_case_0000 --loop
    python -m sim_judge.view failed_case_0001_x_plus_2mm --loop --speed 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco

from sim_judge.loader.case_bundle import BundleError, discover_bundle
from sim_judge.loader.model_loader import load_model
from sim_judge.loader.trace_reader import TraceReader
from sim_judge.world.timeline import Timeline, TimelineError, build_timeline

_STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

_KEY_SPACE = 32
_KEY_RIGHT = 262
_KEY_LEFT = 263
_KEY_R = 82
_KEY_COMMA = 44
_KEY_PERIOD = 46
_KEY_LEFT_BRACKET = 91
_KEY_RIGHT_BRACKET = 93


@dataclass
class Playback:
    paused: bool = False
    restart: bool = False
    step_delta: int = 0
    speed: float = 1.0
    last_label: str = ""

    def on_key(self, key: int) -> None:
        if key == _KEY_SPACE:
            self.paused = not self.paused
            print("paused" if self.paused else "resumed")
        elif key == _KEY_R:
            self.restart = True
            self.paused = False
            print("replaying from the start")
        elif key == _KEY_RIGHT:
            self.step_delta += 1
            self.paused = True
        elif key == _KEY_LEFT:
            self.step_delta -= 1
            self.paused = True
        elif key in (_KEY_PERIOD, _KEY_RIGHT_BRACKET):
            self.speed = min(16.0, self.speed * 2)
            print(f"speed {self.speed:g}x")
        elif key in (_KEY_COMMA, _KEY_LEFT_BRACKET):
            self.speed = max(0.125, self.speed / 2)
            print(f"speed {self.speed:g}x")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mjpython -m sim_judge.view",
        description="open a MuJoCo window and replay a case trace from the recorded states.",
    )
    parser.add_argument("case_dir", type=Path, help="case directory, e.g. correct_case_0000")
    parser.add_argument("--loop", action="store_true", help="restart once playback ends")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, default 1")
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="visual sampling stride, default 16 (about 60 fps at a 1 ms timestep)",
    )
    parser.add_argument("--start", type=int, default=0, help="step to start from")
    parser.add_argument("--stop", type=int, default=None, help="step to stop at (exclusive)")
    parser.add_argument(
        "--show-debug-markers", action="store_true", help="show the sites in group 4"
    )
    parser.add_argument(
        "--show-collision-proxies",
        action="store_true",
        help="show the collision proxies in group 5",
    )
    return parser


def _libpython_dir() -> Path | None:
    exe = Path(sys.executable).resolve()
    dylib = exe.parent.parent / "lib" / f"libpython{sys.version_info.major}.{sys.version_info.minor}.dylib"
    return dylib.parent if dylib.is_file() else None


def _reexec_under_mjpython(argv: list[str]) -> None:
    """A uv / venv Python does not put libpython where mjpython can find it, so add it to
    the search path before handing over."""
    if sys.platform != "darwin":
        return
    from mujoco import viewer

    if getattr(viewer, "_MJPYTHON", None) is not None:
        return
    launcher = Path(sys.executable).parent / "mjpython"
    if not launcher.is_file():
        return

    env = os.environ.copy()
    libdir = _libpython_dir()
    if libdir is not None:
        previous = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{libdir}{os.pathsep}{previous}" if previous else str(libdir)
    env["PYTHONUNBUFFERED"] = "1"
    root = str(Path(__file__).resolve().parent.parent)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{pythonpath}" if pythonpath else root
    os.execve(str(launcher), [str(launcher), "-m", "sim_judge.view", *argv], env)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    _reexec_under_mjpython(raw_argv)
    args = build_parser().parse_args(argv)
    if args.speed <= 0:
        print("--speed must be greater than 0", file=sys.stderr)
        return 2
    if args.stride < 1:
        print("--stride must be a positive integer", file=sys.stderr)
        return 2

    try:
        bundle = discover_bundle(args.case_dir)
    except BundleError as error:
        print(f"bad input: {error}", file=sys.stderr)
        return 2

    print(f"loading model: {bundle.model_path}")
    loaded = load_model(bundle)
    model = loaded.model
    reader = TraceReader(bundle, verify_chunks=False)
    timeline = _try_timeline(bundle, reader)

    start = max(0, args.start)
    stop = reader.step_count if args.stop is None else min(args.stop, reader.step_count)
    if start >= stop:
        print(f"empty replay range: start={start} stop={stop}", file=sys.stderr)
        return 2

    duration_s = (stop - start) * reader.timestep_s
    print(
        f"MuJoCo {mujoco.__version__}  source {loaded.source}  "
        f"nq={model.nq}  steps {stop - start}/{reader.step_count}  "
        f"duration {duration_s:.1f}s"
    )
    if timeline is not None:
        _print_spans(timeline, start, stop)
    print(
        "keys: space pause/resume   left/right single step   R restart   "
        ",/. slower/faster   close the window to quit"
    )

    from mujoco import viewer

    data = mujoco.MjData(model)
    playback = Playback(speed=args.speed)
    try:
        handle = viewer.launch_passive(model, data, key_callback=playback.on_key)
    except RuntimeError as error:
        if "mjpython" in str(error):
            print(
                "on macOS this must be launched with mjpython, for example:\n"
                f"  mjpython -m sim_judge.view {args.case_dir} --loop",
                file=sys.stderr,
            )
            return 2
        raise

    with handle:
        handle.opt.sitegroup[4] = 1 if args.show_debug_markers else 0
        handle.opt.geomgroup[5] = 1 if args.show_collision_proxies else 0
        handle.cam.lookat[:] = (0.28, -0.05, 0.42)
        handle.cam.distance = 1.55
        handle.cam.azimuth = 155
        handle.cam.elevation = -28

        cursor = start
        ended = False
        while handle.is_running():
            if playback.restart:
                cursor = start
                playback.restart = False
                playback.paused = False
                ended = False
            if ended:
                handle.sync()
                cursor = _apply_seek(model, data, handle, reader, timeline, playback, cursor, start, stop)
                if playback.restart:
                    continue
                time.sleep(0.05)
                continue

            jumped = False
            for raw in reader.iter_steps(stride=args.stride, start=cursor, stop=stop):
                if not handle.is_running() or playback.restart:
                    jumped = True
                    break
                cursor = raw.index
                _apply_state(model, data, handle, raw.state_before)
                _announce(raw.index, raw.step_id, timeline, playback)
                _pace(reader.timestep_s, args.stride, playback, handle)
                sought = _apply_seek(
                    model, data, handle, reader, timeline, playback, cursor, start, stop
                )
                if sought != cursor:
                    cursor = sought
                    jumped = True
                    break
                cursor = min(raw.index + args.stride, stop)
            else:
                if args.loop:
                    cursor = start
                    playback.last_label = ""
                    continue
                if not ended:
                    print("replay finished, the window stays open; press R to watch it again.")
                    playback.paused = True
                    ended = True
                continue
            if jumped and playback.restart:
                continue
    return 0


def _try_timeline(bundle, reader: TraceReader) -> Timeline | None:
    try:
        timeline, _ = build_timeline(bundle.read_task_execution(), bundle.read_trace_manifest())
        return timeline
    except TimelineError as error:
        print(f"action segmentation unavailable (the replay still works): {error}")
        return None


def _print_spans(timeline: Timeline, start: int, stop: int) -> None:
    print("actions (use --start/--stop to jump to one of them):")
    for span in timeline.spans:
        if span.stop_step <= start or span.first_step >= stop:
            continue
        t0 = span.first_step * timeline.timestep_s
        t1 = span.stop_step * timeline.timestep_s
        print(
            f"  {span.first_step:6d}–{span.stop_step:<6d}  "
            f"{t0:6.2f}–{t1:<6.2f}s  {span.step_id}  {span.action_id}"
        )


def _apply_state(model, data, handle, state) -> None:
    with handle.lock():
        mujoco.mj_setState(model, data, state, _STATE_SPEC)
        mujoco.mj_forward(model, data)
    handle.sync()


def _announce(step: int, step_id: str, timeline: Timeline | None, playback: Playback) -> None:
    action = timeline.action_id(step) if timeline is not None else ""
    label = f"{step_id} · {action}".strip(" ·")
    if label == playback.last_label:
        return
    playback.last_label = label
    time_s = step * (timeline.timestep_s if timeline is not None else 0.001)
    print(f"[{step:6d} | {time_s:7.3f}s] {label or step_id}")


def _pace(timestep_s: float, stride: int, playback: Playback, handle) -> None:
    deadline = time.perf_counter() + (stride * timestep_s) / max(playback.speed, 1e-6)
    while True:
        if not handle.is_running() or playback.restart or playback.step_delta:
            return
        if not playback.paused:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.02))
            continue
        handle.sync()
        time.sleep(0.05)


def _apply_seek(
    model,
    data,
    handle,
    reader: TraceReader,
    timeline: Timeline | None,
    playback: Playback,
    cursor: int,
    start: int,
    stop: int,
) -> int:
    if not playback.step_delta or not handle.is_running():
        return cursor
    jump = max(1, reader.step_count // 200)
    target = min(stop - 1, max(start, cursor + playback.step_delta * jump))
    playback.step_delta = 0
    samples = reader.sample_steps([target])
    if not samples:
        return cursor
    raw = samples[0]
    _apply_state(model, data, handle, raw.state_before)
    playback.last_label = ""
    _announce(raw.index, raw.step_id, timeline, playback)
    return raw.index


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
