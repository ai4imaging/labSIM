"""One real episode, executed and judged.

The only test here that runs physics. It exists because the case bundle's layout is a
contract with `sim_judge` that nothing else in the suite exercises: every other test builds
the parts of a case without ever asking the judge to read one.

Kept to a couple of seconds of simulated time — the arm reaches down, closes on nothing and
lifts — because the property under test is the shape of the bundle and the honesty of the
verdict, not whether this particular motion is any good.
"""

from __future__ import annotations

import json

import pytest

from amx.loop.judge import judge
from amx.report import Severity
from amx.sim.plan import Grip, Hold, Move, OperationPlan, PoseTarget
from amx.sim.policy import build_policy
from amx.sim.run import run_episode
from amx.sim.scene import Bench, RobotRef, Workcell, build_scene

pytestmark = pytest.mark.slow

TOOL_DOWN = (3.14159, 0.0, 0.0)


@pytest.fixture(scope="module")
def episode(robot_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("episode")
    workcell = Workcell(
        workcell_id="reach",
        robot=RobotRef(model=robot_dir.name),
        bench=Bench(size_xy=(1.2, 0.9)),
        assets=[],
        fixtures=[],
    )
    built = build_scene(workcell, root / "scene")
    plan = OperationPlan(
        plan_id="reach",
        protocol_id="reach",
        actions=[
            Move(step_id="step.p01.approach", duration_s=0.6,
                 target=PoseTarget(pos=(0.45, 0.10, 0.30), euler=TOOL_DOWN)),
            Grip(step_id="step.p01.approach", duration_s=0.3, width_m=0.030),
            Move(step_id="step.p02.descend", duration_s=0.6,
                 target=PoseTarget(pos=(0.45, 0.10, 0.16), euler=TOOL_DOWN)),
            Grip(step_id="step.p03.grasp", duration_s=0.4, width_m=0.010),
            Move(step_id="step.p04.lift", duration_s=0.6,
                 target=PoseTarget(pos=(0.45, 0.10, 0.30), euler=TOOL_DOWN)),
            Hold(step_id="step.p05.settle", duration_s=0.3),
        ],
        settle_s=0.2,
    )
    case = root / "case"
    result = run_episode(built, plan, case)
    (case / "bound-operation.json").write_text(
        json.dumps(build_policy(built, plan, built.load_model(), task_id="reach"), indent=2)
    )
    return built, plan, result


def test_the_episode_runs_without_diverging(episode):
    """An arm reaching into empty space must not break the solver.

    Weaker than it sounds, and worth asserting: this scene diverged for a week of this
    project's life, first from a timestep the stiff position servos could not survive, then
    from gripper self-collisions inside the linkage, then from a gain that crossed its own
    equilibrium in one step. Each time, the arm was simply moving through open air.
    """
    _, _, result = episode
    assert result.diverged_at_step is None, result.divergence_reason
    assert result.step_count > 0


def test_the_case_bundle_is_what_sim_judge_expects(episode):
    """`discover_bundle` has to find this without being told anything about its layout."""
    from sim_judge.loader.case_bundle import discover_bundle

    _, _, result = episode
    bundle = discover_bundle(result.case_dir)
    assert bundle is not None
    for name in ("bound-operation.json", "task-execution.json"):
        assert (result.case_dir / name).is_file()
    assert (result.case_dir / "closed-scene" / "scene.xml").is_file()
    raw = result.case_dir / "raw-dynamics"
    assert (raw / "raw-trace.json").is_file()
    assert (raw / "compiled-model.mjb").is_file()
    assert list(raw.glob("steps-*.npz")), "no state chunks were recorded"


def test_the_recorded_states_replay(episode):
    """A trace that cannot be replayed is not evidence of anything.

    The judge's whole method is to reload each recorded state into the compiled model and
    measure it, so the chunk contents have to match the model's integration-state size
    exactly. A mismatch here is silent — the states load, the numbers are nonsense.
    """
    import glob

    import mujoco
    import numpy as np

    _, _, result = episode
    trace = json.loads((result.case_dir / "raw-dynamics" / "raw-trace.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(result.case_dir / "closed-scene" / "scene.xml"))
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    assert trace["state_size"] == size

    data = mujoco.MjData(model)
    total = 0
    for path in sorted(glob.glob(str(result.case_dir / "raw-dynamics" / "steps-*.npz"))):
        chunk = np.load(path)["after"]
        assert chunk.shape[1] == size
        mujoco.mj_setState(model, data, chunk[-1], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.qpos).all()
        total += chunk.shape[0]
    assert total == trace["step_count"] == result.step_count


def test_the_judge_reads_the_generated_policy(episode):
    """End of the loop: the judge returns a verdict, and its findings carry repair targets.

    Not asserting PASS. The arm closes its fingers on nothing here, and what the verdict
    should be is a question about this motion rather than about the pipeline. What must
    hold is that the judge ran, saw the declared rules, and produced findings the loop can
    route.
    """
    _, _, result = episode
    judgement = judge(result.case_dir, stride=10)
    assert judgement.verdict in {"PASS", "FAIL"}
    assert judgement.counts
    for finding in judgement.report.findings:
        assert finding.repair_target is not None
        assert finding.severity in set(Severity)
        assert finding.summary
