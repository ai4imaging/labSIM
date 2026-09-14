"""Physics plausibility judge.

Given a simulation evidence directory (a compiled MuJoCo model plus a step-by-step
state recording), replay the whole trace and look for physically implausible
behaviour -- floating bodies, penetration, teleports, loss of support -- then emit a
structured diagnostic report.

Usage::

    python -m sim_judge <case_dir>
    python -m sim_judge <case_dir> --format json --output report.json

Or as a library::

    from sim_judge import judge_case
    report = judge_case("correct_case_0000")
"""

from sim_judge.judge import JudgeOptions, judge_case
from sim_judge.report.finding import Finding, Severity, Subject

__all__ = ["JudgeOptions", "judge_case", "Finding", "Severity", "Subject"]
__version__ = "1.0.0"
