"""The detector suite.

Each detector owns exactly one class of physical problem and knows nothing about the
others. :func:`build_detectors` is the single assembly point: adding a new class of
check means appending one line here.
"""

from sim_judge.detectors.base import BaseDetector, Detector, DetectorContext
from sim_judge.detectors.clearance import ClearanceDetector
from sim_judge.detectors.floating import FloatingDetector, SupportedBodyDetector
from sim_judge.detectors.kinematics import EngineWarningDetector, KinematicsDetector
from sim_judge.detectors.overlap import UnmodelledOverlapDetector
from sim_judge.detectors.penetration import (
    ContactForceDetector,
    PenetrationDetector,
    TunnelingDetector,
)
from sim_judge.detectors.receiver import ReceiverSeatingDetector
from sim_judge.detectors.static_scene import StaticSceneDetector

#: Registry of detector classes. The order determines the order in which findings are
#: emitted within a single step; it does not affect the verdict.
DETECTOR_CLASSES = (
    StaticSceneDetector,
    PenetrationDetector,
    UnmodelledOverlapDetector,
    TunnelingDetector,
    ContactForceDetector,
    ClearanceDetector,
    FloatingDetector,
    SupportedBodyDetector,
    KinematicsDetector,
    EngineWarningDetector,
    ReceiverSeatingDetector,
)


def build_detectors(context: DetectorContext) -> list[Detector]:
    """Assemble the detectors that actually have something to judge in this scene.

    Detectors whose backing policy declarations are missing (for example the R class in
    a task that has no receiver socket) are excluded right here, so they do not spin
    uselessly over a hundred thousand frames.
    """
    return [d for d in (cls(context) for cls in DETECTOR_CLASSES) if d.enabled]


__all__ = [
    "BaseDetector",
    "ClearanceDetector",
    "ContactForceDetector",
    "DETECTOR_CLASSES",
    "Detector",
    "DetectorContext",
    "EngineWarningDetector",
    "FloatingDetector",
    "KinematicsDetector",
    "PenetrationDetector",
    "ReceiverSeatingDetector",
    "StaticSceneDetector",
    "SupportedBodyDetector",
    "TunnelingDetector",
    "UnmodelledOverlapDetector",
    "build_detectors",
]
