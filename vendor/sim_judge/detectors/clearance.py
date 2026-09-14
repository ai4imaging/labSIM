"""Class C: declared minimum clearance.

``runtime_feedback.clearance_rules`` requires certain groups of bodies to **stay a
certain distance apart at all times**, not merely to avoid colliding. The canonical
example is the 5 mm safety clearance between the tool body and non-adjacent robot arm
links: on real hardware, cable routing, manufacturing tolerances and calibration error
all eat into the margin, so a trajectory that skims past in simulation will scrape on
the physical machine.

Because the requirement is that even non-contact counts as a violation, the contact
table is of no use here and we must compute the geometric distance for every pair. This
scene expands to 262 geom pairs to test, and running the exact distance query on all of
them at every step would double the total runtime, so we pre-filter with bounding
spheres first: the surface distance between two geoms can never be less than the
distance between their sphere centers minus the two radii, so any pair whose lower bound
already clears the threshold can be skipped outright. The pre-filter is vectorized, and
it can only let pairs through, never discard a real violation, so the result is
identical to computing the exact distance for every pair.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase

import numpy as np

from sim_judge.detectors.base import BaseDetector, DetectorContext, declared, geom_pair_subject
from sim_judge.loader.policy import ClearanceRule
from sim_judge.replay import Frame
from sim_judge.report.finding import Observation
from sim_judge.world.naming import GeomInfo


@dataclass(frozen=True, slots=True)
class _RulePairs:
    """Every geom pair a single rule expands to, plus the values precomputed for the
    bounding-sphere pre-filter."""

    rule: ClearanceRule
    pairs: tuple[tuple[GeomInfo, GeomInfo], ...]
    search_range_m: float

    index_a: np.ndarray
    """The id of the left-hand geom of each pair in ``pairs``, shape ``(n,)``. Used to
    fetch world coordinates in a vectorized way."""

    index_b: np.ndarray
    radius_sum: np.ndarray
    """The sum of the two geoms' bounding-sphere radii for each pair, shape ``(n,)``. The
    pre-filter's lower bound is the center distance minus this."""

    def candidates(self, geom_xpos: np.ndarray) -> np.ndarray:
        """Return the indices of the geom pairs that could be in violation.

        The bounding spheres give a lower bound on the surface distance, so any pair
        whose lower bound already meets the requirement is necessarily compliant and can
        be safely skipped.
        """
        offsets = geom_xpos[self.index_a] - geom_xpos[self.index_b]
        lower_bound = np.linalg.norm(offsets, axis=1) - self.radius_sum
        return np.flatnonzero(lower_bound < self.rule.minimum_clearance_m)


class ClearanceDetector(BaseDetector):
    """C1: the distance between two groups of bodies falls below the declared minimum
    clearance."""

    code_prefix = "C"

    def __init__(self, context: DetectorContext) -> None:
        super().__init__(context)
        self._stride = max(1, self.policy.clearance_sample_stride_steps)
        self._rules = self._expand_rules()
        self._next_sample = 0

    @property
    def enabled(self) -> bool:
        return bool(self._rules)

    def feed(self, frame: Frame) -> Iterable[Observation]:
        if frame.step_index < self._next_sample:
            return ()
        self._next_sample = frame.step_index + self._stride

        distance_of = self.ctx.replayer.geom_distance
        for entry in self._rules:
            rule = entry.rule
            if not rule.covers_phase(frame.phase_labels):
                continue

            closest: tuple[float, GeomInfo, GeomInfo] | None = None
            for index in entry.candidates(frame.geom_xpos):
                geom_a, geom_b = entry.pairs[index]
                distance = distance_of(geom_a.id, geom_b.id, entry.search_range_m)
                if closest is None or distance < closest[0]:
                    closest = (distance, geom_a, geom_b)

            if closest is None or closest[0] >= rule.minimum_clearance_m:
                continue
            distance, geom_a, geom_b = closest

            yield declared(
                code="C1_MINIMUM_CLEARANCE_VIOLATED",
                step=frame.step_index,
                subject=geom_pair_subject(geom_a, geom_b),
                metrics={
                    "clearance_m": distance,
                    "shortfall_m": rule.minimum_clearance_m - distance,
                },
                peak_metric="shortfall_m",
                thresholds={"minimum_clearance_m": rule.minimum_clearance_m},
                rule_id=rule.rule_id,
                merge_gap_steps=self._stride,
                detail={"message": rule.message, "repair_target": rule.repair_target},
            )

    # -- construction --------------------------------------------------------

    def _expand_rules(self) -> list[_RulePairs]:
        """Expand body-level wildcard rules into the geom-level pairs to be tested.

        Only geoms in the ``geom_groups`` named by the rule are kept: collision proxies
        and visual shells commonly coexist on the same body, and measuring clearance
        against a visual shell yields a meaningless number.
        """
        expanded: list[_RulePairs] = []
        for rule in self.policy.clearance_rules:
            pairs = tuple(self._pairs_for(rule))
            if not pairs:
                continue
            index_a = np.array([a.id for a, _ in pairs], dtype=np.int32)
            index_b = np.array([b.id for _, b in pairs], dtype=np.int32)
            expanded.append(
                _RulePairs(
                    rule=rule,
                    pairs=pairs,
                    # We only need to know whether the distance dips below the
                    # threshold, so a search bound slightly above it is enough.
                    search_range_m=rule.minimum_clearance_m * 2.0,
                    index_a=index_a,
                    index_b=index_b,
                    radius_sum=self._bounding_radii(index_a) + self._bounding_radii(index_b),
                )
            )
        return expanded

    def _bounding_radii(self, geom_ids: np.ndarray) -> np.ndarray:
        """Bounding-sphere radii of the geoms.

        MuJoCo records 0 for planes and height fields because they are unbounded. Giving
        such geoms an infinite radius makes the pre-filter always let them through to the
        exact distance query — better to do extra work than to miss a violation.
        """
        radii = np.asarray(self.model.geom_rbound, dtype=np.float64)[geom_ids].copy()
        radii[radii <= 0.0] = np.inf
        return radii

    def _pairs_for(self, rule: ClearanceRule) -> Iterable[tuple[GeomInfo, GeomInfo]]:
        groups = set(rule.geom_groups)
        side_a = self._bodies_matching(rule.body_a)
        side_b = self._bodies_matching(rule.body_b)

        for body_a, body_b in itertools.product(side_a, side_b):
            if body_a == body_b:
                continue
            if not rule.covers_bodies(
                self.resolver.body_name(body_a), self.resolver.body_name(body_b)
            ):
                continue  # matched by excluded_body_pairs
            geoms_a = self._geoms_in_groups(body_a, groups)
            geoms_b = self._geoms_in_groups(body_b, groups)
            yield from itertools.product(geoms_a, geoms_b)

    def _bodies_matching(self, pattern: str) -> list[int]:
        return [
            body_id
            for body_id in range(self.model.nbody)
            if fnmatchcase(self.resolver.body_name(body_id), pattern)
        ]

    def _geoms_in_groups(self, body_id: int, groups: set[int]) -> list[GeomInfo]:
        geoms = self.resolver.geoms_of_body(body_id)
        if not groups:
            return [g for g in geoms if g.collidable]
        return [g for g in geoms if g.group in groups]
