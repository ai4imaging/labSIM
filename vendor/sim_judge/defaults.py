"""Fallback thresholds for the generic geometric criteria.

The judge draws its thresholds from two layers:

1. **Declared layer** -- taken from ``bound-operation.json``. Violating a declared
   rule drives the overall verdict to FAIL.
2. **Generic layer** -- this module. It covers the geom pairs and bodies the policy
   never mentions, and only ever emits warning-level findings; it does not veto the
   verdict unless ``--strict`` is passed on the command line.

The reason for the split: the declared layer is the task author's formal contract
about what counts as a failure, whereas the generic layer is the judge's own
physical common sense. Merging the two would let the judge decide on the author's
behalf wherever the policy is silent, which makes the acceptance criteria
irreproducible.

All lengths are in metres, times in seconds and forces in newtons.

**Times are always expressed in seconds, never in steps.** A step count only means
something physically once you pair it with the simulation timestep, and the
timestep varies from task to task; detectors convert seconds into steps against the
recording's actual timestep via
:meth:`~sim_judge.detectors.base.DetectorContext.steps_for`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenericThresholds:
    """Generic physical criteria used wherever the policy declares nothing."""

    # ---- Class P: penetration ---------------------------------------------
    # Penetration depth is made dimensionless against the minimum half-extent of the
    # thinner of the two geoms in contact:
    #   ratio = depth / min_half_extent. A ratio >= 1 means the contact has eaten
    # through that geom's entire half-thickness.
    #
    # Using a ratio rather than an absolute depth is exactly what makes this portable
    # across scenes: "normal press-in" differs by orders of magnitude in absolute
    # terms between millimetre-scale labware and a metre-scale robot arm, yet as a
    # fraction of each part's own thickness it is the same phenomenon. The specific
    # value 0.5 was calibrated on the centrifuge tube-loading scene and can be
    # overridden with ``--penetration-ratio``. In that scene the three kinds of
    # contact separate cleanly:
    #   gripper guide face / pad  <-> tube   0.29 - 0.39   normal elastic press-in of a soft contact
    #   rotor socket wall         <-> tube   0.65          squeezes the wall while seating, worth a hint
    #   thin socket floor plate   <-> tube   3.33          far beyond the plate thickness, geometrically through it
    # The 0.5 line separates "soft-contact compression" from "genuine overlap"
    # cleanly, and its meaning is plain: penetration has reached half the
    # half-thickness of the thinner party. An absolute cut-off has no discriminating
    # power here -- all three cases above sit between 0.39 mm and 3.33 mm of depth,
    # and the difference only emerges once each is normalised by its own thickness.
    penetration_ratio_warning: float = 0.5
    penetration_ratio_severe: float = 1.0
    # The dimensionless criterion turns hypersensitive on very small geoms, so there
    # is also an absolute floor: anything shallower than this is never reported.
    penetration_absolute_floor_m: float = 2.0e-4

    # Tunneling: if a free body moves further in a single step than this multiple of
    # its own minimum half-extent, it may well have passed through a thin wall.
    tunneling_displacement_ratio: float = 1.0

    # Solver-blind overlap (P5) is scaled against the half-thickness of the *thicker* of
    # the two geoms, which is the opposite of what P1 does, for a concrete reason. P1 asks
    # "was a thin wall breached", so the thin wall has to be the yardstick. P5 only ever
    # looks at pairs with no contact between their bodies at all, where that question does
    # not arise, and normalising by the thinner side there hands the scale to whichever
    # paper-thin decoration happens to be in the way. Measured on the reference scene, with
    # the depth already reduced by whatever the pair overlaps at rest:
    #   robot wrist capsule  -> centrifuge lid shell   20.5 mm   thin 0.63   thick 0.51
    #   robot upper arm mesh -> centrifuge lid shell   14.3 mm   thin 0.44   thick 0.38
    #   robot wrist mesh     -> centrifuge lid shell   11.7 mm   thin 0.36   thick 0.31
    #   button pusher tip    -> printed "open" symbol   0.4 mm   thin 1.58   thick 0.13
    #   gripper finger       -> tube cap                0.5 mm   thin 0.31   thick 0.15
    # Normalised by the thinner side the printed symbol outranks a limb buried in a
    # machine; normalised by the thicker side the real intrusions occupy 0.27 to 0.51 and
    # everything else stays under 0.21, so the two populations separate on their own.
    unmodelled_overlap_ratio_warning: float = 0.25
    unmodelled_overlap_ratio_severe: float = 0.5

    # P5 measures geometry directly instead of reading the
    # contact list, which costs far more per step than any other check. Sampling it on
    # an interval rather than every step is sound because interpenetration is a state
    # that persists, not an instant: two solids that end up sharing volume stay that way
    # for many milliseconds while the motion carries through. A genuinely instantaneous
    # pass-through leaves no overlapping frame to find at any sampling rate, which is
    # exactly why P4 judges it from displacement instead. 5 ms keeps the cost near one
    # twentieth of a full per-step scan while still catching events an order of
    # magnitude shorter than anything a human would call a collision.
    overlap_sample_interval_s: float = 5.0e-3
    # Consecutive samples of the same overlap are the same event and have to be merged
    # into one finding, otherwise a one-second intrusion sampled every 5 ms is reported as
    # two hundred separate single-step findings. The window is generous relative to the
    # sampling interval so that a brief dip below the threshold mid-event does not split
    # it, while genuinely separate intrusions stay separate.
    overlap_merge_gap_s: float = 0.05
    # Fraction of the witness segment that has to fall inside the true triangle mesh
    # before a hull-level overlap is accepted as real. Grazing contact against a curved
    # shell legitimately leaves an endpoint outside, so unanimity is too strict.
    overlap_witness_inside_fraction: float = 0.5

    # ---- Class F: floating ------------------------------------------------
    # Speed below which a free body counts as "at rest". Slower than this with no
    # contact for a sustained period means non-physical levitation.
    floating_speed_epsilon_m_s: float = 5.0e-3
    # How long the hover has to persist before it is reported, which filters out
    # instantaneous zero-speed moments such as the apex of a ballistic arc.
    floating_min_duration_s: float = 0.2
    # Gravity consistency: warn when the vertical acceleration falls below this
    # fraction of g while there is no adequate upward support force.
    gravity_consistency_accel_ratio: float = 0.25
    # Relative tolerance between support force and weight.
    gravity_consistency_force_tolerance: float = 0.5

    # ---- Class K: kinematics ----------------------------------------------
    # Teleport: warn when the actual displacement exceeds this multiple of |v|*dt
    # plus an absolute slack.
    teleport_velocity_slack_ratio: float = 4.0
    teleport_absolute_slack_m: float = 1.0e-4
    # MuJoCo joint limits are soft constraints: the solver lets a joint briefly cross
    # its limit and then pushes it back, with the magnitude set by solref. This
    # allowance separates "normal soft-constraint rebound" from "the joint really ran
    # out of travel". At judging time we take the larger of this and the policy's
    # declared joint_limit_tolerance -- the policy tolerance is meant for commanded
    # values, and applying it directly to solver output would flag every normal press
    # as a limit violation.
    joint_limit_soft_allowance: float = 5.0e-3
    # Velocity blow-up thresholds: linear in m/s, angular in rad/s.
    max_linear_speed_m_s: float = 50.0
    max_angular_speed_rad_s: float = 200.0

    # ---- Event aggregation ------------------------------------------------
    # Largest gap, in steps, tolerated between consecutive hits of the same
    # diagnostic before they stop being merged into one event; this is what collapses
    # jitter. This one is in steps rather than seconds on purpose: it describes
    # "adjacent samples" and has nothing to do with the simulation timestep.
    event_gap_steps: int = 2
    # During processes such as seating or rebound the monitored quantity crosses the
    # threshold repeatedly. A much larger gap merges the whole process into a single
    # event; otherwise one seating gets chopped into a dozen findings that look
    # unrelated to each other.
    oscillation_merge_gap_s: float = 0.2

    # ---- Sampling ---------------------------------------------------------
    # Static support checks (F4) are expensive and the quantity itself barely varies
    # over time, so they are sampled at this interval.
    support_sample_interval_s: float = 0.5
    # Number of sample points for the determinism self-check.
    determinism_sample_count: int = 24
    # Largest deviation the self-check tolerates; anything above means the model or
    # engine version does not match the recording.
    determinism_tolerance: float = 1.0e-8


DEFAULT_THRESHOLDS = GenericThresholds()
