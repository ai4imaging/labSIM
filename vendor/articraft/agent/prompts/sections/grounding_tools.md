# Grounding: matching the specification, not just compiling

This task carries a specification: stated dimensions, required components, an internal
volume, handling the object has to survive. A clean `compile_model` does not speak to any
of it. A solid cylinder of the wrong diameter compiles perfectly, and so does a beaker
with no cavity in it.

You have three tools that do speak to it. Each answers with a `<grounding_signals>` block
in the same form as `<compile_signals>`: a summary, failures, warnings and suggested next
steps.

- `check_physical_grounding` — measures the built object against the stated dimensions,
  checks total mass, matches the body tree against the required component list, and
  integrates the internal volume.
- `check_protocol_grounding` — drives every required control and mechanism along its whole
  travel and back, checks joint type and range, checks that nothing else in the model
  moves while it does, reports the worst penetration anywhere on the path, then simulates
  bench stability, probe insertion and tilt protocols when requested.
- `check_visual_grounding` — renders it from several directions, judges whether each
  required visual feature is actually visible, and attaches the renders for you to look at.

## Mass and inertia are part of the asset, not export-time repair

Every authored `Part` must have an explicit, physically plausible positive mass and
positive-definite inertia tensor. This includes the root, fixed decorative subparts, lids,
doors, rotors, buttons and every other child of an articulation. A visually complete mesh
with `part.inertial is None` is not simulation-ready: MuJoCo will reject it when that body
is movable, or when the assembled asset is given a free joint in a workcell.

Use `Inertial.from_geometry(primitive_proxy, mass=...)` for the normal case. The inertial
proxy is not visual geometry: use a `Box`, `Cylinder`, or `Sphere` that conservatively
covers the part's mass distribution, even when the visible form is a CadQuery mesh. Put
the proxy's centre of mass in `origin=Origin(...)` when the part frame is not centred.
Choose masses in kilograms from plausible material density and part volume; never use zero
or epsilon mass merely to satisfy validation. `compile_model` checks this contract for
every part and reports the exact names still missing physical properties.

## Interaction semantics are executable contracts

Before authoring geometry, make an internal mechanism plan from the
`Physical-operation contract` in the user request. For every operation identify the
independent child `Part`, parent, joint type, axis/origin, limits, mass/inertia and the
endpoint motion you will verify. Keep this plan current when the design changes.

Any visible push button or key must be a separate `Part`, not another visual fused into a
control panel. Mount it as the child of a `PRISMATIC` articulation with a finite,
non-zero, physically plausible press stroke (normally 0.5–5 mm), an inward-facing axis,
and suitable damping. This applies to power, start/stop, open/eject, zero/tare and soft
keys. A coloured rectangle on a panel is not a button.

Use mechanism-appropriate joint types elsewhere. Rotors and shafts may be `CONTINUOUS`.
Hinged lids and doors use bounded `REVOLUTE`; sliding covers use bounded `PRISMATIC`;
genuinely loose removable covers may use `FLOATING`. Never use `CONTINUOUS` merely as an
easy way to make a lid movable: that turns opening into unlimited spinning.
`compile_model` inspects semantic part/visual names and blocks these mismatches before
grounding begins.

`CONTINUOUS` is a claim that the part turns forever under power — a rotor, a turntable, a
stirrer shaft. Everything else gets limits. An unbounded hinge on a lid, a knob or a
button is reported as `G-OP-RANGE` and is also what makes a static object appear to spin
when someone previews it.

A part the specification calls removable comes off; it does not swing. Model it as
`FLOATING`, or as a bounded `PRISMATIC` lift along the axis it is actually withdrawn
along. Giving a tray, an adapter, a gasket or a screw cap a revolute joint so that
"something moves" is wrong twice over: the motion is not the one the object has, and
swinging a part that is seated inside a recess drives it straight through the wall.

Author it where it sits, in contact with whatever holds it — a pestle resting in its
mortar, a lid down on its rim, a tray seated in its slot. A removable part floating in
mid-air above its seat is reported by the compiler as an isolated part, and it is a real
defect rather than a technicality: it means the seat and the part were designed to
different dimensions and nothing would actually locate one in the other.

## Getting the direction of motion right

A joint has an axis, and the sign of the axis decides which way positive travel goes.
Work it out before you author it, and state it to yourself: right-hand rule about the
axis, from the zero pose, at the joint origin. A lid hinged at the back of a housing with
`axis=(1, 0, 0)` opens the opposite way from the same lid with `axis=(-1, 0, 0)`, and one
of the two swings down through the bench.

Then verify it rather than assuming, with a `run_tests` expectation that names the
direction in terms the specification uses — the far edge of the lid ends up *higher* at
the upper limit, the button face ends up *lower* when pressed, the drawer front ends up
*further out* when open. A joint whose limits are right and whose axis is inverted passes
every range check there is.

## Motion has to be clear along the whole path, not just at the ends

`check_protocol_grounding` drives each mechanism through its travel and reports the worst
penetration it finds anywhere on the way, not only at the endpoint. A lid that clears its
housing when shut and when fully open, and ploughs through the rim in between, fails.

It also holds still everything that is not being driven. If operating one control moves
the housing, the bench plate or an unrelated part, that is a parenting error — the part is
hanging off the wrong body, or off a chain of joints it should not be behind — and it is
reported as `G-OP-DISTURBANCE`.

`ctx.allow_overlap(...)` is for interference that is real and intended: a press fit, a
shaft in its bore, an O-ring in its groove. It is not a way past a failing check. Using it
to declare away parts that visibly pass through each other produces an asset that compiles
and is obviously broken in every render, and the grounding checks do not honour it.

The operation check finds the named child bodies in the exported MuJoCo model, requires
the number of independent parts stated in the contract, verifies the mechanism-specific
joint type and limits, imposes an endpoint pose, measures real translation/rotation and
collision penetration, and restores the initial pose. Passing compile is not evidence
that a button presses or a lid opens; run `check_protocol_grounding` and repair every
`G-OP-*` failure before finishing.

## Keep CadQuery repairable

Compile a mechanically complete low-detail baseline before adding repeated perforations,
threads, flutes or cosmetic fillets. Prefer patterning a small number of solids and
performing one boolean against their compound; do not build long chains of pairwise
`union()`/`cut()` calls inside nested loops. Thin seals and membranes need real thickness:
coplanar meshes have no collision volume and MuJoCo cannot load them.

After each detail group, compile. If a new group causes timeout, empty tessellation, or
coplanar collision geometry, revert that group immediately and represent the same function
with fewer features or a primitive-backed thickness. Never increase a 300-second compile
timeout to rescue pathological geometry. If the turn budget is exhausted after a later
regression, the harness restores the last successfully compiled `model.py` so downstream
inspection receives a usable partial asset rather than the broken final edit.

## You cannot finish until these pass

Build a complete, mechanically correct object first, at low detail. An empty
`ArticulatedObject` with no parts is not a starting point you can finish from: it fails to
compile, nothing can be measured on it, and it scores zero however many turns went into
the plan for it. Get one compiling asset with every required component present, then
refine.

When you finish, every applicable check is run for you against the code as it then stands,
and the finish is refused if any of them fails. You do not get to submit an unmeasured
revision: there is no version of this task where the last thing you did was an edit.
Stale results do not count either — editing `model.py` invalidates a measurement exactly
as it invalidates a compile.

So the turn budget is better spent this way: get it compiling early, call the checks
yourself, and fix what they report while the geometry is still simple. A dimension that is
wrong at turn three is much cheaper to fix than the same dimension at turn thirty, when
other geometry has been built on top of it. Work the failures in the order they are
reported — measurements, then motion, then appearance — because fixing geometry usually
changes the motion and redoing the motion first is wasted work.

## How the measurements are actually taken

Knowing this saves you from arguing with numbers that are correct.

- **Outer diameter** is the median radius of a horizontal cross-section, doubled — not the
  bounding box. A spout, a handle or a moulded panel widens the bounding box and does not
  change the diameter, so those are ignored on purpose.
- **Inner diameter and wall thickness** come from the same cross-section, taken at the
  height the specification names. On a tapered part, mid-height and rim are genuinely
  different numbers.
- **Internal volume** is integrated from the floor of the cavity up to the *lowest*
  overflow edge, which is found rather than declared. Scanning upwards, a height counts
  as cavity while its cross-section is still a closed ring. The first height at which the
  ring opens is where the contents would run out. A pouring spout or a notch in the rim
  therefore caps the usable volume however tall the wall is — which is correct, and is
  what a catalogue capacity means.
- **A cavity has to be a real subtraction from the solid.** Modelling an inner surface as
  a separate shell does not produce a ring in cross-section and will be reported as having
  no cavity at all. Use a boolean difference.
- **Component matching is by name.** Name each part for what it is — `pouring_spout`,
  `hinged_lid`, `base_flange` — and the structure check will find it. Names like `part_3`
  will be reported as missing however correct the geometry is.
- **Insertion is simulated against a convex decomposition** of your collision geometry, so
  a probe genuinely has to be able to get in. It is also checked geometrically, which is
  what localises an obstruction to a particular height.

## Look at the renders

When `check_visual_grounding` attaches images, look at them. A feature can be present in
the geometry, pass every numeric check, and still be wrong in silhouette, proportion or
placement — and that is the one thing the measurements cannot tell you.
