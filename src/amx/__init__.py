"""Articraft-MuJoCo wetlab pipeline.

Three stages, each usable on its own:

1. `amx.asset` — a prompt, protocol excerpts and a datasheet become a compiled MJCF asset,
   authored by the vendored Articraft SDK and then checked against the datasheet.
2. `amx.sim` and `amx.codesign` — assets, a robot arm and parametrically generated fixtures
   are composed into a workcell, and an operation plan of four action primitives is
   executed and recorded.
3. `amx.loop` — `sim_judge` grades the recording, and its findings drive a bounded repair
   loop over the plan, the fixture parameters and the layout.
"""

__version__ = "0.1.0"
