"""The objects: the pine board the two arms carry, the two blocks it starts on and the two cradles it goes into.

Each function returns an ``MjSpec`` with one root body.  The board has a free joint; the blocks and cradles stay put."""

from __future__ import annotations

import mujoco

BOX = mujoco.mjtGeom.mjGEOM_BOX
OBJECT_SOLREF = (0.01, 1.0)
OBJECT_SOLIMP = (0.99, 0.995, 0.001, 0.5, 2.0)
"""Contact stiffness for the board, as for the finger pads (``sim.robot``): MuJoCo's default contact is soft in
proportion to how light the object is, and light things sink into what they rest on."""
PINE = (0.86, 0.72, 0.5, 1.0)
DARK_WOOD = (0.45, 0.32, 0.2, 1.0)
TAPE = {"left": (0.15, 0.45, 0.85, 1.0), "right": (0.95, 0.55, 0.1, 1.0)}
"""The grip bands' tape: blue on the controlled arm's end, orange on the other arm's."""

# --- The board ------------------------------------------------------------------------------------------------------
BOARD = (0.03, 0.60, 0.02)
"""Full sizes: 3 cm front to back (x, what the jaw closes across), 60 cm long (y, across the table), 2 cm thick."""
BOARD_MASS = 0.18
"""Pine, about 0.5 g/cm^3."""
BAND_Y = 0.25
"""The grip bands' middles, from the board's middle: 5 cm in from each end, straight in front of each arm's base."""
BAND_WIDTH = 0.08
BOARD_FRICTION = (0.9, 0.02, 5e-4)
"""Sliding 0.9, planed pine on the pads' rubber; torsional 2 cm, as a flat pad pressed on a flat face resists turning
over its area (MuJoCo's contacts are points, and with the default 5 mm the board swung about the jaw)."""


def board_spec() -> mujoco.MjSpec:
  """The board, origin at its middle, length along y.  Sites ``left_band`` and ``right_band``: the middles of the grip
  bands, on the board's axis; ``left_end`` and ``right_end``: the middles of its end faces."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="board")
  body.add_freejoint(name="board_joint")
  x, y, z = BOARD
  body.add_geom(type=BOX, size=(x / 2, y / 2, z / 2), mass=BOARD_MASS, rgba=PINE, friction=BOARD_FRICTION, condim=4,
                contype=1, conaffinity=1)
  for side, s in (("left", 1.0), ("right", -1.0)):
    # The tape: seen only, a hair proud of the wood all round.
    body.add_geom(type=BOX, size=(x / 2 + 0.0005, BAND_WIDTH / 2, z / 2 + 0.0005), pos=(0, s * BAND_Y, 0), mass=0.0,
                  rgba=TAPE[side], contype=0, conaffinity=0)
    body.add_site(name=f"{side}_band", pos=(0, s * BAND_Y, 0), size=(0.004, 0, 0), rgba=(1, 0, 0, 1))
    body.add_site(name=f"{side}_end", pos=(0, s * y / 2, 0), size=(0.004, 0, 0), rgba=(1, 0, 0, 1))
  for geom in spec.geoms:
    if geom.contype:
      geom.solref, geom.solimp, geom.priority = OBJECT_SOLREF, OBJECT_SOLIMP, 1
  return spec


# --- The start blocks -----------------------------------------------------------------------------------------------
REST = (0.06, 0.06, 0.05)
"""Full sizes of each wooden block the board starts on: 5 cm high, so the fingers close round the board without
reaching the table."""


def rest_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="rest")
  x, y, z = REST
  body.add_geom(type=BOX, size=(x / 2, y / 2, z / 2), pos=(0, 0, z / 2), rgba=DARK_WOOD, friction=(0.6, 5e-3, 5e-4))
  return spec


# --- The cradles ----------------------------------------------------------------------------------------------------
CRADLE_BASE = (0.08, 0.06, 0.05)
"""Full sizes of each cradle's base block: the board sits on its top, 5 cm up, the height it started at."""
CRADLE_SLOT = 0.04
"""Gap between the cradle's two uprights, front to back: 1 cm wider than the board."""
CRADLE_POST = (0.012, 0.06, 0.03)
"""Full sizes of each upright: 1.2 cm thick, as wide as the base, standing 3 cm above the seat."""


def cradle_spec() -> mujoco.MjSpec:
  """A U-shaped cradle, origin at the middle of its underside, the slot running along y."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cradle")
  bx, by, bz = CRADLE_BASE
  px, py, pz = CRADLE_POST
  grey = (0.5, 0.52, 0.55, 1.0)
  body.add_geom(type=BOX, size=(bx / 2, by / 2, bz / 2), pos=(0, 0, bz / 2), rgba=grey, friction=(0.6, 5e-3, 5e-4))
  for s in (1, -1):
    body.add_geom(type=BOX, size=(px / 2, py / 2, pz / 2), pos=(s * (CRADLE_SLOT / 2 + px / 2), 0, bz + pz / 2),
                  rgba=grey, friction=(0.4, 5e-3, 5e-4))
  return spec
