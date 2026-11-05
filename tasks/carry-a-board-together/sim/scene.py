"""The rig: two YAM Pro arms side by side on one edge of a table, both reaching across it.

    y
    ^   +-----------------------+
    |   |[L] ->                 |       Seen from above.  The origin is midway between the arm bases, on the floor;
    0 --|                       |--> x  x forward, y toward the left arm.  ``TABLE_TOP`` is the table's top.
    |   |[R] ->                 |
        +-----------------------+

* Base separation, 0.50 m.
* Square, 0.90 m a side.  The bases sit flush with the back edge: their back faces, 35 mm behind the base axes, are on it.
* Height, 0.75 m, over a floor plane: things that go over the edge fall and stay fallen.
"""

from __future__ import annotations

import math

import mujoco

from mjlab.entity import EntityCfg

from sim.robot import _station_camera, get_yam_cfg

TABLE_TOP = 0.75
TABLE_SIZE = (0.90, 0.90)
"""x (front to back) by y (side to side), metres."""
TABLE_TOP_MASS = 15.0
BASE_HALF_DEPTH = 0.035
"""From a base's axis to its back face."""
TABLE_CENTER_X = TABLE_SIZE[0] / 2 - BASE_HALF_DEPTH
"""Puts the back edge on the bases' back faces."""
TABLE_THICKNESS = 0.03
LEG_INSET = 0.10
"""Legs stand this far in from the edges, so anything going over an edge reaches the floor."""
BASE_Y = 0.25
"""Each arm's base is this far from the centre along y."""

LEFT_ARM, RIGHT_ARM = "left_arm", "right_arm"
_SHELL_BITS = {LEFT_ARM: (2, 4), RIGHT_ARM: (4, 2)}
BASE_POSE = {
  LEFT_ARM: ((0.0, BASE_Y, TABLE_TOP), (1.0, 0.0, 0.0, 0.0)),
  RIGHT_ARM: ((0.0, -BASE_Y, TABLE_TOP), (1.0, 0.0, 0.0, 0.0)),
}


def _table_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table", pos=(TABLE_CENTER_X, 0.0, TABLE_TOP))
  # Dream Labs' calibrated YAM station top camera, moved with the arms: it sits where it does relative to the midpoint
  # of the station's arm bases (their 0.2525, 0, 0.76), here our midpoint (0, 0, TABLE_TOP).
  _station_camera(body, "station_top", (0.0860 - 0.2525 - TABLE_CENTER_X, -0.0090, 1.7043 - 0.76),
                  (0.68301, 0.18301, -0.18301, -0.68301), (392.1956, 391.7224), (318.3894, 237.8763))
  # Straight down over the arms' working area; a camera looks along its own -z.
  body.add_camera(name="overhead", pos=(0.28 - TABLE_CENTER_X, 0.0, 0.85), fovy=55.0)
  half_x, half_y = TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2
  body.add_geom(
    name="top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half_x, half_y, TABLE_THICKNESS / 2),
    pos=(0.0, 0.0, -TABLE_THICKNESS / 2),
    mass=TABLE_TOP_MASS,
    rgba=(0.62, 0.48, 0.34, 1.0),
    friction=(0.8, 5e-3, 5e-4),
  )
  leg_half = (TABLE_TOP - TABLE_THICKNESS) / 2
  for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
    body.add_geom(
      name=f"leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.025, 0.025, leg_half),
      pos=(sx * (half_x - LEG_INSET), sy * (half_y - LEG_INSET), -TABLE_THICKNESS - leg_half),
      rgba=(0.25, 0.25, 0.27, 1.0),
    )
  return spec


def make_rig(ready_joints: dict[str, float]) -> dict[str, EntityCfg]:
  """The two arms on the table."""
  entities = {
    arm: get_yam_cfg(pos, rot, ready_joints, *_SHELL_BITS[arm]) for arm, (pos, rot) in BASE_POSE.items()
  }
  entities["table"] = EntityCfg(spec_fn=_table_spec)
  return entities


def world_to_base(arm: str, point: tuple[float, float, float]) -> tuple[float, float, float]:
  """A point in the world (env) frame, in ``arm``'s base frame, whose z = 0 is the table top."""
  (bx, by, bz), (w, _, _, z) = BASE_POSE[arm]
  yaw = 2.0 * math.atan2(z, w)
  dx, dy = point[0] - bx, point[1] - by
  return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy, point[2] - bz)
