"""The scripted arms, each reading the scene's true state.

    arm = Leader(arm, ik, rng, dt)        # the task's other arm: grips its end, waits for yours, and leads the carry
    arm = Follower(arm, ik, rng, dt)      # the demonstration: grips its end and keeps it level with the leader's
    row = arm.act(env, e, now)            # every step: 6 joint targets (rad) and the gripper, 0 closed to 1 open

The leader grips the board in the middle of its own band, then waits until the controlled arm has held the board in the
middle part of the other band for ``READY_FOR`` seconds.  Then, at the same pace in every trial, it lifts its end
``LIFT`` straight up, carries it forward to over its cradle, lowers it onto the seat, lets go and goes back to its ready pose.
It does not watch the board or the other arm once it has started: a partner who has said "lift" and is lifting.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.envs import ManagerBasedRlEnv

from sim import benchmark as b
from sim import props
from sim.env_cfg import READY_JOINTS
from sim.ik import ArmIk
from sim.robot import FINGER_MAX, GRASP_SITE
from sim.scene import LEFT_ARM, world_to_base

READY_Q = np.array(list(READY_JOINTS.values()))
LIFT = 0.08
"""How high the leader lifts its end: the board's underside goes from 5 cm to 13 cm, 5 cm over the cradles' uprights."""
READY_FOR = 1.0
"""Seconds the controlled arm must have held its band before the leader lifts."""
HOLD_REACH = 0.015
"""The controlled arm holds its band when its grasp point is within this of the board's axis..."""
HOLD_BAND = props.BAND_WIDTH / 2
"""...within this of the band's middle along the board..."""
HOLD_OPENING = 0.6
"""...and its gripper is closed to under this (0 closed, 1 open): on the 3 cm board the fingers stop at about 0.32."""


def turn(axis, degrees: float) -> np.ndarray:
  q, m = np.zeros(4), np.zeros(9)
  mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float), np.radians(degrees))
  mujoco.mju_quat2Mat(m, q)
  return m.reshape(3, 3)


def plumb(jaw: tuple[float, float, float]) -> np.ndarray:
  """Fingers straight down, the jaw (closing direction) along the horizontal ``jaw``."""
  y = np.asarray(jaw, dtype=float)
  z = np.array([0.0, 0.0, -1.0])
  return np.column_stack([np.cross(y, z), y, z])


GRASP_ROT = plumb((1.0, 0.0, 0.0))
"""Both arms take the board from above, fingers down, the jaw closing across the board's 3 cm width."""


def pos(env: ManagerBasedRlEnv, name: str, e: int) -> np.ndarray:
  return (env.scene[name].data.root_link_pos_w[e] - env.scene.env_origins[e]).cpu().numpy()


def site(env: ManagerBasedRlEnv, entity: str, name: str, e: int) -> np.ndarray:
  thing = env.scene[entity]
  return (thing.data.site_pos_w[e, thing.site_names.index(name)] - env.scene.env_origins[e]).cpu().numpy()


def hand(env: ManagerBasedRlEnv, arm: str, e: int) -> np.ndarray:
  return site(env, arm, GRASP_SITE, e)


def opening(env: ManagerBasedRlEnv, arm: str, e: int) -> float:
  """The gripper's opening, 0 closed to 1 open."""
  return float(env.scene[arm].data.joint_pos[e, 6].item() / FINGER_MAX)


def band(arm: str) -> str:
  return "left_band" if arm == LEFT_ARM else "right_band"


def in_board(env: ManagerBasedRlEnv, point: np.ndarray, e: int) -> np.ndarray:
  """``point`` in the board's frame: x across it, y along it (toward the left band), z up through it."""
  q = env.scene["board"].data.root_link_quat_w[e].cpu().numpy().astype(float)
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, q)
  return rot.reshape(3, 3).T @ (point - pos(env, "board", e))


def holds_band(env: ManagerBasedRlEnv, arm: str, e: int) -> bool:
  """Whether ``arm`` has the board in its gripper, in the middle part of its own band."""
  local = in_board(env, hand(env, arm, e), e)
  middle = props.BAND_Y if arm == LEFT_ARM else -props.BAND_Y
  return bool(np.hypot(local[0], local[2]) < HOLD_REACH and abs(local[1] - middle) < HOLD_BAND
              and opening(env, arm, e) < HOLD_OPENING)


class Mover:
  """A queue of (joint targets, gripper) rows, refilled stage by stage."""

  def __init__(self, arm: str, ik: ArmIk, rng: np.random.Generator, dt: float) -> None:
    self.arm, self.ik, self.rng, self.dt = arm, ik, rng, dt
    self.q = READY_Q.copy()
    self.current = READY_Q.copy()
    self.grip = 1.0
    self.queue: list[tuple[np.ndarray, float]] = []
    self.rot = GRASP_ROT
    self.stage = "wait"
    self.log: list[str] = []
    self.failed = None

  def solve(self, point) -> np.ndarray:
    q = self.ik.solve(world_to_base(self.arm, tuple(float(v) for v in point)), rot=self.rot, seed=self.q, starts=60,
                      iters=200)
    if q is None:
      raise RuntimeError(f"no IK at {np.round(point, 3).tolist()}")
    return q

  def glide(self, q: np.ndarray, seconds: float, grip: float) -> None:
    n = max(1, int(seconds / self.dt))
    for i in range(1, n + 1):
      s = i / n
      self.queue.append((self.q + (q - self.q) * (3 * s ** 2 - 2 * s ** 3), grip))
    self.q = q

  def line(self, a, b_, seconds: float, grip: float) -> None:
    """A straight line, eased in and out."""
    n = max(1, int(seconds / self.dt))
    a, b_ = np.asarray(a, dtype=float), np.asarray(b_, dtype=float)
    for i in range(1, n + 1):
      s = i / n
      self.q = self.solve(a + (b_ - a) * (3 * s ** 2 - 2 * s ** 3))
      self.queue.append((self.q, grip))

  def hold(self, seconds: float, grip: float) -> None:
    self.queue += [(self.q, grip)] * max(1, int(seconds / self.dt))

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    raise NotImplementedError

  def act(self, env: ManagerBasedRlEnv, e: int, now: float) -> np.ndarray:
    if not self.queue and self.stage not in ("done", "failed"):
      try:
        self.plan(env, e, now)
      except RuntimeError as err:
        self.log.append(f"t={now:.1f} {self.stage}: {err}")
        self.failed = str(err)
        self.queue.clear()
        self.q = self.current.copy()
        self.hold(0.5, 1.0)
        self.glide(READY_Q, 1.2, 1.0)
        self.stage = "failed"
    if self.queue:
      self.current, self.grip = self.queue.pop(0)
    return np.r_[self.current, self.grip]

  def _take(self, env: ManagerBasedRlEnv, e: int) -> np.ndarray:
    """Down onto the middle of this arm's band from 6 cm above it, and close.  Returns the grasp point."""
    grasp = site(env, "board", band(self.arm), e)
    above = grasp + [0.0, 0.0, 0.06]
    self.glide(self.solve(above), 1.5, 1.0)
    self.line(above, grasp, 0.8, 1.0)
    self.hold(0.6, 0.0)
    return grasp


class Leader(Mover):
  """The task's other arm."""

  START = 1.0
  """Seconds before it reaches for the board."""

  def __init__(self, arm: str, ik: ArmIk, rng: np.random.Generator, dt: float) -> None:
    super().__init__(arm, ik, rng, dt)
    self.ready_since: float | None = None
    self.carrying = False
    """True from the moment it starts to lift until it has set its end down and starts to open."""
    self.lifted_at: float | None = None

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    if self.stage == "wait":
      if now >= self.START:
        self.grasp = self._take(env, e)
        self.stage = "waiting"
        self.log.append(f"t={now:.1f}: taking its end")
      return
    if self.stage == "waiting":
      if holds_band(env, b.LEFT_ARM, e):
        self.ready_since = now if self.ready_since is None else self.ready_since
      else:
        self.ready_since = None
      if self.ready_since is None or now - self.ready_since < READY_FOR:
        self.hold(self.dt, 0.0)
        return
      self.log.append(f"t={now:.1f}: the other end is held; lifting")
      self.carrying, self.lifted_at = True, now
      g = self.grasp
      up = g + [0.0, 0.0, LIFT]
      # To the cradles, wherever on its blocks the board was: over the slot, then straight down onto the seat.
      down = np.array([b.GOAL_X, g[1], b.Z + b.BOARD_Z + 0.002])
      ahead = down + [0.0, 0.0, up[2] - down[2]]
      self.set_at = down
      self.line(g, up, 2.0, 0.0)
      self.hold(0.5, 0.0)
      self.line(up, ahead, 3.0, 0.0)
      self.hold(0.5, 0.0)
      self.line(ahead, down, 2.0, 0.0)
      self.hold(0.5, 0.0)
      self.stage = "set_down"
      return
    if self.stage == "set_down":
      self.carrying = False
      self.log.append(f"t={now:.1f}: set its end down; opening")
      self.hold(0.8, 1.0)
      self.stage = "open"
      return
    if self.stage == "open":
      self.log.append(f"t={now:.1f}: let go; going back")
      g = self.set_at
      self.line(g, g + [0.0, 0.0, 0.07], 0.8, 1.0)
      self.glide(READY_Q, 1.5, 1.0)
      self.stage = "back"
      return
    if self.stage == "back":
      self.stage = "done"


class Follower(Mover):
  """The demonstration: takes its own band, then every step puts its hand where the leader's hand is, mirrored across
  the centre line (same height, same distance forward), until the leader has set its end down and let go; then lets
  go too and goes back.

  It reads the leader's hand as it is, not where the leader is about to go, so it trails by about one control step."""

  def __init__(self, arm: str, ik: ArmIk, rng: np.random.Generator, dt: float) -> None:
    super().__init__(arm, ik, rng, dt)
    self.leader: Leader | None = None
    self.offset = np.zeros(3)

  def plan(self, env: ManagerBasedRlEnv, e: int, now: float) -> None:
    if self.stage == "wait":
      if now < 0.3:  # Until the simulation has stepped, the board's pose is not yet read back.
        return
      grasp = self._take(env, e)
      self.grasp = grasp
      self.stage = "holding"
      self.log.append(f"t={now:.1f}: taking its end")
      return
    if self.stage == "holding":
      leader = self.leader
      if leader is None or leader.stage in ("back", "done"):
        self.log.append(f"t={now:.1f}: the leader let go; letting go")
        self.hold(0.6, 1.0)
        here = self.grasp_now(env, e)
        self.line(here, here + [0.0, 0.0, 0.07], 0.8, 1.0)
        self.glide(READY_Q, 1.5, 1.0)
        self.stage = "back"
        return
      if not leader.carrying:
        self.hold(self.dt, 0.0)
        return
      lead = hand(env, leader.arm, e)
      target = np.array([lead[0], self.grasp[1], lead[2]])
      self.q = self.solve(target)
      self.queue.append((self.q, 0.0))
      return
    if self.stage == "back":
      self.stage = "done"

  def grasp_now(self, env: ManagerBasedRlEnv, e: int) -> np.ndarray:
    return hand(env, self.arm, e)
