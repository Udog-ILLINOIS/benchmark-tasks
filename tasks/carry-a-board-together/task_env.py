"""The task: your YAM arm carries a board with a second YAM arm that leads, and sets it into two cradles.

    from task_env import TaskEnv
    env = TaskEnv()
    obs = env.reset()
    while not env.done:
      obs = env.step(action)              # 7 numbers: six joint targets (rad), then the gripper, 0 closed to 1 open
    print(env.score())

The controlled arm is the left one (positive y): on the right of the screenshot, which looks toward the arms.  The
second arm (the right one) is scripted (``sim/partner.py``): it grips the orange band, waits until the controlled arm
holds the blue band, then lifts, carries and sets down its end the same way in every trial.  Control is at 20 Hz,
``RUN_LIMIT`` steps at most.  Observations: ``joints`` (13: six joint angles, the gripper opening 0 to 1, six joint
speeds), and two 240 by 320 colour images, ``overhead`` and ``wrist``.
"""

from __future__ import annotations

import numpy as np
import torch

from cameras import Cameras
from sim import benchmark as b
from sim.benchmark import make_carry_cfg
from sim.env_cfg import ACTION_SCALE, READY_JOINTS
from sim.ik import ArmIk
from sim.partner import Leader, hand, in_board, opening, pos, site
from sim.robot import ARM_JOINTS, FINGER_MAX
from sim.scene import LEFT_ARM, RIGHT_ARM

from mjlab.envs import ManagerBasedRlEnv

CONTROL_HZ = 20
RUN_LIMIT = 600
"""Control steps: 30 seconds at 20 Hz.  The other arm's carry takes about 12 s once it starts."""
READY = np.array([READY_JOINTS[j] for j in ARM_JOINTS])
PARTNER_SEED = 0
"""Seeds the second arm (it draws nothing at random; kept for the shared interface)."""
LEVEL = 0.04
"""While the other arm carries, the two bands' middles must stay within 4 cm of the same height."""
SLIP = 0.02
"""While the other arm carries, the board must not move more than 2 cm in its grip."""
EARLY = 0.02
"""Before the other arm lifts, the board must not move more than 2 cm (either band's middle)."""
LIFTED = 0.03
"""Lifted together: both bands' middles at least 3 cm above where they started, at the same moment."""
SEATED_X = 0.015
"""In the cradles: the board's middle within 1.5 cm of the cradles' line front to back (the slot leaves 5 mm each
side)..."""
SEATED_Y = 0.03
"""...within 3 cm of the centre line along its length..."""
SEATED_Z = 0.006
"""...and both bands' middles within 6 mm of their resting height: on the seats, level."""
LET_GO = 0.5
"""The controlled arm has let go when its gripper is more than half open."""


def _row(action) -> np.ndarray:
  return np.r_[(np.asarray(action[:6]) - READY) / ACTION_SCALE, np.clip(action[6], 0.0, 1.0)]


class TaskEnv:
  def __init__(self, device: str | None = None) -> None:
    cfg = make_carry_cfg()
    cfg.episode_length_s = RUN_LIMIT / CONTROL_HZ + 10.0
    cfg.auto_reset = False  # Keep the final state to score.
    del cfg.terminations["arm_collision"]  # Watched here, every simulation step.
    self.env = ManagerBasedRlEnv(cfg, device=device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    self.dt = 1.0 / CONTROL_HZ
    self.ticks_per_s = round(1.0 / self.env.step_dt)
    self.cameras = Cameras(self.env)
    self.ik = ArmIk()

  def _obs(self) -> dict:
    data = self.env.scene[LEFT_ARM].data
    joints = torch.cat([data.joint_pos[0, :6], (data.joint_pos[0, 6] / FINGER_MAX)[None], data.joint_vel[0, :6]])
    self.cameras.sync()
    return {"joints": joints.cpu().numpy(), "overhead": self.cameras.render("left", "overhead", sync=False),
            "wrist": self.cameras.render("left", "wrist", sync=False)}

  def reset(self) -> dict:
    self.env.reset(seed=0)
    self.steps = self.ticks = 0
    self.done = False
    self.fault: str | None = None
    self.partner = Leader(RIGHT_ARM, self.ik, np.random.default_rng(PARTNER_SEED), self.env.step_dt)
    self.start_bands: dict[str, np.ndarray] | None = None
    self.grip_at: np.ndarray | None = None
    self.lifted_together = False
    self.worst_level = 0.0
    self.worst_slip = 0.0
    return self._obs()

  def _bands(self) -> dict[str, np.ndarray]:
    return {s: site(self.env, "board", f"{s}_band", 0) for s in ("left", "right")}

  def _check(self) -> str | None:
    """What went wrong this simulation step, if anything."""
    if bool(self.env.scene["arm_contact"].data.found[0, 0] > 0):
      return "the arms touched"
    bands = self._bands()
    if not all(np.isfinite(p).all() for p in bands.values()):
      raise RuntimeError(f"t={self.ticks * self.env.step_dt:.1f}: the simulation became unstable; the trial is invalid")
    if self.start_bands is None:
      self.start_bands = bands
    if min(bands[s][2] - self.start_bands[s][2] for s in bands) >= LIFTED:
      self.lifted_together = True
    leader = self.partner
    if not leader.carrying and leader.lifted_at is None:
      moved = max(np.linalg.norm(bands[s] - self.start_bands[s]) for s in bands)
      if moved > EARLY:
        return "the board moved more than 2 cm before the other arm lifted"
      return None
    if leader.carrying:
      local = in_board(self.env, hand(self.env, RIGHT_ARM, 0), 0)
      if self.grip_at is None:
        self.grip_at = local
      slip = float(np.linalg.norm(local - self.grip_at))
      self.worst_slip = max(self.worst_slip, slip)
      if slip > SLIP:
        return "the board moved more than 2 cm in the other arm's grip"
      level = float(abs(bands["left"][2] - bands["right"][2]))
      self.worst_level = max(self.worst_level, level)
      if level > LEVEL:
        return "the two ends were more than 4 cm apart in height"
    return None

  def seated(self) -> bool:
    middle = pos(self.env, "board", 0)
    bands = self._bands()
    return bool(abs(middle[0] - b.GOAL_X) < SEATED_X and abs(middle[1]) < SEATED_Y
                and all(abs(p[2] - (b.Z + b.BOARD_Z)) < SEATED_Z for p in bands.values()))

  def let_go(self) -> bool:
    return opening(self.env, LEFT_ARM, 0) > LET_GO

  def step(self, action) -> dict:
    if self.done:
      raise RuntimeError("the trial is over; call reset()")
    row = np.asarray(action, dtype=np.float64)
    if row.shape != (7,) or not np.isfinite(row).all():
      raise ValueError("action must be 7 finite numbers")
    own = _row(row)
    self.steps += 1
    # The simulation steps at 50 Hz, the second arm with it; the controlled arm holds its command for the control step.
    until = self.steps * self.ticks_per_s // CONTROL_HZ
    while self.ticks < until and not self.done:
      other = _row(self.partner.act(self.env, 0, self.ticks * self.env.step_dt))
      vector = torch.tensor(np.r_[own, other], dtype=torch.float32, device=self.env.device)[None]
      self.env.step(vector)
      self.ticks += 1
      self.fault = self._check()
      self.done = self.fault is not None or (self.partner.stage == "done" and self.seated() and self.let_go())
    self.done = self.done or self.steps >= RUN_LIMIT
    return self._obs()

  def score(self) -> dict:
    """1.0 if the board ends in both cradles and the controlled arm has let go; 0.5 if it is in the cradles but still
    held; 0.25 if it was lifted together but did not end in the cradles; otherwise 0.  0 whatever else if the arms
    touched, the board moved more than 2 cm before the other arm lifted, the ends were ever more than 4 cm apart in
    height while the other arm carried, or the board moved more than 2 cm in the other arm's grip."""
    seated, let_go = self.seated(), self.let_go()
    if self.fault:
      value = 0.0
    elif seated:
      value = 1.0 if let_go else 0.5
    else:
      value = 0.25 if self.lifted_together else 0.0
    return {"score": value, "in_cradles": seated, "let_go": let_go, "lifted_together": self.lifted_together,
            "fault": self.fault, "worst_level_cm": round(100 * self.worst_level, 1),
            "worst_slip_cm": round(100 * self.worst_slip, 1), "other_arm": self.partner.stage, "steps": self.steps}
