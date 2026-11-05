"""Damped least-squares IK for one YAM on a tabletop, for checking the rig and scripting motions; not used in training.

Works in the arm's base frame: the table top is z = 0.  A solution counts only if it respects the soft joint limits
and no link shell or finger pad is in the table.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sim.robot import ARM_JOINTS, FINGER_MAX, GRASP_SITE, get_spec

SOFT_LIMIT_FACTOR = 0.9
"""As the actuators train under (``sim.robot``)."""


def down_rotation(jaw: tuple[float, float, float], pos: tuple[float, float, float], lean_deg: float = 6.0) -> np.ndarray:
  """Grasp-site rotation at ``pos`` (base frame) with the jaw opening along the horizontal ``jaw`` and the fingertips
  down, leaning ``lean_deg`` away from the base.  Plumb vertical puts joint 4 on its limit across most of the table."""
  y = np.asarray(jaw, dtype=float)
  z = np.array([0.0, 0.0, -1.0])
  down = np.column_stack([np.cross(y, z), y, z])
  axis = np.cross([pos[0], pos[1], 0.0], [0.0, 0.0, 1.0])
  rot = np.zeros(9)
  quat = np.zeros(4)
  mujoco.mju_axisAngle2Quat(quat, axis / np.linalg.norm(axis), np.radians(lean_deg))
  mujoco.mju_quat2Mat(rot, quat)
  return rot.reshape(3, 3) @ down


class ArmIk:
  def __init__(self) -> None:
    spec = get_spec()
    spec.worldbody.add_geom(
      name="tabletop", type=mujoco.mjtGeom.mjGEOM_BOX, size=(3.0, 3.0, 0.05), pos=(0.0, 0.0, -0.05),
    )
    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    self.site = self.model.site(GRASP_SITE).id
    self.qadr = np.array([self.model.joint(j).qposadr[0] for j in ARM_JOINTS])
    self.vadr = np.array([self.model.joint(j).dofadr[0] for j in ARM_JOINTS])
    lo, hi = np.array([self.model.joint(j).range for j in ARM_JOINTS]).T
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * SOFT_LIMIT_FACTOR
    self.lower, self.upper = mid - half, mid + half
    for j in ("joint7", "joint8"):
      self.data.joint(j).qpos = FINGER_MAX
    self.rng = np.random.default_rng(0)

  def _set(self, q: np.ndarray) -> None:
    self.data.qpos[self.qadr] = q
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_comPos(self.model, self.data)

  def in_table(self, q: np.ndarray) -> bool:
    self._set(q)
    mujoco.mj_collision(self.model, self.data)
    table = self.model.geom("tabletop").id
    return any(table in (c.geom1, c.geom2) and c.dist < 0.0 for c in self.data.contact[:self.data.ncon])

  def _solve_from(
    self, q: np.ndarray, pos: np.ndarray, rot: np.ndarray | None, tilt_deg: float, iters: int,
  ) -> np.ndarray | None:
    jacp, jacr = np.zeros((2, 3, self.model.nv))
    pos_tol, rot_tol = 0.002, np.radians(1.0)
    for _ in range(iters):
      self._set(q)
      site_rot = self.data.site_xmat[self.site].reshape(3, 3)
      e_pos = pos - self.data.site_xpos[self.site]
      if rot is None:
        # Only the fingertip direction matters, and it may lean up to tilt_deg from straight down.
        z = site_rot[:, 2]
        off = np.arccos(np.clip(-z[2], -1.0, 1.0))
        e_rot = np.cross(z, [0.0, 0.0, -1.0])
        norm = np.linalg.norm(e_rot)
        e_rot = e_rot / norm * max(0.0, off - np.radians(tilt_deg)) if norm > 1e-9 else np.zeros(3)
      else:
        quat, err = np.zeros(4), np.zeros(3)
        mujoco.mju_mat2Quat(quat, (rot @ site_rot.T).flatten())
        mujoco.mju_quat2Vel(err, quat, 1.0)
        e_rot = err
      if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
        return q
      mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
      jac = np.vstack([jacp[:, self.vadr], 0.3 * jacr[:, self.vadr]])
      err = np.concatenate([e_pos, 0.3 * e_rot])
      step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
      q = np.clip(q + np.clip(step, -0.2, 0.2), self.lower, self.upper)
    return None

  def solve(
    self, pos, rot: np.ndarray | None = None, tilt_deg: float = 15.0,
    seed: np.ndarray | None = None, starts: int = 24, iters: int = 150, through_table: bool = False,
  ) -> np.ndarray | None:
    """Joints putting the grasp site at ``pos``: at orientation ``rot``, or with the fingertips within ``tilt_deg`` of
    straight down.  Tries ``seed`` first, then random starts pointed roughly at the target.  ``through_table`` accepts
    a pose inside the table, for testing that the table stops it."""
    pos = np.asarray(pos, dtype=float)
    azimuth = np.arctan2(pos[1], pos[0])
    guesses = [] if seed is None else [seed]
    for _ in range(starts):
      q = self.rng.uniform(self.lower, self.upper)
      q[0] = np.clip(azimuth + self.rng.normal(0.0, 0.2), self.lower[0], self.upper[0])
      guesses.append(q)
    for guess in guesses:
      q = self._solve_from(np.array(guess), pos, rot, tilt_deg, iters)
      if q is not None and (through_table or not self.in_table(q)):
        return q
    return None
