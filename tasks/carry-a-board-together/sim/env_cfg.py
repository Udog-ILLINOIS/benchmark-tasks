"""How the arms are driven: per arm, six joint position targets about ``READY_JOINTS`` and a gripper command.

Physics steps at 5 ms; the task drives the arms at 20 Hz (``task_env``).
"""

from __future__ import annotations

from mjlab.envs.mdp.actions import JointPositionActionCfg

from sim.robot import ARM_JOINTS, FINGER_JOINTS, FINGER_MAX

READY_JOINTS = {"joint1": 0.0, "joint2": 0.63, "joint3": 0.74, "joint4": -0.89, "joint5": 0.0, "joint6": 0.0}
"""Gripper 0.22 m out and 0.18 m above the table, pointing 45 deg down, well clear of the other arm."""
ACTION_SCALE = 0.5
"""rad per unit of action, about ``READY_JOINTS``."""
PHYSICS_DT = 0.005
DECIMATION = 4
"""Physics steps per control step before ``task_env`` sets the 20 Hz control rate."""


def arm_actions(arm: str) -> dict[str, JointPositionActionCfg]:
  return {
    f"{arm}_joints": JointPositionActionCfg(
      entity_name=arm, actuator_names=ARM_JOINTS, scale=ACTION_SCALE, use_default_offset=True, preserve_order=True,
    ),
    f"{arm}_gripper": JointPositionActionCfg(
      entity_name=arm, actuator_names=(FINGER_JOINTS[0],), scale=FINGER_MAX, use_default_offset=False,
    ),
  }
