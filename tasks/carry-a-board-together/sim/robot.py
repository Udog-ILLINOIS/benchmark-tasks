"""The sim YAM: the real arm and gripper, under the real controller.

**Model.**  i2rt's own MJCF for the YAM Pro with the linear_4310 gripper
(``assets/yam_pro_linear_4310``), not mjlab's YAM, which carries the crank
gripper, whose grasp point is about 35 mm nearer the wrist.

**Controller.**  i2rt's PD gains with gravity compensation
(``i2rt/robots/config/yam_pro.yml``).  mjlab's stock gains are a soft 2 Hz
overdamped PD that trails a moving target by about 0.3 s, which leaves the arm
10-20 cm behind a reference recorded under the real, much stiffer one.

**Motors.**  Each arm joint's torque is bounded by its motor's torque-speed line:
full torque standing still, none at the motor's no-load speed (``NO_LOAD_SPEED``,
Damiao's figures at 24 V: DM-J4340 52 rpm on joints 1-3, DM-J4310 200 rpm on 4-6).
Without it the sim swung the wrist at 19 rad/s and the elbow at 9.  The model's
own joints cap every motor at 10 Nm (``actuatorfrcrange``), the DM-J4340s on 1-3
included; ``get_spec`` sets them to ``EFFORT`` (28 Nm there, i2rt's and Damiao's
peak) -- 10 Nm cannot brake the elbow against the shoulder's swing.

**Fingers.**  The model's meshes are visual; contact is through box pads fitted to
the inner face of each fingertip, so a held object is pinched between two flat
faces as it is on the robot.

**Wrist damping.**  i2rt's kd on joints 4-6 is too much for an explicit controller at 5 ms: joint 6 carries about
0.002 kg m^2, kd dt / I is 3.6 and anything over 2 diverges, so it rings at full torque while commanded to stay still.
``WRIST_JOINTS`` take the same kd as MuJoCo joint damping, which the integrator treats implicitly.

**Shell.**  With two arms over one table the links have to be solid too.  Each link gets a capsule fitted to its mesh
(``_add_shell``), massless so the dynamics stay the model's.  The contact bits in ``get_spec`` let a shell hit the
table, the objects and the *other* arm's shell, never its own: the fit is coarse at the joints, where neighbouring
capsules overlap.  The base has none; it is bolted to the table.  A finger's capsule meets only the other arm's shell,
so the pads alone touch what the gripper holds.

Not measured, and worth measuring: ``GRIPPER_FORCE``, and how fast the real
gripper closes (``GRIPPER_NATURAL_FREQ`` reproduces the half second the demos
show, no more).
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

YAM_XML = (pathlib.Path(__file__).resolve().parent.parent
           / "assets" / "yam_pro_linear_4310" / "yam_pro_linear_4310.xml")

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGER_JOINTS = ("joint7", "joint8")
"""Left then right; only the left is actuated, the right follows by an equality."""
FINGER_MAX = 0.0475
"""Finger-joint travel, metres: each finger's distance from the closed centre line."""
GRASP_SITE = "grasp_site"
PAD_GEOMS = ("left_pad", "left_pad_root", "right_pad", "right_pad_root")

I2RT_KP = {"joint1": 80.0, "joint2": 80.0, "joint3": 80.0, "joint4": 40.0, "joint5": 10.0, "joint6": 10.0}
I2RT_KD = {"joint1": 5.0, "joint2": 5.0, "joint3": 5.0, "joint4": 1.5, "joint5": 1.5, "joint6": 1.5}
WRIST_JOINTS = ARM_JOINTS[3:]
ARMATURE = {**{j: 0.032 for j in ARM_JOINTS[:3]}, **{j: 0.0018 for j in ARM_JOINTS[3:]}}   # DM4340, DM4310
EFFORT = {**{j: 28.0 for j in ARM_JOINTS[:3]}, **{j: 10.0 for j in ARM_JOINTS[3:]}}
NO_LOAD_SPEED = {**{j: 52.0 * 2 * np.pi / 60 for j in ARM_JOINTS[:3]}, **{j: 200.0 * 2 * np.pi / 60 for j in ARM_JOINTS[3:]}}
"""rad/s: DM-J4340 (40:1) and DM-J4310 (10:1) at 24 V, from Damiao's data."""

# The gripper motor's rotor, seen through the 96 mm / 6.57 rad transmission, as a
# mass on one finger.
FINGER_ARMATURE = 0.0018 / (0.5 * 0.096 / 6.57) ** 2
GRIPPER_NATURAL_FREQ = 4.0 * 2.0 * np.pi   # 4 Hz, critically damped
# N: the force limit of the one gripper actuator (the left finger; the right follows by an equality), so each
# pad presses with half of it -- 50 N of squeeze, as i2rt's force limiter holds the real gripper to
# (get_yam_robot's limit_gripper_force=50.0) once its fingers stall on an object.
GRIPPER_FORCE = 100.0

# Contact stiffness for the pads.  MuJoCo's default contact is soft in proportion to
# 1/mass of what is being touched, and light objects let each finger sink several
# millimetres under the grip force.  These bring that under a millimetre.  0.01 s is
# the stiffest time constant a 5 ms step allows.
PAD_SOLREF = (0.01, 1.0)
PAD_SOLIMP = (0.99, 0.995, 0.001, 0.5, 2.0)

# Fingertip pads in the grasp-site frame with the gripper closed: (centre, half
# extents) for the left finger; the right is its mirror image in y.  Read off the
# tip meshes -- the last 40 mm of each finger is a flat plate, then it thickens.
_PADS = {
  "pad": ((0.0, 0.003, -0.010), (0.014, 0.003, 0.020)),
  "pad_root": ((0.0, 0.0105, -0.040), (0.019, 0.0085, 0.010)),
}


def _add_pads(spec: mujoco.MjSpec) -> None:
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  site = model.site(GRASP_SITE).id
  site_pos, site_rot = data.site_xpos[site], data.site_xmat[site].reshape(3, 3)
  for side, body_name, mirror in (("left", "tip_left", 1.0), ("right", "tip_right", -1.0)):
    body_id = model.body(body_name).id
    body_pos, body_rot = data.xpos[body_id], data.xmat[body_id].reshape(3, 3)
    rot = body_rot.T @ site_rot
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    for name, (centre, half) in _PADS.items():
      centre = np.array(centre) * (1.0, mirror, 1.0)
      spec.body(body_name).add_geom(
        name=f"{side}_{name}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=body_rot.T @ (site_pos + site_rot @ centre - body_pos),
        quat=quat,
        mass=0.0,
        friction=(1.0, 5e-3, 5e-4),
        condim=4,
        priority=1,
        solref=PAD_SOLREF,
        solimp=PAD_SOLIMP,
        contype=1,
        conaffinity=1,
        group=3,
        rgba=(0.9, 0.3, 0.3, 0.4),
      )


def _add_shell(spec: mujoco.MjSpec, shell_bit: int, other_shell_bit: int) -> None:
  for geom in list(spec.geoms):
    body = geom.parent.name
    if body == "arm":
      continue  # Bolted to the table.
    # A finger's capsule meets only the other arm's shell: anywhere else it would push objects off the pads.
    conaffinity = other_shell_bit if body in ("tip_left", "tip_right") else other_shell_bit | 1
    geom.parent.add_geom(
      name=f"{geom.meshname}_shell",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      meshname=geom.meshname,
      pos=geom.pos,
      quat=geom.quat,
      mass=0.0,
      friction=(0.5, 5e-3, 5e-4),
      contype=shell_bit,
      conaffinity=conaffinity,
      group=3,
      rgba=(0.3, 0.6, 0.9, 0.3),
    )


def get_spec(shell_bit: int = 2, other_shell_bit: int = 4) -> mujoco.MjSpec:
  """The arm's spec.  ``shell_bit`` is this arm's contact bit, ``other_shell_bit`` the bit of the arm it must not pass
  through; give the two arms the two opposite ways round.  Everything else in the scene is on bit 1."""
  spec = mujoco.MjSpec.from_file(str(YAM_XML))
  for geom in spec.geoms:
    geom.contype, geom.conaffinity = 0, 0
  _add_shell(spec, shell_bit, other_shell_bit)
  for body in spec.bodies:
    if body.name != "world":
      body.gravcomp = 1.0
  for name in ARM_JOINTS:
    spec.joint(name).actfrcrange = (-EFFORT[name], EFFORT[name])
  for name in WRIST_JOINTS:
    spec.joint(name).damping = (I2RT_KD[name], 0.0, 0.0)
  # Only the left finger gets an actuator, and with it the motor's armature;
  # give the follower the same so the pair closes symmetrically.
  spec.joint(FINGER_JOINTS[1]).armature = FINGER_ARMATURE
  _add_pads(spec)
  # Between the fingers' root and the palm's edge, looking along the fingers.
  spec.body("gripper").add_camera(name="wrist", pos=(0.06, 0.0, -0.02), euler=(0.0, -0.45, 0.0), fovy=70.0)
  _add_station_wrist_camera(spec)
  return spec


# Dream Labs' calibrated YAM station wrist camera (dreamlabs SDK, dream_examples/yam_mujoco/assets/sim-eval-yam.json),
# given in their link_6 frame.  Our link6 frame is theirs turned by LINK6_FLIP (constant to 1e-5 over their reference
# poses), so the mount is turned the same way: a camera Dream-1 was trained with sees what it would see on the station.
STATION_WRIST = {"pos": (0.0, 0.069638, 0.072), "quat": (-0.003227, 0.002817, 0.975619, 0.21943),
                 "focal": (390.666, 390.162), "centre": (317.526, 236.146)}
LINK6_FLIP = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _station_camera(body, name: str, pos, quat, focal, centre) -> None:
  quat = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
  body.add_camera(
    name=name, pos=pos, quat=quat, resolution=(640, 480), sensor_size=(0.0064, 0.0048),
    focal_pixel=focal, principal_pixel=(320.0 - centre[0], centre[1] - 240.0),
  )


def _add_station_wrist_camera(spec: mujoco.MjSpec) -> None:
  rotation = np.empty(9)
  mujoco.mju_quat2Mat(rotation, np.asarray(STATION_WRIST["quat"], dtype=float) / np.linalg.norm(STATION_WRIST["quat"]))
  quat = np.empty(4)
  mujoco.mju_mat2Quat(quat, (LINK6_FLIP @ rotation.reshape(3, 3)).flatten())
  _station_camera(spec.body("link6"), "station_wrist", LINK6_FLIP @ np.array(STATION_WRIST["pos"]), quat,
                  STATION_WRIST["focal"], STATION_WRIST["centre"])


def get_yam_cfg(
  pos: tuple[float, float, float],
  rot: tuple[float, float, float, float],
  joint_pos: dict[str, float],
  shell_bit: int = 2,
  other_shell_bit: int = 4,
) -> EntityCfg:
  """YAM Pro + linear_4310 with its base at ``pos``, ``rot`` (wxyz), starting at ``joint_pos``."""
  arm = tuple(
    DcMotorActuatorCfg(
      target_names_expr=(name,),
      stiffness=I2RT_KP[name],
      damping=0.0 if name in WRIST_JOINTS else I2RT_KD[name],
      effort_limit=EFFORT[name],
      saturation_effort=EFFORT[name],
      velocity_limit=NO_LOAD_SPEED[name],
      armature=ARMATURE[name],
    )
    for name in ARM_JOINTS
  )
  gripper = BuiltinPositionActuatorCfg(
    target_names_expr=(FINGER_JOINTS[0],),
    stiffness=FINGER_ARMATURE * GRIPPER_NATURAL_FREQ ** 2,
    damping=2.0 * FINGER_ARMATURE * GRIPPER_NATURAL_FREQ,
    effort_limit=GRIPPER_FORCE,
    armature=FINGER_ARMATURE,
  )
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=pos,
      rot=rot,
      joint_pos={**joint_pos, "joint[78]": FINGER_MAX},
      joint_vel={".*": 0.0},
    ),
    spec_fn=lambda: get_spec(shell_bit, other_shell_bit),
    articulation=EntityArticulationInfoCfg(
      actuators=(*arm, gripper),
      soft_joint_pos_limit_factor=0.9,
    ),
  )
