"""The rig's cameras, rendered from the simulator's first env.  What a real rig would film, and nothing else."""

from __future__ import annotations

import mujoco
import numpy as np

CAMERAS = ("overhead", "wrist")
IMAGE = (240, 320)
ARM_NAMES = {"left": "left_arm", "right": "right_arm"}


HIDE_SITES = mujoco.MjvOption()
HIDE_SITES.sitegroup[:] = 0
"""Policy images show the scene, not the model's marker sites (the green grasp point, for one)."""


class Cameras:
  def __init__(self, env) -> None:
    self.env = env
    self.model = env.sim.mj_model
    self.data = mujoco.MjData(self.model)
    self.renderer = mujoco.Renderer(self.model, height=IMAGE[0], width=IMAGE[1])

  def sync(self) -> None:
    data = self.env.sim.data
    self.data.qpos[:] = data.qpos[0].cpu().numpy()
    self.data.qvel[:] = data.qvel[0].cpu().numpy()
    if self.model.nmocap:
      self.data.mocap_pos[:] = data.mocap_pos[0].cpu().numpy()
      self.data.mocap_quat[:] = data.mocap_quat[0].cpu().numpy()
    mujoco.mj_forward(self.model, self.data)

  def render(self, arm: str, camera: str, sync: bool = True) -> np.ndarray:
    if sync:
      self.sync()
    name = "table/overhead" if camera == "overhead" else f"{ARM_NAMES[arm]}/wrist"
    self.renderer.update_scene(self.data, camera=name, scene_option=HIDE_SITES)
    return self.renderer.render().copy()


STATION_CAMERAS = {"top": "table/station_top", "left": "left_arm/station_wrist", "right": "right_arm/station_wrist"}


class StationCameras:
  """Dream Labs' calibrated YAM station cameras, as their simulator renders them: 640 by 480, rows 60 to 420 kept."""

  def __init__(self, cameras: Cameras) -> None:
    self.cameras = cameras
    self.renderer = mujoco.Renderer(cameras.model, height=480, width=640)

  def images(self) -> dict[str, np.ndarray]:
    self.cameras.sync()
    out = {}
    for name, camera in STATION_CAMERAS.items():
      self.renderer.update_scene(self.cameras.data, camera=camera, scene_option=HIDE_SITES)
      out[name] = self.renderer.render()[60:420].copy()
    return out


class Overview:
  """A wide, high view of the whole table, for the setup screenshot and the trial videos.  Not given to the policy."""

  def __init__(self, env, width: int = 960, height: int = 720) -> None:
    self.cameras = Cameras(env)
    self.renderer = mujoco.Renderer(self.cameras.model, height=height, width=width)
    self.camera = mujoco.MjvCamera()
    self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.camera.lookat[:] = (0.24, 0.0, 0.84)
    self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.45, 180.0, -38.0

  def render(self) -> np.ndarray:
    self.cameras.sync()
    self.renderer.update_scene(self.cameras.data, camera=self.camera, scene_option=HIDE_SITES)
    return self.renderer.render().copy()
