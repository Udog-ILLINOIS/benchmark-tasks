"""Run Dream-1 on the task, as Dream Labs' own YAM MuJoCo example drives a simulator.

    uv sync --extra dream
    DREAM_API_KEY=... MUJOCO_GL=egl uv run python dream_run.py --out dream_run/       # one trial; uses account credits
    MUJOCO_GL=egl uv run python dream_run.py --dry-run                                # checks the request; no API call

Dream-1 is a two-arm YAM model, so it is sent what a YAM station sends: both arms' joints and gripper openings, and the
last nine frames, 0.2 s apart, of the station's three calibrated cameras (top, left wrist, right wrist; 640 by 480,
rows 60 to 420, sent as 320 by 176 JPEG). The request keys and session settings follow the SDK's
``dream_examples/yam_mujoco/dream_control.py`` for ``dreamlabs/dream-1-beta``. The controlled arm is the left one
(``q_l_chunk``); the right arm is the task's scripted second arm, so Dream-1's right-arm rows are not executed. Simulated
time stands still while a request is out, as in the SDK's example. The camera windows start filled with the starting
frame, so the trial's first step is Dream-1's.

The key is read by the dream client from ``DREAM_API_KEY`` or ``~/.dreamlabs/credentials``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import time
from collections import deque

import mediapy
import numpy as np
from PIL import Image

from cameras import Overview, StationCameras
from task_env import TaskEnv
from sim.robot import FINGER_MAX
from sim.scene import LEFT_ARM, RIGHT_ARM

HERE = pathlib.Path(__file__).resolve().parent
INSTRUCTION = json.loads((HERE / "task.json").read_text())["instruction"]
MODEL = "dreamlabs/dream-1-beta"
FRAMES, FRAME_EVERY_STEPS, FRAME_SIZE = 9, 4, (320, 176)
STATION_ROBOT = {"robot.arm": "yam", "robot.exec_head": "joint", "robot.max_joint_vel_rad_s": 0.0, "robot.gripper_open": 1.0}
SLOTS = {"left": ("arm_l", LEFT_ARM), "right": ("arm_r", RIGHT_ARM)}


def jpeg(rgb: np.ndarray) -> str:
  buffer = io.BytesIO()
  Image.fromarray(rgb).resize(FRAME_SIZE, Image.BILINEAR).save(buffer, format="JPEG", quality=85)
  return base64.b64encode(buffer.getvalue()).decode("ascii")


def arm_state(env: TaskEnv) -> dict:
  state = {}
  for slot, entity in SLOTS.values():
    data = env.env.scene[entity].data
    state[f"state.{slot}.joint_pos"] = data.joint_pos[0, :6].cpu().numpy().astype(float).tolist()
    state[f"state.{slot}.gripper_pos"] = [float(np.clip(data.joint_pos[0, 6].item() / FINGER_MAX, 0.0, 1.0))]
  return {**state, **STATION_ROBOT, "prompt": INSTRUCTION}


def feed(client, env: TaskEnv, windows: dict[str, deque], steps: int, all_frames: bool = False) -> None:
  """Hand the client the newest frame of each camera, or the whole windows, oldest first, for a new session."""
  state = arm_state(env)
  count = len(windows["top"])
  for i in range(count) if all_frames else [count - 1]:
    frames = {f"image_window.{name}": windows[name][i] for name in windows}
    seconds = max((steps - (count - 1 - i) * FRAME_EVERY_STEPS) * env.dt, 0.0)
    client.set_observation(frames, state, timestamp=seconds + i * 1e-6)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--model", default=MODEL)
  parser.add_argument("--actions", type=int, default=24, help="rows of each reply executed before asking again (1 to 48)")
  parser.add_argument("--out", type=pathlib.Path, default=HERE / "dream_run")
  parser.add_argument("--dry-run", action="store_true", help="build and check the first request, call nothing")
  parser.add_argument("--key-file", type=pathlib.Path, help="a file holding only the API key; it is never printed")
  args = parser.parse_args()
  if args.key_file:
    os.environ["DREAM_API_KEY"] = args.key_file.read_text().strip()
  if not 1 <= args.actions <= 48:
    parser.error("--actions is 1 to 48")

  env = TaskEnv()
  env.reset()
  station = StationCameras(env.cameras)
  view = Overview(env.env)
  first = {name: jpeg(rgb) for name, rgb in station.images().items()}
  windows = {name: deque([frame] * FRAMES, maxlen=FRAMES) for name, frame in first.items()}

  if args.dry_run:
    state = arm_state(env)
    sizes = {name: Image.open(io.BytesIO(base64.b64decode(w[0]))).size for name, w in windows.items()}
    print(json.dumps({"model": args.model, "prompt": state["prompt"], "frames_per_camera": {k: len(w) for k, w in windows.items()},
                      "frame_size": sizes, "state_keys": sorted(k for k in state if k.startswith("state.")),
                      "left_state": state["state.arm_l.joint_pos"] + state["state.arm_l.gripper_pos"]}, indent=1))
    return

  import dream

  args.out.mkdir(parents=True, exist_ok=True)
  client = dream.DreamClient()
  client.policy({"checkpoint": args.model, "timeout_ms": 45000, "strict": False, "action_hz": 20, "_record": False})
  steps, requests, rtts, rows_run = 0, 0, [], 0
  film = [view.render()]
  try:
    feed(client, env, windows, steps, all_frames=True)
    while not env.done:
      started = time.monotonic()
      try:
        reply = client.get_action()
      except Exception as error:
        if type(error).__name__ == "DreamSessionExpired":
          feed(client, env, windows, steps, all_frames=True)
          continue
        raise
      rtts.append(time.monotonic() - started)
      requests += 1
      left = np.asarray(reply["q_l_chunk"], dtype=float)
      if left.ndim != 2 or left.shape[1] != 7 or not np.isfinite(left).all():
        raise ValueError(f"unexpected left chunk shape {left.shape}")
      for row in left[:args.actions]:
        if env.done:
          break
        row = row.copy()
        row[6] = np.clip(row[6], 0.0, 1.0)
        env.step(row)
        steps += 1
        rows_run += 1
        if steps % 2 == 0:
          film.append(view.render())
        if steps % FRAME_EVERY_STEPS == 0:
          for name, rgb in station.images().items():
            windows[name].append(jpeg(rgb))
          feed(client, env, windows, steps)
      print(f"request {requests}: {rtts[-1]:.2f} s, {rows_run} rows run, step {env.steps}", flush=True)
  finally:
    client.close()
  result = {**env.score(), "model": args.model, "requests": requests, "rows_per_request": args.actions,
            "mean_rtt_s": round(float(np.mean(rtts)), 2) if rtts else None}
  mediapy.write_video(args.out / "dream_run.mp4", film, fps=10, codec="h264")
  (args.out / "score.json").write_text(json.dumps(result, indent=2))
  print(json.dumps(result))


if __name__ == "__main__":
  main()
