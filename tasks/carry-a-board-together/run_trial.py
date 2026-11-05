"""Run and score one trial.

    python run_trial.py --policy hold                    # the arm stays still: scores 0
    python run_trial.py --policy demo --video demo.mp4   # the full-score demonstration
    python run_trial.py --policy mymodule:MyPolicy       # yours: MyPolicy().act(obs) -> 7 numbers
    python run_trial.py --screenshot setup.png           # the starting layout, from a wide view of the table

Prints the score as JSON.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib

import mediapy
import numpy as np

from cameras import Overview
from sim.partner import Follower
from sim.scene import LEFT_ARM
from task_env import CONTROL_HZ, READY, TaskEnv

HERE = pathlib.Path(__file__).resolve().parent
DEMO_SEED = 1


class Hold:
  def act(self, obs):
    return np.r_[READY, 1.0]


class Demo:
  """The demonstration: the scripted Follower of ``sim/partner.py`` driving the controlled arm at 20 Hz.  Unlike a
  policy it reads the simulator's true state (where the board and the other arm's hand are), not the cameras: it
  shows that the task can be done with this arm in this scene."""

  def __init__(self, env: TaskEnv) -> None:
    self.env = env
    self.arm = Follower(LEFT_ARM, env.ik, np.random.default_rng(DEMO_SEED), env.dt)
    self.arm.leader = env.partner

  def act(self, obs):
    return self.arm.act(self.env.env, 0, self.env.steps * self.env.dt)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", default="hold")
  parser.add_argument("--video")
  parser.add_argument("--screenshot")
  args = parser.parse_args()
  env = TaskEnv()
  obs = env.reset()
  view = Overview(env.env) if args.video or args.screenshot else None
  if args.screenshot:
    mediapy.write_image(args.screenshot, view.render())
  if args.policy == "hold":
    policy = Hold()
  elif args.policy == "demo":
    policy = Demo(env)
  else:
    module, name = args.policy.split(":")
    policy = getattr(importlib.import_module(module), name)()
  frames = [view.render()] if args.video else []
  while not env.done:
    obs = env.step(policy.act(obs))
    if args.video and env.steps % 2 == 0:
      frames.append(view.render())
  if args.policy == "demo":
    print("\n".join(["follower:", *policy.arm.log, "leader:", *env.partner.log]))
  if args.video:
    mediapy.write_video(args.video, frames, fps=CONTROL_HZ / 2, codec="h264")
  print(json.dumps(env.score()))


if __name__ == "__main__":
  main()
