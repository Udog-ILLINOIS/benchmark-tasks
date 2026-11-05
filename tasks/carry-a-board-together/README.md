# Carry a board with another robot arm, following its lead

A 60 cm pine board lies across two blocks in front of two YAM arms. The other arm grips the orange tape at its end and
waits. Once your arm has held the blue tape at your end for a second, the other arm lifts its end 8 cm, carries it 15 cm
forward and lowers it into its cradle, at the same pace in every trial. Your arm has to move with it: keep your end
level with theirs, don't drag the board through their grip, set your end into your cradle and let go.

Nothing tells your arm when or where the other arm will move; it has to read that from what it sees. Getting ahead
(lifting before the other arm does), lagging behind, or pulling against the leader scores 0.

## Running a trial

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/). Uses a CUDA GPU if there is one, otherwise the CPU.

    uv sync
    MUJOCO_GL=egl uv run python run_trial.py --policy hold                  # arm stays still: 0.0
    MUJOCO_GL=egl uv run python run_trial.py --policy demo --video demo.mp4  # the demo: 1.0
    MUJOCO_GL=egl uv run python run_trial.py --policy mymodule:MyPolicy      # your policy
    MUJOCO_GL=egl uv run python run_trial.py --screenshot setup.png          # starting layout

On a Mac, leave out `MUJOCO_GL=egl`. Without a GPU or EGL on Linux, use `MUJOCO_GL=glfw xvfb-run -a uv run python ...`.
Each command prints the score as JSON, for example:

    {"score": 1.0, "in_cradles": true, "let_go": true, "lifted_together": true, "fault": null, "worst_level_cm": 0.7, "worst_slip_cm": 0.0, "other_arm": "done", "steps": 331}

**Runtime:** about 4 to 5 minutes per trial on a 4-core Linux CPU, mostly compiling and stepping the simulation; the
first run also compiles the simulator's kernels.

**Seeds:** none. The layout is fixed, and the other arm moves the same way every trial once your arm holds its end.

## Interface

`task_env.TaskEnv`: `reset()` and `step(action)` return the observation, `done` says when the trial is over, and
`score()` returns the result.

- **Control:** 20 Hz, at most 600 steps (30 s). The simulation runs at 50 Hz; your arm holds each command for the
  whole control step.
- **Action:** 7 numbers: six joint angle targets in radians, then the gripper, from 0 (closed) to 1 (open).
- **Observation:** `joints` (13 floats: six joint angles, gripper opening from 0 to 1, six joint speeds), and
  `overhead` and `wrist`, 240 x 320 RGB images from a camera above the table and one on your wrist.
- **Your arm:** the left arm in the scene (positive y). It's on the right in the screenshot, behind the blue tape.

## The other arm

`Leader` in `sim/partner.py`. After 1 s it grips the middle of the orange band from above. Then it waits until your
arm counts as holding the board: your gripper closed to under 60% open, with its grasp point within 1.5 cm of the
board's axis and within 4 cm of the middle of the blue band, for 1 s straight. Then it lifts its end 8 cm in 2 s,
pauses 0.5 s, carries it forward in 3 s to over its cradle, pauses 0.5 s, lowers it onto the cradle's seat in 2 s,
pauses 0.5 s, opens over 0.8 s and moves back to its ready pose. Once it has started lifting it doesn't watch your arm.
If your arm never holds its end, it waits for the whole trial.

## Scoring

- **1.0:** at the end the board rests in both cradles (its middle within 1.5 cm of the cradles' line front to back
  and within 3 cm of the centre line; both tape middles within 6 mm of their resting height) and your gripper is more
  than half open.
- **0.5:** the board rests in both cradles but your arm still grips it.
- **0.25:** both tape middles were at some moment 3 cm or more above where they started, but the board doesn't end in
  both cradles.
- **0** otherwise.
- **0 in any case, and the trial stops,** if the arms touch; if the board moves more than 2 cm before the other arm
  starts lifting; or while the other arm carries (from when it starts lifting until it starts to open), if the two
  tape middles differ in height by more than 4 cm, or the board moves more than 2 cm in the other arm's grip.
- The trial ends at 600 steps, or earlier once the other arm has finished and the board is in the cradles with your
  gripper open.

The rules close off the shortcuts: carrying the board alone before the other arm takes hold moves it early; gripping
and staying put lets the other arm's lift tip the board past 4 cm; pulling or lagging sideways drags the board through
the other arm's grip.

## The demo

`--policy demo` drives your arm with the scripted `Follower` in `sim/partner.py`. It isn't a policy: it reads the
simulator's true state. It takes the middle of the blue band, then at every control step puts its hand where the other
arm's hand is, mirrored across the centre line (same height, same distance forward), and lets go once the other arm
has. It scored 1.0 in 331 steps (ends at most 0.7 cm out of level, no slip). In 2 more runs, with the board moved 1 cm
front to back and 1 cm side to side and the other arm starting 0.3 s early or late, it scored 1.0 each time.

Checked to score 0: `--policy hold` (0.0, the other arm never lifts); gripping the blue band and staying put (0.0, the
ends were more than 4 cm apart in height); gripping and carrying the board to the cradles alone before the other arm
lifts (0.0, the board moved more than 2 cm before the other arm lifted).

## Running Dream-1

Not run. `dream_run.py` runs Dream-1 (`dreamlabs/dream-1-beta`) the way Dream Labs' YAM MuJoCo example does. Dream-1
controls your arm, its rows for the other arm are ignored, and the prompt is the instruction in `task.json`.

    uv sync --extra dream
    MUJOCO_GL=egl uv run python dream_run.py --out dream_run       # needs DREAM_API_KEY; uses credits
    MUJOCO_GL=egl uv run python dream_run.py --dry-run             # checks the request; no API call

## Files

- `task_env.py`: the task, the other arm and the scoring.
- `run_trial.py`: runs and scores one trial; also the demo, screenshot and videos.
- `dream_run.py`: runs Dream-1 on the task.
- `cameras.py`: the policy's two cameras, the YAM station's three cameras for Dream-1, and the wide view for the
  screenshot and videos (not given to the policy).
- `sim/`: the scene and objects (`benchmark.py`, `props.py`), the YAM model and controller (`robot.py`,
  `env_cfg.py`), the table (`scene.py`), inverse kinematics (`ik.py`), and the scripts for the other arm and the demo
  (`partner.py`).
- `assets/`: the YAM Pro arm and linear_4310 gripper model, from i2rt.
- `media/demo.mp4`, `media/setup.png`: the demo run, scoring 1.0, and the starting layout.
