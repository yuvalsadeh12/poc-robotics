# POC: Robotic Arm RL on CPU (MuJoCo + Gymnasium + Stable-Baselines3)

A minimal proof of concept for our throw-and-catch project. It shows that this stack
runs on an ordinary Windows laptop **with no GPU**:

| Script | What it proves |
|---|---|
| `train_and_demo.py` | Trains a PPO agent to move a **Franka Emika Panda** (a real 7-joint industrial arm built from the manufacturer's meshes) to random targets. It saves the model, evaluates it, and shows it live in a 3D viewer. |
| `panda_catch_env.py` | **Catch task:** a ball is thrown at the Panda from a random spot and the arm learns to catch it in a cup (`--env catch`). |
| `panda_reach_env.py` | Our own Gymnasium environment around the Panda (observation, action, reward, reset). This is the template for the throw/catch env. |
| `custom_ball_test.py` | Loads a hand-written MuJoCo XML scene (2-link arm + cup + falling ball) and opens the interactive viewer for 10 s. |

## Setup (once)

Install [uv](https://docs.astral.sh/uv/) if you don't have it (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, in the project folder:

```powershell
uv sync
```

`uv sync` creates `.venv\`, downloads Python 3.12 if needed, and installs pinned
versions from `uv.lock`. PyTorch comes from the **CPU-only** index (see
`pyproject.toml`), so there is no multi-GB CUDA download.

## Run

```powershell
# 1. Panda arm: train PPO (500k steps, ~5 min on CPU) -> save -> evaluate -> 3D demo
uv run train_and_demo.py

#    ...or skip training and replay the saved model
uv run train_and_demo.py --skip-train

# 2. Custom physics scene: arm with a cup catching a falling ball (10 s)
uv run custom_ball_test.py
```

Catch a thrown ball (Panda with a cup):

```powershell
uv run train_and_demo.py --env catch               # train (~1.5M steps) + demo
uv run train_and_demo.py --env catch --skip-train  # demo the saved model
```

In the catch task the agent outputs a 3-D **cup velocity**. Inverse kinematics
(`_solve_ik` in `panda_catch_env.py`) turns it into 7 joint angles and keeps the cup level,
so RL only has to learn where and when to put the cup.

Useful options:

```powershell
uv run train_and_demo.py --env reacher        # tiny 2-joint toy arm, trains in ~20 s
uv run train_and_demo.py --episodes 30        # longer demo
uv run train_and_demo.py --timesteps 1000000  # custom training budget
uv run train_and_demo.py --no-demo            # train without opening a window
uv run custom_ball_test.py --headless         # physics only, prints ball position
```

Viewer controls: left-drag rotates, right-drag pans, scroll zooms. Closing the
window ends the program.

## What to look for

- **Training log:** `ep_rew_mean` climbs from about -40 to about +45. The reward is
  minus the distance to the target, plus a bonus for each step within 5 cm, so it
  only turns positive once the arm reliably reaches the target.
- **Evaluation:** the reference run ends a median **0.3 cm** from the target, and within
  5 cm in **100%** of episodes.
- **Demo window:** the Panda swings its gripper onto the red ball within about 1 s and
  holds it there. The ball jumps to a new random spot every 5 s.
- **Custom scene:** the red ball falls into the green cup at about 0.5 s and stays in while
  the arm sways.

Reference run on a CPU-only Windows 11 laptop: about 1,600 steps/s with 8 parallel envs,
so 500k steps take about 5 min.

## Project layout

```
pyproject.toml                  dependencies + CPU-only PyTorch index
uv.lock                         exact pinned versions (commit this)
train_and_demo.py               PPO training -> save/load -> evaluate -> live viewer
panda_reach_env.py              custom Gymnasium env: Panda reaches a random target
panda_catch_env.py              custom Gymnasium env: Panda catches a thrown ball in a cup
custom_ball_test.py             raw MJCF XML scene via the mujoco Python bindings
assets/franka_emika_panda/      Panda model from MuJoCo Menagerie (Apache-2.0),
                                panda_cup.xml (Panda + cup), reach_scene.xml, catch_scene.xml
models/                         trained models (committed: run demos with --skip-train right after cloning)
```

## Next steps toward throw and catch

1. Catching works (`panda_catch_env.py`). Next, the thrower: a second Panda that learns
   to release the ball so it lands in a target zone.
2. Put both Pandas facing each other in one scene and replace the scripted throw with the
   thrower arm.
3. Train them with the same PPO code, first one at a time and then together.

Credits: the Franka Emika Panda model is from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0,
see `assets/franka_emika_panda/LICENSE`).
