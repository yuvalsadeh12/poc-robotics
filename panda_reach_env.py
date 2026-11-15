"""
panda_reach_env.py - A custom Gymnasium environment: Franka Panda reaches a target.

This is the template for our future throw/catch env: a real robot model loaded
from XML, wrapped in the standard Gymnasium API (reset / step / observation /
reward) so any Stable-Baselines3 algorithm can train on it.

Task: move the gripper to a red ball placed at a random spot in front of the
robot. A new random target is chosen every episode (100 steps = 5 seconds).
"""

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

SCENE_XML = Path(__file__).resolve().parent / "assets" / "franka_emika_panda" / "reach_scene.xml"

N_ARM_JOINTS = 7          # Panda has 7 rotating joints (+ 2 finger joints)
MAX_DELTA = 0.05          # max change of each joint's target angle per step (rad)
GRIPPER_OPEN = 255.0      # actuator 8 controls the fingers: 0 = closed, 255 = open
TCP_OFFSET = 0.105        # distance from the "hand" body to between the fingertips (m)
SUCCESS_DIST = 0.05       # within 5 cm counts as "reached"

# Random targets are sampled inside this box (metres, robot base at the origin).
TARGET_LOW = np.array([0.30, -0.35, 0.10])
TARGET_HIGH = np.array([0.65, 0.35, 0.60])


class PandaReachEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 20}

    def __init__(self, render_mode=None, **kwargs):
        # OBSERVATION (23 numbers):
        #   7 joint angles + 7 joint velocities
        #   + gripper position (3) + target position (3) + gripper->target vector (3)
        obs_space = spaces.Box(-np.inf, np.inf, shape=(23,), dtype=np.float64)

        super().__init__(
            model_path=str(SCENE_XML),
            frame_skip=25,  # 25 physics steps x 2 ms = 50 ms per RL step (20 Hz)
            observation_space=obs_space,
            render_mode=render_mode,
            width=1280,
            height=720,
            default_camera_config={"distance": 1.9, "azimuth": 145.0,
                                   "elevation": -22.0, "lookat": np.array([0.35, 0.0, 0.35])},
            **kwargs,
        )

        # ACTION (7 numbers in [-1, 1]): how much to move each joint's target
        # angle this step. The robot's built-in position servos (from the
        # Menagerie model) then drive the joints there - like a real Panda's
        # joint-position control mode. Small deltas keep the motion smooth.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_ARM_JOINTS,), dtype=np.float32)

        self._hand_id = self.model.body("hand").id
        self._home = self.model.key("home")
        self._ctrl_low = self.model.actuator_ctrlrange[:N_ARM_JOINTS, 0]
        self._ctrl_high = self.model.actuator_ctrlrange[:N_ARM_JOINTS, 1]

    # ------------------------------------------------------------------ helpers
    def _gripper_pos(self) -> np.ndarray:
        """World position of the point between the two fingertips."""
        hand_pos = self.data.xpos[self._hand_id]
        hand_z_axis = self.data.xmat[self._hand_id].reshape(3, 3)[:, 2]
        return hand_pos + TCP_OFFSET * hand_z_axis

    def _target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[0].copy()

    def _get_obs(self) -> np.ndarray:
        grip, target = self._gripper_pos(), self._target_pos()
        return np.concatenate([
            self.data.qpos[:N_ARM_JOINTS],
            self.data.qvel[:N_ARM_JOINTS],
            grip, target, target - grip,
        ])

    # --------------------------------------------------------------- Gym API
    def step(self, action):
        # 1. ACTION -> new joint targets for the position servos.
        action = np.clip(action, -1.0, 1.0)
        arm_ctrl = np.clip(self.data.ctrl[:N_ARM_JOINTS] + MAX_DELTA * action,
                           self._ctrl_low, self._ctrl_high)
        ctrl = np.append(arm_ctrl, GRIPPER_OPEN)

        # 2. Advance the physics simulation by one RL step (25 x 2 ms).
        self.do_simulation(ctrl, self.frame_skip)

        # 3. REWARD: closer to the target = higher (less negative) reward.
        #    A small penalty on large actions encourages smooth, efficient motion.
        dist = np.linalg.norm(self._target_pos() - self._gripper_pos())
        reward = -dist - 0.01 * float(np.square(action).sum())
        if dist < SUCCESS_DIST:
            reward += 0.5  # bonus for being right on the target

        if self.render_mode == "human":
            self.render()

        # 4. Return (observation, reward, terminated, truncated, info).
        #    The episode never "terminates" early; the 100-step time limit
        #    (set in register()) handles truncation.
        return self._get_obs(), reward, False, False, {"distance": dist,
                                                       "is_success": dist < SUCCESS_DIST}

    def reset_model(self):
        """Called by reset(): put the arm at 'home' (slightly perturbed) and move the target."""
        qpos = self._home.qpos.copy()
        qpos[:N_ARM_JOINTS] += self.np_random.uniform(-0.1, 0.1, N_ARM_JOINTS)
        self.set_state(qpos, np.zeros(self.model.nv))
        self.data.ctrl[:] = np.append(qpos[:N_ARM_JOINTS], GRIPPER_OPEN)

        self.data.mocap_pos[0] = self.np_random.uniform(TARGET_LOW, TARGET_HIGH)
        mujoco.mj_forward(self.model, self.data)  # update positions for the first observation
        return self._get_obs()


# Register so it can be created with gym.make("PandaReach-v0") like built-in envs.
# max_episode_steps adds a TimeLimit wrapper -> truncated=True after 100 steps.
from gymnasium.envs.registration import register  # noqa: E402

register(id="PandaReach-v0", entry_point=PandaReachEnv, max_episode_steps=100)
