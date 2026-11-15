"""
panda_catch_env.py - Custom Gymnasium env: a Franka Panda catches a thrown ball in a cup.

Each episode (2.5 s):
  1. A ball is launched from a random spot 1.5-2 m in front of the robot, aimed
     so that it comes down somewhere in the robot's reachable workspace.
  2. The agent moves the cup to get under the ball and keep it inside.

How the agent controls the robot (important design choice):
  The ACTION is a 3-D velocity command for the CUP (move left/right, forward/
  back, up/down). Each step we convert that into 7 joint angles with
  inverse kinematics (IK), which also keeps the cup level (opening facing up).
  The robot's position servos then drive the joints there.
  -> RL only has to learn "where to put the cup, and when". Learning raw joint
     commands for a catch is possible but needs far more training.
"""

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register

SCENE_XML = Path(__file__).resolve().parent / "assets" / "franka_emika_panda" / "catch_scene.xml"

N_ARM = 7                    # Panda arm joints (qpos[0:7]); fingers are qpos[7:9]
BALL_QPOS = 9                # ball free joint: qpos[9:12] position, qpos[12:16] quaternion
BALL_QVEL = 9                # ball free joint: qvel[9:12] linear velocity, qvel[12:15] angular
GRIPPER_OPEN = 255.0
MAX_CUP_SPEED = 1.6          # m/s: fastest the agent may move the cup
GRAVITY = 9.81

# Box the cup is allowed to move in (metres, robot base at the origin).
CUP_LOW = np.array([0.25, -0.45, 0.20])
CUP_HIGH = np.array([0.75, 0.45, 0.75])
CUP_START = np.array([0.45, 0.0, 0.40])

# Throws: launch point, where the ball should come down, and how long it flies.
LAUNCH_LOW, LAUNCH_HIGH = np.array([1.5, -0.5, 0.6]), np.array([2.0, 0.5, 1.0])
LAND_LOW, LAND_HIGH = np.array([0.32, -0.32, 0.30]), np.array([0.65, 0.32, 0.55])
FLIGHT_TIME = (0.75, 0.95)   # seconds

# Ball-in-cup test, in the cup's own frame (cup floor top is at z = 0.01).
CUP_HALF_WIDTH = 0.08        # inner half-width of the cup
CUP_RIM = 0.09               # height of the rim above the floor centre


class PandaCatchEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 20}

    def __init__(self, render_mode=None, **kwargs):
        # OBSERVATION (20 numbers):
        #   cup position (3), cup velocity (3), ball position (3), ball velocity (3),
        #   ball - cup vector (3), commanded cup position (3),
        #   episode progress 0..1 (1), ball-in-cup flag (1)
        obs_space = spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float64)
        super().__init__(
            model_path=str(SCENE_XML),
            frame_skip=25,  # 25 x 2 ms = 50 ms per RL step
            observation_space=obs_space,
            render_mode=render_mode,
            width=1280,
            height=720,
            # Side view: the throw arcs from right (launcher) to left (robot).
            default_camera_config={"distance": 2.9, "azimuth": 250.0, "elevation": -12.0,
                                   "lookat": np.array([0.95, 0.0, 0.75])},
            **kwargs,
        )
        # ACTION (3 numbers in [-1, 1]): cup velocity along x, y, z.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self._site = self.model.site("cup_center").id
        self._cup_body = self.model.body("cup").id
        self._jnt_low = self.model.jnt_range[:N_ARM, 0]
        self._jnt_high = self.model.jnt_range[:N_ARM, 1]
        self._max_steps = 50  # must match max_episode_steps in register()

        # A second MjData used only for IK math ("where should the joints go?"),
        # so we never disturb the real physics state while solving.
        self._kin = mujoco.MjData(self.model)

        # Find a comfortable "home" pose with the cup level at CUP_START.
        q = self.model.key("home").qpos[:N_ARM].copy()
        q[6] = 0.0  # unwind the wrist so the cup can face up
        self._q_rest = q.copy()
        self._q_home = self._solve_ik(q, CUP_START, iters=300)
        self._q_rest = self._q_home.copy()

    # ------------------------------------------------------------- kinematics
    def _solve_ik(self, q, target, iters=3):
        """Damped least-squares IK: joint angles that put the cup at `target`, level.

        Task (5-D): cup position (3) + cup opening pointing straight up (2).
        Extra freedom (7 joints > 5 task dims) is used to stay near a rest pose.
        """
        kin, m = self._kin, self.model
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        q = q.copy()
        for _ in range(iters):
            kin.qpos[:N_ARM] = q
            mujoco.mj_kinematics(m, kin)
            mujoco.mj_comPos(m, kin)

            pos_err = target - kin.site_xpos[self._site]
            cup_up = kin.xmat[self._cup_body].reshape(3, 3)[:, 2]
            rot_err = np.cross(cup_up, [0.0, 0.0, 1.0])[:2]  # tilt error (x, y)
            err = np.concatenate([pos_err, rot_err])

            mujoco.mj_jacSite(m, kin, jacp, jacr, self._site)
            J = np.vstack([jacp[:, :N_ARM], jacr[:2, :N_ARM]])
            JJt = J @ J.T + 1e-4 * np.eye(5)
            J_pinv = J.T @ np.linalg.inv(JJt)
            dq = J_pinv @ err
            # Null-space: drift gently toward the rest pose without disturbing the task.
            dq += (np.eye(N_ARM) - J_pinv @ J) @ (0.1 * (self._q_rest - q))
            q = np.clip(q + dq, self._jnt_low, self._jnt_high)
        return q

    # ------------------------------------------------------------- helpers
    def _cup_pos(self):
        return self.data.site_xpos[self._site].copy()

    def _ball_pos(self):
        return self.data.qpos[BALL_QPOS:BALL_QPOS + 3].copy()

    def _ball_vel(self):
        return self.data.qvel[BALL_QVEL:BALL_QVEL + 3].copy()

    def _ball_in_cup(self) -> bool:
        """Is the ball inside the cup's walls (expressed in the cup's frame)?"""
        R = self.data.xmat[self._cup_body].reshape(3, 3)
        rel = R.T @ (self._ball_pos() - self.data.xpos[self._cup_body])
        return bool(abs(rel[0]) < CUP_HALF_WIDTH and abs(rel[1]) < CUP_HALF_WIDTH
                    and 0.0 < rel[2] < CUP_RIM)

    def _get_obs(self):
        cup, ball = self._cup_pos(), self._ball_pos()
        return np.concatenate([
            cup, self._cup_vel, ball, self._ball_vel(), ball - cup, self._cmd,
            [self._t / self._max_steps, float(self._ball_in_cup())],
        ])

    # ------------------------------------------------------------- Gym API
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # 1. ACTION -> new commanded cup position (clipped to the workspace box).
        self._cmd = np.clip(self._cmd + MAX_CUP_SPEED * self.dt * action, CUP_LOW, CUP_HIGH)

        # 2. IK -> joint angle targets for the robot's position servos.
        self._q_cmd = self._solve_ik(self._q_cmd, self._cmd)
        ctrl = np.append(self._q_cmd, GRIPPER_OPEN)

        # 3. Physics: advance 50 ms.
        cup_before = self._cup_pos()
        self.do_simulation(ctrl, self.frame_skip)
        self._cup_vel = (self._cup_pos() - cup_before) / self.dt
        self._t += 1

        # 4. REWARD (all terms >= 0 except the small effort penalty, so the agent
        #    never "wants" the episode to end early by dropping the ball):
        #    - closeness of the cup to the ball (dense guidance while it flies)
        #    - +2 every step the ball is inside the cup (the real goal)
        ball, cup = self._ball_pos(), self._cup_pos()
        dist = float(np.linalg.norm(ball - cup))
        in_cup = self._ball_in_cup()
        reward = np.exp(-dist / 0.2) + 2.0 * in_cup - 0.05 * float(np.square(action).sum())

        # 5. TERMINATED: ball hit the floor (or flew away) -> episode over, no more reward.
        terminated = bool(ball[2] < 0.08 or ball[0] < -0.5 or abs(ball[1]) > 1.5)

        if self.render_mode == "human":
            self.render()

        info = {"distance": dist, "in_cup": in_cup, "is_success": in_cup}
        return self._get_obs(), reward, terminated, False, info

    def reset_model(self):
        """Called by reset(): arm to home (cup level), then throw a new ball."""
        rng = self.np_random
        qpos, qvel = self.init_qpos.copy(), np.zeros(self.model.nv)

        # Arm at home pose with a bit of noise, cup command at its current position.
        self._q_cmd = np.clip(self._q_home + rng.uniform(-0.03, 0.03, N_ARM),
                              self._jnt_low, self._jnt_high)
        qpos[:N_ARM] = self._q_cmd
        qpos[N_ARM:N_ARM + 2] = 0.04  # fingers open

        # THE THROW: pick a launch point, a landing point and a flight time, then
        # solve projectile motion  p(T) = p0 + v0*T - 0.5*g*T^2*z  for v0.
        p0 = rng.uniform(LAUNCH_LOW, LAUNCH_HIGH)
        p_land = rng.uniform(LAND_LOW, LAND_HIGH)
        T = rng.uniform(*FLIGHT_TIME)
        v0 = (p_land - p0) / T
        v0[2] += 0.5 * GRAVITY * T
        qpos[BALL_QPOS:BALL_QPOS + 3] = p0
        qpos[BALL_QPOS + 3:BALL_QPOS + 7] = [1, 0, 0, 0]
        qvel[BALL_QVEL:BALL_QVEL + 3] = v0

        self.set_state(qpos, qvel)
        self.data.ctrl[:] = np.append(self._q_cmd, GRIPPER_OPEN)
        self._cmd = self._cup_pos()
        self._cup_vel = np.zeros(3)
        self._t = 0
        return self._get_obs()


# 50 steps x 50 ms = 2.5 s per episode (ball flies ~1 s, then must stay in the cup).
register(id="PandaCatch-v0", entry_point=PandaCatchEnv, max_episode_steps=50)
