"""
custom_ball_test.py - A hand-written MuJoCo scene: 2-link arm + cup + falling ball.

Goal: prove we can build OUR OWN physics scenes (the future throw/catch task
will need a custom scene with two arms and a ball - no ready-made Gymnasium env
exists for that). Nothing is learned here; the arm is held in place by simple
position controllers while a ball drops into its cup under gravity.

Run:
    uv run custom_ball_test.py              # opens the 3D viewer for 10 s
    uv run custom_ball_test.py --headless   # no window, just print the physics
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

SIM_SECONDS = 10.0

# ----------------------------------------------------------------------------
# The scene, described in MJCF (MuJoCo's XML format).
#   * <worldbody>  : bodies arranged as a tree (parent -> child = attached)
#   * <joint>      : how a body may move relative to its parent
#                    (hinge = rotation, free = 6-DoF flying object)
#   * <geom>       : shapes used for collisions and drawing
#   * <actuator>   : motors the controller (later: the RL agent) can command
# Units: metres, kilograms, seconds, radians.
# ----------------------------------------------------------------------------
SCENE_XML = """
<mujoco model="arm_cup_ball">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.2 0.3 0.4" rgb2="0.15 0.2 0.3"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>

    <!-- Fixed pedestal the arm is mounted on -->
    <geom name="pedestal" type="cylinder" size="0.06 0.3" pos="0 0 0.3"
          rgba="0.4 0.4 0.4 1"/>

    <!-- ARM: shoulder -> upper link -> elbow -> forearm -> cup.
         gravcomp="1" cancels gravity on the arm itself (like a real robot's
         gravity compensation), so the motors only need to position it. -->
    <body name="upper_arm" pos="0 0 0.6" gravcomp="1">
      <joint name="shoulder" type="hinge" axis="0 -1 0" range="-90 90" damping="2"/>
      <geom type="capsule" fromto="0 0 0  0.4 0 0" size="0.03" rgba="0.9 0.5 0.1 1"/>

      <body name="forearm" pos="0.4 0 0" gravcomp="1">
        <joint name="elbow" type="hinge" axis="0 -1 0" range="-120 120" damping="2"/>
        <geom type="capsule" fromto="0 0 0  0.4 0 0" size="0.025" rgba="0.9 0.6 0.2 1"/>

        <!-- CUP / BASKET at the end-effector: a floor plus 4 walls -->
        <body name="cup" pos="0.4 0 0" gravcomp="1">
          <geom type="box" size="0.08 0.08 0.01" pos="0 0 0"        rgba="0.2 0.7 0.3 1"/>
          <geom type="box" size="0.01 0.08 0.06" pos=" 0.08 0 0.05" rgba="0.2 0.7 0.3 0.6"/>
          <geom type="box" size="0.01 0.08 0.06" pos="-0.08 0 0.05" rgba="0.2 0.7 0.3 0.6"/>
          <geom type="box" size="0.08 0.01 0.06" pos="0  0.08 0.05" rgba="0.2 0.7 0.3 0.6"/>
          <geom type="box" size="0.08 0.01 0.06" pos="0 -0.08 0.05" rgba="0.2 0.7 0.3 0.6"/>
          <site name="cup_center" pos="0 0 0.02" size="0.01"/>
        </body>
      </body>
    </body>

    <!-- BALL: a free body (6 DoF) dropped from above the cup -->
    <body name="ball" pos="0.8 0 1.4">
      <freejoint/>
      <geom type="sphere" size="0.03" mass="0.05" rgba="0.9 0.1 0.1 1"/>
    </body>
  </worldbody>

  <!-- Position servos: ctrl = desired joint angle (rad). An RL agent would
       write to data.ctrl exactly like the controller below does. -->
  <actuator>
    <position name="shoulder_servo" joint="shoulder" kp="100" ctrlrange="-1.57 1.57"/>
    <position name="elbow_servo"    joint="elbow"    kp="100" ctrlrange="-2.1 2.1"/>
  </actuator>
</mujoco>
"""


def controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Scripted arm motion: hold still to catch, then gently sway the cup.

    The shoulder and elbow move in opposite directions, so the cup stays
    roughly level and the ball does not spill out.
    """
    if data.time < 3.0:
        data.ctrl[:] = [0.0, 0.0]          # hold the cup under the falling ball
    else:
        a = 0.25 * np.sin(1.5 * (data.time - 3.0))
        data.ctrl[:] = [a, -a]


def report(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Print the ball's height and its distance from the cup centre."""
    ball = data.body("ball").xpos
    cup = data.site("cup_center").xpos
    dist = np.linalg.norm(ball - cup)
    status = "IN CUP" if dist < 0.08 else "flying"
    print(f"t={data.time:5.2f}s  ball z={ball[2]:.3f} m  "
          f"dist to cup={dist:.3f} m  [{status}]")


def run_headless(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Simulate as fast as possible without a window."""
    next_print = 0.0
    while data.time < SIM_SECONDS:
        controller(model, data)
        mujoco.mj_step(model, data)
        if data.time >= next_print:
            report(model, data)
            next_print += 0.5


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Simulate in real time inside the interactive MuJoCo viewer.

    launch_passive gives US control of the simulation loop (we call mj_step),
    while the viewer just draws whatever is in `data` each time we sync().
    """
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Point the camera at the arm.
        with viewer.lock():
            viewer.cam.lookat[:] = [0.4, 0.0, 0.7]
            viewer.cam.distance = 2.2
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20

        wall_start = time.time()
        next_print = 0.0
        while viewer.is_running() and data.time < SIM_SECONDS:
            step_start = time.time()

            controller(model, data)
            mujoco.mj_step(model, data)  # advance physics by one timestep
            viewer.sync()                # push the new state to the window

            if data.time >= next_print:
                report(model, data)
                next_print += 0.5

            # Sleep so simulated time matches wall-clock time.
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        print(f"Done: simulated {data.time:.1f} s in {time.time() - wall_start:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom MuJoCo arm + cup + ball scene")
    parser.add_argument("--headless", action="store_true", help="run without a window")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(SCENE_XML)  # compile XML -> physics model
    data = mujoco.MjData(model)                        # mutable simulation state
    print(f"Loaded scene: {model.nbody} bodies, {model.njnt} joints, "
          f"{model.nu} actuators, timestep={model.opt.timestep}s")

    if args.headless:
        run_headless(model, data)
    else:
        run_viewer(model, data)


if __name__ == "__main__":
    main()
