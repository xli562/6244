"""Scripted oracle for the cup-in-drawer task.

Open the drawer, pick the cup from the counter, place it in the drawer, close
the drawer. Renders an MP4 headlessly (MUJOCO_GL=egl) and prints a success check.

    python sim/demo_cup_drawer.py --out sim/outputs/demo.mp4
    python sim/demo_cup_drawer.py --xml sim/outputs/family/variant_003.xml
"""

import argparse
from pathlib import Path

import imageio
import mink
import mujoco
import numpy as np

import scene
from scene import SceneConfig

SAFE_Z = 0.42  # travel height for the TCP
HANDLE_GRASP_Z = scene.DRAWER_Z + 0.12
CUP_GRASP_HEIGHT = 0.035  # TCP height above the cup bottom when grasping
OPEN_TARGET = 0.20  # drawer opening before placing the cup (m)


def slerp(q0, q1, t):
    dot = np.dot(q0, q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


class Robot:
    """Differential-IK reference (mink) tracked by the Panda position actuators."""

    def __init__(self, model, data, fps=30, cameras=("overview", "side"), size=(480, 640)):
        self.model, self.data = model, data
        self.dt = model.opt.timestep
        self.config = mink.Configuration(model)
        self.config.update(data.qpos.copy())
        self.tcp_task = mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=1.0,
                                       lm_damping=1.0)
        posture = mink.PostureTask(model, cost=1e-2)
        posture.set_target(data.qpos.copy())
        self.tasks = [self.tcp_task, posture]
        self.limits = [mink.ConfigurationLimit(model)]
        self.grip = 255.0
        self.pos = data.site("tcp").xpos.copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site("tcp").xmat)
        self.quat = quat
        self.cameras = cameras
        self.renderer = mujoco.Renderer(model, *size) if cameras else None
        self.frames = []
        self.render_every = max(1, round(1 / (fps * self.dt)))
        self.steps = 0

    def _step(self):
        target = mink.SE3.from_rotation_and_translation(mink.SO3(self.target_quat), self.target_pos)
        self.tcp_task.set_target(target)
        vel = mink.solve_ik(self.config, self.tasks, self.dt, solver="daqp", damping=1e-3,
                            limits=self.limits)
        self.config.integrate_inplace(vel, self.dt)
        self.data.ctrl[:7] = self.config.q[:7]
        self.data.ctrl[7] = self.grip
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.renderer and self.steps % self.render_every == 0:
            views = []
            for cam in self.cameras:
                self.renderer.update_scene(self.data, camera=cam)
                views.append(self.renderer.render())
            self.frames.append(np.concatenate(views, axis=1))

    def move(self, pos, quat=None, duration=1.5):
        quat = self.quat if quat is None else np.asarray(quat, float)
        start_pos, start_quat = self.pos.copy(), self.quat.copy()
        n = max(1, int(duration / self.dt))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)  # smoothstep
            self.target_pos = (1 - s) * start_pos + s * np.asarray(pos, float)
            self.target_quat = slerp(start_quat, quat, s)
            self._step()
        self.pos, self.quat = np.asarray(pos, float), quat

    def hold(self, duration):
        self.move(self.pos, self.quat, duration)

    def gripper(self, open_, duration=0.6):
        self.grip = 255.0 if open_ else 0.0
        self.hold(duration)

    def close(self):
        if self.renderer:
            self.renderer.close()


def handle_xy(open_amount):
    x = scene.COUNTER_FRONT_X - open_amount + scene.HANDLE_OFFSET_X
    return np.array([x, scene.DRAWER_Y])


def run(model, data, video=None, log=print):
    robot = Robot(model, data, cameras=("overview", "side") if video else ())
    down = scene.TOP_DOWN_QUAT
    drawer_q = model.joint("drawer_slide").qposadr[0]

    def phase(name):
        log(f"[{data.time:5.2f}s] {name}")

    phase("orient gripper top-down")
    robot.move(robot.pos, down, 1.5)

    phase("open drawer")
    h = handle_xy(data.qpos[drawer_q])
    robot.move([*h, SAFE_Z], duration=2.0)
    robot.move([*h, HANDLE_GRASP_Z], duration=1.2)
    robot.gripper(open_=False)
    pull = OPEN_TARGET - data.qpos[drawer_q]
    robot.move([h[0] - pull, h[1], HANDLE_GRASP_Z], duration=2.5)
    robot.gripper(open_=True, duration=0.5)
    robot.move([h[0] - pull, h[1], SAFE_Z], duration=1.0)
    log(f"        drawer opening = {data.qpos[drawer_q]:.3f} m")

    phase("pick cup")
    cup = data.site("cup_center").xpos.copy()
    cup_bottom = scene.COUNTER_TOP_Z
    # Rotate the gripper with the cup so the fingers stay clear of the handle.
    rot = data.body("cup_model").xmat.reshape(3, 3)
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    grasp_quat = np.zeros(4)
    mujoco.mju_mulQuat(grasp_quat, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), down)
    robot.move([cup[0], cup[1], SAFE_Z], grasp_quat, duration=2.0)
    robot.move([cup[0], cup[1], cup_bottom + CUP_GRASP_HEIGHT], duration=1.5)
    robot.gripper(open_=False, duration=0.8)
    robot.move([cup[0], cup[1], SAFE_Z + 0.03], duration=1.5)

    phase("place cup in drawer")
    opening = data.qpos[drawer_q]
    drawer_front = scene.COUNTER_FRONT_X - opening + scene.DRAWER_PANEL
    place = np.array([(drawer_front + scene.COUNTER_FRONT_X) / 2, scene.DRAWER_Y])
    floor_z = scene.DRAWER_Z + 2 * scene.DRAWER_WALL
    robot.move([*place, SAFE_Z + 0.03], duration=2.0)
    robot.move([*place, floor_z + CUP_GRASP_HEIGHT + 0.008], duration=1.8)
    robot.gripper(open_=True, duration=0.6)
    robot.move([*place, SAFE_Z], duration=1.2)

    phase("close drawer")
    h = handle_xy(data.qpos[drawer_q])
    robot.move([*h, SAFE_Z], down, duration=1.5)
    robot.move([*h, HANDLE_GRASP_Z], duration=1.2)
    robot.gripper(open_=False)
    robot.move([scene.COUNTER_FRONT_X + scene.HANDLE_OFFSET_X + 0.01, h[1], HANDLE_GRASP_Z],
               duration=2.5)
    robot.gripper(open_=True, duration=0.5)
    robot.move([robot.pos[0], robot.pos[1], SAFE_Z], duration=1.0)
    robot.hold(1.0)

    result = check_success(model, data)
    phase(f"done: {result}")
    if video:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video, robot.frames, fps=30, quality=8, macro_block_size=8)
        log(f"saved {video} ({len(robot.frames)} frames)")
    robot.close()
    return result


def check_success(model, data):
    """Cup upright, inside the drawer volume, and drawer closed."""
    drawer = data.body("drawer")
    rel = drawer.xmat.reshape(3, 3).T @ (data.site("cup_center").xpos - drawer.xpos)
    inside = (scene.DRAWER_PANEL < rel[0] < scene.DRAWER_DEPTH
              and abs(rel[1]) < scene.DRAWER_HALF_WIDTH
              and 0 < rel[2] < scene.DRAWER_WALL_HEIGHT)
    upright = data.body("cup_model").xmat.reshape(3, 3)[2, 2] > 0.95
    opening = data.qpos[model.joint("drawer_slide").qposadr[0]]
    closed = opening < 0.01
    return {"success": bool(inside and upright and closed), "cup_in_drawer": bool(inside),
            "cup_upright": bool(upright), "drawer_closed": bool(closed),
            "drawer_opening_m": round(float(opening), 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(Path(__file__).parent / "outputs/demo_cup_drawer.mp4"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--xml", help="run on an exported scene file instead of the default scene")
    args = parser.parse_args()
    model, data = scene.load_model(args.xml) if args.xml else scene.build_model(SceneConfig())
    run(model, data, video=None if args.no_video else args.out)
