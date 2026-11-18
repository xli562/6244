"""Low-level arm control: a differential-IK reference (mink) tracked by the Panda position actuators."""

import mink
import mujoco
import numpy as np


class EpisodeTimeout(Exception):
    """Raised when the simulated time budget of an episode is used up."""


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
    def __init__(self, model, data, cameras=(), fps=30, size=(480, 640), max_time=None,
                 on_step=None):
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
        self.target_pos, self.target_quat = self.pos.copy(), self.quat.copy()
        self.cameras = cameras
        self.renderer = mujoco.Renderer(model, *size) if cameras else None
        self.frames = []
        self.render_every = max(1, round(1 / (fps * self.dt)))
        self.steps = 0
        self.max_time = max_time
        self.on_step = on_step

    def _step(self):
        if self.max_time is not None and self.data.time >= self.max_time:
            raise EpisodeTimeout(f"episode exceeded {self.max_time:.0f} s of simulated time")
        target = mink.SE3.from_rotation_and_translation(mink.SO3(self.target_quat), self.target_pos)
        self.tcp_task.set_target(target)
        vel = mink.solve_ik(self.config, self.tasks, self.dt, solver="daqp", damping=1e-3,
                            limits=self.limits)
        self.config.integrate_inplace(vel, self.dt)
        self.data.ctrl[:7] = self.config.q[:7]
        self.data.ctrl[7] = self.grip
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.on_step:
            self.on_step(self.steps)
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

    def tcp_pos(self):
        return self.data.site("tcp").xpos.copy()

    def gripper_width(self):
        return float(self.data.qpos[7] + self.data.qpos[8])

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
