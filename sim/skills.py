"""Mid-level skill API handed to a policy: noisy perception and top-down Cartesian motion.

A policy is a Python file that defines `run(api)`. The API deliberately stops
short of task-level skills (there is no `open_drawer()` or `place_in()`): the
policy decides where to grasp, how far to pull and where to put the cup, using
perception that carries an error, as a real perception stack would.
"""

import mujoco
import numpy as np

import scene

API_DOC = """\
# Skill API

Your program is a Python file that defines `run(api)`. It may import only
`numpy` and `math`. All quantities are in metres, radians and seconds, in the
world frame: the robot base is at the origin, +x points from the robot toward
the counter, +y is to the robot's left, +z is up. The drawer is in the counter
front and slides out toward the robot (the -x direction).

The gripper always points straight down. `yaw` is its rotation about the
vertical axis: at yaw = 0 the two fingers close along the y axis. The gripper
opens to 0.08 m.

## Perception (every reading has an error of several millimetres; the error is
## different in every episode)

- `api.get_cup()` -> `{"pos": [x, y, z], "yaw": a, "radius": r, "height": h}`
  `pos` is the centre of the cup body. The cup has a handle on one side; at
  yaw = 0 the handle points toward the robot (-x), and it rotates with yaw.
- `api.get_handle()` -> `{"pos": [x, y, z]}`
  Grasp point on the drawer handle, a vertical bar in front of the drawer front.
- `api.get_drawer()` -> `{"opening": d, "interior_center": [x, y, z],
  "interior_size": [sx, sy, sz], "floor_z": z}`
  `opening` is how far the drawer is pulled out (0 = closed). The interior box
  is the usable volume inside the drawer at its current position; it moves
  with the drawer. `floor_z` is the height of the drawer floor.
- `api.get_counter()` -> `{"top_z": z, "front_x": x}`
  Height of the counter top and x of the counter front face. The part of the
  drawer with x > front_x is under the counter and cannot be reached from above.
- `api.get_objects()` -> list of `{"name": s, "pos": [x, y, z], "size": [sx, sy, sz],
  "location": "counter" | "drawer"}`
  Other objects in the scene (bounding-box centre and full extents). They are
  not part of the task and should be left where they are.
- `api.get_ee()` -> `{"pos": [x, y, z], "yaw": a, "gripper_width": w}`
  Fingertip position (exact) and the current finger opening.

## Motion

- `api.move_to(pos, yaw=None, speed=0.15)` -> `{"reached": bool, "error": e}`
  Move the fingertips in a straight line to `pos`. `yaw=None` keeps the current
  yaw. There is no collision avoidance: route the gripper yourself. `error` is
  the remaining distance to the target; `reached` is False if it is above 0.015.
- `api.close_gripper()` -> finger opening after closing (0 = fully closed on nothing)
- `api.open_gripper()` -> finger opening after opening
- `api.wait(seconds)`
- `api.log(message)` puts a line in the execution trace you get back.

An episode has 90 s of simulated time; when it runs out the program is stopped.
"""


def _yaw_of(mat9):
    r = np.asarray(mat9).reshape(3, 3)
    return float(np.arctan2(r[1, 0], r[0, 0]))


class SkillAPI:
    def __init__(self, robot, model, data, noise_sigma=0.0, noise_seed=0, trace_limit=400, exact_readings=()):
        self._robot, self._model, self._data = robot, model, data
        self._sigma = float(noise_sigma)
        self._exact_readings = frozenset(exact_readings)
        self._rng = np.random.default_rng(noise_seed)
        self._bias = {}
        self._trace = []
        self._trace_limit = trace_limit
        self._yaw = 0.0
        self._clutter = [model.body(i).name for i in range(model.nbody)
                         if model.body(i).name.startswith("clutter_")]

    # --- internals --------------------------------------------------------
    def _record(self, call, args, result):
        if len(self._trace) < self._trace_limit:
            self._trace.append({"t": round(float(self._data.time), 2), "call": call,
                                "args": args, "result": result})

    def _noisy(self, key, value):
        """Per-episode bias for each perceived quantity, plus a smaller per-call jitter."""
        value = np.asarray(value, float)
        if self._sigma == 0:
            return value
        if key not in self._bias:
            self._bias[key] = self._rng.normal(0, self._sigma, value.shape)
        noisy = value + self._bias[key] + self._rng.normal(0, self._sigma / 4, value.shape)
        # Consume identical random draws on matched calls during sensor interventions.
        return value if key in self._exact_readings else noisy

    @staticmethod
    def _r(x, n=4):
        return [round(float(v), n) for v in np.atleast_1d(x)]

    # --- perception -------------------------------------------------------
    def get_cup(self):
        d, m = self._data, self._model
        radius, height = m.numeric("cup_dims").data[:2]
        pos = self._noisy("cup_pos", d.site("cup_center").xpos)
        # Yaw error: 5 rad per metre of position sigma (0.005 m -> 0.025 rad).
        yaw = _yaw_of(d.body("cup_model").xmat) + float(self._noisy("cup_yaw", [0.0])[0]) * 5
        out = {"pos": self._r(pos), "yaw": round(yaw, 3), "radius": round(float(radius), 4),
               "height": round(float(height), 4)}
        self._record("get_cup", {}, out)
        return out

    def get_handle(self):
        out = {"pos": self._r(self._noisy("handle", self._data.site("handle_grasp").xpos))}
        self._record("get_handle", {}, out)
        return out

    def get_drawer(self):
        d, m = self._data, self._model
        site = d.site("drawer_interior")
        half = m.site("drawer_interior").size
        opening = d.qpos[m.joint("drawer_slide").qposadr[0]]
        center = self._noisy("drawer", site.xpos)
        out = {"opening": round(float(self._noisy("opening", [opening])[0]), 4),
               "interior_center": self._r(center),
               "interior_size": self._r(2 * half),
               "floor_z": round(float(center[2] - half[2]), 4)}
        self._record("get_drawer", {}, out)
        return out

    def get_counter(self):
        site, half = self._data.site("counter_top"), self._model.site("counter_top").size
        noise = self._noisy("counter", [0.0, 0.0])
        out = {"top_z": round(float(site.xpos[2] + noise[0] / 2), 4),
               "front_x": round(float(site.xpos[0] - half[0] + noise[1] / 2), 4)}
        self._record("get_counter", {}, out)
        return out

    def get_objects(self):
        d, m = self._data, self._model
        drawer_top = d.site("drawer_interior").xpos[2] + m.site("drawer_interior").size[2]
        out = []
        for name in self._clutter:
            geom = m.geom(name)
            if geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
                ext = 2 * geom.size
            else:
                ext = np.array([2 * geom.size[0], 2 * geom.size[0], 2 * geom.size[1]])
            pos = self._noisy(name, d.body(name).xpos)
            out.append({"name": name[len("clutter_"):], "pos": self._r(pos), "size": self._r(ext),
                        "location": "drawer" if d.body(name).xpos[2] < drawer_top else "counter"})
        self._record("get_objects", {}, out)
        return out

    def get_ee(self):
        out = {"pos": self._r(self._robot.tcp_pos()), "yaw": round(self._yaw, 3),
               "gripper_width": round(self._robot.gripper_width(), 4)}
        self._record("get_ee", {}, out)
        return out

    # --- motion -----------------------------------------------------------
    def move_to(self, pos, yaw=None, speed=0.15):
        pos = np.asarray(pos, float).reshape(3)
        if not np.all(np.isfinite(pos)):
            raise ValueError(f"move_to: non-finite target {pos}")
        speed = float(np.clip(speed, 0.02, 0.4))
        yaw = self._yaw if yaw is None else float(yaw)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
                           scene.TOP_DOWN_QUAT)
        dist = np.linalg.norm(pos - self._robot.pos)
        turn = abs(yaw - self._yaw)
        duration = max(0.4, dist / speed, turn / 1.0)
        self._robot.move(pos, quat, duration)
        self._robot.hold(0.15)
        self._yaw = yaw
        err = float(np.linalg.norm(self._robot.tcp_pos() - pos))
        out = {"reached": bool(err < 0.015), "error": round(err, 4)}
        self._record("move_to", {"pos": self._r(pos), "yaw": round(yaw, 3), "speed": speed}, out)
        return out

    def close_gripper(self):
        self._robot.gripper(open_=False, duration=0.8)
        width = round(self._robot.gripper_width(), 4)
        self._record("close_gripper", {}, width)
        return width

    def open_gripper(self):
        self._robot.gripper(open_=True, duration=0.6)
        width = round(self._robot.gripper_width(), 4)
        self._record("open_gripper", {}, width)
        return width

    def wait(self, seconds):
        self._robot.hold(float(np.clip(seconds, 0.0, 10.0)))
        self._record("wait", {"seconds": seconds}, None)

    def log(self, message):
        self._record("log", {"message": str(message)[:300]}, None)
