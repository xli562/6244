"""Run one policy on one scene and judge the outcome from ground-truth state.

An episode is described by a plain dict:
    {"scene": SceneConfig.to_dict(), "noise_seed": int, "noise_sigma": float}

The judge (success check, stage, failure label, contact events) reads the
simulator state directly. The policy only ever sees the SkillAPI object.
"""

import ast
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

MAX_SIM_TIME = 90.0
ALLOWED_IMPORTS = {"numpy", "math"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit"}
STAGES = ["start", "drawer_opened", "cup_lifted", "cup_in_drawer", "success"]


class PolicyError(Exception):
    """The policy file breaks the rules (bad import, private attribute access, no run())."""


def check_policy_source(source):
    """Static check: only numpy/math imports, no private attributes, no file or code execution."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise PolicyError(f"syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise PolicyError(f"line {node.lineno}: import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise PolicyError(f"line {node.lineno}: import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise PolicyError(f"line {node.lineno}: access to private attribute '{node.attr}'")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise PolicyError(f"line {node.lineno}: use of '{node.id}' is not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body):
        raise PolicyError("the file must define a top-level function run(api)")


def load_policy(source):
    check_policy_source(source)
    namespace = {"__name__": "policy"}
    exec(compile(source, "policy.py", "exec"), namespace)  # noqa: S102 - source passed the static check
    return namespace["run"]


class Monitor:
    """Ground-truth event tracking during an episode (never shown to the policy as it runs)."""

    def __init__(self, model, data, every=5):
        self.model, self.data, self.every = model, data, every
        self.drawer_adr = model.joint("drawer_slide").qposadr[0]
        self.cup_body = model.body("cup_model").id
        self.cup_z0 = data.site("cup_center").xpos[2]
        self.max_opening = float(data.qpos[self.drawer_adr])
        self.max_cup_lift = 0.0
        self.clutter = [model.body(i).name for i in range(model.nbody)
                        if model.body(i).name.startswith("clutter_")]
        self.clutter_start = {n: data.body(n).xpos.copy() for n in self.clutter}
        self.contact_steps = {}
        root = model.body("link0").id
        self.kind = []
        for g in range(model.ngeom):
            b = model.geom_bodyid[g]
            name = model.body(b).name
            if model.body_rootid[b] == root:
                kind = "finger" if "finger" in name else "hand" if name == "hand" else "arm"
            elif name == "cup_model":
                kind = "cup"
            elif name == "drawer":
                kind = "handle" if model.geom(g).name.startswith("handle") else "drawer"
            elif name == "counter":
                kind = "counter"
            elif name.startswith("clutter_"):
                kind = name
            else:
                kind = "floor"
            self.kind.append(kind)

    def __call__(self, step):
        if step % self.every:
            return
        d = self.data
        self.max_opening = max(self.max_opening, float(d.qpos[self.drawer_adr]))
        self.max_cup_lift = max(self.max_cup_lift, float(d.site("cup_center").xpos[2] - self.cup_z0))
        for pair in self._robot_contacts():
            self.contact_steps[pair] = self.contact_steps.get(pair, 0) + self.every

    def _robot_contacts(self):
        pairs = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a, b = self.kind[c.geom1], self.kind[c.geom2]
            robot_kinds = ("arm", "hand", "finger")
            if a in robot_kinds and b in robot_kinds:
                continue
            for robot, other in ((a, b), (b, a)):
                if robot in robot_kinds:
                    # Gripper on the cup or the handle is the task; everything else is a collision.
                    if robot in ("finger", "hand") and other in ("cup", "handle"):
                        continue
                    pairs.add(f"{'gripper' if robot != 'arm' else 'arm'}-{other}")
        return pairs

    def cup_held(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if {self.kind[c.geom1], self.kind[c.geom2]} == {"finger", "cup"}:
                return True
        return False


def judge(model, data, monitor):
    """Success check and diagnosis from the final ground-truth state."""
    site = data.site("drawer_interior")
    half = model.site("drawer_interior").size
    rel = site.xmat.reshape(3, 3).T @ (data.site("cup_center").xpos - site.xpos)
    cup_h = model.numeric("cup_dims").data[1]
    inside = bool(abs(rel[0]) < half[0] and abs(rel[1]) < half[1]
                  and -half[2] < rel[2] < half[2] + cup_h)
    zz = data.body("cup_model").xmat.reshape(3, 3)[2, 2]
    tilt = float(np.degrees(np.arccos(np.clip(zz, -1, 1))))
    upright = tilt < 18.0
    opening = float(data.qpos[monitor.drawer_adr])
    closed = opening < 0.01
    released = not monitor.cup_held()
    counter_top = data.site("counter_top").xpos[2]
    moved, fallen = {}, []
    for name in monitor.clutter:
        start, now = monitor.clutter_start[name], data.body(name).xpos
        was_in_drawer = start[2] < counter_top - 0.02
        delta = now - start
        if was_in_drawer:  # compare in the drawer frame: it legitimately moves with the drawer
            delta = delta + np.array([opening - monitor_initial_opening(model), 0, 0])
        dist = float(np.linalg.norm(delta[:2]))
        if dist > 0.02:
            moved[name[len("clutter_"):]] = round(dist, 3)
        if now[2] < start[2] - 0.05:
            fallen.append(name[len("clutter_"):])
    success = inside and upright and closed and released and not fallen

    cup_z = data.site("cup_center").xpos[2]
    if inside:
        location = "drawer"
    elif not released:
        location = "gripper"
    elif cup_z < counter_top - 0.05 and not inside:
        location = "floor"
    elif abs(cup_z - monitor.cup_z0) < 0.03:
        location = "counter"
    else:
        location = "other"

    need_open = 2 * model.numeric("cup_dims").data[0] + 0.04
    if success:
        stage = "success"
    elif inside:
        stage = "cup_in_drawer"
    elif monitor.max_cup_lift > 0.03:
        stage = "cup_lifted"
    elif monitor.max_opening > need_open:
        stage = "drawer_opened"
    else:
        stage = "start"

    checks = {"cup_in_drawer": inside, "cup_upright": bool(upright), "drawer_closed": bool(closed),
              "cup_released": bool(released), "no_object_fell": not fallen}
    final = {"cup_location": location, "cup_tilt_deg": round(tilt, 1),
             "cup_offset_from_drawer_interior_center": [round(float(v), 3) for v in rel],
             "drawer_opening": round(opening, 3)}
    events = {"max_drawer_opening": round(monitor.max_opening, 3),
              "max_cup_lift": round(monitor.max_cup_lift, 3),
              "collision_seconds": {k: round(v * model.opt.timestep, 2)
                                    for k, v in sorted(monitor.contact_steps.items())},
              "objects_moved_m": moved, "objects_fallen": fallen}
    return bool(success), stage, checks, final, events


def monitor_initial_opening(model):
    key = model.key("init")
    return float(key.qpos[model.joint("drawer_slide").qposadr[0]])


def failure_label(success, stage, checks, final, error):
    if success:
        return None
    if error:
        kind = "timeout" if "EpisodeTimeout" in error else "exception"
        return f"{kind}_at_stage_{stage}"
    if not checks["no_object_fell"]:
        return "knocked_object_off"
    if stage == "start":
        return "drawer_not_opened_enough"
    if stage == "drawer_opened":
        return "cup_not_lifted"
    if stage == "cup_lifted":
        return f"cup_not_placed_in_drawer (cup ended on: {final['cup_location']})"
    if not checks["cup_upright"]:
        return "cup_tipped_over_in_drawer"
    if not checks["cup_released"]:
        return "cup_still_in_gripper"
    if not checks["drawer_closed"]:
        return "drawer_not_closed"
    return "unknown"


def run_episode(episode, policy_source, video=None, trace_limit=400):
    """Returns a JSON-serialisable result dict. Never raises for policy errors."""
    import scene
    from robot import Robot
    from skills import SkillAPI

    cfg = scene.SceneConfig.from_dict(episode["scene"])
    model, data = scene.build_model(cfg)
    monitor = Monitor(model, data)
    robot = Robot(model, data, cameras=("overview", "side") if video else (),
                  max_time=MAX_SIM_TIME, on_step=monitor)
    api = SkillAPI(robot, model, data, noise_sigma=episode.get("noise_sigma", 0.0),
                   noise_seed=episode.get("noise_seed", 0), trace_limit=trace_limit,
                   exact_readings=episode.get("exact_readings", ()))
    error = None
    try:
        policy = load_policy(policy_source)
        policy(api)
        robot.max_time = None
        robot.hold(1.0)  # let things settle before judging
    except Exception:  # noqa: BLE001 - any policy failure becomes feedback
        lines = traceback.format_exc().strip().splitlines()
        error = "\n".join(lines[-12:])
        try:
            robot.max_time = None
            robot.hold(1.0)
        except Exception:  # noqa: BLE001
            pass
    success, stage, checks, final, events = judge(model, data, monitor)
    result = {"id": episode.get("id"), "success": success, "stage": stage,
              "failure": failure_label(success, stage, checks, final, error),
              "checks": checks, "final": final, "events": events, "error": error,
              "sim_time": round(float(data.time), 1), "trace": api._trace}
    if video:
        import imageio
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video, robot.frames, fps=30, quality=8, macro_block_size=8)
    robot.close()
    return result


def _worker(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    episode, source, trace_limit = args
    try:
        return run_episode(episode, source, trace_limit=trace_limit)
    except Exception:  # noqa: BLE001 - harness-side failure (scene build etc.)
        return {"id": episode.get("id"), "success": False, "stage": "start",
                "failure": "harness_error", "checks": {}, "final": {}, "events": {},
                "error": traceback.format_exc()[-1500:], "sim_time": 0.0, "trace": []}


def run_episodes(episodes, policy_source, workers=None, trace_limit=400):
    """Run one policy on many episodes in parallel; results come back in order."""
    check_policy_source(policy_source)
    workers = workers or min(len(episodes), 48, os.cpu_count() or 1)
    jobs = [(e, policy_source, trace_limit) for e in episodes]
    if workers <= 1:
        return [_worker(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_worker, jobs))


def summarize(results):
    n = len(results)
    k = sum(r["success"] for r in results)
    failures = {}
    for r in results:
        if not r["success"]:
            failures[r["failure"]] = failures.get(r["failure"], 0) + 1
    stages = {s: sum(STAGES.index(r["stage"]) >= i for r in results) for i, s in enumerate(STAGES)}
    return {"episodes": n, "successes": k, "success_rate": round(k / n, 3) if n else 0.0,
            "reached_stage": stages, "failures": failures}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, mid - half)), float(min(1.0, mid + half)))
