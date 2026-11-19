"""LIBERO-Long state-based manipulation with official tasks, assets, initial states and judge."""
import argparse
import hashlib
import json
import os
import pickle
import signal
import sys
import zipfile
from pathlib import Path

import numpy as np

import episode

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "third_party"
UPSTREAM = BASE / "LIBERO/libero/libero"
CONFIG = BASE / "libero-config"
MANIFEST = json.loads((HERE / "data/libero_long.json").read_text())
TRANSFER = json.loads((HERE / "data/libero_transfer.json").read_text())
COMPOSITION = json.loads((HERE / "data/libero_composition.json").read_text())
HORIZON = 500

API_DOC = """Write a single run(api) program for multiple LIBERO kitchen, living room and study tasks.
At runtime api.instruction is the task in English; api.names() lists perceived object and region names.
api.pos(name) returns a numpy world xyz position in metres for a body or region, api.eef() the fingertip midpoint.
api.move(xyz, steps=30) servos the gripper in world coordinates; returns remaining distance.
api.grip(closed) closes (True) or opens (False) the fingers and waits.
api.action([dx,dy,dz,rx,ry,rz,g], steps=1) applies normalized relative OSC commands; +g closes.
api.transfer(item, target, grasp=(0,0,0), drop=(0,0,0.08)) tries a top-down grasp and placement.
Offsets are world xyz relative to the named item/target. It returns the observed lift in metres.
It uses physical fingers: a poor grasp can slip; check object positions. You may call primitives directly.
No teleportation. Each episode ends on success or after 500 control steps. Only numpy/math imports.
Programs are evaluated on additional task instructions as well as new starting configurations."""


def paths(task):
    suite = next((t["suite"] for t in COMPOSITION["tasks"] if t["task"] == task), None)
    if suite is None:
        suite = TRANSFER["suite"] if any(task == t["task"] for t in TRANSFER["tasks"]) else MANIFEST["suite"]
    return (UPSTREAM / "bddl_files" / suite / (task + ".bddl"),
            UPSTREAM / "init_files" / suite / (task + ".pruned_init"))


def prepare():
    import yaml
    CONFIG.mkdir(exist_ok=True)
    cfg = {k: str(UPSTREAM / sub) for k, sub in {
        "benchmark_root": ".", "assets": "assets", "init_states": "init_files",
        "bddl_files": "bddl_files", "datasets": "."}.items()}
    (CONFIG / "config.yaml").write_text(yaml.safe_dump(cfg))
    tasks = {t["task"]: t for t in MANIFEST["tasks"] + TRANSFER["tasks"] + COMPOSITION["tasks"]}
    for task in tasks.values():
        bddl, archive = paths(task["task"])
        for file, key in ((bddl, "bddl_sha256"), (archive, "states_sha256")):
            if hashlib.sha256(file.read_bytes()).hexdigest() != task[key]:
                raise ValueError(f"Unexpected upstream content: {file}")
        # Official pinned archives store a single NumPy array inside torch.save's zip.
        with zipfile.ZipFile(archive) as zipped:
            states = pickle.loads(zipped.read("archive/data.pkl"))
        if states.shape[0] != 50 or not np.isfinite(states).all():
            raise ValueError(f"Invalid start states in {archive}")
        np.save(CONFIG / (task["task"] + ".npy"), states, allow_pickle=False)
    print(f"Prepared {len(tasks)} verified tasks and {len(tasks)*50} official initial states.", flush=True)


def cases(split, limit=None):
    lo, hi = MANIFEST["state_ranges"][split]
    if limit is not None and not 1 <= limit <= hi - lo:
        raise ValueError(f"limit must be between 1 and {hi-lo} for {split}")
    for item in MANIFEST["tasks"]:
        if (item["split"] == "unseen") == (split == "unseen"):
            yield item["task"], list(range(lo, lo + limit if limit else hi))


def environment(task, extra_steps=0):
    upstream_root = str(BASE / "LIBERO")
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)
    os.environ["LIBERO_CONFIG_PATH"] = str(CONFIG)
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero.envs.env_wrapper import ControlEnv
    return ControlEnv(str(paths(task)[0]), use_camera_obs=False, has_offscreen_renderer=False,
                      hard_reset=False, horizon=HORIZON + 5 + extra_steps)


class EpisodeEnd(Exception):
    pass


def stop_slow_policy(signum, frame):
    raise TimeoutError("Policy exceeded the 90-second wall-clock budget.")


class Skills:
    def __init__(self, env):
        self._env = env
        self.instruction = env.language_instruction
        self._steps = 0
        self._closed = False
        self._trace = []
        self._names = sorted(env.env.object_states_dict)

    def names(self):
        return list(self._names)

    def pos(self, name):
        return self._env.env.object_states_dict[name].get_geom_state()["pos"].copy()

    def eef(self):
        robot = self._env.robots[0]
        return self._env.sim.data.site_xpos[robot.eef_site_id].copy()

    def action(self, action, steps=1):
        a = np.asarray(action, dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError("Expected 7 finite action values.")
        self._closed = a[-1] > 0
        for _ in range(int(steps)):
            if self._steps >= HORIZON:
                raise EpisodeEnd()
            _, _, done, _ = self._env.step(np.clip(a, -1, 1))
            self._steps += 1
            if done:
                raise EpisodeEnd()

    def move(self, xyz, steps=30):
        target = np.asarray(xyz, float)
        for _ in range(int(steps)):
            error = target - self.eef()
            if np.linalg.norm(error) < .006:
                break
            self.action([*(error * 15), 0, 0, 0, 1 if self._closed else -1])
        return float(np.linalg.norm(target - self.eef()))

    def grip(self, closed):
        self.action([0, 0, 0, 0, 0, 0, 1 if closed else -1], steps=10)

    def transfer(self, item, target, grasp=(0, 0, 0), drop=(0, 0, .08)):
        initial = self.pos(item)
        p, q = initial + grasp, self.pos(target) + drop
        height = max(p[2], q[2], self.eef()[2]) + .09
        self.grip(False)
        self.move([p[0], p[1], height])
        self.move(p)
        self.grip(True)
        self.move([p[0], p[1], height])
        lift = float(self.pos(item)[2] - initial[2])
        self._trace.append([item, target, "lift_m", round(lift, 3)])
        if lift > .02:
            self.move([q[0], q[1], height])
            self.move(q)
        self.grip(False)
        self.move([self.eef()[0], self.eef()[1], height])
        return lift


def run_task(job):
    task, indices, source, *options = job
    api_class = Skills
    if options:
        from placement import PlacementSkills
        api_class = PlacementSkills
    episode.check_policy_source(source)
    states = np.load(CONFIG / (task + ".npy"), allow_pickle=False)
    env = environment(task, extra_steps=40 if options else 0)
    records = []
    try:
        for i in indices:
            # Reset controller goals and object tracker state before restoring the official state.
            np.random.seed(i)
            env.env.reset()
            env.set_init_state(states[i])
            api = api_class(env, *options)
            for _ in range(5):
                env.step([0, 0, 0, 0, 0, 0, -1])
            if options:
                initial_goals = {" ".join(g): bool(env.env._eval_predicate(g))
                                 for g in env.env.parsed_problem["goal_state"]}
            error = None
            previous = signal.signal(signal.SIGALRM, stop_slow_policy)
            signal.alarm(90)
            try:
                episode.load_policy(source)(api)
            except EpisodeEnd:
                pass
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            goals = {" ".join(g): bool(env.env._eval_predicate(g)) for g in env.env.parsed_problem["goal_state"]}
            target_positions = {}
            for predicate in env.env.parsed_problem["goal_state"]:
                for name in predicate[1:]:
                    if name in api.names():
                        target_positions[name] = api.pos(name).round(4).tolist()
            records.append(dict(task=task, index=i, instruction=api.instruction,
                                success=bool(env.check_success()), goals=goals, steps=api._steps,
                                objects=api.names(), final_positions=target_positions,
                                trace=api._trace[-6:], error=error))
            if options:
                record = records[-1]
                record["initial_goals"] = initial_goals
                record["holds_after_release"] = record["success"] and api.holds_after_release()
    finally:
        env.close()
    return records


def evaluate(source, split, limit=None, workers=None):
    from concurrent.futures import ProcessPoolExecutor
    workers = int(os.environ.get("LIBERO_WORKERS", "4")) if workers is None else workers
    jobs = []
    for task, indices in cases(split, limit):
        for start in range(0, len(indices), 5):
            jobs.append((task, indices[start:start + 5], source))
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return [r for group in pool.map(run_task, jobs) for r in group]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "score"))
    p.add_argument("--policy", type=Path)
    p.add_argument("--split", choices=list(MANIFEST["state_ranges"]), default="debug")
    p.add_argument("--limit", type=int, help="initial states per task; default uses full split")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        if not args.policy:
            p.error("score requires --policy")
        rows = evaluate(args.policy.read_text(), args.split, args.limit)
        if args.out:
            args.out.write_text(json.dumps(rows, indent=1))
        print(f"{sum(r['success'] for r in rows)}/{len(rows)}")
