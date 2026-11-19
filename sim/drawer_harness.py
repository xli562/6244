"""Paired, single-call evaluation of ordered drawer actions and grounded spatial commands."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import fcntl
from pathlib import Path
import signal

import numpy as np

import episode
import libero_long as bench
from composition import PROMPT as PREVIOUS_PROMPT, base, read, save_lock
from drawer_skills import DrawerSkills
from multitask import digest, dump
from placement import PlacementSkills
from repair_cli import AGENTS, propose, repair_prompt
from task_adapter import instruction

HERE = Path(__file__).resolve().parent
MANIFEST = read(HERE / "data/libero_drawers.json")
OUT = HERE.parent / "runs/grounded_drawers_01"
METHODS = ("plan", "free", "free_skills")
API = """The shared skills are:
api.arrange([(movable_name, target_name), ...]) physically places objects in sequence.
api.relative_target(reference_body, direction, relation) returns the observed region name.
direction is left/right/front/back, relation 'inside' (a compartment IN a fixture) or
'outside' (a table location NEXT TO a fixture). Preserve the exact spatial wording.
api.drawer(drawer_region_name, 'open' or 'close') physically operates a sliding drawer.
Use its named top/middle/bottom region, not the countertop; for a placement on top use top_side.
Call the actions in the requested order. Opening then putting an object inside requires both.
api.nearest(region_name, candidates) resolves a location; api.fallback() runs the previous policy.
Use physical commands for unsupported actions or fall back to the old policy."""
PLAN = """Compile the instruction as a short ordered program using the shared skills below.
Return one fenced python block defining run(api). No tools or simulator access during generation.
Use the observed names, include every requested action, and distinguish placements inside from
placements outside. Prefer shared skills when present."""


def register():
    # Extend the pinned asset catalogue for this process; historical experiment files stay valid.
    known = {r["task"] for r in bench.COMPOSITION["tasks"]}
    bench.COMPOSITION = dict(
        bench.COMPOSITION,
        tasks=bench.COMPOSITION["tasks"] + [r for r in MANIFEST["tasks"] if r["task"] not in known],
    )


def fingerprint():
    files = [
        "drawer_harness.py",
        "drawer_skills.py",
        "placement.py",
        "libero_long.py",
        "composition.py",
        "repair_cli.py",
        "task_adapter.py",
        "episode.py",
        "multitask.py",
        "data/libero_drawers.json",
    ]
    return dict(
        files={name: digest((HERE / name).read_text()) for name in files},
        bases={a: digest(base(a)) for a in AGENTS},
        models=AGENTS,
    )


def compile_plans(out, group, method, workers):
    rows = [r for r in MANIFEST["tasks"] if r["group"] == group]
    descriptors = {}
    for row in rows:
        native, names = instruction(row["task"])
        descriptors[row["task"]] = dict(
            native=native, text=row.get("user_instruction", native), names=names
        )
    with (out / (group + "_inputs.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        save_lock(out / (group + "_inputs.json"), descriptors)

    def one(job):
        agent, task = job
        info = descriptors[task]
        descriptor = (
            "New instruction: "
            + info["text"]
            + "\nAvailable object/region names: "
            + json.dumps(info["names"])
        )
        if method == "old_plan":
            prompt = (
                PREVIOUS_PROMPT
                + "\nInstruction: "
                + info["text"]
                + "\nPerceived names: "
                + json.dumps(info["names"])
            )
        elif method == "plan":
            prompt = PLAN + "\n" + API + "\n" + descriptor
        else:
            interface = bench.API_DOC + ("\n" + API if method == "free_skills" else "")
            request = (
                descriptor
                + "\nFreely adapt the old program to this instruction. You may rewrite any part and use any available skills. No trial, demonstration, goal predicates or test feedback."
            )
            prompt = repair_prompt(interface, base(agent), request)
        folder = out / group / method / agent / task
        if not (folder / "proposal.py").exists():
            propose(agent, AGENTS[agent]["model"], AGENTS[agent]["effort"], prompt, folder)
        if (folder / "prompt.txt").read_text() != prompt:
            raise ValueError("Cached proposal has a different input.")
        return agent + "/" + task, digest((folder / "proposal.py").read_text())

    with ThreadPoolExecutor(workers) as pool:
        records = dict(pool.map(one, [(a, t) for t in descriptors for a in AGENTS]))
    save_lock(out / f"{group}_{method}_lock.json", records)


class ObserveOrder:
    """Record first attainment of each goal after an action; the policy only sees its motor API."""

    def __init__(self, env):
        self.raw = env
        self.goals = env.env.parsed_problem["goal_state"]
        self.steps = 0
        self.first = {}
        self.observe()

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def observe(self):
        for g in self.goals:
            key = " ".join(g)
            if key not in self.first and self.raw.env._eval_predicate(g):
                self.first[key] = self.steps

    def step(self, action):
        result = self.raw.step(action)
        self.steps += 1
        self.observe()
        return result


def rollout(job):
    task, text, indices, src, agent, method = job
    register()
    policy = episode.load_policy(src)
    states = np.load(bench.CONFIG / (task + ".npy"), allow_pickle=False)
    env = bench.environment(task, extra_steps=40)
    results = []
    try:
        for idx in indices:
            np.random.seed(idx)
            env.env.reset()
            env.set_init_state(states[idx])
            for _ in range(5):
                env.step([0, 0, 0, 0, 0, 0, -1])
            observed = ObserveOrder(env)
            cls = PlacementSkills if method in ("free", "previous", "old_plan") else DrawerSkills
            api = cls(observed, "full", base(agent))
            if method in ("rim_off", "tilt_off"):
                api.diagnostic = method
            api.instruction = text
            initial = dict(observed.first)
            error = None
            old = signal.signal(signal.SIGALRM, bench.stop_slow_policy)
            signal.alarm(90)
            try:
                policy(api)
            except bench.EpisodeEnd:
                pass
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
            success = bool(env.check_success())
            goals = {" ".join(g): bool(env.env._eval_predicate(g)) for g in observed.goals}
            results.append(
                dict(
                    task=task,
                    index=idx,
                    success=success,
                    goals=goals,
                    initial=initial,
                    first=dict(observed.first),
                    steps=api._steps,
                    error=error,
                    trace=api._trace,
                    settled=success and api.holds_after_release(),
                )
            )
    finally:
        env.close()
    return results


def score(out, phase, method, workers):
    group = "fresh" if phase == "transfer" else "repair"
    inputs = read(out / f"{group}_inputs.json")
    plan_method = "plan" if method in ("tilt_off", "rim_off") else method
    hashes = read(out / f"{group}_{plan_method}_lock.json") if method != "previous" else {}
    jobs, agents = [], []
    indices = {
        "debug": range(6),
        "validation": range(6, 10),
        "repair-test": range(10, 50),
        "transfer": range(50),
    }[phase]
    for agent in AGENTS:
        for task, info in inputs.items():
            if method == "previous":
                archive = HERE.parent / "runs/composition_01"
                source = (archive / agent / task / "proposal.py").read_text()
                expected = read(archive / "test_plans.json")[agent + "/" + task]
            else:
                source = (out / group / plan_method / agent / task / "proposal.py").read_text()
                expected = hashes[agent + "/" + task]
            if digest(source) != expected:
                raise ValueError("Changed proposal")
            for i in range(0, len(indices), 5):
                jobs.append((task, info["text"], list(indices[i : i + 5]), source, agent, method))
                agents.append(agent)
    results = {a: [] for a in AGENTS}
    with ProcessPoolExecutor(workers) as pool:
        for agent, rows in zip(agents, pool.map(rollout, jobs)):
            results[agent].extend(rows)
    dump(out / f"{phase}_{method}.json", results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=["prepare", "compile", "freeze", "debug", "validation", "repair-test", "transfer"],
    )
    p.add_argument(
        "--method",
        choices=(*METHODS, "previous", "old_plan", "tilt_off", "rim_off"),
        default="plan",
    )
    p.add_argument("--group", choices=["repair", "fresh"], default="repair")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    register()
    args.out.mkdir(exist_ok=True, parents=True)
    if args.stage in ("repair-test", "transfer") or (
        args.stage == "compile" and args.group == "fresh"
    ):
        if read(args.out / "system_freeze.json") != fingerprint():
            raise ValueError("First freeze the current skills, prompts and task split.")
    if args.stage == "prepare":
        bench.prepare()
    elif args.stage == "freeze":
        save_lock(args.out / "system_freeze.json", fingerprint())
    elif args.stage == "compile":
        if args.method not in (*METHODS, "old_plan"):
            p.error("This method reuses a frozen proposal")
        compile_plans(args.out, args.group, args.method, args.workers)
    else:
        score(args.out, args.stage, args.method, args.workers)
