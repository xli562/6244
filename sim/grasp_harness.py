"""Select a physical grasp in a short plan, or freely rewrite a program with the same skills."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import signal
import subprocess

import numpy as np

import drawer_harness as prior
from drawer_harness import (
    AGENTS,
    HERE,
    ObserveOrder,
    base,
    bench,
    digest,
    dump,
    episode,
    instruction,
    propose,
    read,
    save_lock,
)
from grasp_skills import GraspSkills
from record_rejected_proposal import save_failure

DATA = read(HERE / "data/libero_grasps.json")
DEFAULT_OUT = HERE.parent / "runs/object_grasps_01"
INTERFACE = """api.place_with(item, target, grasp) physically grasps and places one object.
Select 'rim' to pinch the wall of an open bowl, 'thin_edge' to align the jaws across a
narrow book-like object and orient it for placement, or 'standard' for ordinary solid
objects and earlier cup skills. It can retry a grasp. Resolve any ambiguous object/region
using the instruction and perceived names. When a placement requires operation of a
drawer, include all actions that the instruction requires."""
STYLES = {
    "plan": "Write a short ordered program, naming your choice of grasp for each placement and composing the shared skills.",
    "free": "Freely rewrite the policy for this task. You may use the same grasp skills, choose other physical actions, replace the starter or combine approaches.",
}


def register():
    prior.register()
    known = {r["task"] for r in bench.COMPOSITION["tasks"]}
    bench.COMPOSITION = dict(
        bench.COMPOSITION,
        tasks=bench.COMPOSITION["tasks"] + [r for r in DATA["tasks"] if r["task"] not in known],
    )


def system_spec():
    files = [
        "grasp_harness.py",
        "grasp_skills.py",
        "record_rejected_proposal.py",
        "data/libero_grasps.json",
    ]
    return dict(
        files={n: digest((HERE / n).read_text()) for n in files},
        previous=prior.fingerprint(),
        styles=STYLES,
    )


def proposals(out, group, workers):
    descriptors = {}
    for row in DATA["tasks"]:
        if row["group"] == group:
            text, names = instruction(row["task"])
            descriptors[row["task"]] = dict(text=text, names=names)
    save_lock(out / (group + "_inputs.json"), descriptors)

    def generate(job):
        method, agent, task = job
        info = descriptors[task]
        common = (
            bench.API_DOC
            + "\n"
            + prior.API
            + "\n"
            + INTERFACE
            + "\nThe same previous policy is available at api.fallback(). "
            "Below is its source for optional reuse. Return one fenced Python block defining run(api). "
            "You have one call, no tools, trials, demonstrations, goal predicates or test results. "
            "You may submit just a new run(api) if it uses the API directly. "
            "Source:\n"
            + base(agent)
            + "\nInstruction: "
            + info["text"]
            + "\nPerceived names: "
            + json.dumps(info["names"])
        )
        prompt = common + "\n" + STYLES[method]
        folder = out / group / method / agent / task
        folder.mkdir(parents=True, exist_ok=True)
        policy = folder / "proposal.py"
        with (folder / "compile.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not policy.exists():
                # Never give a failed arm a second model attempt for the same instruction.
                if (folder / "calls.jsonl").exists():
                    raise ValueError(f"Unresolved previous call in {folder}")
                try:
                    propose(agent, AGENTS[agent]["model"], AGENTS[agent]["effort"], prompt, folder)
                except (
                    subprocess.TimeoutExpired,
                    episode.PolicyError,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    if not (folder / "calls.jsonl").exists():
                        raise
                    save_failure(folder, str(exc), None)
            if (folder / "prompt.txt").read_text() != prompt:
                raise ValueError(f"Prompt changed: {folder}")
        return "/".join(job), digest(policy.read_text())

    jobs = [(m, a, t) for t in descriptors for a in AGENTS for m in STYLES]
    with ThreadPoolExecutor(workers) as pool:
        save_lock(out / (group + "_programs.json"), dict(pool.map(generate, jobs)))


def rollout(job):
    task, indices, source, agent, arm = job
    register()
    run = episode.load_policy(source)
    env = bench.environment(task, extra_steps=40)
    states = np.load(bench.CONFIG / (task + ".npy"), allow_pickle=False)
    rows = []
    try:
        for idx in indices:
            np.random.seed(idx)
            env.env.reset()
            env.set_init_state(states[idx])
            for _ in range(5):
                env.step([0, 0, 0, 0, 0, 0, -1])
            observed = ObserveOrder(env)
            api = GraspSkills(observed, "full", base(agent))
            api.diagnostic = arm if arm not in STYLES else None
            error = None
            old = signal.signal(signal.SIGALRM, bench.stop_slow_policy)
            signal.alarm(90)
            try:
                run(api)
            except bench.EpisodeEnd:
                pass
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
            official = bool(env.check_success())
            goals = {" ".join(g): bool(env.env._eval_predicate(g)) for g in observed.goals}
            rows.append(
                dict(
                    task=task,
                    index=idx,
                    success=official,
                    goals=goals,
                    first=dict(observed.first),
                    steps=api._steps,
                    trace=api._trace,
                    error=error,
                    settled=official and api.holds_after_release(),
                )
            )
    finally:
        env.close()
    return rows


def score(out, phase, workers):
    group = "fresh" if phase == "transfer" else "repair"
    locks = read(out / (group + "_programs.json"))
    tasks = read(out / (group + "_inputs.json"))
    indices = {
        "debug": DATA["debug"],
        "validation": DATA["validation"],
        "repair-test": list(range(10, 50)),
        "transfer": list(range(50)),
    }[phase]
    results = {}
    jobs, keys = [], []
    arms = ["plan", "free", "fixed"]
    if phase in ("debug", "validation"):
        arms.extend(
            ["old_rim_offset", "no_pick_yaw", "no_slot_yaw", "legacy_pick_motion", "low_edge"]
        )
    for arm in arms:
        method = arm if arm in STYLES else "plan"
        for agent in AGENTS:
            key = arm + "/" + agent
            results[key] = []
            for task in tasks:
                source = (out / group / method / agent / task / "proposal.py").read_text()
                if digest(source) != locks["/".join([method, agent, task])]:
                    raise ValueError("Frozen program changed")
                for i in range(0, len(indices), 5):
                    jobs.append((task, indices[i : i + 5], source, agent, arm))
                    keys.append(key)
    with ProcessPoolExecutor(workers) as pool:
        for key, rows in zip(keys, pool.map(rollout, jobs)):
            results[key].extend(rows)
    dump(out / (phase + ".json"), results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "compile", "debug", "validation", "freeze", "repair-test", "transfer"),
    )
    parser.add_argument("--group", choices=("repair", "fresh"), default="repair")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    register()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("repair-test", "transfer") or (
        args.stage == "compile" and args.group == "fresh"
    ):
        if read(args.out / "system_freeze.json") != system_spec():
            raise ValueError("Freeze the current system before test.")
    if args.stage == "prepare":
        bench.prepare()
    elif args.stage == "compile":
        proposals(args.out, args.group, args.workers)
    elif args.stage == "freeze":
        save_lock(args.out / "system_freeze.json", system_spec())
    else:
        score(args.out, args.stage, args.workers)
