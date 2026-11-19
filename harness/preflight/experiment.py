"""Freeze the code and blind policies, select unseen tasks, then run paired evaluation."""

import argparse
import fcntl
import json
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from prompts import COMMON, DIRECT, CHECKED
from select_tasks import select
from skill_library import SkillLibrary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "harness/interaction"))
import run as previous
from tools_api import InteractionSkills

h = previous.old
OUT = ROOT / "runs/preflight_01"
REPS = 3
MODES = ("receipt", "checked", "replay")
FAMILIES = {"receipt": DIRECT, "checked": CHECKED}


class PublicAPI:
    def __init__(self, lib):
        self._lib = lib

    def __getattr__(self, name):
        if name not in ("act", "preview", "pos", "catalog", "remaining", "instruction"):
            raise AttributeError(name)
        return getattr(self._lib, name)


def lock():
    files = [
        *[
            HERE / n
            for n in (
                "experiment.py",
                "skill_library.py",
                "binding.py",
                "prompts.py",
                "select_tasks.py",
            )
        ],
        ROOT / "harness/interaction/tools_api.py",
        ROOT / "harness/interaction/run.py",
        ROOT / "testsets/libero_holdout_v1/collect.py",
        ROOT / "testsets/libero_holdout_v1/manifest.json",
        ROOT / "harness/interaction/manifest.json",
        ROOT / "runs/interaction_01/practice_inputs.json",
        *sorted((ROOT / "sim/data").glob("libero_*.json")),
    ]
    h.save_lock(
        OUT / "system.json",
        dict(
            hashes={str(p.relative_to(ROOT)): h.digest(p.read_text()) for p in files},
            prior=h.system_spec(),
            repetitions=REPS,
            models=h.AGENTS,
            modes=list(MODES),
            primary="Stable success after 40 release/raise/wait steps; mean all three generations, no selection",
            budget="One CLI call per independently authored universal program; identical physics; zero test calls",
        ),
    )


def compile_group(workers):
    inputs = h.read(ROOT / "runs/interaction_01/practice_inputs.json")

    def generate(job):
        agent, rep, family = job
        prompt = "\n".join(
            [
                COMMON,
                FAMILIES[family],
                "Previous practice instructions and observed names:",
                json.dumps(inputs),
            ]
        )
        key = f"{agent}/{rep}/{family}"
        source = previous.generate(agent, OUT / key, prompt)
        return key, h.digest(source)

    jobs = [(a, r, f) for r in range(1, REPS + 1) for a in h.AGENTS for f in FAMILIES]
    with ThreadPoolExecutor(workers) as pool:
        h.save_lock(OUT / "programs.json", dict(pool.map(generate, jobs)))


def load_programs():
    hashes = h.read(OUT / "programs.json")
    expected = {
        f"{a}/{r}/{f}" for a in h.AGENTS for r in range(1, REPS + 1) for f in FAMILIES
    }
    if hashes.keys() != expected:
        raise ValueError("Incomplete program freeze")
    result = {}
    for key, signature in hashes.items():
        src = (OUT / key / "proposal.py").read_text()
        if h.digest(src) != signature:
            raise ValueError("Program freeze mismatch")
        result[key] = src
    return result


def register(manifest):
    h.register()
    known = {r["task"] for r in h.bench.COMPOSITION["tasks"]}
    h.bench.COMPOSITION = dict(
        h.bench.COMPOSITION,
        tasks=h.bench.COMPOSITION["tasks"]
        + [r for r in manifest["tasks"] if r["task"] not in known],
    )


def json_value(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(f"Invalid IR value {type(v).__name__}")


def selected(manifest):
    return {
        "manifest": h.digest(json.dumps(manifest, sort_keys=True)),
        "programs": h.digest((OUT / "programs.json").read_text()),
    }


def episodes(job):
    key, task, indices, src = job
    agent, _, mode = key.split("/")
    np.random.seed(indices[0])
    env = h.bench.environment(task, extra_steps=40)
    states = np.load(h.bench.CONFIG / (task + ".npy"), allow_pickle=False)
    program = h.episode.load_policy(src)
    rows = []
    try:
        for i in indices:
            np.random.seed(i)
            env.env.reset()
            env.set_init_state(states[i])
            for _ in range(5):
                env.step([0, 0, 0, 0, 0, 0, -1])
            api = InteractionSkills(env, h.base(agent))
            library = SkillLibrary(api, mode == "checked")
            error = None
            previous_handler = signal.signal(signal.SIGALRM, h.bench.stop_slow_policy)
            signal.alarm(90)
            try:
                program(PublicAPI(library))
            except h.bench.EpisodeEnd:
                pass
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
            official = bool(env.check_success())
            rows.append(
                dict(
                    task=task,
                    index=i,
                    success=official,
                    settled=official and api.holds_after_release(),
                    steps=api._steps,
                    error=error,
                    bindings=json.loads(json.dumps(library.events, default=json_value)),
                    counts=library.counts,
                    trace=api._trace,
                )
            )
    finally:
        env.close()
    return rows


def test(workers):
    programs = load_programs()
    manifest = h.read(OUT / "manifest.json")
    if selected(manifest) != h.read(OUT / "selection_lock.json"):
        raise ValueError("Test set or programs changed")
    if (OUT / "results.json").exists():
        raise FileExistsError("Already scored")
    register(manifest)
    h.bench.prepare()
    jobs = []
    results = {}
    for key, src in programs.items():
        for mode in (
            [key]
            if key.endswith("/receipt")
            else [key, key.replace("/checked", "/replay")]
        ):
            results[mode] = []
            for row in manifest["tasks"]:
                ids = manifest["indices"]
                for i in range(0, len(ids), 5):
                    jobs.append((mode, row["task"], ids[i : i + 5], src))
    with ProcessPoolExecutor(workers) as pool:
        for job, rows in zip(jobs, pool.map(episodes, jobs)):
            results[job[0]].extend(rows)
    h.dump(OUT / "results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["compile", "select", "test", "verify"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    with (OUT / "experiment.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        lock()
        if args.stage == "compile":
            compile_group(args.workers)
        elif args.stage == "select":
            load_programs()
            m = select()
            h.save_lock(OUT / "manifest.json", m)
            h.save_lock(OUT / "selection_lock.json", selected(m))
        elif args.stage == "test":
            test(args.workers)
        else:
            load_programs()
        lock()
