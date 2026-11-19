"""Develop a shared policy on practice tasks, lock it, then evaluate unseen instructions."""

import argparse
import fcntl
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sim"))
import grasp_harness as old
from tools_api import TOOL_DOC, InteractionSkills, PolicyInterface

SPEC = old.read(HERE / "manifest.json")
OUT = ROOT / "runs/interaction_01"
AGENTS = old.AGENTS
REPLICAS = range(1, SPEC["repetitions"] + 1)
BRANCHES = ("review", "feedback")
DRAFT = """Write one reusable Python run(api) for household manipulation. At deployment its English
instruction, entity names, positions and relations change. Resolve roles from words, entity types,
region owners, and geometry. Use api.attempt to compose measurable actions, inspect their effects, and budget retries.
Handle compositional instructions. Preserve a reasonable fallback for unsupported operations.
No table keyed by complete instructions or scene IDs, no judge calls. Use only the documented API.
Return one fenced Python block. The program is frozen across unseen tasks. No online model calls."""
REVIEW = """Review the draft and return a full revised program for unseen instructions. Preserve
working behavior. Prefer improvements that apply to multiple objects and tasks. You may keep the
draft unchanged. You have the same runtime tools and control budget as the initial draft."""


def register():
    old.register()
    catalog = old.bench.COMPOSITION
    known = {t["task"] for t in catalog["tasks"]}
    old.bench.COMPOSITION = dict(
        catalog,
        tasks=catalog["tasks"]
        + [r for r in SPEC["practice"] + SPEC["tasks"] if r["task"] not in known],
    )


def lock_system():
    inputs = [HERE / n for n in ("manifest.json", "tools_api.py", "run.py")]
    inputs += sorted((ROOT / "sim/data").glob("libero_*.json"))
    old.save_lock(
        OUT / "freeze.json",
        {
            "previous": old.system_spec(),
            "draft": DRAFT,
            "review": REVIEW,
            "tools": TOOL_DOC,
            "hashes": {
                str(p.relative_to(ROOT)): old.digest(p.read_text()) for p in inputs
            },
        },
    )


def generate(agent, folder, prompt):
    cfg = AGENTS[agent]
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / "proposal.py"
    with (folder / "compile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if (folder / "prompt.txt").read_text() != prompt:
                raise ValueError(f"Prompt changed in {folder}")
        else:
            if (folder / "calls.jsonl").exists():
                raise RuntimeError(f"Interrupted completed call in {folder}")
            try:
                old.propose(agent, cfg["model"], cfg["effort"], prompt, folder)
            except (
                subprocess.TimeoutExpired,
                old.episode.PolicyError,
                RuntimeError,
                ValueError,
            ) as exc:
                if not (folder / "calls.jsonl").exists():
                    raise
                old.save_failure(folder, str(exc), None)
    return path.read_text()


def run_episodes(job):
    task, indices, program, agent, practice = job
    env = old.bench.environment(task, extra_steps=40)
    try:
        states = np.load(old.bench.CONFIG / (task + ".npy"), allow_pickle=False)
        policy = old.episode.load_policy(program)
        records = []
        for index in indices:
            np.random.seed(index)
            env.env.reset()
            env.set_init_state(states[index])
            for _ in range(5):
                env.step([0, 0, 0, 0, 0, 0, -1])
            api = InteractionSkills(env, old.base(agent))
            initial = (
                {name: api.pos(name) for name in env.env.objects_dict}
                if practice
                else {}
            )
            error = None
            prev = signal.signal(signal.SIGALRM, old.bench.stop_slow_policy)
            signal.alarm(90)
            try:
                policy(PolicyInterface(api))
            except old.bench.EpisodeEnd:
                pass
            except Exception as exc:  # noqa: BLE001 - a faulty policy gets an episode record.
                error = f"{type(exc).__name__}: {exc}"[:300]
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev)
            if practice:
                deltas = {
                    name: (api.pos(name) - p).round(3).tolist()
                    for name, p in initial.items()
                }
                records.append(
                    {
                        "instruction": api.instruction,
                        "steps": api._steps,
                        "error": error,
                        "moved_xyz": {
                            n: v for n, v in deltas.items() if np.linalg.norm(v) > 0.005
                        },
                        "receipts": api.receipts[:3],
                        "skill_trace": api._trace[-6:],
                    }
                )
            else:
                success = bool(env.check_success())
                records.append(
                    {
                        "task": task,
                        "index": index,
                        "steps": api._steps,
                        "error": error,
                        "tools": api.tool_calls,
                        "trace": api._trace,
                        "success": success,
                        "settled": success and api.holds_after_release(),
                    }
                )
        return records
    finally:
        env.close()


def develop(workers):
    import json

    descriptors = [list(old.instruction(r["task"])) for r in SPEC["practice"]]
    old.save_lock(OUT / "practice_inputs.json", descriptors)
    pairs = [(agent, rep) for agent in AGENTS for rep in REPLICAS]

    def context(agent):
        return "\n".join(
            [
                old.bench.API_DOC,
                old.prior.API,
                old.INTERFACE,
                TOOL_DOC,
                DRAFT,
                "Existing optional api.fallback() source:",
                old.base(agent),
                "Practice instructions and visible entity names:",
                json.dumps(descriptors),
            ]
        )

    def directory(agent, rep, branch):
        return OUT / agent / str(rep) / branch

    def draft(pair):
        agent, rep = pair
        return generate(agent, directory(agent, rep, "draft"), context(agent))

    with ThreadPoolExecutor(min(workers, len(pairs))) as pool:
        drafts = dict(zip(pairs, pool.map(draft, pairs)))
    feedback = {pair: [] for pair in pairs}
    jobs = [
        (pair, (r["task"], SPEC["practice_indices"], drafts[pair], pair[0], True))
        for pair in pairs
        for r in SPEC["practice"]
    ]
    with ProcessPoolExecutor(workers) as pool:
        for (pair, _), rows in zip(jobs, pool.map(run_episodes, (j for _, j in jobs))):
            feedback[pair].extend(rows)

    def revise(job):
        agent, rep, arm = job
        pair = (agent, rep)
        message = context(agent) + "\nDraft:\n" + drafts[pair] + "\n" + REVIEW
        if arm == "feedback":
            message += "\nPractice observations from this draft:\n" + json.dumps(
                feedback[pair], separators=(",", ":")
            )
        return generate(agent, directory(agent, rep, arm), message)

    revisions = [(a, r, b) for a, r in pairs for b in BRANCHES]
    for a, r in pairs:
        old.save_lock(directory(a, r, "draft") / "practice.json", feedback[a, r])
    hashes = {f"{a}/{r}/draft": old.digest(s) for (a, r), s in drafts.items()}
    with ThreadPoolExecutor(min(len(revisions), workers)) as pool:
        for (a, r, b), source in zip(revisions, pool.map(revise, revisions)):
            hashes[f"{a}/{r}/{b}"] = old.digest(source)
    old.save_lock(OUT / "programs.json", hashes)


def prior(workers):
    # Invoke the original one-call free-rewrite prompt and CLI, with only the task list replaced.
    old.DATA = SPEC
    old.STYLES = {"free": old.STYLES["free"]}
    folder = OUT / "prior"
    folder.mkdir(exist_ok=True)
    old.proposals(folder, "fresh", workers)


def score(workers):
    hashes = old.read(OUT / "programs.json")
    original = old.read(OUT / "prior/fresh_programs.json")
    indexes = SPEC["test_indices"]
    if (OUT / "test.json").exists():
        raise FileExistsError("Results already recorded")
    results = {}
    jobs = []
    prior_jobs = []
    for row in SPEC["tasks"]:
        task = row["task"]
        for key, signature in hashes.items():
            agent = key.split("/")[0]
            source = (OUT / key / "proposal.py").read_text()
            if old.digest(source) != signature:
                raise ValueError("Candidate changed: " + key)
            for i in range(0, len(indexes), 5):
                jobs.append((key, (task, indexes[i : i + 5], source, agent, False)))
        for agent in AGENTS:
            key = f"free/{agent}/{task}"
            source = (OUT / "prior/fresh" / key / "proposal.py").read_text()
            if old.digest(source) != original[key]:
                raise ValueError("Prior program changed: " + key)
            for i in range(0, len(indexes), 5):
                prior_jobs.append(
                    (
                        agent + "/prior",
                        (task, indexes[i : i + 5], source, agent, "free"),
                    )
                )
    with ProcessPoolExecutor(workers) as pool:
        for batch, function in [(jobs, run_episodes), (prior_jobs, old.rollout)]:
            for (key, _), rows in zip(batch, pool.map(function, (j for _, j in batch))):
                results.setdefault(key, []).extend(rows)
    expected = {(r["task"], i) for r in SPEC["tasks"] for i in indexes}
    for rows in results.values():
        if (
            len(rows) != len(expected)
            or {(r["task"], r["index"]) for r in rows} != expected
        ):
            raise ValueError("Incomplete paired results")
    old.dump(OUT / "test.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "develop", "prior", "test", "verify"]
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_lock = (OUT / (args.stage + ".lock")).open("a")
    fcntl.flock(stage_lock, fcntl.LOCK_EX)
    lock_system()
    register()
    if args.stage == "prepare":
        old.bench.prepare()
    elif args.stage == "develop":
        develop(args.workers)
    elif args.stage == "prior":
        prior(args.workers)
    elif args.stage == "test":
        score(args.workers)
