"""Freeze a placement executor, compile one plan per instruction with each CLI, then compare."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import libero_long as bench
from multitask import digest, dump
from repair_cli import AGENTS, propose
from task_adapter import compile_tasks, instruction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = bench.COMPOSITION
OUT = ROOT / "runs/composition_01"
PROMPT = """Write one short Python function run(api) for the English instruction and perceived names below.
Return your answer in a fenced python block. You have no trials, goal predicates or test results.
All physical actions are performed by api.arrange(list_of_pairs). List one (item_name, target_name)
for EVERY requested placement, including both instances if the instruction says both. For onto an
object use its name; for a named semantic area on/in/next to an object choose the corresponding
perceived region if present (e.g. a directional region versus an object's own centre). Use a
container's interior site or an appliance's support site if available. Never use an initial
region as the destination unless the instruction explicitly calls for it. You may resolve an
ambiguous source with api.nearest(region_name, [candidate_names]) at runtime when initial regions
mark left/right/front/back etc. Each pair means pick up the physical item and put it down once.
If the whole instruction requires actions outside placement (e.g. open/close/switch), use only
api.fallback(), which runs this model's previous multi-task program. Do not invent objects, motor
commands, thresholds, helpers or new API methods. Use no external tools."""
MODES = ("baseline", "plan_only", "full")


def read(path):
    return json.loads(path.read_text())


def group_tasks(group):
    return [t["task"] for t in DATA["tasks"] if t["group"] == group]


def base(agent):
    return (ROOT / f"runs/long_{agent}_01/policy_calibrated.py").read_text()


def spec():
    files = [
        Path(__file__),
        HERE / "placement.py",
        HERE / "libero_long.py",
        HERE / "repair_cli.py",
        HERE / "task_adapter.py",
        HERE / "episode.py",
        HERE / "multitask.py",
        HERE / "data/libero_composition.json",
    ]
    return dict(
        files={f.name: digest(f.read_text()) for f in files},
        base={a: digest(base(a)) for a in AGENTS},
        models=AGENTS,
    )


def save_lock(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Mismatch with frozen record {path}")
    else:
        dump(path, value)


def verify_lock(out):
    if read(out / "system_freeze.json") != spec():
        raise ValueError("The execution protocol differs from the frozen version.")


def compile_group(out, tasks, workers):
    descriptors = {t: instruction(t) for t in tasks}

    def compile_one(job):
        a, t = job
        language, objects = descriptors[t]
        prompt = PROMPT + "\nInstruction: " + language + "\nPerceived names: " + json.dumps(objects)
        folder = out / a / t
        if (folder / "proposal.py").exists():
            if (folder / "prompt.txt").read_text() != prompt:
                raise ValueError("Cached plan has a different prompt.")
        else:
            propose(a, AGENTS[a]["model"], AGENTS[a]["effort"], prompt, folder)
        return a + "/" + t, digest((folder / "proposal.py").read_text())

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(compile_one, [(a, t) for a in AGENTS for t in tasks]))


def freeform(args):
    """Equal-call control using the previous instruction-only adaptation harness."""
    tasks = group_tasks("new_test")
    folder = args.out / "freeform" / args.agent
    if args.stage == "freeform-plan":
        compile_tasks(
            SimpleNamespace(
                agent=args.agent,
                model=None,
                out=folder,
                base=ROOT / f"runs/long_{args.agent}_01/policy_calibrated.py",
            ),
            tasks,
        )
        return
    hashes = read(folder / "freeze.json")
    jobs = []
    for task in tasks:
        source = (folder / task / "proposal.py").read_text()
        if digest(source) != hashes[task]:
            raise ValueError("The baseline adaptation changed after compilation.")
        jobs.extend(
            (task, list(range(i, i + 5)), source, "baseline", base(args.agent))
            for i in range(0, 50, 5)
        )
    with ProcessPoolExecutor(args.workers) as pool:
        results = [r for group in pool.map(bench.run_task, jobs) for r in group]
    dump(folder / "test.json", results)


def run(args):
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    dev = group_tasks("debug")
    other = group_tasks("previous_test") + group_tasks("new_test")
    if args.stage == "plan-debug":
        save_lock(out / "debug_plans.json", compile_group(out, dev, args.workers))
        return
    if args.stage == "freeze":
        save_lock(out / "system_freeze.json", spec())
        return
    if args.stage.startswith("freeform-"):
        verify_lock(out)
        if not args.agent:
            raise ValueError("Select a CLI with --agent.")
        freeform(args)
        return
    if args.stage in ("plan-test", "test"):
        verify_lock(out)
    if args.stage == "plan-test":
        save_lock(out / "test_plans.json", compile_group(out, other, args.workers))
        return
    tasks = dev if args.stage != "test" else dev + other
    hashes = read(out / "debug_plans.json")
    if args.stage == "test":
        hashes.update(read(out / "test_plans.json"))
    rows_by_key = {}
    jobs, keys = [], []
    modes = MODES if args.stage == "test" else (*MODES, "no_spacing", "center_grasp")
    for a in AGENTS:
        for t in tasks:
            plan = (out / a / t / "proposal.py").read_text()
            if digest(plan) != hashes[a + "/" + t]:
                raise ValueError("Changed CLI proposal.")
            indices = DATA["debug_indices"] if args.stage == "debug" else DATA["validation_indices"]
            if args.stage == "test":
                indices = list(range(10, 50)) if t in dev else list(range(50))
            for mode in modes:
                key = (a, t, mode)
                rows_by_key[key] = []
                options = (mode, base(a))
                source = base(a) if mode == "baseline" else plan
                for start in range(0, len(indices), 5):
                    jobs.append((t, indices[start : start + 5], source, *options))
                    keys.append(key)
    with ProcessPoolExecutor(args.workers) as pool:
        for key, rows in zip(keys, pool.map(bench.run_task, jobs)):
            rows_by_key[key].extend(rows)
    results = {
        a: {mode: [r for t in tasks for r in rows_by_key[a, t, mode]] for mode in modes}
        for a in AGENTS
    }
    dump(out / (args.stage + ".json"), results)
    for (a, t, m), rows in rows_by_key.items():
        print(a, t, m, sum(r["success"] for r in rows), "/", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "plan-debug",
            "debug",
            "validation",
            "freeze",
            "plan-test",
            "test",
            "freeform-plan",
            "freeform-test",
        ],
    )
    parser.add_argument("--agent", choices=list(AGENTS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=OUT)
    run(parser.parse_args())
