"""Compile one policy per new instruction without simulator feedback, then freeze and test."""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import libero_long as bench
from multitask import digest, dump, quality
from repair_cli import AGENTS, propose, repair_prompt


def instruction(task):
    env = bench.environment(task)
    try:
        # Only read information available to the deployed program. No reset state or score is sent.
        return env.language_instruction, sorted(env.env.object_states_dict)
    finally:
        env.close()


def compile_tasks(args, tasks):
    source = args.base.read_text()
    spec = dict(base_sha256=digest(source), agent=args.agent, model=args.model or AGENTS[args.agent]["model"],
                tasks=tasks, protocol="one instruction-only proposal per task")
    args.out.mkdir(exist_ok=True, parents=True)
    config = args.out / "protocol.json"
    if config.exists() and json.loads(config.read_text()) != spec:
        raise ValueError("Run directory already contains a different experiment.")
    dump(config, spec)
    (args.out / "base.py").write_text(source)
    descriptors = {task: instruction(task) for task in tasks}

    def convert(task):
        folder = args.out / task
        if not (folder / "proposal.py").exists():
            language, objects = descriptors[task]
            request = (f"New instruction: {language}\nAvailable object/region names: {objects}. "
                       "Adapt the supplied program to this instruction using its reusable skills. "
                       "The policy must work over many starting positions. You get no trial, "
                       "demonstration, goal predicates or test feedback.")
            prompt = repair_prompt(bench.API_DOC, source, request)
            propose(args.agent, spec["model"], AGENTS[args.agent]["effort"], prompt, folder)
        return task, digest((folder / "proposal.py").read_text())

    with ThreadPoolExecutor(min(3, len(tasks))) as pool:
        frozen = dict(pool.map(convert, tasks))
    dump(args.out / "freeze.json", frozen)


def refine(args, tasks):
    """One targeted repair with 3 debug starts; select on 3 different validation starts."""
    frozen = json.loads((args.out / "freeze.json").read_text())
    spec = json.loads((args.out / "protocol.json").read_text())
    base = (args.out / "base.py").read_text()
    if digest(base) != spec["base_sha256"]:
        raise ValueError("Base program changed.")

    def work(task):
        folder = args.out / task
        compiled = (folder / "proposal.py").read_text()
        if digest(compiled) != frozen[task]:
            raise ValueError("Compiled program changed.")
        choices = [base, compiled]
        # Initialization and wall timers run in simulator processes, independent of the CLI.
        debug_jobs = [(task, [0, 1, 2], src) for src in choices]
        with ProcessPoolExecutor(2, mp_context=get_context("spawn")) as pool:
            debug = list(pool.map(bench.run_task, debug_jobs))
        winner = max(range(2), key=lambda i: quality(debug[i]))
        folder = folder / "repair"
        folder.mkdir(exist_ok=True)
        dump(folder / "debug.json", debug)
        policy = folder / "proposal.py"
        if not policy.exists():
            examples = [{k: r[k] for k in ("instruction", "objects", "goals", "final_positions",
                                             "trace", "steps", "error")} for r in debug[winner]]
            prompt = repair_prompt(bench.API_DOC, choices[winner], json.dumps(examples))
            propose(spec["agent"], spec["model"], AGENTS[spec["agent"]]["effort"], prompt, folder)
        choices.append(policy.read_text())
        with ProcessPoolExecutor(3, mp_context=get_context("spawn")) as pool:
            val = list(pool.map(bench.run_task, [(task, [5, 6, 7], src) for src in choices]))
        dump(folder / "validation.json", val)
        chosen = max(range(3), key=lambda i: quality(val[i]))
        (folder / "selected.py").write_text(choices[chosen])
        return task, dict(selected=chosen, sha256=digest(choices[chosen]), validation=[quality(r) for r in val])

    with ThreadPoolExecutor(3) as pool:
        selected = dict(pool.map(work, tasks))
    dump(args.out / "freeze_repaired.json", selected)


def test_repair(args, tasks):
    selected = json.loads((args.out / "freeze_repaired.json").read_text())
    spec = json.loads((args.out / "protocol.json").read_text())
    compiled = json.loads((args.out / "freeze.json").read_text())
    programs = {mode: {} for mode in ("direct", "adapted", "repaired")}
    for task in tasks:
        folder = args.out / task
        for mode, path, expected in (
            ("direct", args.out / "base.py", spec["base_sha256"]),
            ("adapted", folder / "proposal.py", compiled[task]),
            ("repaired", folder / "repair/selected.py", selected[task]["sha256"]),
        ):
            source = path.read_text()
            if digest(source) != expected:
                raise ValueError("Frozen program changed.")
            programs[mode][task] = source
    results = {}
    # Share identical rollouts across methods only when the frozen source and task match.
    rows_by_program = {}
    for mode, group in programs.items():
        jobs = [(task, list(range(i, i+5)), src) for task, src in group.items()
                if (task, digest(src)) not in rows_by_program for i in range(10, 50, 5)]
        if jobs:
            with ProcessPoolExecutor(min(args.workers, len(jobs))) as pool:
                for (task, _, src), rows in zip(jobs, pool.map(bench.run_task, jobs)):
                    rows_by_program.setdefault((task, digest(src)), []).extend(rows)
        results[mode] = [r for task, src in group.items() for r in rows_by_program[task, digest(src)]]
    dump(args.out / "scores_repaired.json", results)

def score(args, tasks):
    manifest = json.loads((args.out / "freeze.json").read_text())
    protocol = json.loads((args.out / "protocol.json").read_text())
    base = (args.out / "base.py").read_text()
    if digest(base) != protocol["base_sha256"] or set(manifest) != set(tasks):
        raise ValueError("Incomplete or changed frozen experiment.")
    programs = {}
    for task in tasks:
        source = (args.out / task / "proposal.py").read_text()
        if digest(source) != manifest[task]:
            raise ValueError("Frozen policy has changed.")
        programs[task] = source
    report = {}
    for mode in ("direct", "adapted"):
        jobs = [(t, list(range(i, min(i+5, args.limit))), base if mode == "direct" else programs[t])
                for t in tasks for i in range(0, args.limit, 5)]
        with ProcessPoolExecutor(min(len(jobs), args.workers)) as pool:
            rows = [r for group in pool.map(bench.run_task, jobs) for r in group]
        report[mode] = rows
        print(mode, sum(r["success"] for r in rows), "/", len(rows), flush=True)
    dump(args.out / f"scores_{args.limit}.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest="stage", required=True)
    dev = stages.add_parser("compile")
    dev.add_argument("--agent", choices=list(AGENTS), required=True)
    dev.add_argument("--base", type=Path, required=True)
    dev.add_argument("--model", help="CLI model override")
    test = stages.add_parser("test")
    test.add_argument("--limit", type=int, choices=range(1, 51), default=50)
    test.add_argument("--workers", type=int, default=8)
    repair = stages.add_parser("refine")
    repaired = stages.add_parser("test-repair")
    repaired.add_argument("--workers", type=int, default=8)
    for parser in (dev, test, repair, repaired):
        parser.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    tasks = [t["task"] for t in bench.TRANSFER["tasks"]]
    if args.stage == "compile":
        compile_tasks(args, tasks)
    elif args.stage == "refine":
        refine(args, tasks)
    elif args.stage == "test-repair":
        test_repair(args, tasks)
    else:
        score(args, tasks)
