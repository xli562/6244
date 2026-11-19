"""Paired 2x2 single-repair experiment. Generate all candidates before opening the test set."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import episode
import harness
import variants
from repair_cli import propose, repair_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/noise_factorial"
ARMS = [(v, label) for v in (False, True) for label in (False, True)]


def prefix(agent, rep):
    return ROOT / f"runs/main_{agent}_family_r{rep}/evals/eval_01/policy.py"


def arm_dir(agent, rep, varied, label):
    return OUT / f"{agent}_r{rep}" / f"{'varied' if varied else 'fixed'}_{'shown' if label else 'hidden'}"


def episodes(rep, varied, label):
    name = "family_v2_64" if varied else "family_64"
    return [dict(ep, noise_shown=label) for ep in variants.load_set(name)[8*(rep-1):8*rep]]


def develop(job):
    agent, rep, varied, label = job
    folder = arm_dir(*job)
    if (folder / "proposal.py").exists():
        return
    source = prefix(agent, rep).read_text()
    eps = episodes(rep, varied, label)
    results = episode.run_episodes(eps, source, workers=3, trace_limit=80)
    if any(r["failure"] == "harness_error" for r in results):
        raise RuntimeError("Simulator error in development.")
    feedback = harness.feedback_text(1, 1, eps, results, "").split("\n\nFull record")[0]
    for r in results:
        if not r["success"]:
            feedback += "\nExample failure trajectory:\n" + harness.format_trace(r["trace"], last=16)
            break
    task = ("In simulation place the cup upright in the drawer, release it, close the drawer "
            "without knocking other objects off. The development set varies layouts.\n" +
            harness.skills.API_DOC + "\n" + harness.scene_doc(variants.make_twin()).split("- Counter:")[1])
    if label:
        task += "\nDevelopment position noise standard deviations are given for each scene."
    spec = harness.AGENTS[agent]
    propose(agent, spec["model"], spec["effort"], repair_prompt(task, source, feedback), folder)
    print("generated", folder.name, agent, rep, flush=True)


def score(jobs, workers):
    # Refuse partial cells so an interrupted generation never silently changes the denominator.
    programs = {arm_dir(*j): (arm_dir(*j) / "proposal.py").read_text() for j in jobs}
    for agent, rep in sorted({j[:2] for j in jobs}):
        programs[OUT / f"{agent}_r{rep}" / "initial"] = prefix(agent, rep).read_text()
    path = variants.SETS_DIR / "attribution_test_605.json"
    if not path.exists():
        variants.save_set(variants.generate("test_real", 64, 605, workers=workers),
                          "attribution_test_605")
    eps = variants.load_set(str(path))
    if len(eps) != 64:
        raise ValueError("Expected 64 frozen test scenes.")
    rows = []
    for folder, source in programs.items():
        results = episode.run_episodes(eps, source, workers=workers, trace_limit=0)
        if any(r["failure"] == "harness_error" for r in results):
            raise RuntimeError("Simulator error in test.")
        rows.append(dict(run=folder.parent.name, arm=folder.name,
                         sha256=hashlib.sha256(source.encode()).hexdigest(),
                         outcomes=[r["success"] for r in results],
                         failures=episode.summarize(results)["failures"]))
        print(folder.parent.name, folder.name, sum(rows[-1]["outcomes"]), flush=True)
    report = {"test_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "test_ids": [e["id"] for e in eps], "rows": rows}
    groups = sorted({agent for agent, _, _, _ in jobs})
    reps = max(rep for _, rep, _, _ in jobs)
    filename = "results.json" if groups == ["claude", "codex"] and reps == 3 else (
        f"results_{'_'.join(groups)}_{reps}reps.json")
    (OUT / filename).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("develop", "score"))
    p.add_argument("--agents", nargs="+", choices=["codex", "claude"], default=["claude", "codex"])
    p.add_argument("--reps", type=int, default=3, choices=range(1, 9))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    jobs = [(a, r, v, label) for r in range(1, args.reps+1)
            for v, label in ARMS for a in args.agents]
    if args.stage == "develop":
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(develop, jobs))
    else:
        score(jobs, args.workers)
