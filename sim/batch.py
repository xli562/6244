"""Main experiment: several development runs per (agent, condition), then score every final
program on the held-out test sets.

    python sim/batch.py run   --tag main --reps 3     # skips runs that already finished well
    python sim/batch.py score --tag main              # scores already in scores.json are reused

Run ids are <tag>_<agent>_<condition>_r<rep>. Repetition r develops on scenes
[8 (r-1), 8 r) of the development set, so the repetitions of a condition see different draws.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

import episode
import harness
import variants

HERE = Path(__file__).resolve().parent
DEV_SETS = {"twin": "twin_64", "family": "family_64", "family_v2": "family_v2_64", "naive": "naive"}  # same first 24 scenes as twin_24 / family_24
# test_real_b_100: a second draw of the hidden real-scene set, made after test_real_100 had
# been used to decide what family_v2 varies; no design decision has looked at it.
TEST_SETS = ["test_id_100", "test_real_100", "test_real_b_100"]
PAIRS = [("family", "twin"), ("family_v2", "twin"), ("family_v2", "family"),
         ("family", "oneshot"), ("twin", "oneshot")]
REFERENCES = {"reference_oracle": HERE / "policies" / "oracle.py",
              "reference_twin_overfit": HERE / "policies" / "naive_twin.py"}


def run_id(tag, agent, condition, rep):
    return f"{tag}_{agent}_{condition}_r{rep}"


def load_meta(rid):
    path = harness.RUNS_DIR / rid / "run.json"
    return json.loads(path.read_text()) if path.exists() else None


def problem_of(meta):
    """None if the run finished properly, else a short reason."""
    if meta is None:
        return "no run.json"
    if meta.get("timed_out"):
        return "timed out"
    if meta.get("loop_problem"):
        return meta["loop_problem"]
    if meta.get("exit_code") not in (0, None):
        return f"exit code {meta['exit_code']}"
    if meta.get("usage", {}).get("is_error"):
        return "agent reported an error"
    if not meta.get("final_policy_valid"):
        return f"no valid final program ({meta.get('final_policy_problem')})"
    if meta["condition"] != "oneshot" and not meta.get("evaluations_used"):
        return "no evaluation was used"
    return None


def launch(job, n_dev, budget, timeout):
    tag, agent, condition, rep = job
    rid = run_id(*job)
    command = [sys.executable, str(HERE / "harness.py"), "--agent", agent, "--condition", condition,
               "--run-id", rid, "--n-dev", str(n_dev), "--budget", str(budget),
               "--timeout", str(timeout)]
    if condition in DEV_SETS:
        command += ["--dev-set", DEV_SETS[condition], "--dev-offset", str(n_dev * (rep - 1))]
    harness.RUNS_DIR.mkdir(exist_ok=True)
    with open(harness.RUNS_DIR / f"{rid}.out", "w") as out:
        subprocess.run(command, cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
    return problem_of(load_meta(rid))


def cmd_run(args):
    jobs = [(args.tag, agent, condition, rep) for rep in range(1, args.reps + 1)
            for condition in args.conditions for agent in args.agents]
    slots = {agent: threading.Semaphore(args.parallel) for agent in args.agents}

    def work(job):
        with slots[job[1]]:
            started = time.time()
            problem = launch(job, args.n_dev, args.budget, args.timeout)
            meta = load_meta(run_id(*job)) or {}
            history = [f"{h['successes']}/{h['episodes']}" for h in meta.get("evaluation_history", [])]
            print(f"{time.strftime('%H:%M:%S')} {run_id(*job)}: {problem or 'ok'} "
                  f"({time.time() - started:.0f} s, dev {history})", flush=True)
            return problem

    for attempt in range(1, args.attempts + 1):
        todo = [j for j in jobs if problem_of(load_meta(run_id(*j)))]
        if not todo:
            break
        if attempt > 1:
            print(f"waiting {args.retry_wait} s before retrying {len(todo)} runs", flush=True)
            time.sleep(args.retry_wait)
        print(f"attempt {attempt}: {len(todo)} runs to do", flush=True)
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            list(pool.map(work, todo))
    bad = {run_id(*j): problem_of(load_meta(run_id(*j))) for j in jobs}
    bad = {k: v for k, v in bad.items() if v}
    print(f"done: {len(jobs) - len(bad)}/{len(jobs)} runs finished properly", flush=True)
    for rid, problem in bad.items():
        print(f"  NOT OK {rid}: {problem}", flush=True)


def _score_job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    r = episode._worker(args)
    return {"success": bool(r["success"]), "failure": r["failure"], "stage": r["stage"]}


def scene_bootstrap(a, b, n_boot=10000, seed=0):
    """95% interval of mean(a) - mean(b), resampling test scenes. a, b: runs x scenes arrays."""
    diff = a.mean(axis=0) - b.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def run_scene_bootstrap(a, b, n_boot=10000, seed=0):
    """95% interval of mean(a) - mean(b), resampling development runs and test scenes."""
    rng = np.random.default_rng(seed)
    n = a.shape[1]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        cols = rng.integers(0, n, n)
        ra = a[rng.integers(0, len(a), len(a))][:, cols]
        rb = b[rng.integers(0, len(b), len(b))][:, cols]
        diffs[i] = ra.mean() - rb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def run_permutation(a, b, n_perm=20000, seed=0):
    """Two-sided permutation test on the per-run success rates."""
    ra, rb = a.mean(axis=1), b.mean(axis=1)
    observed = abs(ra.mean() - rb.mean())
    pooled = np.concatenate([ra, rb])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        hits += abs(pooled[:len(ra)].mean() - pooled[len(ra):].mean()) >= observed - 1e-12
    return (hits + 1) / (n_perm + 1)


def cmd_score(args):
    out_dir = HERE / "outputs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = {name: variants.load_set(name) for name in TEST_SETS}
    rows = []
    for name, path in REFERENCES.items():
        rows.append({"run_id": name, "agent": "reference", "condition": name, "source": path.read_text()})
    for run_dir in sorted(harness.RUNS_DIR.glob(f"{args.tag}_*_r*")):
        meta = load_meta(run_dir.name)
        if not run_dir.is_dir() or meta is None:
            continue
        rows.append({"run_id": run_dir.name, "agent": meta["agent"], "condition": meta["condition"],
                     "source": (run_dir / "final_policy.py").read_text(), "problem": problem_of(meta),
                     "dev_history": [h["successes"] for h in meta["evaluation_history"]],
                     "dev_episodes": meta["dev_episodes"], "wall_seconds": meta.get("wall_seconds"),
                     "usage": meta.get("usage")})
    scores_path = out_dir / "scores.json"
    old = {r["run_id"]: r for r in json.loads(scores_path.read_text())} if scores_path.exists() else {}
    todo = []  # (row, set name) pairs that still have to be simulated
    for row in rows:
        row["source_sha"] = hashlib.sha1(row["source"].encode()).hexdigest()
        prev = old.get(row["run_id"], {})
        same = prev.get("source_sha", row["source_sha"]) == row["source_sha"]
        row["scores"] = {n: prev["scores"][n] for n in sets if same and n in prev.get("scores", {})}
        try:
            episode.check_policy_source(row["source"])
        except episode.PolicyError as e:
            for name, episodes in sets.items():
                row["scores"][name] = {"success": [False] * len(episodes), "successes": 0,
                                       "episodes": len(episodes),
                                       "failures": {f"invalid program: {e}": len(episodes)}}
        todo += [(row, name) for name in sets if name not in row["scores"]]
    jobs = [(ep, row["source"], 0) for row, name in todo for ep in sets[name]]
    print(f"{len(todo)} program x set pairs to simulate ({len(jobs)} episodes), "
          f"{len(rows) * len(sets) - len(todo)} reused", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        flat = list(pool.map(_score_job, jobs, chunksize=4)) if jobs else []
    i = 0
    for row, name in todo:
        results = flat[i:i + len(sets[name])]
        i += len(results)
        ok = [r["success"] for r in results]
        row["scores"][name] = {"success": ok, "successes": sum(ok), "episodes": len(ok),
                               "failures": episode.summarize(results)["failures"]}
    for row in rows:
        del row["source"]
        row["scores"] = {n: row["scores"][n] for n in sets}
        print(f"{row['run_id']:36s} " + "  ".join(
            f"{n} {s['successes']:3d}/{s['episodes']}" for n, s in row["scores"].items()), flush=True)
    scores_path.write_text(json.dumps(rows, indent=1))

    lines = [f"# Results of experiment '{args.tag}'", ""]
    cells = {}
    for row in rows:
        cells.setdefault((row["agent"], row["condition"]), []).append(row)
    for name in TEST_SETS:
        lines += [f"## {name}", "", "| agent | condition | runs | per run | pooled | 95% Wilson (pooled) "
                  "| run mean, SD | failures |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for (agent, condition), group in cells.items():
            k = sum(r["scores"][name]["successes"] for r in group)
            n = sum(r["scores"][name]["episodes"] for r in group)
            lo, hi = episode.wilson(k, n)
            failures = {}
            for r in group:
                for label, count in r["scores"][name]["failures"].items():
                    failures[label] = failures.get(label, 0) + count
            per_run = [r["scores"][name]["successes"] for r in group]
            top = ", ".join(f"{label} {count}" for label, count in
                            sorted(failures.items(), key=lambda kv: -kv[1])[:3])
            lines.append(f"| {agent} | {condition} | {len(group)} | "
                         + ", ".join(str(r["scores"][name]["successes"]) for r in group)
                         + f" | {k}/{n} = {100 * k / n:.1f}% | {100 * lo:.1f}-{100 * hi:.1f}% | "
                         + (f"{np.mean(per_run):.1f}, {np.std(per_run, ddof=1):.1f}" if len(group) > 1 else "-")
                         + f" | {top} |")
        lines.append("")
        lines += ["Differences in success rate (first minus second). 'scenes': 95% interval from "
                  "resampling test scenes only. 'runs and scenes': resampling development runs and "
                  "test scenes. p: permutation test on the per-run success rates.", ""]
        for agent in sorted({a for a, _ in cells if a != "reference"}):
            for first, second in PAIRS:
                if (agent, first) in cells and (agent, second) in cells:
                    a = np.array([r["scores"][name]["success"] for r in cells[(agent, first)]], float)
                    b = np.array([r["scores"][name]["success"] for r in cells[(agent, second)]], float)
                    d, lo, hi = scene_bootstrap(a, b)
                    lo2, hi2 = run_scene_bootstrap(a, b)
                    lines.append(f"- {agent}: {first} - {second} = {100 * d:+.1f} points; scenes "
                                 f"({100 * lo:+.1f} to {100 * hi:+.1f}); runs and scenes "
                                 f"({100 * lo2:+.1f} to {100 * hi2:+.1f}); p = {run_permutation(a, b):.3f}")
        lines.append("")
    lines += ["## Development runs", "", "| run | problem | dev successes per evaluation | minutes |",
              "| --- | --- | --- | --- |"]
    for row in rows:
        if row["agent"] != "reference":
            minutes = f"{row['wall_seconds'] / 60:.0f}" if row.get("wall_seconds") else "?"
            lines.append(f"| {row['run_id']} | {row['problem'] or '-'} | "
                         f"{row['dev_history']} of {row['dev_episodes']} | {minutes} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"saved {out_dir / 'scores.json'} and {out_dir / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tag", default="main")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--agents", nargs="+", default=["claude", "codex"])
    p.add_argument("--conditions", nargs="+", default=["oneshot", "twin", "family"])
    p.add_argument("--n-dev", type=int, default=8)
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument("--parallel", type=int, default=3, help="simultaneous runs per agent")
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--retry-wait", type=int, default=600)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("score")
    p.add_argument("--tag", default="main")
    p.add_argument("--workers", type=int, default=200)
    p.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
