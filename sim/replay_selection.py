"""Compare returning the agent's last drawer program with validation-based selection.

Both methods use exactly the same archived proposals. The selector uses a fresh family
validation draw. Evaluation uses the frozen, previously collected second proxy test set.
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import episode
import variants

ROOT = Path(__file__).resolve().parents[1]


def evaluate(jobs, workers):
    with ProcessPoolExecutor(workers) as pool:
        observations = list(pool.map(episode._worker, jobs, chunksize=2))
    if any(row["failure"] == "harness_error" for row in observations):
        raise RuntimeError("Simulator infrastructure failure.")
    return [row["success"] for row in observations]


def sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def validation_set(workers):
    tag = "selection_validation_706"
    path = variants.SETS_DIR / (tag + ".json")
    if not path.exists():
        data = variants.generate("family", 16, 706, workers=workers)
        if len(data["episodes"]) != 16:
            raise RuntimeError("Incomplete validation set.")
        for i, ep in enumerate(data["episodes"]):
            ep["noise_sigma"] = (.0025, .005, .0075, .010)[i % 4]
        variants.save_set(data, tag)
    return variants.load_set(tag)


def select(run, val, workers):
    paths = sorted(run.glob("evals/eval_*/policy.py")) + [run / "final_policy.py"]
    candidates = {}
    for path in paths:
        source = path.read_text()
        try:
            episode.check_policy_source(source)
        except episode.PolicyError:
            continue
        candidates.setdefault(source, str(path.relative_to(run)))
    if not candidates:
        raise ValueError("No valid checkpoint to select.")
    jobs = [(ep, src, 0) for src in candidates for ep in val]
    hits = evaluate(jobs, workers)
    votes = [sum(hits[i:i+len(val)]) for i in range(0, len(hits), len(val))]
    best = list(candidates)[votes.index(max(votes))]
    return dict(run=run.name, files=list(candidates.values()), votes=votes,
                selected=candidates[best], best=best, final=paths[-1].read_text())


def publish_selection(run, workers=8):
    val = validation_set(workers)
    record = select(run, val, workers)
    chosen = record.pop("best")
    original = record.pop("final")
    record.update(selected_sha256=sha(chosen), original_sha256=sha(original),
                  validation_ids=[ep["id"] for ep in val])
    # Preserve the historical final policy and score it using the existing batch entry point.
    (run / "selected_policy.py").write_text(chosen)
    (run / "selection.json").write_text(json.dumps(record, indent=1))


def main(workers):
    val = validation_set(workers)
    test = variants.load_set("test_real_b_100")
    output = ROOT / "runs/selection_replay"
    output.mkdir(exist_ok=True)
    just_scored = []
    for agent in ("claude", "codex"):
        for rep in range(1, 9):
            run = ROOT / "runs" / f"main_{agent}_family_v2_r{rep}"
            just_scored.append(select(run, val, workers))
    # All 16 selections are frozen before any test feedback is read.
    historical = {r["run_id"]: r for r in json.loads((ROOT/"sim/outputs/main/scores.json").read_text())}
    for row in just_scored:
        best, final = row.pop("best"), row.pop("final")
        ref = historical[row["run"]]
        assert hashlib.sha1(final.encode()).hexdigest() == ref["source_sha"]
        old = ref["scores"]["test_real_b_100"]["success"]
        new = old if best == final else evaluate([(ep, best, 0) for ep in test], workers)
        row.update(selected_sha256=sha(best), final_sha256=sha(final), old=old, new=new)
        print(row["run"], sum(old), sum(new), flush=True)
    (output / "results.json").write_text(json.dumps(just_scored, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=16)
    main(p.parse_args().workers)
