"""Verify the frozen proposals and the paired design; export per-task results."""
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "runs/libero_holdout_v1"


def read(path):
    return json.loads(path.read_text())


def main():
    spec = read(HERE / "manifest.json")
    tasks = {row["task"]: row["suite"] for row in spec["tasks"]}
    expected = {(t, i) for t in tasks for i in spec["indices"]}
    programs = read(OUT / "fresh_programs.json")
    results = read(OUT / "transfer.json")
    expected_calls = {f"{arm}/{agent}/{task}" for task in tasks
                      for arm in ("plan", "free") for agent in ("claude", "codex")}
    assert set(programs) == expected_calls
    call_stats = Counter()
    for key, digest in programs.items():
        folder = OUT / "fresh" / key
        calls = (folder / "calls.jsonl").read_text().splitlines()
        assert len(calls) == 1
        call = json.loads(calls[0])
        assert call["agent"] == key.split("/")[1]
        assert sha256((folder / "proposal.py").read_bytes()).hexdigest() == digest
        status = "rejected" if (folder / "rejection.json").exists() else "accepted"
        call_stats[f"{key.split('/')[1]}/{status}"] += 1
    rows = []
    by_suite = {}
    assert set(results) == {f"{arm}/{agent}" for arm in ("plan", "free", "fixed")
                            for agent in ("claude", "codex")}
    for arm, episodes in sorted(results.items()):
        assert len(episodes) == len(expected)
        assert {(r["task"], r["index"]) for r in episodes} == expected
        for r in episodes:
            assert not r["settled"] or r["success"]
        for task, suite in tasks.items():
            subset = [r for r in episodes if r["task"] == task]
            rows.append(dict(suite=suite, task=task, arm=arm, episodes=len(subset),
                             success=sum(r["success"] for r in subset),
                             settled=sum(r["settled"] for r in subset),
                             errors=sum(r["error"] is not None for r in subset)))
        for suite in sorted(set(tasks.values())):
            subset = [r for r in rows if r["arm"] == arm and r["suite"] == suite]
            by_suite[f"{arm}/{suite}"] = {k: sum(r[k] for r in subset) for k in
                                           ("episodes", "success", "settled", "errors")}
    csv_path = OUT / "per_task.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info = {"calls": dict(call_stats), "results": by_suite}
    (OUT / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
