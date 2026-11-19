"""Verify call and state pairing; summarize stable gains and losses with and without the checker."""

import csv
import json
from collections import Counter

from experiment import MODES, OUT, REPS, h, load_programs, lock, selected


def main():
    lock()
    manifest = h.read(OUT / "manifest.json")
    assert h.read(OUT / "selection_lock.json") == selected(manifest)
    programs = load_programs()
    data = h.read(OUT / "results.json")
    assert set(data) == {
        f"{agent}/{rep}/{mode}"
        for rep in range(1, REPS + 1)
        for agent in h.AGENTS
        for mode in MODES
    }
    calls = list(OUT.glob("*/*/*/calls.jsonl"))
    assert len(calls) == len(programs)
    rejected = {}
    for f in calls:
        folder = f.parent
        arm = str(folder.relative_to(OUT))
        assert arm in programs and len(f.read_text().splitlines()) == 1
        if (folder / "rejection.json").exists():
            rejected[arm] = h.read(folder / "rejection.json")["reason"]
        assert all(
            r["task"] not in (folder / "prompt.txt").read_text()
            for r in manifest["tasks"]
        )
    states = {(r["task"], i) for r in manifest["tasks"] for i in manifest["indices"]}

    def describe(rows):
        return {
            "n": len(rows),
            "official": sum(x["success"] for x in rows),
            "stable": sum(x["settled"] for x in rows),
            "errors": sum(x["error"] is not None for x in rows),
            "executed_steps": sum(x["steps"] for x in rows),
            "rebound_episodes": sum(x["counts"]["corrected"] > 0 for x in rows),
            "blocked_episodes": sum(x["counts"]["blocked"] > 0 for x in rows),
        }

    totals, per_task, paired = {}, [], []
    for key, rows in data.items():
        assert (
            len(rows) == len(states)
            and {(r["task"], r["index"]) for r in rows} == states
        )
        assert all(not x["settled"] or x["success"] for x in rows)
        totals[key] = describe(rows)
        for spec in manifest["tasks"]:
            task = spec["task"]
            per_task.append(
                {
                    "program": key,
                    "task": task,
                    **describe([x for x in rows if x["task"] == task]),
                }
            )
    for a in h.AGENTS:
        for rep in range(1, REPS + 1):
            prefix = f"{a}/{rep}/"
            before = {(r["task"], r["index"]): r for r in data[prefix + "replay"]}
            counts = Counter()
            for r in data[prefix + "checked"]:
                baseline = before[r["task"], r["index"]]
                delta = int(r["settled"]) - int(baseline["settled"])
                counts[{1: "gained", -1: "lost", 0: "unchanged"}[delta]] += 1
            paired.append({"agent": a, "rep": rep, **counts})
    with (OUT / "by_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(per_task[0]))
        w.writeheader()
        w.writerows(per_task)
    summary = {
        "calls": len(calls),
        "unique_states": len(states),
        "episodes": sum(len(rs) for rs in data.values()),
        "rejected": rejected,
        "programs": totals,
        "paired_checker_vs_replay": paired,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
