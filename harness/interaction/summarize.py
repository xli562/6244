"""Audit budgets, pair states, and export run- and task-level scores."""

import csv
import json
from hashlib import sha256
from pathlib import Path

from run import AGENTS, BRANCHES, OUT, REPLICAS, SPEC, lock_system


def main():
    lock_system()
    test = json.loads((OUT / "test.json").read_text())
    signatures = json.loads((OUT / "programs.json").read_text())
    prior = json.loads((OUT / "prior/fresh_programs.json").read_text())
    tasks = {r["task"] for r in SPEC["tasks"]}
    states = {(t, i) for t in tasks for i in SPEC["test_indices"]}
    keys = {
        f"{a}/{r}/{b}" for a in AGENTS for r in REPLICAS for b in ("draft", *BRANCHES)
    }
    assert keys == set(signatures)
    assert set(test) == keys | {a + "/prior" for a in AGENTS}
    assert set(prior) == {f"free/{a}/{t}" for a in AGENTS for t in tasks}
    calls = {**signatures, **{"prior/fresh/" + k: v for k, v in prior.items()}}
    rejections = []
    for key, signature in calls.items():
        folder = OUT / key
        assert sha256((folder / "proposal.py").read_bytes()).hexdigest() == signature
        assert len((folder / "calls.jsonl").read_text().splitlines()) == 1
        if (folder / "rejection.json").exists():
            rejections.append(key)
    for a in AGENTS:
        for r in REPLICAS:
            p = OUT / a / str(r)
            trace = json.loads((p / "draft/practice.json").read_text())
            assert len(trace) == len(SPEC["practice"]) * len(SPEC["practice_indices"])
            assert all(
                set(row)
                == {
                    "instruction",
                    "steps",
                    "error",
                    "moved_xyz",
                    "receipts",
                    "skill_trace",
                }
                for row in trace
            )
            control = (p / "review/prompt.txt").read_text()
            treatment = (p / "feedback/prompt.txt").read_text()
            assert (
                treatment
                == control
                + "\nPractice observations from this draft:\n"
                + json.dumps(trace, separators=(",", ":"))
            )
            assert not any(task in treatment for task in tasks)
    per_task = []
    totals = {}
    for key, rows in sorted(test.items()):
        assert (
            len(rows) == len(states)
            and {(r["task"], r["index"]) for r in rows} == states
        )
        assert all(not r["settled"] or r["success"] for r in rows)

        def stats(group):
            return {
                "n": len(group),
                "official": sum(r["success"] for r in group),
                "stable": sum(r["settled"] for r in group),
                "errors": sum(r["error"] is not None for r in group),
                "catalog_episodes": sum(
                    r.get("tools", {}).get("catalog", 0) > 0 for r in group
                ),
                "attempt_episodes": sum(
                    r.get("tools", {}).get("attempt", 0) > 0 for r in group
                ),
            }

        totals[key] = stats(rows)
        for task in sorted(tasks):
            per_task.append(
                {
                    "program": key,
                    "task": task,
                    **stats([r for r in rows if r["task"] == task]),
                }
            )
    with (OUT / "by_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, per_task[0])
        w.writeheader()
        w.writerows(per_task)
    paired = {}
    for agent in AGENTS:
        for rep in REPLICAS:
            prefix = f"{agent}/{rep}/"
            tables = [
                {(r["task"], r["index"]): r["settled"] for r in test[prefix + arm]}
                for arm in BRANCHES
            ]
            review, feedback = tables
            paired[prefix.rstrip("/")] = {
                "review_only": sum(review[k] and not feedback[k] for k in states),
                "feedback_only": sum(feedback[k] and not review[k] for k in states),
                "both": sum(review[k] and feedback[k] for k in states),
            }
    summary = dict(
        stable_pairs=paired,
        calls=len(calls),
        rejections=rejections,
        test_episodes=sum(len(v) for v in test.values()),
        programs=totals,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
