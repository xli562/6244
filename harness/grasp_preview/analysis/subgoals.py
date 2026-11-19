"""Read each subgoal at the scorer boundary for a fixed sample, outside the policy."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from unittest.mock import patch

from score import b, key, records


def one(job):
    model_key, task, index = job
    snapshots = []
    environment = b.base.h.bench.environment

    def instrument(*args, **kwargs):
        env = environment(*args, **kwargs)
        check = env.check_success

        def read_goals():
            success = check()
            goals = env.env.parsed_problem["goal_state"]
            snapshots.append([bool(env.env._eval_predicate(goal)) for goal in goals])
            return success

        env.check_success = read_goals
        return env

    with patch.object(b.base.h.bench, "environment", instrument):
        row = b.play((model_key, "new", task, [index]))[0]
    return dict(key=model_key, task=task, index=index, stable=row["stable"],
                steps=row["steps"], error=row["error"], last=snapshots[-1])


def main():
    manifest = json.loads((b.OUT / "manifest.json").read_text())
    keys = json.loads((b.OUT / "programs.json").read_text())
    jobs = [(k, t["task"], i) for k in keys if k.endswith("/checked")
            for t in manifest["tasks"] for i in (0, 9, 19, 29, 39, 49)]
    with ProcessPoolExecutor(16) as pool:
        results = list(pool.map(one, jobs))
    (b.OUT / "sampled_subgoals.json").write_text(json.dumps(results, indent=1) + "\n")


def verify():
    sample = json.loads((b.OUT / "sampled_subgoals.json").read_text())
    test = {key(r): r for r in records("test") if r["arm"] == "new"}
    counts = {}
    for r in sample:
        k = (*r["key"].split("/")[:2], r["task"], r["index"])
        ref = test[k]
        assert all(ref[field] == r[field] for field in ("stable", "steps", "error"))
        assert r["stable"] == all(r["last"])
        counts.setdefault(r["task"], Counter())[tuple(r["last"])] += 1
    print(counts)


if __name__ == "__main__":
    import sys
    verify() if "--verify" in sys.argv else main()
