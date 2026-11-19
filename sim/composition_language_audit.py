"""Also score the stacking relation named by LIBERO's conflicting language annotation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
from pathlib import Path

import libero_long as bench
from composition import OUT, base, read, verify_lock
from multitask import dump, digest

TASK = "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl"


class InspectRelation:
    def __init__(self, env, checks):
        self.env_wrapper = env
        self.checks = checks

    def __getattr__(self, key):
        return getattr(self.env_wrapper, key)

    def check_success(self):
        # Both checks are observational. Step termination is still LIBERO's official predicate.
        value = self.env_wrapper.check_success()
        self.checks.append(
            bool(self.env._eval_predicate(["on", "akita_black_bowl_2", "akita_black_bowl_1"]))
        )
        return value


def compare(job):
    make_env = bench.environment
    checks = []
    with patch.object(
        bench, "environment", lambda *a, **kw: InspectRelation(make_env(*a, **kw), checks)
    ):
        rows = bench.run_task(job)
    seen = iter(checks)
    for row in rows:
        row["language_goal"] = next(seen)
        if row["success"]:
            next(seen)  # check made during the separate release diagnostic
    if next(seen, None) is not None:
        raise ValueError("Unexpected check count.")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    verify_lock(args.out)
    jobs, keys = [], []
    for a in ("claude", "codex"):
        plan = (args.out / a / TASK / "proposal.py").read_text()
        free = args.out / "freeform" / a
        rewrite = (free / TASK / "proposal.py").read_text()
        assert digest(plan) == read(args.out / "test_plans.json")[a + "/" + TASK]
        assert digest(rewrite) == read(free / "freeze.json")[TASK]
        for method, source, mode in [
            ("baseline", base(a), "baseline"),
            ("plan_only", plan, "plan_only"),
            ("full", plan, "full"),
            ("freeform", rewrite, "baseline"),
        ]:
            for i in range(0, 50, 5):
                jobs.append((TASK, list(range(i, i + 5)), source, mode, base(a)))
                keys.append(a + "/" + method)
    out = {}
    with ProcessPoolExecutor(20) as pool:
        for key, rows in zip(keys, pool.map(compare, jobs)):
            out.setdefault(key, []).extend(rows)
    dump(args.out / "language_audit.json", out)
