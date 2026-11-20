"""Score the newest frozen system (grasp_preview, checked programs, 'new' arm) on official LIBERO suites.

    python harness/official_eval/run.py prepare
    python harness/official_eval/run.py test [--workers 56] [--states 50] [--fix-receipts]
    python harness/official_eval/run.py test --suite unseen --fix-receipts   # tasks from unseen.py select

Policies are reused unchanged from runs/preview_01; no model calls. Official tasks are registered in a
separate asset folder (runs/official_eval_01/assets) so frozen evidence folders stay untouched.
"""
import argparse, gzip, json, pickle, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness/grasp_preview"))
import benchmark as latest            # grasp_preview entry; pulls in the whole harness chain
from control import PreviewLibrary

base, previous = latest.base, latest.previous
UP = ROOT / "third_party/LIBERO/libero/libero"
OUT = ROOT / "runs/official_eval_01"
ASSETS = OUT / "assets"
SUITES = {"libero_10": "10", "libero_goal": "goal", "libero_object": "object", "libero_spatial": "spatial"}
ARM = "new"
FIX_RECEIPTS = False   # set by --fix-receipts before the worker pool forks


class ReceiptFixOverlay:
    """Stage-program view that hides non-placement receipts (their numeric fields are None).

    The generation prompt only promises numbers on placements and says the receipt may be None,
    so a push/open/turn receipt is shown as None rather than as a dict holding None values.
    """
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def instruction(self):
        return self._inner.instruction

    def last(self):
        r = self._inner.last()
        if r and r.get("op") != "place" and (r.get("max_lift") is None or r.get("xy_gap") is None):
            return None
        return r


def tasks():
    return [f"{tag}__{p.stem}" for suite, tag in SUITES.items()
            for p in sorted((UP / "bddl_files" / suite).glob("*.bddl"))]


def prepare():
    import yaml
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "config.yaml").write_text(yaml.safe_dump({k: str(UP / sub) for k, sub in {
        "benchmark_root": ".", "assets": "assets", "init_states": "init_files",
        "bddl_files": "bddl_files", "datasets": "."}.items()}))
    suite_of = {tag: suite for suite, tag in SUITES.items()}
    for t in tasks():
        tag, name = t.split("__", 1)
        suite = suite_of[tag]
        (ASSETS / (t + ".bddl")).write_text((UP / "bddl_files" / suite / (name + ".bddl")).read_text())
        with zipfile.ZipFile(UP / "init_files" / suite / (name + ".pruned_init")) as z:
            member = next(n for n in z.namelist() if n.endswith("data.pkl"))
            states = pickle.loads(z.read(member))
        assert states.shape[0] == 50 and np.isfinite(states).all(), t
        np.save(ASSETS / (t + ".npy"), states, allow_pickle=False)
    print(f"Prepared {len(tasks())} official tasks in {ASSETS}")


def play(job):
    """Same as grasp_preview.benchmark.play, but loads tasks from ASSETS."""
    k, arm, t, ids = job
    prior = previous.prior
    ancestor, planner, original = prior.OLD_SOURCES(k)
    libs = []

    def factory(*args):
        lib = PreviewLibrary(*args)
        libs.append(lib)
        return lib

    def entry(ctrl):
        stage, review, newest = {}, {}, {}
        exec(prior.compiled(k), stage)
        exec(previous.code(k), review)
        exec(latest.compiled(k), newest)
        lib = libs[-1]
        lib.parent_handler = stage.get("checkpoint")
        decide = review.get("decide")
        lib.policy = (lambda e: decide(lib.ledger(e))) if lib.enhanced and decide else decide
        lib.next_choice = newest.get("choose")
        lib.release_choice = newest.get("release")
        lib.handler = lib.arbitrate
        view = prior.Overlay(ctrl, ancestor["run"])
        stage["run"](ReceiptFixOverlay(view) if FIX_RECEIPTS else view)
        for finish in (review["finish"], newest["finish"]):
            receipt = ctrl.last()
            if receipt and receipt["op"] == "place" and finish(lib.ledger(receipt)) == "regrasp":
                ctrl.adjust("regrasp")

    with ExitStack() as s:
        s.enter_context(patch.object(base, "EffectLibrary", factory))
        s.enter_context(patch.object(base, "sources", lambda _: ({**ancestor, "run": entry}, planner, original)))
        s.enter_context(patch.object(base, "OUT", OUT))
        rows = base.play(job)
    return [dict(key=r["key"], task=r["task"], i=r["i"], official=r["official"], stable=r["stable"],
                 steps=r["steps"], error=r["error"]) for r in rows]


def test(workers, n_states, fix=False, suite="official"):
    global FIX_RECEIPTS, OUT
    FIX_RECEIPTS = fix
    if suite == "unseen":            # globals are set before the worker pool forks
        import unseen
        OUT = unseen.OUT
        names = [r["task"] for r in unseen.tasks()]
    else:
        names = tasks()
    keys = [k for k in base.h.read(latest.OUT / "programs.json") if k.endswith("/checked")]
    jobs = [(k, ARM, t, list(range(i, min(i + 5, n_states)))) for k in keys for t in names
            for i in range(0, n_states, 5)]
    dest = OUT / f"test_{n_states}{'_fixed' if fix else ''}.jsonl.gz"
    if dest.exists():
        raise FileExistsError(dest)
    tmp, n = dest.with_suffix(".tmp"), 0
    with ProcessPoolExecutor(workers) as pool, gzip.open(tmp, "wt") as f:
        for rows in pool.map(play, jobs):
            for r in rows:
                f.write(json.dumps(r, default=base.old.json_value) + "\n")
            n += len(rows)
            if n % 500 < len(rows):
                print(f"{n}/{len(jobs) * 5}", flush=True)
    tmp.replace(dest)
    print("wrote", dest, n)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=("prepare", "test"))
    p.add_argument("--workers", type=int, default=56)
    p.add_argument("--states", type=int, default=50)
    p.add_argument("--suite", choices=("official", "unseen"), default="official")
    p.add_argument("--fix-receipts", action="store_true",
                   help="hide non-placement receipts from stage programs (see ReceiptFixOverlay)")
    a = p.parse_args()
    prepare() if a.stage == "prepare" else test(a.workers, a.states, a.fix_receipts, a.suite)
