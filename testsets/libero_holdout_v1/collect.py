"""Lock the three complete official suites before any model calls or rollouts."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import io
import json
import pickle
import subprocess
import urllib.request
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REV = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SUITES = ("libero_goal", "libero_object", "libero_spatial")
URL = f"https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/{REV}/libero/libero"
CHECKOUT = ROOT / "third_party/LIBERO"
UP = CHECKOUT / "libero/libero"


def digest(data):
    return sha256(data).hexdigest()


def state_keys(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        states = pickle.loads(archive.read("archive/data.pkl"))
    assert len(states) == 50 and np.isfinite(states).all()
    return [digest(np.asarray(state, dtype="<f8").tobytes()) for state in states]


def collect(suite_task):
    suite, task = suite_task
    hashes = {}
    for category, ext, key in (("bddl_files", "bddl", "bddl_sha256"),
                               ("init_files", "pruned_init", "states_sha256")):
        path = f"{category}/{suite}/{task}.{ext}"
        with urllib.request.urlopen(f"{URL}/{path}", timeout=60) as r:
            data = r.read()
        expected = subprocess.check_output(["git", "-C", str(CHECKOUT), "show",
                                            f"{REV}:libero/libero/{path}"])
        assert data == expected == (UP / path).read_bytes(), path
        hashes[key] = digest(data)
    return dict(task=task, suite=suite, group="fresh", **hashes), state_keys(data)


def main():
    previous = {}
    for p in sorted((ROOT / "sim/data").glob("libero_*.json")):
        spec = json.loads(p.read_text())
        for row in spec["tasks"]:
            previous[row["task"]] = row.get("suite", spec.get("suite"))
    old_states = set()
    for name, suite in previous.items():
        old_states.update(state_keys((UP / "init_files" / suite / (name + ".pruned_init")).read_bytes()))
    files = sorted((suite, p.stem) for suite in SUITES for p in
                   (UP / "bddl_files" / suite).glob("*.bddl"))
    assert all(sum(s == suite for s, _ in files) == 10 for suite in SUITES)
    assert not ({t for _, t in files} & previous.keys())
    with ThreadPoolExecutor(6) as pool:
        collected = list(pool.map(collect, files))
    keys = [k for _, states in collected for k in states]
    assert not old_states.intersection(keys)
    spec = dict(revision=REV, source=URL, selection="All ten official tasks in each of three previously unused suites.",
                debug=[], validation=[], tasks=[row for row, _ in collected], indices=list(range(50)),
                prior_tasks=sorted(previous), prior_unique_states=len(old_states),
                unique_test_states=len(set(keys)), duplicates_with_history=0,
                protocol="One proposal per model/task/style. Same 50 starts. No score feedback or second call. Primary official success; secondary settled; fixed replay is diagnostic.")
    serialized = json.dumps(spec, indent=2) + "\n"
    path = HERE / "manifest.json"
    if path.exists():
        assert path.read_text() == serialized
    else:
        path.write_text(serialized)
    print(len(files), "tasks;", len(keys), "states;", len(set(keys)), "unique;", len(previous), "historical tasks")


if __name__ == "__main__":
    main()
