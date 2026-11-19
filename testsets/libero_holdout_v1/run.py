"""Pass an external test catalogue to the previous frozen experiment, without editing sim/."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sim"))
import grasp_harness as h

OUT = ROOT / "runs/libero_holdout_v1"
PREVIOUS = ROOT / "runs/object_grasps_01/system_freeze.json"


def verify():
    if h.system_spec() != json.loads(PREVIOUS.read_text()):
        raise ValueError("Harness differs from the previous experiment freeze")
    files = sorted((ROOT / "sim").rglob("*.py")) + sorted((ROOT / "sim/data").glob("*.json"))
    files += [HERE / "manifest.json", HERE / "run.py", PREVIOUS]
    hashes = {str(f.relative_to(ROOT)): sha256(f.read_bytes()).hexdigest() for f in files}
    h.save_lock(OUT / "input_hashes.json", hashes)
    h.save_lock(OUT / "system_freeze.json", h.system_spec())


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=["prepare", "compile", "test", "verify"])
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    verify()
    h.DATA = h.read(HERE / "manifest.json")
    h.register()
    if args.stage == "prepare":
        h.bench.prepare()
    elif args.stage == "compile":
        h.proposals(OUT, "fresh", args.workers)
    elif args.stage == "test":
        if (OUT / "transfer.json").exists():
            raise FileExistsError("This test run has already been scored")
        h.score(OUT, "transfer", args.workers)
    verify()
