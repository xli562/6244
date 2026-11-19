"""Replay the two debug instructions with only the hand/object placement correction disabled."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import libero_long as bench
import placement
from composition import DATA, OUT, base, group_tasks, read, verify_lock
from multitask import digest, dump


def without_correction(job):
    # Use an exact one-line intervention on the frozen skill. Preserve every other waypoint.
    filename = Path(placement.__file__)
    source = filename.read_text()
    line = "hand_offset = self.eef() - held"
    if source.count(line) != 1:
        raise ValueError("Cannot isolate the frozen hand-to-object correction.")
    namespace = dict(vars(placement))
    exec(
        compile(source.replace(line, "hand_offset = np.zeros(3)"), str(filename), "exec"), namespace
    )
    with patch.object(placement, "PlacementSkills", namespace["PlacementSkills"]):
        return bench.run_task(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify_lock(args.out)
    hashes = read(args.out / "debug_plans.json")
    jobs = []
    # The two CLIs emitted the same grounded plans. One physical replay suffices.
    for task in group_tasks("debug"):
        source = (args.out / "codex" / task / "proposal.py").read_text()
        if digest(source) != hashes["codex/" + task]:
            raise ValueError("Changed plan")
        jobs.extend(
            (task, [i], source, "full", base("codex"))
            for i in DATA["debug_indices"] + DATA["validation_indices"]
        )
    with ProcessPoolExecutor(12) as pool:
        results = [row for group in pool.map(without_correction, jobs) for row in group]
    dump(args.out / "no_hand_correction.json", results)
