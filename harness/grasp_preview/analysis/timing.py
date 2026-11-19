"""Post-freeze physical controls: restore just full-duration checks or repeated opening."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import gzip
import inspect
import json
from unittest.mock import patch

from score import b, key, records
from controller import ReviewLibrary


def one(job):
    variant, policy, task, indices = job
    new = b.PreviewLibrary
    configure = new.configure
    inspect_release = new.inspect_release

    def repeated_open(self, *args):
        configure(self, *args)
        # Bypass only budgeted_grip. Its delegate still measures all native
        # closures and runs all inherited contact/carry/release observations.
        grip = self._sensors.grip
        assert grip.__name__ == "budgeted_grip"
        self._sensors.grip = inspect.getclosurevars(grip).nonlocals["grip"]

    def full_inspection(self):
        if self.seating:
            return inspect_release(self)
        # The old 40-step open/retract/settle sequence uses the same loss rule.
        # Keep the new local repair and its cost bound during an actual reseat.
        return ReviewLibrary.inspect_release(self)

    with ExitStack() as stack:
        if variant == "repeat_open":
            stack.enter_context(patch.object(new, "configure", repeated_open))
        else:
            stack.enter_context(patch.object(new, "inspect_release", full_inspection))
        rows = b.play((policy, "new", task, indices))
    return [dict(key=r["key"], task=task, i=r["i"], variant=variant, stable=r["stable"],
                 steps=r["steps"], error=r["error"]) for r in rows]


def main():
    manifest = json.loads((b.OUT / "manifest.json").read_text())
    policies = json.loads((b.OUT / "programs.json").read_text())
    jobs = [(variant, policy, task["task"], list(range(start, start + 5)))
            for policy in policies if policy.endswith("checked") for task in manifest["tasks"]
            for variant in ("repeat_open", "full_inspection") for start in range(0, 50, 5)]
    with gzip.open(b.OUT / "timing.jsonl.gz", "wt") as handle, ProcessPoolExecutor(58) as pool:
        for rows in pool.map(one, jobs):
            for row in rows:
                handle.write(json.dumps(row) + "\n")


def verify():
    new = {key(r): r for r in records("test") if r["arm"] == "new"}
    results = {}
    for variant in ("repeat_open", "full_inspection"):
        rows = {key(r): r for r in records("timing") if r["variant"] == variant}
        assert rows.keys() == new.keys()
        section = {}
        for k, row in rows.items():
            assert not row["error"] and row["steps"] <= 460, row
            for group in (row["task"], "ALL"):
                c = section.setdefault(group, dict(new_wins=0, new_losses=0))
                c["new_wins"] += new[k]["stable"] and not row["stable"]
                c["new_losses"] += row["stable"] and not new[k]["stable"]
        results[variant] = section
    (b.OUT / "timing_summary.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    import sys
    verify() if "--verify" in sys.argv else main()
