"""At the same loss compare a physical correction with the same time spent waiting."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import numpy as np
from score import b, key, records


def replay(job):
    audit, baseline = job
    repairs = [r for r in baseline["repairs"] if r.get("source") == "post_release"]
    actual = b.PreviewLibrary.compact_repair
    events = []

    def passive(self, source, event):
        if source != "post_release":
            return actual(self, source, event)
        j = len(events)
        sim = self._sensors._env
        state = np.asarray(sim.get_sim_state(), dtype="<f8").tobytes()
        matched = hashlib.sha256(state).hexdigest() == audit["episode"]["unchanged_state"]
        # Match the measured time of the actual intervention. Any additional
        # intervention on a diverged trajectory is marked for separate review.
        duration = repairs[j]["spent"] if j < len(repairs) else 0
        env = sim.env
        goals = audit["episode"]["goals"]
        def attained():
            return any(env._eval_predicate(g) for g in goals)
        info = dict(same_start=matched, remaining=self.remaining(), ticks=duration,
                    before=bool(attained()), extra=j >= len(repairs))
        events.append(info)
        self.adjustments.append(dict(source=source, strategy="wait", at=self.remaining(),
                                     event=len(self.records) - 1))
        self.motor([0, 0, 0, 0, 0, 0, -1], duration)
        info["after"] = bool(attained())
        self.normalize()
        return True

    with patch.object(b.PreviewLibrary, "compact_repair", passive):
        row = b.play((baseline["key"], "new", baseline["task"], [baseline["i"]]))[0]
    assert events and events[0]["same_start"] and not row["error"]
    return dict(key=row["key"], task=row["task"], i=row["i"], stable=row["stable"],
                events=events, old_failed=audit["also_old_failed"])


def main(stem):
    by_key = {key(r): r for r in records(stem) if r["arm"] == "new"}
    jobs = [(a, by_key[key(a)]) for a in records(stem + "_audit") if a["proven"]]
    with ProcessPoolExecutor(12) as pool:
        results = list(pool.map(replay, jobs))
    (b.OUT / (stem + "_passive.json")).write_text(json.dumps(results, indent=1) + "\n")
    print(stem, len(results), sum(not r["stable"] for r in results),
          sum(not r["events"][0]["after"] for r in results),
          sum(any(e["extra"] for e in r["events"]) for r in results))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
