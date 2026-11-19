"""Match the original pan regression and basket gains to the same policy and layout."""
from collections import Counter
import gzip
import json

from score import b, key, records


def main():
    by_task = {task: Counter() for task in b.DEV}
    earlier = {"old": {}, "new": {}}
    with gzip.open(b.previous.OUT / "test.jsonl.gz", "rt") as handle:
        for row in map(json.loads, handle):
            if row["arm"] in earlier and row["task"] in by_task:
                earlier[row["arm"]][key(row)] = row["stable"]

    for row in records("dev_final"):
        if row["arm"] != "new":
            continue
        first = earlier["old"][key(row)]
        second = earlier["new"][key(row)]
        now = row["stable"]
        by_task[row["task"]].update(
            original=first, previous=second, current=now,
            original_lost=first and not second,
            restored_original=first and not second and now,
            previous_retained=second and now,
            previous_gains_retained=not first and second and now,
            gained_over_previous=not second and now,
            lost_from_previous=second and not now,
        )
    (b.OUT / "paired_history.json").write_text(json.dumps(by_task, indent=2) + "\n")


if __name__ == "__main__":
    main()
