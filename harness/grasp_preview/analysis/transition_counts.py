"""Summarize paired wins, losses, retry costs and available observations."""
from collections import Counter, defaultdict
import json

from score import b, key, records


def summarize(stem):
    arms = defaultdict(dict)
    for row in records(stem):
        arms[row["arm"]][key(row)] = row
    result = {}
    for k, new in arms["new"].items():
        other = {arm: rows[k] for arm, rows in arms.items()}
        for group in ("ALL", new["task"]):
            count = result.setdefault(group, Counter())
            count.update(
                episodes=1,
                zero_steps=new["steps"] == 0,
                exhausted=new["steps"] == 460,
                misses_verification=new["official"] and not new["stable"],
                receipt_paired_wins=new["stable"] and not other["receipt"]["stable"],
                receipt_paired_losses=not new["stable"] and other["receipt"]["stable"],
            )
            for arm in ("old", "new"):
                losses = [e for e in other[arm]["releases"] if e["loss"]]
                count[arm + "_alerted"] += bool(losses)
                count[arm + "_actions"] += sum(bool(e.get("acted")) for e in losses)
            for e in new["releases"]:
                if not e["loss"]:
                    continue
                count["loss_events"] += 1
                if "retry_cost" not in e:
                    count["no_new_decision"] += 1
                    continue
                enough = e["remaining"] >= e["retry_cost"]
                available = e["retries_left"] > 0
                count["steps_and_slot"] += enough and available
                count["steps_only"] += enough and not available
                count["slot_only"] += available and not enough
                count["neither"] += not enough and not available
            for repair in new["repairs"]:
                if repair.get("strategy") == "reseat":
                    count["local_repair"] += 1
                    count["post_release_repair"] += repair["source"] == "post_release"
                    count["positive_spend"] += repair["spent"] > 0
            if new["stable"] != other["old"]["stable"]:
                outcome = "win" if new["stable"] else "loss"
                if new["stable"] == other["no_preview"]["stable"]:
                    count[outcome + "_also_without_preview"] += 1
                if new["stable"] == other["no_release"]["stable"]:
                    count[outcome + "_also_without_release"] += 1
    (b.OUT / (stem + "_transitions.json")).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    import sys
    summarize(sys.argv[1])
