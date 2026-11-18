"""Score one or more policy files on one or more episode sets.

    python sim/evalset.py --policy sim/policies/oracle.py --sets twin family test_id test_real
"""

import argparse
import json
from pathlib import Path

import episode
import variants


def score(policy_path, set_name, workers=None):
    episodes = variants.load_set(set_name)
    results = episode.run_episodes(episodes, Path(policy_path).read_text(), workers=workers,
                                   trace_limit=0)
    summary = episode.summarize(results)
    lo, hi = episode.wilson(summary["successes"], summary["episodes"])
    summary["wilson95"] = [round(lo, 3), round(hi, 3)]
    return summary, results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", nargs="+", required=True)
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--out", help="write the full results to this JSON file")
    args = parser.parse_args()
    table = {}
    for policy in args.policy:
        for set_name in args.sets:
            summary, results = score(policy, set_name)
            table[f"{policy}|{set_name}"] = {"summary": summary, "results": results}
            lo, hi = summary["wilson95"]
            print(f"{Path(policy).name:28s} {set_name:10s} {summary['successes']:3d}/{summary['episodes']:<3d} "
                  f"({100 * lo:3.0f}-{100 * hi:3.0f}%)  failures: {summary['failures']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
