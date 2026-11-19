"""Develop and freeze one program for many LIBERO tasks, then independently score it."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import libero_long as bench
from episode import PolicyError
from repair_cli import AGENTS, propose, repair_prompt


def quality(rows):
    complete = sum(row["success"] for row in rows)
    partial = sum(sum(row["goals"].values()) / len(row["goals"]) for row in rows)
    return complete, round(partial, 8)


def feedback(rows):
    lines = []
    for task in sorted({r["task"] for r in rows}):
        group = [r for r in rows if r["task"] == task]
        r = next((r for r in group if not r["success"]), group[0])
        lines.append(json.dumps(dict(instruction=r["instruction"], objects=r["objects"],
                                     solved=sum(x["success"] for x in group), total=len(group),
                                     example_final_goals=r["goals"], attempted_grasps=r["trace"],
                                     final_positions=r.get("final_positions", {}), error=r["error"])))
    return "\n".join(lines)


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def dump(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def develop(args):
    out = args.run
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(agent=args.agent, model=args.model or AGENTS[args.agent]["model"],
                effort=AGENTS[args.agent]["effort"], rounds=args.rounds, dev=args.dev, val=args.val,
                manifest=digest(json.dumps(bench.MANIFEST)), api=digest(Path(bench.__file__).read_text()))
    plan = out / "plan.json"
    if plan.exists() and json.loads(plan.read_text()) != spec:
        raise ValueError("This run has a different protocol. Choose a new run directory.")
    if (out / "freeze.json").exists():
        print("Already frozen:", out)
        return
    dump(plan, spec)
    # Observe the available debug scenes once. This is also a no-action calibration.
    intro = out / "debug_inventory.json"
    if not intro.exists():
        dump(intro, bench.evaluate("def run(api):\n    pass\n", "debug", 1))
    task = bench.API_DOC + "\nDevelopment task inventory:\n" + feedback(json.loads(intro.read_text()))
    best_dev, best_val = None, None
    stop_reason = None
    history = []
    for i in range(args.rounds):
        folder = out / f"candidate_{i:02d}"
        proposal = folder / "proposal.py"
        if proposal.exists():
            source = proposal.read_text()
        else:
            if best_dev is None:
                prompt = repair_prompt(task, "# Write the first policy.", "No trials yet.")
            else:
                incumbent = out / f"candidate_{best_dev:02d}"
                prompt = repair_prompt(task, (incumbent / "proposal.py").read_text(),
                                       feedback(json.loads((incumbent / "debug.json").read_text())))
                if i > 1 and best_dev != i-1:
                    rejected = out / f"candidate_{i-1:02d}" / "debug.json"
                    prompt += "\nA later attempt failed to improve. Its debug outcomes:\n" + feedback(json.loads(rejected.read_text()))
            try:
                source = propose(args.agent, spec["model"], spec["effort"], prompt, folder)
            except (subprocess.TimeoutExpired, RuntimeError, ValueError, PolicyError) as exc:
                if not history:
                    raise
                stop_reason = f"Candidate {i}: {type(exc).__name__}: {exc}"
                print("Freezing previously evaluated candidates.", stop_reason, flush=True)
                break
        scores = {}
        for split, count in (("debug", args.dev), ("validation", args.val)):
            data_path = folder / (split + ".json")
            if not data_path.exists():
                dump(data_path, bench.evaluate(source, split, count))
            scores[split] = quality(json.loads(data_path.read_text()))
        history.append(dict(index=i, sha256=digest(source), **scores))
        if best_dev is None or scores["debug"] > history[best_dev]["debug"]:
            best_dev = i
        if best_val is None or scores["validation"] > history[best_val]["validation"]:
            best_val = i
        print("candidate", i, scores, flush=True)
    selected = out / f"candidate_{best_val:02d}" / "proposal.py"
    (out / "policy.py").write_text(selected.read_text())
    dump(out / "freeze.json", dict(selected=best_val, candidates=history,
                                   requested_rounds=args.rounds, stop_reason=stop_reason))


def calibrate(args):
    """Rank the already frozen candidate pool on all 10 validation states per task."""
    frozen = json.loads((args.run / "freeze.json").read_text())
    for entry in frozen["candidates"]:
        folder = args.run / f"candidate_{entry['index']:02d}"
        source = (folder / "proposal.py").read_text()
        if digest(source) != entry["sha256"]:
            raise ValueError("Frozen candidate changed.")
        rows = bench.evaluate(source, "validation")
        entry["validation"] = quality(rows)
        dump(folder / "validation_full.json", rows)
    winner = max(frozen["candidates"], key=lambda entry: entry["validation"])
    frozen["selected"] = winner["index"]
    frozen["validation_states_per_task"] = 10
    source = (args.run / f"candidate_{winner['index']:02d}" / "proposal.py").read_text()
    (args.run / "policy_calibrated.py").write_text(source)
    dump(args.run / "freeze_val10.json", frozen)
    print("full validation", [(c["index"], c["validation"]) for c in frozen["candidates"]],
          "selected", frozen["selected"], flush=True)


def test(args):
    freeze = "freeze_val10.json" if args.calibrated else "freeze.json"
    frozen = json.loads((args.run / freeze).read_text())
    indices = sorted({0, frozen["selected"], len(frozen["candidates"])-1})
    scores = {}
    for i in indices:
        source = (args.run / f"candidate_{i:02d}" / "proposal.py").read_text()
        if digest(source) != frozen["candidates"][i]["sha256"]:
            raise ValueError("Frozen candidate changed.")
        scores[i] = {}
        for split in ("test", "unseen"):
            rows = bench.evaluate(source, split, args.limit)
            scores[i][split] = rows
            print("candidate", i, split, quality(rows), "/", len(rows), flush=True)
    suffix = "_val10" if args.calibrated else ""
    dump(args.run / f"test_{args.limit or 'full'}{suffix}.json", scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    d = sub.add_parser("develop")
    d.add_argument("--agent", choices=list(AGENTS), required=True)
    d.add_argument("--model")
    d.add_argument("--rounds", type=int, default=3)
    d.add_argument("--dev", type=int, default=2)
    d.add_argument("--val", type=int, default=2)
    c = sub.add_parser("calibrate")
    t = sub.add_parser("test")
    t.add_argument("--calibrated", action="store_true")
    t.add_argument("--limit", type=int, help="number of starts per task")
    for cmd in (d, c, t):
        cmd.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "develop":
        if args.rounds < 1:
            parser.error("rounds must be positive")
        develop(args)
    elif args.stage == "calibrate":
        calibrate(args)
    else:
        test(args)
