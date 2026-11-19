"""Replay frozen Claude v1/v2 policies while removing only selected sensor errors."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import episode
import variants

SENSORS = {"normal": (), "accurate_drawer": ("handle", "opening", "drawer"),
           "accurate_cup": ("cup_pos", "cup_yaw")}


def run(workers):
    root = Path(__file__).resolve().parents[1]
    path = variants.SETS_DIR / "attribution_test_605.json"
    if not path.exists():
        variants.save_set(variants.generate("test_real", 64, 605, workers=workers),
                          "attribution_test_605")
    episodes = variants.load_set(str(path))[:32]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for condition in ("family", "family_v2"):
            for rep in range(1, 9):
                name = f"main_claude_{condition}_r{rep}"
                source = (root / "runs" / name / "final_policy.py").read_text()
                for sensor, keys in SENSORS.items():
                    jobs = [(dict(ep, exact_readings=keys), source, 0) for ep in episodes]
                    results = list(pool.map(episode._worker, jobs))
                    if any(r["failure"] == "harness_error" for r in results):
                        raise RuntimeError("Simulator infrastructure failure.")
                    rows.append(dict(run=name, sensor=sensor, successes=[r["success"] for r in results],
                                     failures=episode.summarize(results)["failures"]))
                print(name, flush=True)
    out = root / "runs/sensor_ablation"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(dict(episodes=[ep["id"] for ep in episodes], rows=rows)))


def closing_examples(workers):
    """Targeted post-hoc interventions on two documented failures; never used for selecting policies."""
    root = Path(__file__).resolve().parents[1]
    eps = variants.load_set("attribution_test_605")

    def assess(source, scenes, sensors=()):
        batch = [dict(ep, exact_readings=sensors) for ep in scenes]
        rows = episode.run_episodes(batch, source, workers=workers, trace_limit=0)
        if any(r["failure"] == "harness_error" for r in rows):
            raise RuntimeError("Simulator infrastructure failure.")
        return {"outcomes": [r["success"] for r in rows], "failures": episode.summarize(rows)["failures"]}

    branch = root / "runs/noise_factorial"
    src = (branch / "claude_r2/varied_hidden/proposal.py").read_text()
    before = "CLOSE_ZOFFS = [-0.050, -0.062]"
    if src.count(before) != 1:
        raise ValueError("Expected the archived low-grasp candidate.")
    record = assess(src.replace(before, "CLOSE_ZOFFS = [0.0, 0.0]"), eps)
    record.update(source="claude_r2/varied_hidden", change="CLOSE_ZOFFS [-0.050, -0.062] to [0.0, 0.0]",
                  test_ids=[ep["id"] for ep in eps])
    (branch / "closing_intervention.json").write_text(json.dumps(record))

    src = (root / "runs/main_claude_family_v2_r1/final_policy.py").read_text()
    head, close = src.split("def close_drawer(rb):")
    if close.count("gz = h[2] + 0.02") != 1:
        raise ValueError("Expected the archived push-height candidate.")
    adjusted = head + "def close_drawer(rb):" + close.replace("gz = h[2] + 0.02", "gz = h[2]")
    report = {}
    for name, sensors, policy in (("handle_only", ("handle",), src), ("opening_only", ("opening",), src),
                                  ("drawer_center_only", ("drawer",), src),
                                  ("height_fix_exact", SENSORS["accurate_drawer"], adjusted),
                                  ("height_fix_normal", (), adjusted)):
        report[name] = assess(policy, eps[:32], sensors)
    out = root / "runs/sensor_ablation"
    out.mkdir(exist_ok=True)
    (out / "close_height.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--closing-examples", action="store_true")
    args = parser.parse_args()
    (closing_examples if args.closing_examples else run)(args.workers)
