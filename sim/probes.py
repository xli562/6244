"""Fidelity probes: change one aspect of the development scenes at a time and see how much
the developed programs lose.

The base is test_id_100 (the twin with family layouts, 5 mm perception error). Every probe
keeps the same 100 layouts and noise seeds and changes one thing toward what the hidden
real-scene test set has; `wide_layout` is a fresh set (twin, wider layout ranges) and
`all_together` is test_real_100 itself.

    python sim/probes.py --tag main
"""

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import episode
import harness
import variants

HERE = Path(__file__).resolve().parent


def with_scene(episodes, **changes):
    out = copy.deepcopy(episodes)
    for ep in out:
        ep["scene"].update(changes)
    return out


def with_noise(episodes, sigma):
    out = copy.deepcopy(episodes)
    for ep in out:
        ep["noise_sigma"] = sigma
    return out


def conditions():
    base = variants.load_set("test_id_100")
    real, twin = variants.real_scene().to_dict(), variants.make_twin().to_dict()
    geometry = {k: real[k] for k in variants.TWIN_ERROR}
    c = {"base": base}
    c["geometry_real"] = with_scene(base, **geometry)
    for key in variants.TWIN_ERROR:  # one dimension at a time
        c[f"geom:{key}"] = with_scene(base, **{key: real[key]})
    c["drawer_friction_real"] = with_scene(base, drawer_frictionloss=real["drawer_frictionloss"])
    c["drawer_friction_high"] = with_scene(base, drawer_frictionloss=1.5 * real["drawer_frictionloss"])
    c["drawer_damping_low"] = with_scene(base, drawer_damping=0.5 * real["drawer_damping"])
    c["drawer_damping_high"] = with_scene(base, drawer_damping=1.5 * real["drawer_damping"])
    c["cup_mass_low"] = with_scene(base, cup_mass=0.7 * real["cup_mass"])
    c["cup_mass_high"] = with_scene(base, cup_mass=1.3 * real["cup_mass"])
    c["cup_friction_low"] = with_scene(base, cup_friction=0.6 * real["cup_friction"])
    c["physics_real"] = with_scene(base, **{k: real[k] for k in variants.TWIN_SCALE})
    for mm in (0, 2.5, 7.5, 10, 15):
        c[f"noise_{mm}mm"] = with_noise(base, mm / 1000)
    c["wide_layout"] = variants.load_set("wide_layout_100")
    c["all_together"] = variants.load_set("test_real_100")
    return c, {"twin": twin, "real": real}


def programs(tag):
    out = {"reference_oracle": (HERE / "policies" / "oracle.py").read_text()}
    for run_dir in sorted(harness.RUNS_DIR.glob(f"{tag}_*_r*")):
        meta = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else None
        if run_dir.is_dir() and meta and meta["condition"] in ("twin", "family", "family_v2"):
            out[run_dir.name] = (run_dir / "final_policy.py").read_text()
    return out


def _job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    r = episode._worker(args)
    f, e = r.get("final") or {}, r.get("events") or {}
    return {"success": bool(r["success"]), "failure": r["failure"], "sim_time": r["sim_time"],
            "drawer_opening": f.get("drawer_opening"), "max_drawer_opening": e.get("max_drawer_opening")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="main")
    parser.add_argument("--workers", type=int, default=200)
    parser.add_argument("--only", nargs="*", help="run only these conditions (merged into the file)")
    parser.add_argument("--programs", help="run only programs whose run id contains this text "
                                           "(merged into the file)")
    args = parser.parse_args()
    conds, scenes = conditions()
    if args.only:
        conds = {k: v for k, v in conds.items() if k in args.only}
    all_progs = programs(args.tag)
    progs = {k: v for k, v in all_progs.items() if not args.programs or args.programs in k}
    out_path = HERE / "outputs" / args.tag / "probes.json"
    data = json.loads(out_path.read_text()) if out_path.exists() else {"results": {}}
    data.update({"programs": sorted(all_progs), "scenes": scenes})
    keys = [(c, p) for c in conds for p in progs]
    jobs = [(ep, progs[p], 0) for c, p in keys for ep in conds[c]]
    print(f"{len(conds)} conditions x {len(progs)} programs = {len(jobs)} episodes", flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        flat = list(pool.map(_job, jobs, chunksize=4))
    i = 0
    for c, p in keys:
        n = len(conds[c])
        data["results"].setdefault(c, {})[p] = flat[i:i + n]
        i += n
    out_path.write_text(json.dumps(data))
    print(f"done in {time.time() - started:.0f} s, saved {out_path}", flush=True)
    for c in conds:
        row = []
        for group in ("reference_oracle", "claude_twin_", "claude_family_r", "claude_family_v2_",
                      "codex_twin_", "codex_family_r", "codex_family_v2_"):
            rs = [r for p in all_progs if group in p for r in data["results"][c].get(p, [])]
            if rs:
                row.append(f"{group.rstrip('_r')} {100 * sum(r['success'] for r in rs) / len(rs):5.1f}")
        print(f"{c:26s} " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
