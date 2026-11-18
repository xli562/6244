"""Generate and verify sets of scene variants (episode lists) for development and testing.

Scenes:
  real  - the reference scene from data/real_scene.json. Coding agents never see it.
  twin  - the real scene with a fixed reconstruction error (what a real-to-sim
          pipeline would hand over). All development scenes derive from it.

Sets (profiles):
  twin       - the twin itself, repeated with different perception-noise seeds
  family     - task-preserving variants of the twin: cup pose, drawer opening, clutter
  naive      - same sampling as `family` but with no checks at all (baseline)
  test_id    - fresh `family` samples
  test_real  - unseen configurations of the *real* scene: wider ranges, perturbed
               physics, larger perception noise

Every candidate goes through layered checks, cheapest first, and every
rejection is logged with its reason:
  1. layout   (analytic)  objects on the counter, no overlapping footprints,
                          drawer clutter inside the drawer, room left for the cup
  2. spawn    (1 forward) no interpenetration at the initial state
  3. settle   (0.5 s)     nothing moves or tips on its own
  4. oracle   (1 rollout) the reference policy solves it without disturbing clutter

    python sim/variants.py --profile family --n 16 --seed 1
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

import scene
from scene import Clutter, SceneConfig

SETS_DIR = Path(__file__).parent / "outputs/sets"
ORACLE = Path(__file__).parent / "policies/oracle.py"
NOISE_SIGMA = 0.005  # m, perception error (per-episode bias) on development scenes

# Reconstruction error of the twin relative to the real scene (fixed, see make_twin).
TWIN_ERROR = {
    "drawer_y": +0.012, "counter_top_z": -0.008, "counter_front_x": +0.007,
    "drawer_half_width": -0.004, "drawer_wall_height": +0.004,
    "handle_offset_x": +0.004, "cup_diameter": -0.003, "cup_height": +0.003,
}
TWIN_SCALE = {"drawer_frictionloss": 0.6, "drawer_damping": 1.3, "cup_mass": 0.85,
              "cup_friction": 1.2}

PROFILES = {
    "family": dict(
        base="twin", cup_x=(0.61, 0.76), cup_y=(-0.30, 0.36), cup_yaw=(-math.pi, math.pi),
        drawer_open=(0.0, 0.30), counter_clutter=(0, 3), drawer_clutter=(0, 1),
        noise_sigma=NOISE_SIGMA),
    "test_real": dict(
        base="real", cup_x=(0.60, 0.78), cup_y=(-0.34, 0.38), cup_yaw=(-math.pi, math.pi),
        drawer_open=(0.0, 0.34), counter_clutter=(0, 4), drawer_clutter=(0, 2),
        physics={"drawer_frictionloss": (0.5, 1.5), "drawer_damping": (0.5, 1.5),
                 "cup_mass": (0.7, 1.3), "cup_friction": (0.6, 1.2)},
        noise_sigma=1.5 * NOISE_SIGMA),
}
# Probe set: the twin with the wider layout ranges of test_real, nothing else changed.
PROFILES["wide_layout"] = {**PROFILES["test_real"], "base": "twin", "physics": {},
                           "noise_sigma": NOISE_SIGMA}
PROFILES["test_id"] = PROFILES["family"]
PROFILES["naive"] = PROFILES["family"]
# family_v2: the layouts of a `family` set, with the size of the perception error varied too.
# Every block of 8 scenes (one development run) gets each of these levels once.
NOISE_LEVELS_V2 = tuple(float(s) for s in np.linspace(0.5, 2.0, 8) * NOISE_SIGMA)


def real_scene():
    return scene.load_config(scene.REAL_SCENE)


def make_twin(real=None):
    real = real or real_scene()
    changes = {k: getattr(real, k) + v for k, v in TWIN_ERROR.items()}
    changes.update({k: getattr(real, k) * v for k, v in TWIN_SCALE.items()})
    return replace(real, **changes)


def base_scene(name):
    return real_scene() if name == "real" else make_twin()


# --- sampling ---------------------------------------------------------------
def _sample_clutter(rng, cfg, n_counter, n_drawer):
    items = []
    for i in range(n_counter):
        if rng.random() < 0.5:
            shape, size = "box", (rng.uniform(0.02, 0.04), rng.uniform(0.02, 0.04),
                                  rng.uniform(0.02, 0.06))
        else:
            shape, size = "cylinder", (rng.uniform(0.02, 0.035), rng.uniform(0.03, 0.06))
        xy = (rng.uniform(cfg.counter_front_x + 0.05, cfg.counter_front_x + 0.32),
              rng.uniform(-cfg.counter_half_y + 0.05, cfg.counter_half_y - 0.05))
        items.append(Clutter(name=f"c{i}", shape=shape, size=tuple(map(float, size)),
                             xy=tuple(map(float, xy)), yaw=float(rng.uniform(-1.5, 1.5)),
                             where="counter", mass=float(rng.uniform(0.1, 0.4))))
    hx, hy, _ = cfg.interior_half
    for i in range(n_drawer):
        size = (rng.uniform(0.02, 0.04), rng.uniform(0.02, 0.05), rng.uniform(0.012, 0.03))
        xy = (cfg.interior_center[0] + rng.uniform(-hx, hx), rng.uniform(-hy, hy))
        items.append(Clutter(name=f"d{i}", shape="box", size=tuple(map(float, size)),
                             xy=tuple(map(float, xy)), yaw=0.0, where="drawer",
                             mass=float(rng.uniform(0.1, 0.4))))
    return tuple(items)


def sample(rng, profile):
    p = PROFILES[profile]
    cfg = base_scene(p["base"])
    for key, (lo, hi) in p.get("physics", {}).items():
        cfg = replace(cfg, **{key: getattr(cfg, key) * float(rng.uniform(lo, hi))})
    cfg = replace(
        cfg,
        cup_xy=(float(rng.uniform(*p["cup_x"])), float(rng.uniform(*p["cup_y"]))),
        cup_yaw=float(rng.uniform(*p["cup_yaw"])),
        drawer_open=float(rng.uniform(*p["drawer_open"])),
    )
    n_counter = int(rng.integers(p["counter_clutter"][0], p["counter_clutter"][1] + 1))
    n_drawer = int(rng.integers(p["drawer_clutter"][0], p["drawer_clutter"][1] + 1))
    return replace(cfg, clutter=_sample_clutter(rng, cfg, n_counter, n_drawer))


# --- checks -----------------------------------------------------------------
def _footprint_radius(c):
    return math.hypot(c.size[0], c.size[1]) if c.shape == "box" else c.size[0]


def check_layout(cfg):
    """Analytic checks. Returns None if the layout is acceptable, else the reason."""
    r_cup = cfg.cup_diameter / 2 + 0.02  # cup body plus its handle
    x_lo, x_hi = cfg.counter_front_x, cfg.counter_front_x + cfg.counter_depth
    if not (x_lo + r_cup < cfg.cup_xy[0] < x_hi - r_cup
            and abs(cfg.cup_xy[1]) < cfg.counter_half_y - r_cup):
        return "layout:cup_off_counter"
    counter = [c for c in cfg.clutter if c.where == "counter"]
    drawer = [c for c in cfg.clutter if c.where == "drawer"]
    for i, c in enumerate(counter):
        r = _footprint_radius(c)
        if not (x_lo + r < c.xy[0] < x_hi - r and abs(c.xy[1]) < cfg.counter_half_y - r):
            return "layout:clutter_off_counter"
        if math.dist(c.xy, cfg.cup_xy) < r + r_cup + 0.01:
            return "layout:clutter_overlaps_cup"
        for other in counter[:i]:
            if math.dist(c.xy, other.xy) < r + _footprint_radius(other) + 0.01:
                return "layout:clutter_overlaps_clutter"
    hx, hy, hz = cfg.interior_half
    cx = cfg.interior_center[0]
    for i, c in enumerate(drawer):
        if (abs(c.xy[0] - cx) > hx - c.size[0] - 0.005 or abs(c.xy[1]) > hy - c.size[1] - 0.005):
            return "layout:drawer_clutter_outside_drawer"
        if 2 * c.size[2] > 2 * hz + 0.02:
            return "layout:drawer_clutter_too_tall"
        for other in drawer[:i]:
            if (abs(c.xy[0] - other.xy[0]) < c.size[0] + other.size[0] + 0.01
                    and abs(c.xy[1] - other.xy[1]) < c.size[1] + other.size[1] + 0.01):
                return "layout:clutter_overlaps_clutter"
    # Clearance that decides the task: the cup must fit under the counter when the drawer shuts,
    # and there must be a free cell for it in the part of the drawer that can be opened.
    cavity_clear = cfg.drawer_front_height + scene.CAVITY_GAP - 2 * cfg.drawer_wall
    if cfg.cup_height + 0.005 > cavity_clear:
        return "layout:cup_taller_than_drawer_cavity"
    need = cfg.cup_diameter / 2 + 0.05  # cup radius plus an open finger
    reach = min(cfg.drawer_max_open, 0.30)  # how far the robot can usefully open it
    free = False
    for x in np.linspace(cfg.drawer_panel + need, max(cfg.drawer_panel + need, reach - need), 6):
        for y in np.linspace(-hy + need, hy - need, 9):
            if all(abs(x - c.xy[0]) > c.size[0] + need or abs(y - c.xy[1]) > c.size[1] + need
                   for c in drawer):
                free = True
    if not free:
        return "layout:no_free_spot_in_drawer"
    return None


def _free_bodies(model):
    return [model.body(i).name for i in range(model.nbody)
            if model.body(i).name == "cup_model" or model.body(i).name.startswith("clutter_")]


def check_physics(cfg):
    """Spawn and settle checks on the compiled scene. Returns None or the reason."""
    try:
        model, data = scene.build_model(cfg)
    except Exception as e:  # noqa: BLE001
        return f"spawn:build_failed ({type(e).__name__})"
    robot_root = model.body("link0").id
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if c.dist < -0.002 and model.body_rootid[b1] != robot_root and model.body_rootid[b2] != robot_root:
            return "spawn:penetration"
    names = _free_bodies(model)
    start = {n: (data.body(n).xpos.copy(), data.body(n).xmat.reshape(3, 3)[2, 2]) for n in names}
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    for n in names:
        pos0, zz0 = start[n]
        if np.linalg.norm(data.body(n).xpos - pos0) > 0.01:
            return "settle:object_moved"
        if abs(data.body(n).xmat.reshape(3, 3)[2, 2] - zz0) > 0.02:
            return "settle:object_tipped"
    return None


def verify(cfg_dict):
    """Worker: physics checks, then the noise-free reference policy. Returns None or the reason."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import episode
    cfg = SceneConfig.from_dict(cfg_dict)
    reason = check_physics(cfg)
    if reason:
        return reason
    result = episode.run_episode({"scene": cfg_dict, "noise_sigma": 0.0}, ORACLE.read_text(),
                                 trace_limit=0)
    if not result["success"]:
        return f"oracle:{result['failure']}"
    if result["events"]["objects_moved_m"]:
        return "oracle:disturbed_clutter"
    return None


# --- set generation -----------------------------------------------------------
def generate(profile, n, seed, workers=48, log=print):
    """Returns {"episodes": [...], "rejections": {...}, ...} with n accepted episodes."""
    rng = np.random.default_rng(seed)
    noise = PROFILES[profile]["noise_sigma"] if profile in PROFILES else NOISE_SIGMA
    episodes, rejections, tried = [], {}, 0

    def add(cfg):
        i = len(episodes)
        episodes.append({"id": f"{profile}_{seed}_{i:03d}", "scene": cfg.to_dict(),
                         "noise_seed": int(seed * 100003 + i), "noise_sigma": noise})

    if profile == "twin":
        twin = make_twin()
        for _ in range(n):
            add(twin)
        tried = n
    elif profile == "naive":
        for _ in range(n):
            add(sample(rng, profile))
        tried = n
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while len(episodes) < n and tried < 40 * n + 200:
                batch = [sample(rng, profile) for _ in range(max(16, 2 * (n - len(episodes))))]
                tried += len(batch)
                passed = []
                for cfg in batch:
                    reason = check_layout(cfg)
                    if reason:
                        rejections[reason] = rejections.get(reason, 0) + 1
                    else:
                        passed.append(cfg)
                for cfg, reason in zip(passed, pool.map(verify, [c.to_dict() for c in passed])):
                    if reason:
                        rejections[reason] = rejections.get(reason, 0) + 1
                    elif len(episodes) < n:
                        add(cfg)
                log(f"  {profile}: {len(episodes)}/{n} accepted after {tried} candidates")
    return {"profile": profile, "seed": seed, "candidates": tried, "accepted": len(episodes),
            "rejections": dict(sorted(rejections.items(), key=lambda kv: -kv[1])),
            "episodes": episodes}


def vary_noise(source, seed, levels=NOISE_LEVELS_V2):
    """A copy of the set `source` in which the perception error differs from scene to scene:
    every block of len(levels) scenes gets each level once, in random order."""
    rng = np.random.default_rng(seed)
    episodes = []
    for i, ep in enumerate(load_set(source)):
        if i % len(levels) == 0:
            order = rng.permutation(len(levels))
        episodes.append({**ep, "id": f"family_v2_{seed}_{i:03d}", "layout_of": ep["id"],
                         "noise_sigma": levels[order[i % len(levels)]], "noise_shown": True})
    return {"profile": "family_v2", "seed": seed, "source": source, "noise_levels": list(levels),
            "candidates": len(episodes), "accepted": len(episodes), "rejections": {},
            "episodes": episodes}


def save_set(result, name=None):
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    path = SETS_DIR / f"{name or result['profile']}.json"
    path.write_text(json.dumps(result, indent=1))
    return path


def load_set(name):
    path = Path(name) if str(name).endswith(".json") else SETS_DIR / f"{name}.json"
    return json.loads(path.read_text())["episodes"]


def snapshot_grid(episodes, path, cols=4, size=(240, 320)):
    import imageio
    tiles = []
    for ep in episodes:
        model, data = scene.build_model(SceneConfig.from_dict(ep["scene"]))
        with mujoco.Renderer(model, *size) as renderer:
            renderer.update_scene(data, camera="overview")
            tiles.append(renderer.render())
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    imageio.imwrite(path, np.concatenate(rows, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True,
                        choices=["twin", "family", "naive", "test_id", "test_real", "wide_layout",
                                 "family_v2"])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="file name of the set (default: the profile name)")
    parser.add_argument("--source", default="family_64",
                        help="family_v2 only: the set whose layouts are reused")
    parser.add_argument("--grid", action="store_true", help="also save a picture of the first 8 scenes")
    args = parser.parse_args()
    if args.profile == "family_v2":
        result = vary_noise(args.source, args.seed)
    else:
        result = generate(args.profile, args.n, args.seed)
    path = save_set(result, args.name)
    print(f"saved {path}: {result['accepted']} episodes from {result['candidates']} candidates")
    for reason, count in result["rejections"].items():
        print(f"  rejected {count:4d}  {reason}")
    if args.grid:
        grid = path.with_suffix(".png")
        snapshot_grid(result["episodes"][:8], grid)
        print(f"saved {grid}")


if __name__ == "__main__":
    main()
