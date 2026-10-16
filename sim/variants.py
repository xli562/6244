"""Generate a family of scene variants as standalone MJCF files and verify each one.

Varies cup position and yaw on the counter and the drawer's initial opening.
Each variant is exported to <out>/variant_NNN.xml (assets shared in <out>/assets/),
then the scripted oracle is run on the exported file to check it is solvable.

    python sim/variants.py --n 20 --seed 0
    python sim/demo_cup_drawer.py --xml sim/outputs/family/variant_000.xml
"""

import argparse
import json
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import imageio
import mujoco
import numpy as np

import demo_cup_drawer
import scene
from scene import SceneConfig

RANGES = {
    "cup_x": (0.60, 0.75),
    "cup_y": (0.05, 0.30),
    "cup_yaw": (-0.5, 0.5),
    "drawer_open": (0.0, 0.08),
}


def sample(rng, base=SceneConfig()):
    u = {k: rng.uniform(*r) for k, r in RANGES.items()}
    return replace(base, cup_xy=(u["cup_x"], u["cup_y"]), cup_yaw=u["cup_yaw"],
                   drawer_open=u["drawer_open"])


def verify(xml_path):
    model, data = scene.load_model(xml_path)
    return demo_cup_drawer.run(model, data, video=None, log=lambda *_: None)


def snapshot_grid(xml_paths, path, cols=4, size=(240, 320)):
    tiles = []
    for xml_path in xml_paths:
        model, data = scene.load_model(xml_path)
        with mujoco.Renderer(model, *size) as renderer:
            renderer.update_scene(data, camera="overview")
            tiles.append(renderer.render())
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    imageio.imwrite(path, np.concatenate(rows, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(Path(__file__).parent / "outputs/family"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cfgs = [sample(rng) for _ in range(args.n)]
    xml_paths = [out / f"variant_{i:03d}.xml" for i in range(args.n)]
    for cfg, xml_path in zip(cfgs, xml_paths):
        scene.export_xml(scene.build_spec(cfg), xml_path, out / "assets")

    with Pool(min(args.n, 16)) as pool:
        results = pool.map(verify, [str(p) for p in xml_paths])

    manifest = [{"file": p.name, "config": asdict(c), "oracle": r}
                for p, c, r in zip(xml_paths, cfgs, results)]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    snapshot_grid(xml_paths[:8], out / "grid.png")

    n_ok = sum(r["success"] for r in results)
    print(f"exported {len(xml_paths)} variants to {out}")
    print(f"oracle success on exported files: {n_ok}/{len(results)}")
    for p, c, r in zip(xml_paths, cfgs, results):
        if not r["success"]:
            print(f"  {p.name} failed: cup_xy=({c.cup_xy[0]:.3f}, {c.cup_xy[1]:.3f}) "
                  f"yaw={c.cup_yaw:.2f} drawer_open={c.drawer_open:.3f} -> {r}")


if __name__ == "__main__":
    main()
