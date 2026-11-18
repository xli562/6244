"""Run one policy on one scene, print the judge's verdict and optionally render an MP4.

    python sim/demo_cup_drawer.py                                    # reference policy, real scene
    python sim/demo_cup_drawer.py --set test_real --index 3 --policy runs/<id>/final_policy.py
    python sim/demo_cup_drawer.py --out sim/outputs/demo.mp4

Headless rendering needs MUJOCO_GL=egl.
"""

import argparse
import json
from pathlib import Path

import episode
import variants

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", default=str(HERE / "policies/oracle.py"))
    parser.add_argument("--set", help="episode set name (see sim/variants.py); default: the real scene")
    parser.add_argument("--index", type=int, default=0, help="which episode of the set")
    parser.add_argument("--noise", type=float, default=None, help="override perception noise sigma (m)")
    parser.add_argument("--out", help="write a video of the episode to this path")
    args = parser.parse_args()
    if args.set:
        ep = variants.load_set(args.set)[args.index]
    else:
        ep = {"id": "real_scene", "scene": variants.real_scene().to_dict(), "noise_seed": 0,
              "noise_sigma": 0.0}
    if args.noise is not None:
        ep["noise_sigma"] = args.noise
    result = episode.run_episode(ep, Path(args.policy).read_text(), video=args.out)
    result.pop("trace")
    print(json.dumps(result, indent=1))
    if args.out:
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
