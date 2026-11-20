"""Record a frozen policy on one LIBERO task to MP4 (agentview | frontview)."""
import argparse, os, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "third_party/LIBERO")); sys.path.insert(0, str(R / "sim"))
os.environ["LIBERO_CONFIG_PATH"] = str(R / "third_party/libero-config")
import imageio, episode, libero_long as L
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

p = argparse.ArgumentParser()
p.add_argument("--task", default="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it")
p.add_argument("--state", type=int, default=30)
p.add_argument("--policy", type=Path, required=True)
p.add_argument("--out", type=Path, default=R / "build/renders/task.mp4")
p.add_argument("--size", type=int, default=384)
p.add_argument("--every", type=int, default=2, help="keep every Nth control step")
a = p.parse_args()

cams = ["agentview", "frontview"]
env = OffScreenRenderEnv(bddl_file_name=str(L.paths(a.task)[0]), camera_heights=a.size,
                         camera_widths=a.size, camera_names=cams, horizon=L.HORIZON + 5)
frames, n = [], [0]
step = env.step
def recording_step(action):
    obs, *rest = step(action)
    n[0] += 1
    if n[0] % a.every == 0:
        frames.append(np.concatenate([obs[f"{c}_image"][::-1] for c in cams], 1))
    return (obs, *rest)
env.step = recording_step

np.random.seed(a.state)
env.reset()
env.set_init_state(np.load(L.CONFIG / (a.task + ".npy"))[a.state])
api = L.Skills(env)
for _ in range(5):
    env.step([0, 0, 0, 0, 0, 0, -1])
source = a.policy.read_text()
episode.check_policy_source(source)
try:
    episode.load_policy(source)(api)
except L.EpisodeEnd:
    pass
for _ in range(20):  # short tail so the final state is visible
    env.step([0, 0, 0, 0, 0, 0, -1])
ok = bool(env.check_success())
a.out.parent.mkdir(parents=True, exist_ok=True)
imageio.mimwrite(a.out, frames, fps=30 // a.every * 2, quality=8, macro_block_size=8)
print(f"{api.instruction!r} success={ok} steps={api._steps} frames={len(frames)} -> {a.out}")
