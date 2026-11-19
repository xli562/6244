"""One independent CLI call per universal policy, shared by stop and retry arms."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness/preflight"))
import experiment as old
from second_batch import COMPACT

OUT = ROOT / "runs/reselection_01"
SAVED = old.OUT / "second_batch"
REPS = range(1, 4)
EXTRA = """The harness can ask your policy to reread the ORIGINAL instruction when a role has tied candidates.
In the same block, add a small separate top-level function reread(instruction, step, role) that returns
a contiguous VERBATIM substring identifying ONLY the requested role (source item or destination target)
for the current operation, or '' if uncertain. Preserve adjectives and spatial relations. It receives
the current step dictionary. No extra tools. The verifier independently checks this quote's scope,
the observed entity labels and positions. Some deployments stop on ties or disable checks. Your
main run(api) must work in each. Do not rely on serial IDs or hardcode any task or instruction."""
def generate(key):
    agent, rep, family = key.split("/")
    prompt = (SAVED / key / "prompt.txt").read_text()
    if family == "checked":
        prompt += "\n" + EXTRA
    source = old.previous.generate(agent, OUT / key, prompt)
    return key, old.h.digest(source)

if __name__ == "__main__":
    keys = [f"{a}/{r}/{f}" for r in REPS for a in old.h.AGENTS for f in ("receipt","checked")]
    with ThreadPoolExecutor(12) as pool:
        old.h.save_lock(OUT / "programs.json", dict(pool.map(generate, keys)))
