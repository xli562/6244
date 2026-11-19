"""One additional CLI compilation for each inherited planner and repetition."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"harness/effect"))
import experiment_runner as base
OUT=ROOT/"runs/stages_01"
PROMPT="""Return a single fenced Python module for a simulated robot. No imports, private attributes,
object names or task-specific exceptions. Implement run(api): call api.original() to run your frozen prior,
then optionally use api.last() (None or dict op,max_lift,xy_gap,remaining) and api.adjust(strategy) to
improve a missed placement. Strategies 'regrasp' (use observed collision geometry to acquire the item)
and 'align' (correct its live pose relative to the destination); up to two shared adjustments per episode,
all subject to 500 robot control steps. Returning is allowed. Do not implement your own motor primitives."""
HOOK="""Implement checkpoint(event) returning 'regrasp', 'align' or None. It is called for a measured fault
during a placement, with phase 'grasp','carry','release', reason 'empty_grasp','slip','off_target' or 'missed_release',
remaining integer steps, held Boolean, lift, xy_gap, drift (metres), previous (list of prior adjustments).
Recover immediately on an empty/slipped grasp if enough steps for a new acquisition; if held and off target
realign without dropping; after missed release reacquire. Preserve budget and avoid repeated futile corrections.
Some deployments only deliver the identical decision when the original placement returns. No model calls at runtime."""
def main():
    def compile_one(k):
        prompt=PROMPT+("\n"+HOOK if k.endswith("/checked") else "")
        code=base.old.previous.generate(k.split("/")[0],OUT/k,prompt)
        return k,base.h.digest(code)
    with ThreadPoolExecutor(12) as executor:
        base.h.save_lock(OUT/"programs.json",dict(executor.map(compile_one,base.h.read(base.OUT/"programs.json"))))
if __name__=="__main__":main()
