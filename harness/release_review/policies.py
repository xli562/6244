"""Generate one additional, task-blind decision module per frozen parent policy."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"harness/placement_stages"))
import runner as prior
base=prior.base
OUT=ROOT/"runs/release_01"
COMMON="""Define exactly this top-level entry in addition to the requested functions:
def run(api):
    return finish(api.last())
Use only plain dict .get, simple conditionals and numeric arithmetic. The sandbox forbids getattr, hasattr, imports, exceptions introspection, type annotations and private attributes. No other accessors.
Generate a single fenced Python module with function finish(receipt), returning None or
'regrasp' to request an additional physical retry at the end, only for a failed-looking placement with
enough steps left. It wraps an already frozen household robot program. Receipt may be None; fields
op, remaining, xy_gap, max_lift are safe numbers on placements. No imports, object names, instructions,
task exceptions, or model calls at runtime. Two shared corrections per episode, including any earlier
corrections. 460 steps for motion and observation, 40 reserved for final independent release verification.
All versions have identical physical primitives. Prefer no extra retry unless strongly justified."""
CHECK="""Also implement decide(event), returning 'continue', 'switch' or 'retry_release'. At each prior
grasp warning, 'continue' preserves the running skill, including its remaining alternative grasp offsets.
'switch' accepts the existing proposed intervention. event contains phase ('grasp','carry','release'),
reason, remaining, contacts (0-2 fingers), lift (m), object_motion (m), axis_change, misses (distinct
failed attempts in this skill, one per jaw closure), wide (bool, rim grip may be needed), local_options
(bool, another native offset available). On the FIRST failed lift continuing a skill that has another offset
is usually cheaper and safer than replacing it, particularly with no contacts or little movement.
Repeated empty lifts, a moving/tilting object or exhausted options can justify switching. Preserve
budget for the release. After opening and clearing the fingers, an event with phase 'settled_release'
reports actual loss with reason 'lost_support' or 'drifted', previous_support (bool), support (bool),
drift (metres relative to release), xy_gap and remaining. Use 'retry_release' to reacquire and seat
the item if observed loss with enough remaining steps. The controller supplies the original proposed
choice via default if needed. No evaluator signals."""
def main():
    def one(k):
        prompt=COMMON+("\n"+CHECK if k.endswith("/checked") else "")
        code=base.old.previous.generate(k.split("/")[0],OUT/k,prompt)
        return k,base.h.digest(code)
    with ThreadPoolExecutor(12) as ex:
        base.h.save_lock(OUT/"programs.json",dict(ex.map(one,base.h.read(prior.OUT/"programs.json"))))
if __name__=="__main__":main()
