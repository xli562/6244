"""Compile feedback policies; each comparable arm inherits an equally budgeted frozen baseline."""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"harness/affordance"))
import bench as previous
from build import old
OUT=ROOT/"runs/effect_01"
BASE=previous.OUT
GUIDE="""Write a compact general robot policy module in a single fenced Python block. No imports, tools,
private attributes, task names, object-specific branches or extra model calls. Use ONLY these methods.
run(api) first calls api.original(), running your prior universal planner, and may then use api.last() (None or
a dictionary with op, steps_used, remaining, max_lift, xy_gap), api.instruction and api.adjust(strategy).
If the prior physically finishes an episode, control ends immediately. Up to two adjustments total, with
all normal controls and corrections sharing 500 control steps. Strategies (same actuators for every arm):
'joint_low','joint_high' grasp/sweep the accessible moving link toward a joint limit using its geometry;
'grasp_next' try a different grasp on the same bound item and destination; 'recenter' realign then place;
'repeat' reuse the original physical action. No other calls. Return if evidence or budget is insufficient.
Use only ordinary action receipts in run. Be concise; do not copy a large planner. Use instruction-level
operation semantics, never full instructions or object names. Baseline is available by calling original()."""
AFTER="""Also implement after(api,event) -> strategy string or None. New deployment calls it immediately after
an executed action when measured scene evidence indicates a failed intended effect, before the prior planner
continues. Old deployment shares your exact run and never invokes after. event contains op, instruction, reason
('no_effect','wrong_direction','shortfall'), desired ('low','high','at_target'), remaining, previous_strategies,
before and after (dicts containing joint_fraction for an articulated role or item_xyz and gap for placement),
hand_motion in metres. No task score. Use this measured residual to adjust the actual action. Do not call
api.adjust inside after; return a strategy. For articulated roles prefer the desired joint endpoint; placement
can vary grasp when the item stayed put, or realign if the item moved but missed the destination. Up to two retries."""
def one(k):
    a,_,f=k.split("/")
    src=old.previous.generate(a,OUT/k,GUIDE+("\n"+AFTER if f=="checked" else " You use action receipts only; implement run(api)."))
    return k,old.h.digest(src)
if __name__=="__main__":
    keys=previous.programs().keys()
    with ThreadPoolExecutor(12) as ex:
        old.h.save_lock(OUT/"programs.json",dict(ex.map(one,keys)))
