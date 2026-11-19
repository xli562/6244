"""Compile reusable actions once per independent inherited policy."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"harness/release_review"))
import run_review as previous
base=previous.base
OUT=ROOT/"runs/preview_01"
COMMON="""Return one fenced Python module. Mandatory entry:
def run(api):
    return finish(api.last())
Implement finish(receipt) returning 'regrasp' or None. It is one last opportunity after an inherited
placement program; only correct a substantial miss with ample budget. Receipt is None or a dict with
op, xy_gap, max_lift, remaining, retries_left. Use .get and normal numbers, no imports, getattr, private
attributes or object names. 460 control steps and separate 40 final verification, two actual corrections.
Reusing the old program is fine. No additional model calls or scene-specific rules."""
CHECK="""Also implement choose(event), returning 'switch', 'continue', or 'inherit'. It is invoked on a failed
lift before deciding whether to try the actual next local candidate. The sensor tool provides:
available(bool), reach_error(m, measured previous descent tracking error), width(m), aperture(m),
edge_access(bool, whether jaws can straddle a cross-section or pinch a reachable edge at that waypoint),
displaced(m, change of proposed contact from last), landing_margin(m, usable horizontal placement
tolerance), uncertainty(m, estimated lateral uncertainty after a weak grasp), retries_left.
Use 'switch' to use the geometry regrasp immediately for a poor next candidate when a precise landing
is needed (no edge_access and landing_margin <= uncertainty). Preserve the working native grip search
with 'continue' when the receiver can absorb that error and a distinct next point is available.
If evidence is missing, or this isn't a failed grasp, inherit the previous decision. No names or task IDs.
Also implement release(event): return 'retry_release' on observed loss after opening when retries_left>0
and remaining>=retry_cost, otherwise None. loss, previous_support, support, drift, remaining and retry_cost
are observations from the physical controller. Use only the public ledger; no global retry counters."""
def main():
    def compile_one(k):
        src=base.old.previous.generate(k.split("/")[0],OUT/k,COMMON+("\n"+CHECK if k.endswith("checked") else ""))
        return k,base.h.digest(src)
    with ThreadPoolExecutor(12) as ex:
        base.h.save_lock(OUT/"programs.json",dict(ex.map(compile_one,base.h.read(previous.OUT/"programs.json"))))
if __name__=="__main__":main()
