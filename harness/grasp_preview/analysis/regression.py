"""On already-used data check the missing-ledger regression and a known unstable release."""
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import benchmark as bench
from causal import audit
from score import main as score
from reserve_replay import main as reserve_replay
def run():
    bench.freeze()
    # No changes are made to the generated policy. Isolate the failed counter read.
    with gzip.open(bench.previous.OUT/"test.jsonl.gz","rt") as f:
        candidates=[(r,e) for r in map(json.loads,f) if r["arm"]=="new" for e in r["releases"]
                    if e["loss"] and e["remaining"]==133]
    row,event=candidates[0]
    ns={}
    exec(bench.previous.code(row["key"]),ns)
    lib=bench.PreviewLibrary.__new__(bench.PreviewLibrary)
    lib.adjustments=[{}]*event["corrections_used"]
    before=ns["decide"](dict(event))
    after=ns["decide"](lib.ledger(event))
    assert (before,after)==("continue","retry_release")
    (bench.OUT/"counter_replay.json").write_text(json.dumps(dict(before=before,after=after,ledger=lib.ledger(event)),indent=1)+"\n")
    reserve_replay("archived_alerts")
    tasks=["stage_0375814487c77a"]
    bench.deploy("ketchup",tasks,36,list(range(50)))
    audit("ketchup")
    score("ketchup",tasks)
if __name__=="__main__":run()
