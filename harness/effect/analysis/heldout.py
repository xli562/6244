"""Final text-vs-goal audit before test deployment: honor implicit or unscored subgoals."""
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment_runner as b
from holdout import verify,sha

def main(stage):
    b.freeze()
    verify()
    m=b.h.read(b.OUT/"manifest.json")
    kept=[]
    removed={}
    for row in m["tasks"]:
        words=row["instruction"]
        stated=set(re.findall(r"\b(open|close|turn on|turn off)\b",words))
        scored={a[0] for a in row["atoms"]}
        missing={v for v in stated if v.replace(" ","") not in scored}
        if missing:
            removed[row["task"]]=dict(instruction=words,unscored_actions=sorted(missing))
        else:
            kept.append(row["task"])
    b.h.save_lock(b.OUT/"text_audit.json",dict(kept=kept,removed=removed,manifest_sha=sha((b.OUT/"manifest.json").read_bytes()),
                                              screen_sha=sha(Path(__file__).read_bytes())))
    if stage=="test":
        jobs=[(k,a,t,list(range(i,i+5))) for k in b.h.read(b.OUT/"programs.json") for t in kept
              for a in (("receipt",) if k.endswith("/receipt") else ("old","new")) for i in range(0,50,5)]
    elif stage=="repeat":
        with gzip.open(b.OUT/"test.jsonl.gz","rt") as z:
            jobs=[(r["key"],"repeat",r["task"],[r["i"]]) for r in map(json.loads,z)
                  if r["arm"]=="new" and r["stable"] and any(a["source"]=="observation" for a in r["adjustments"])]
    else:
        raise ValueError(stage)
    b.deploy(jobs,stage,48)
if __name__=="__main__":
    main(sys.argv[1])
