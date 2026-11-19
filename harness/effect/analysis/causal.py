"""Privileged, offline replay only. Verify the relevant native goal changed on a corrective action."""
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment_runner as b
from feedback import EffectLibrary

def rows(stem):
    with gzip.open(b.OUT/(stem+".jsonl.gz"),"rt") as f:
        yield from map(json.loads,f)
def key(r):
    return (r["key"],r["task"],r["i"])
def first(r):
    return next((a for a in r["adjustments"] if a["source"]=="observation"),None)

def judged(job):
    src,old,repeat=job
    before_trigger=first(src)
    i=before_trigger["event"]
    traces=[]
    orig=EffectLibrary.perform
    def inspect(lib,strategy):
        env=lib._sensors._env.env
        goals=env.parsed_problem["goal_state"]
        op=lib.current[0]["op"]
        bound=lib.current[1]
        def owner(n):
            return (lib.scene.get(n,{}).get("parent") or n)
        relevant=[]
        for g in goals:
            if op=="place" and len(g)==3 and g[0] in ("in","on"):
                if g[1]==bound["item"] and (g[2]==bound["target"] or g[2]==owner(bound["target"]) or owner(g[2])==bound["target"]):
                    relevant.append(g)
            if len(g)==2 and g[0] in ("turnoff","turnon","open","close") and owner(g[1])==owner(bound["target"]):
                relevant.append(g)
        def truth():
            return [bool(env._eval_predicate(g)) for g in relevant]
        before=truth()
        try:
            return orig(lib,strategy)
        finally:
            traces.append(dict(goals=relevant,before=before,after=truth()))
    with patch.object(EffectLibrary,"perform",inspect):
        rr=b.play((src["key"],"new",src["task"],[src["i"]]))[0]
    assert len(traces)==len(rr["measurements"])
    matched=(rr["stable"]==src["stable"] and rr["measurements"]==src["measurements"] and rr["adjustments"]==src["adjustments"])
    matched_old=old["measurements"][:i+1]==src["measurements"][:i+1]
    matched_repeat=(repeat["measurements"][:i+1]==src["measurements"][:i+1] and first(repeat) is not None and first(repeat)["event"]==i)
    corrected=False
    if matched:
        for a in rr["adjustments"]:
            if a["source"]!="observation": continue
            n=a["event"]
            failed,retry=traces[n:n+2]
            meas=rr["measurements"][n]
            corrected |= (meas["reason"] is not None and meas["at"]>meas["remaining"] and meas["hand_motion"]>=.01 and
                          any(g in retry["goals"] and not failed["after"][j] and
                              not retry["before"][retry["goals"].index(g)] and retry["after"][retry["goals"].index(g)]
                              for j,g in enumerate(failed["goals"])))
    return dict(key=src["key"],task=src["task"],i=src["i"],matched=matched,matched_old=matched_old,
                matched_repeat=matched_repeat,corrected=corrected,first_reason=src["measurements"][i]["reason"],
                old_stable=old["stable"],repeat_stable=repeat["stable"],observations=traces)

def main(stem):
    if stem=="dev": b.old.register(b.h.read(b.BASE/"manifest.json"))
    else:
        import heldout
        heldout.verify()
    data=list(rows(stem))
    old={key(r):r for r in data if r["arm"]=="old"}
    repeat={key(r):r for r in rows("dev_repeat" if stem=="dev" else "repeat")}
    jobs=[(r,old[key(r)],repeat[key(r)]) for r in data if r["arm"]=="new" and r["stable"] and first(r)]
    dest=b.OUT/(stem+"_causal.jsonl.gz")
    if dest.exists():raise FileExistsError(dest)
    tmp=dest.with_suffix(".tmp")
    with ProcessPoolExecutor(36) as p,gzip.open(tmp,"wt") as f:
        for row in p.map(judged,jobs):
            f.write(json.dumps(row)+"\n")
    tmp.replace(dest)
    print(stem,len(jobs))
if __name__=="__main__":
    main(sys.argv[1])
