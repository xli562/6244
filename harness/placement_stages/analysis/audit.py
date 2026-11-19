"""Offline scoring. The model and motor controller never receive predicate truth."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
from statistics import median
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runner as run
from checkpoints import StageLibrary

def read(stem):
    with gzip.open(run.OUT/(stem+".jsonl.gz"),"rt") as f:
        return list(map(json.loads,f))

def episode_key(r):
    return r["key"],r["task"],r["i"]

def check(job):
    early,old,late=job
    events=[]
    original=StageLibrary.recover
    def probe(lib,strategy,source,warning=None):
        env=lib._sensors._env.env
        b=lib.current[1]
        def owner(n):
            return lib.scene.get(n,{}).get("parent") or n
        goals=[g for g in env.parsed_problem["goal_state"] if len(g)==3 and g[0] in ("on","in")
               and g[1]==b.get("item") and owner(g[2])==owner(b.get("target"))]
        def truth():
            return [bool(env._eval_predicate(g)) for g in goals]
        before=truth()
        try:
            return original(lib,strategy,source,warning)
        finally:
            events.append(dict(source=source,goals=goals,before=before,after=truth(),remaining=lib.remaining()))
    k,t,i=episode_key(early)
    with patch.object(StageLibrary,"recover",probe):
        repeat=run.play((k,"early",t,[i]))[0]
    deterministic=all(repeat[field]==early[field] for field in ("stable","steps","requests","adjustments","repairs","error","checks"))
    req=early["requests"][0]
    n=req["check"]+1
    matched=late["requests"][:1]==[req] and early["checks"][:n]==old["checks"][:n]==late["checks"][:n]
    er=early["repairs"][0]
    delayed=next((a for a in late["repairs"] if a["source"]=="stage"),None)
    later_budget=delayed["at"] if delayed else None
    timing=(er["source"]=="stage" and er["at"]==req["event"]["remaining"] and
            (later_budget is None or later_budget<er["at"]))
    first=next((e for e in events if e["source"]=="stage"),None)
    corrected=first is not None and any(not a and b for a,b in zip(first["before"],first["after"]))
    return dict(key=k,task=t,i=i,matched=matched,deterministic=deterministic,
                reason=req["event"]["reason"],phase=req["event"]["phase"],early_remaining=er["at"],
                late_remaining=later_budget,late_terminal_steps=late["steps"],timing=timing,goals=events,
                control_errors=[old["error"],late["error"]],
                credited=bool(matched and deterministic and timing and corrected and not any(e["error"] for e in (early,old,late))))

def main(stem,workers):
    run.freeze()
    if stem=="test":
        from suite import verify
        verify()
    if stem=="challenge":
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"challenge"))
        from relations import verify
        verify()
    rows=read(stem)
    by_arm=defaultdict(dict)
    for r in rows:
        assert 0<=r["steps"]<=500 and len(r["adjustments"])<=2
        assert episode_key(r) not in by_arm[r["arm"]]
        by_arm[r["arm"]][episode_key(r)]=r
    expected=50*3*2*(len(run.DEV) if stem=="dev" else len(verify()))
    assert set(by_arm)=={"old","early","late","receipt"} and all(len(v)==expected for v in by_arm.values())
    jobs=[]
    for key,e in by_arm["early"].items():
        old,late=by_arm["old"][key],by_arm["late"][key]
        if e["stable"] and not old["stable"] and not late["stable"] and e["requests"] and e["repairs"]:
            jobs.append((e,old,late))
    with ProcessPoolExecutor(workers) as p:
        audited=list(p.map(check,jobs,chunksize=2))
    with gzip.open(run.OUT/(stem+"_audit.jsonl.gz"),"wt") as f:
        for r in audited:f.write(json.dumps(r)+"\n")
    credited={episode_key(r) for r in audited if r["credited"]}
    scored=defaultdict(lambda:defaultdict(int))
    for key,e in by_arm["early"].items():
        k,t,i=key
        model,rep,_=k.split("/")
        o=by_arm["old"][key]
        l=by_arm["late"][key]
        receipt=by_arm["receipt"][(model+"/"+rep+"/receipt",t,i)]
        for group in (t,"ALL"):
            scores=scored[(model,group)]
            scores["n"]+=1
            for arm,r in (("early",e),("old",o),("late",l),("receipt",receipt)):
                scores[arm]+=r["stable"]
                scores[arm+"_errors"]+=bool(r["error"])
            scores["gain"]+=e["stable"] and not o["stable"]
            scores["loss"]+=o["stable"] and not e["stable"]
            scores["gain_also_late"]+=e["stable"] and not o["stable"] and l["stable"]
            scores["credited"]+=key in credited
            scores["alerted"]+=bool(e["checks"])
            scores["used_stage"]+=any(a["source"]=="stage" for a in e["repairs"])
    with (run.OUT/(stem+"_scores.csv")).open("w") as f:
        writer=csv.DictWriter(f,fieldnames=["model","task",*next(iter(scored.values()))])
        writer.writeheader()
        for (model,task),score in sorted(scored.items()):
            writer.writerow(dict(model=model,task=task,**score))
    print(stem,len(jobs),len(credited),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stem",choices=("dev","test","challenge"))
    parser.add_argument("--workers",type=int,default=24)
    args=parser.parse_args()
    main(args.stem,args.workers)
