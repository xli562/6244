"""Independent physical replay, including a no-release-intervention counterfactual."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_review as runner
from controller import ReviewLibrary

def records(stem):
    with gzip.open(runner.OUT/(stem+".jsonl.gz"),"rt") as f:return list(map(json.loads,f))

def key(r):return r["key"],r["task"],r["i"]

def replay(ref):
    frames=[]
    def watch(lib,tag):
        if tag=="step" and not lib.opening:return
        env=lib._sensors._env.env
        ctx=lib.approach
        i,t=ctx["item"],ctx["target"]
        goals=[g for g in env.parsed_problem["goal_state"] if len(g)==3 and g[0] in ("on","in") and g[1]==i and
               (g[2]==t or lib.scene.get(g[2],{}).get("parent")==t or lib.scene.get(t,{}).get("parent")==g[2])]
        truth=any(env._eval_predicate(g) for g in goals)
        if tag=="open":
            frames.append(dict(item=i,target=t,goals=goals,at=lib.remaining(),before=truth,seen=truth,
                               after=None,unchanged_state=None,acted=False,corrected=False))
        elif tag=="step" and frames:frames[-1]["seen"]|=truth
        elif tag=="settled":
            frames[-1].update(after=truth,at_settle=lib.remaining(),support_after=lib.touch(),
                   relative=(lib.pos(i)-lib.pos(t)).round(5).tolist(),
                   unchanged_state=hashlib.sha256(np.asarray(lib._sensors._env.get_sim_state(),dtype="<f8").tobytes()).hexdigest())
        elif tag=="retry":frames[-1]["acted"]=True
        elif tag=="retried":
            for e in reversed(frames):
                if e["acted"] and e["item"]==i and e["target"]==t:
                    e["corrected"]=truth
                    break
    configure=ReviewLibrary.configure
    def configure_and_watch(lib,*args):
        configure(lib,*args)
        lib.audit_hook=watch
    k,t,i=key(ref)
    with patch.object(ReviewLibrary,"configure",configure_and_watch):
        rr=runner.play((k,ref["arm"],t,[i]))[0]
    identical=all(ref[x]==rr[x] for x in ("decisions","releases","adjustments","error","stable","steps"))
    return frames,identical

def one(job):
    new,no,old=job
    frames,same=replay(new)
    other,same_control=replay(no)
    first=next((j for j,e in enumerate(frames) if e["acted"]),None)
    event=frames[first] if first is not None else {}
    ctrl=other[first] if first is not None and first<len(other) else {}
    matched=(bool(event.get("unchanged_state")) and event["unchanged_state"]==ctrl.get("unchanged_state")
             and event["at_settle"]==ctrl.get("at_settle") and ctrl.get("seen")==event["seen"] and not ctrl["after"])
    proven=bool(same and same_control and matched and event["seen"] and not event["after"] and event["corrected"]
                and not new["error"] and not no["error"])
    return dict(key=new["key"],task=new["task"],i=new["i"],same=same and same_control,matched=matched,
                proven=proven,also_old_failed=not old["stable"] and old["error"] is None,episode=event)

def main(stem):
    rows=records(stem)
    groups={mode:{key(r):r for r in rows if r["arm"]==mode} for mode in ("old","new","no_release")}
    jobs=[]
    for k,n in groups["new"].items():
        no=groups["no_release"][k]
        if n["stable"] and not no["stable"] and any(e.get("acted") for e in n["releases"]):
            jobs.append((n,no,groups["old"][k]))
    with ProcessPoolExecutor(30) as ex:
        audits=list(ex.map(one,jobs))
    with gzip.open(runner.OUT/(stem+"_audit.jsonl.gz"),"wt") as f:
        for r in audits:f.write(json.dumps(r,default=runner.base.old.json_value)+"\n")
    print(stem,len(jobs),sum(r["proven"] for r in audits))
if __name__=="__main__":main(sys.argv[1])
