import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
import gzip
import json
from pathlib import Path
from unittest.mock import patch

from policies import ROOT,OUT,prior,base
from controller import ReviewLibrary
DEV=("stage_06de888025d376","stage_0375814487c77a")

@lru_cache(None)
def code(k):
    return compile((OUT/k/"proposal.py").read_text(),"review-policy","exec")

def play(job):
    k,mode,t,ids=job
    ancestor,planner,original=prior.OLD_SOURCES(k)
    libs=[]
    def factory(*args):
        lib=ReviewLibrary(*args)
        libs.append(lib)
        return lib
    def entry(ctrl):
        parent={}
        ns={}
        exec(prior.compiled(k),parent)
        exec(code(k),ns)
        lib=libs[-1]
        lib.parent_handler=parent.get("checkpoint")
        lib.policy=ns.get("decide")
        lib.handler=lib.arbitrate
        parent["run"](prior.Overlay(ctrl,ancestor["run"]))
        if ctrl.last() and ctrl.last()["op"]=="place" and ns["finish"](ctrl.last())=="regrasp":
            ctrl.adjust("regrasp")
    with ExitStack() as s:
        s.enter_context(patch.object(base,"EffectLibrary",factory))
        s.enter_context(patch.object(base,"sources",lambda _:({**ancestor,"run":entry},planner,original)))
        s.enter_context(patch.object(base,"OUT",prior.OUT if t in DEV else OUT))
        rows=base.play(job)
    for r,l in zip(rows,libs):
        r.update(decisions=l.decisions,releases=l.releases,repairs=l.repairs,height_fixes=l.height_fixes)
    return rows

def freeze():
    prior.freeze()
    files=[*Path(__file__).parent.glob("*.py"),prior.OUT/"freeze.json",prior.OUT/"challenge_design.json",OUT/"programs.json"]
    keys=base.h.read(OUT/"programs.json")
    for k,digest in keys.items():
        p=OUT/k/"proposal.py"
        assert base.digest(p)==digest and len((OUT/k/"calls.jsonl").read_text().splitlines())==1
        assert not (OUT/k/"rejection.json").exists()
        files.extend((p,OUT/k/"calls.jsonl",OUT/k/"prompt.txt"))
    base.h.save_lock(OUT/"freeze.json",dict(files={str(p.relative_to(ROOT)):base.digest(p) for p in files},
        budget=[460,40],repetitions=3,attribution="same new program succeeds, no-release replay and old fail. Observe target native relation true near opening, false after open/clear/settle, true again after physical retry; stable terminal result; no control errors."))

def deploy(stage,tasks,workers,indices):
    keys=base.h.read(OUT/"programs.json")
    jobs=[(k,m,t,indices[i:i+5]) for k in keys for t in tasks for m in
          (("receipt",) if k.endswith("receipt") else ("old","new","no_release","literal","continue"))
          for i in range(0,len(indices),5)]
    dest=OUT/(stage+".jsonl.gz")
    assert not dest.exists()
    tmp=dest.with_suffix(".tmp")
    with ProcessPoolExecutor(workers) as pool,gzip.open(tmp,"wt",compresslevel=2) as f:
        for rows in pool.map(play,jobs):
            for r in rows:f.write(json.dumps(r,default=base.old.json_value,separators=(",",":"))+"\n")
    tmp.replace(dest)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("--workers",type=int,default=64)
    args=p.parse_args()
    if args.stage=="select":
        freeze()
        from fresh import select
        select()
    elif args.stage=="test":
        freeze()
        from fresh import verify
        deploy("test",verify(),args.workers,list(range(50)))
    else:deploy(args.stage,DEV,args.workers,list(range(50)) if args.stage.startswith("dev") else [0,1,6,7,13,21,44])
