"""Use the existing LIBERO reset, physical cap and release stability scorer unchanged."""
import argparse
from contextlib import ExitStack
from functools import lru_cache
import gzip
import json
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor

from programs import ROOT,OUT,base
from checkpoints import StageLibrary
DEV=("pro_05bb20fac5c129","pro_9de76238809694")
OLD_SOURCES=base.sources

class Overlay:
    def __init__(self,ctrl,prior):
        self._ctrl=ctrl
        self._prior=prior
    @property
    def instruction(self):return self._ctrl.instruction
    def original(self):return self._prior(self._ctrl)
    def last(self):return self._ctrl.last()
    def adjust(self,strategy):return self._ctrl.adjust(strategy)

@lru_cache(None)
def compiled(k):
    return compile((OUT/k/"proposal.py").read_text(),"stage-policy.py","exec")

def play(job):
    k,mode,t,ids=job
    prior,planner,orig=OLD_SOURCES(k)
    libs=[]
    def factory(*args):
        l=StageLibrary(*args)
        libs.append(l)
        return l
    def run(ctrl):
        ns={}
        exec(compiled(k),ns)
        libs[-1].handler=ns.get("checkpoint")
        return ns["run"](Overlay(ctrl,prior["run"]))
    with ExitStack() as s:
        s.enter_context(patch.object(base,"EffectLibrary",factory))
        s.enter_context(patch.object(base,"sources",lambda _:({**prior,"run":run},planner,orig)))
        if t not in DEV:s.enter_context(patch.object(base,"OUT",OUT))
        rows=base.play(job)
    for r,l in zip(rows,libs):
        r.update(checks=l.checks,requests=l.requests,repairs=l.repairs)
    return rows

def deploy(jobs,stage,workers):
    path=OUT/(stage+".jsonl.gz")
    if path.exists(): raise FileExistsError(path)
    tmp=path.with_suffix(".tmp")
    n=0
    with ProcessPoolExecutor(workers) as pool,gzip.open(tmp,"wt",compresslevel=3) as f:
        for rows in pool.map(play,jobs):
            for r in rows:f.write(json.dumps(r,separators=(",",":"),default=base.old.json_value)+"\n")
            n+=len(rows)
            if n%1000==0:print(n,flush=True)
    tmp.replace(path)
    print(stage,n,flush=True)

def freeze():
    base.freeze()
    files=[*Path(__file__).parent.glob("*.py"),base.OUT/"freeze.json",base.OUT/"manifest.json",base.OUT/"text_audit.json",OUT/"programs.json"]
    files.extend((ROOT/"sim").glob("*.py"))
    for k,sha in base.h.read(OUT/"programs.json").items():
        p=OUT/k/"proposal.py"
        assert base.digest(p)==sha and len((OUT/k/"calls.jsonl").read_text().splitlines())==1
        files.append(p)
    base.h.save_lock(OUT/"freeze.json",dict(files={str(p.relative_to(ROOT)):base.digest(p) for p in files},limit=500,
              adjustments=2,models=base.h.AGENTS,controls=["old","receipt","late"],repetitions=3, attribution="Identical first request and pre-intervention observations, old and late fail, early stable, first stage repair changes matching native subgoal false to true in independent replay."))

def main(stage,workers):
    if stage=="dev": tasks=DEV
    else:
        freeze()
        from suite import select,verify
        if stage=="select":return select()
        tasks=verify()
    keys=base.h.read(OUT/"programs.json")
    jobs=[(k,mode,t,list(range(i,i+5))) for k in keys for t in tasks for mode in
          (("receipt",) if k.endswith("receipt") else ("old","early","late")) for i in range(0,50,5)]
    deploy(jobs,stage,workers)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["dev","select","test"])
    p.add_argument("--workers",type=int,default=48)
    args=p.parse_args()
    main(args.stage,args.workers)
