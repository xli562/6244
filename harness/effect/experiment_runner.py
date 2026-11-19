"""Paired simulator execution. Policies never see goals, task IDs, or evaluator status."""
import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch
from compile_policies import BASE, OUT, ROOT, old, previous
from feedback import Controller,EffectLibrary
from interface import Public,PlannerStalled

h=old.h
DEV="KITCHEN_SCENE8_turn_off_the_stove"

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

@lru_cache(None)
def sources(key):
    return [previous.module((p/key/"proposal.py").read_text()) for p in (OUT,BASE,previous.PRIOR)]

def file_paths(t):
    return OUT/"assets"/(t+".bddl"),None

def play(job):
    key,arm,task,ids=job
    new,prior,original=sources(key)
    libs=[]
    def factory(sensor,_):
        family="receipt" if key.endswith("/receipt") else "new"
        lib=EffectLibrary(sensor,family,prior.get("reread",original.get("reread")),original["run"])
        lib.configure(arm,new.get("after"))
        libs.append(lib)
        return lib
    def entry(api):
        lib=libs[-1]
        def legacy():
            try:
                prior["run"](api)
            except h.bench.EpisodeEnd:
                raise
            except PlannerStalled:
                pass
            except Exception as exc:
                if lib.variant!="new" or lib.events: raise
                lib.pending=dict(reason="planner_error",detail=str(exc)[:180],step={},evidence={})
            if lib.variant!="new" or (lib.events and not lib.pending): return
            for _ in range(2):
                fb=lib.pending or dict(reason="no_proposal",step={},evidence={})
                lib.feedback.append(deepcopy(fb))
                if not callable(prior.get("retry")): break
                lib.phase="retry"
                lib.pending=None
                start=lib.remaining()
                try:
                    prior["retry"](api,fb)
                except PlannerStalled:
                    pass
                if lib.remaining()!=start or (lib.events and not lib.pending): break
        lib.controller=Controller(lib,legacy)
        new["run"](lib.controller)
    from contextlib import ExitStack
    with ExitStack() as s:
        for obj,name,value in [(old,"SkillLibrary",factory),(old,"PublicAPI",Public),(h.episode,"load_policy",lambda _:entry)]:
            s.enter_context(patch.object(obj,name,value))
        if task!=DEV:
            s.enter_context(patch.object(h.bench,"paths",file_paths))
            s.enter_context(patch.object(h.bench,"CONFIG",OUT/"assets"))
        rows=old.episodes((key,task,ids,""))
    return [dict(key=key,arm=arm,task=task,i=r["index"],stable=r["settled"],official=r["success"],
                 steps=r["steps"],error=r["error"],measurements=l.records,adjustments=l.adjustments)
            for r,l in zip(rows,libs)]

def deploy(jobs,name,workers):
    dest=OUT/(name+".jsonl.gz")
    if dest.exists(): raise FileExistsError(dest)
    tmp=dest.with_suffix(".tmp")
    n=0
    with gzip.open(tmp,"wt",compresslevel=3) as z,ProcessPoolExecutor(workers) as ex:
        for batch in ex.map(play,jobs):
            for row in batch:
                z.write(json.dumps(row,default=old.json_value,separators=(",",":"))+"\n")
            n+=len(batch)
            if n%1000==0: print(n,flush=True)
    tmp.replace(dest)
    print(name,n,flush=True)

def freeze():
    previous.lock()
    files=[*Path(__file__).parent.glob("*.py"),OUT/"programs.json",BASE/"system.json",BASE/"programs.json",
           BASE/"manifest.json",BASE/"additional_manifest.json",*list((ROOT/"harness/affordance").glob("*.py"))]
    for key,sha in h.read(OUT/"programs.json").items():
        for p in (OUT,BASE,previous.PRIOR):
            files.append(p/key/"proposal.py")
        assert digest(OUT/key/"proposal.py")==sha and len((OUT/key/"calls.jsonl").read_text().splitlines())==1
    h.save_lock(OUT/"freeze.json",dict(hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
        physics_steps=500,max_adjustments=2,models=h.AGENTS,
        eligibility="LIBERO-Pro redesigned tasks and object perturbations. No old goal atoms, duplicate goal atoms, or prior task paraphrases. Stock simulator assets only. Fill any shortage up to eight with novel substitutions on pinned official receiving regions under the same atomic filter.",
        attribution="Same checked policy stable with measured post-action retry, fails without it, exact observation/action prefix at first intervention. Blind repeat diagnostic."))
def main(stage,workers):
    dev=stage.startswith("dev")
    if dev:
        old.register(h.read(BASE/"manifest.json"))
        tasks=[DEV]
    else:
        freeze()
        from holdout import select,verify
        if stage=="select":
            select()
            return
        tasks=verify()
    keys=h.read(OUT/"programs.json")
    if stage in ("dev_repeat","repeat"):
        stem="dev" if dev else "test"
        with gzip.open(OUT/(stem+".jsonl.gz"),"rt") as z:
            rows=[r for r in map(json.loads,z) if r["arm"]=="new" and r["stable"] and any(a["source"]=="observation" for a in r["adjustments"])]
        jobs=[(r["key"],"repeat",r["task"],[r["i"]]) for r in rows]
    else:
        jobs=[(k,a,t,list(range(i,i+5))) for k in keys for t in tasks
              for a in (("receipt",) if k.endswith("/receipt") else ("old","new"))
              for i in range(0,50,5)]
    deploy(jobs,stage,workers)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["dev","dev_repeat","select","test","repeat"])
    p.add_argument("--workers",type=int,default=48)
    args=p.parse_args()
    main(args.stage,args.workers)
