import argparse
import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

from build import old, ROOT, OUT, PRIOR, REPS
from interface import ActionLibrary, PlannerStalled, Public
from evaluate import witnesses

h=old.h
DEV={"codex":"KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
     "claude":"LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray"}
@lru_cache(None)
def module(src):
    h.episode.check_policy_source(src)
    ns={}
    exec(compile(src,"agent.py","exec"),ns)
    return ns

def play(job):
    key,mode,task,indices,practice,force=job
    src=(OUT/key/"proposal.py").read_text()
    base=module((PRIOR/key/"proposal.py").read_text())
    generated=module(src)
    libs=[]
    evaluator_goals=[]
    def factory(sensors, unused):
        lib=ActionLibrary(sensors,mode,generated.get("reread",base.get("reread")),base["run"],force)
        libs.append(lib)
        evaluator_goals.append(deepcopy(sensors._env.env.parsed_problem["goal_state"]))
        return lib
    def execute(api):
        lib=libs[-1]
        callback=generated.get("retry")
        primary=base["run"] if practice else generated["run"]
        reason="no_proposal"
        try:
            primary(api)
        except h.bench.EpisodeEnd:
            raise
        except PlannerStalled:
            reason="blocked"
        except Exception as exc:
            if mode!="new" or lib.events:
                raise
            reason="planner_error"
            lib.pending=dict(reason=reason,detail=str(exc)[:180],step={},evidence={})
        if mode!="new" or (lib.events and not lib.pending):
            return
        for _ in range(2):
            feedback=lib.pending or dict(reason=reason,step={},evidence={})
            lib.feedback.append(deepcopy(feedback))
            if not callable(callback):
                break
            lib.phase="retry"
            lib.pending=None
            before=lib.remaining()
            try:
                callback(api,deepcopy(feedback))
            except PlannerStalled:
                pass
            if lib.remaining()!=before or (lib.events and not lib.pending):
                break
    with patch.object(old,"SkillLibrary",factory),patch.object(old,"PublicAPI",Public),patch.object(h.episode,"load_policy",lambda src:execute):
        rows=old.episodes((key,task,indices,src))
    logs=[]
    for r,lib,goals in zip(rows,libs,evaluator_goals):
        logs.append(dict(key=key,mode=mode,task=task,i=r["index"],stable=r["settled"],official=r["success"],steps=r["steps"],
                         error=r["error"],feedback=lib.feedback,attempts=lib.attempts,queries=lib.query_count,
                         actions=[dict(at=e["start"],step=e["intent"],bound=e["bound"]) for e in lib.events],
                         trace=r["trace"],force=force,force_hit=lib.forced_hit,names=list(lib.scene),
                         goals=goals))
    return logs

def deploy(jobs,workers,stem):
    path=OUT/(stem+".jsonl.gz")
    if path.exists():
        raise FileExistsError(path)
    n=0
    tmp=path.with_suffix(".tmp")
    with gzip.open(tmp,"wt",compresslevel=2) as out, ProcessPoolExecutor(workers) as pool:
        for rows in pool.map(play,jobs):
            for row in rows:
                out.write(json.dumps(row,default=old.json_value,separators=(",",":"))+"\n")
            n+=len(rows)
            if n%1000==0:
                print(n,flush=True)
    tmp.replace(path)
    print(stem,n,flush=True)

def programs():
    keys=h.read(OUT/"programs.json")
    assert keys.keys()=={f"{a}/{r}/{f}" for a in h.AGENTS for r in REPS for f in ("checked","receipt")}
    for key,sha in keys.items():
        assert h.digest((OUT/key/"proposal.py").read_text())==sha
        assert len((OUT/key/"calls.jsonl").read_text().splitlines())==1
    return keys

def lock():
    paths=[*Path(__file__).parent.glob("*.py"),
           *list((ROOT/"harness/reselection").glob("*.py")),PRIOR/"manifest.json",PRIOR/"programs.json",
           OUT/"programs.json",ROOT/"testsets/libero_holdout_v1/collect.py",ROOT/"harness/interaction/tools_api.py",
           *list((ROOT/"harness/preflight").glob("*.py"))]
    for key in programs():
        paths.append(PRIOR/key/"proposal.py")
    h.save_lock(OUT/"system.json",dict(files={str(p.relative_to(ROOT)):h.digest(p.read_text()) for p in sorted(paths)},
                                      sim=h.system_spec(),reps=3,budget="one frozen prior + one new CLI compilation; no model calls during tests; 500 control steps",
                                      modes=["old","new","receipt"],diagnosis="Restore first inconsistent nomination for each changed official goal role on a stable success."))
def main(stage,workers):
    keys=programs()
    if stage=="practice":
        m=h.read(PRIOR/"manifest.json")
    else:
        lock()
        if stage=="select":
            from draw import draw
            m=draw()
            h.save_lock(OUT/"manifest.json",m)
            h.save_lock(OUT/"selection.json",{p:h.digest((OUT/p).read_text()) for p in ("manifest.json","system.json")})
            return
        assert h.read(OUT/"selection.json")=={p:h.digest((OUT/p).read_text()) for p in ("manifest.json","system.json")}
        m=h.read(OUT/"manifest.json")
    old.register(m)
    h.bench.prepare()
    jobs=[]
    if stage in ("test","practice"):
        practice=stage=="practice"
        for key in keys:
            a,_,family=key.split("/")
            if practice and family!="checked":
                continue
            for t in ([DEV[a]] if practice else [r["task"] for r in m["tasks"]]):
                for mode in (("new","old") if family=="checked" else ("receipt",)):
                    for i in range(0,50,5):
                        jobs.append((key,mode,t,list(range(i,i+5)),practice,None))
    else:
        with gzip.open(OUT/"test.jsonl.gz","rt") as f:
            for r in map(json.loads,f):
                for w in witnesses(r):
                    jobs.append((r["key"],"new",r["task"],[r["i"]],False,w))
    deploy(jobs,workers,stage)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("practice","select","test","diagnose"))
    p.add_argument("--workers",type=int,default=56)
    args=p.parse_args()
    main(args.stage,args.workers)
