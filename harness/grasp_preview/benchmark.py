import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
import gzip
import json
from pathlib import Path
from unittest.mock import patch

from generate_candidates import ROOT,OUT,base,previous
from control import PreviewLibrary
DEV=("review_c408ab53315386","review_bd0bead078ecaa")

@lru_cache(None)
def compiled(k):
    return compile((OUT/k/"proposal.py").read_text(),"candidate-policy","exec")

def play(job):
    k,arm,t,ids=job
    prior=previous.prior
    ancestor,planner,original=prior.OLD_SOURCES(k)
    libs=[]
    def factory(*args):
        lib=PreviewLibrary(*args)
        libs.append(lib)
        return lib
    def entry(ctrl):
        stage,review,latest={},{},{}
        exec(prior.compiled(k),stage)
        exec(previous.code(k),review)
        exec(compiled(k),latest)
        lib=libs[-1]
        lib.parent_handler=stage.get("checkpoint")
        decide=review.get("decide")
        lib.policy=(lambda e:decide(lib.ledger(e))) if lib.enhanced and decide else decide
        lib.next_choice=latest.get("choose")
        lib.release_choice=latest.get("release")
        lib.handler=lib.arbitrate
        stage["run"](prior.Overlay(ctrl,ancestor["run"]))
        for finish in (review["finish"],latest["finish"]):
            receipt=ctrl.last()
            if receipt and receipt["op"]=="place" and finish(lib.ledger(receipt))=="regrasp":
                ctrl.adjust("regrasp")
    with ExitStack() as s:
        s.enter_context(patch.object(base,"EffectLibrary",factory))
        s.enter_context(patch.object(base,"sources",lambda _:({**ancestor,"run":entry},planner,original)))
        asset_root=next(root for root in (OUT,previous.OUT,previous.prior.OUT) if (root/"assets"/(t+".bddl")).is_file())
        s.enter_context(patch.object(base,"OUT",asset_root))
        rows=base.play(job)
    for row,lib in zip(rows,libs):
        row.update(decisions=lib.decisions,releases=lib.releases,repairs=lib.repairs,
                   preview=lib.preview_log,reserves=lib.reserves)
    return rows

def freeze():
    previous.freeze()
    files=[*Path(__file__).parent.glob("*.py"),previous.OUT/"freeze.json",previous.OUT/"manifest.json",OUT/"programs.json"]
    for k,d in base.h.read(OUT/"programs.json").items():
        path=OUT/k
        assert base.digest(path/"proposal.py")==d and not (path/"rejection.json").exists()
        assert len((path/"calls.jsonl").read_text().splitlines())==1
        files.extend(path/n for n in ("proposal.py","calls.jsonl","prompt.txt"))
    base.h.save_lock(OUT/"freeze.json",dict(files={str(p.relative_to(ROOT)):base.digest(p) for p in files},
        budget=[460,40],reps=3,attribution="matched full state at release; native subgoal true during release, false after, true after retry; stable new win, same policy suppressing retry fails"))

def deploy(stem,tasks,workers,indices):
    keys=base.h.read(OUT/"programs.json")
    jobs=[(k,m,t,indices[j:j+5]) for k in keys for t in tasks for m in
          (("receipt",) if k.endswith("receipt") else ("old","new","no_release","continue","no_preview"))
          for j in range(0,len(indices),5)]
    dst=OUT/(stem+".jsonl.gz")
    assert not dst.exists()
    tmp=dst.with_suffix(".tmp")
    with ProcessPoolExecutor(workers) as pool,gzip.open(tmp,"wt",compresslevel=2) as f:
        for rows in pool.map(play,jobs):
            for row in rows:f.write(json.dumps(row,default=base.old.json_value,separators=(",",":"))+"\n")
    tmp.replace(dst)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("--workers",type=int,default=64)
    args=p.parse_args()
    if args.stage=="select":
        if not (OUT/"dev_final_summary.json").is_file():raise RuntimeError("Finish the development replay before selecting.")
        freeze()
        from holdback import select
        select()
    elif args.stage=="test":
        freeze()
        from holdback import verify
        deploy("test",verify(),args.workers,list(range(50)))
    else:deploy(args.stage,DEV,args.workers,list(range(50)) if args.stage.startswith('dev') else list(range(8)))
