"""Replay the literal previous release as an additional baseline, without the shared wrapper."""
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runner
from suite import verify

def play(job):
    k,t,ids=job
    with patch.object(runner.base,"OUT",runner.OUT):
        # Load the original program while its own source directory is active.
        with patch.object(runner.base,"sources",lambda _:frozen):
            with patch.object(runner.base,"OUT",PREVIOUS):
                frozen=runner.OLD_SOURCES(k)
            return runner.base.play((k,"new",t,ids))

PREVIOUS=runner.base.OUT
if __name__=="__main__":
    runner.freeze()
    stem=sys.argv[1] if len(sys.argv)>1 else "test"
    if stem=="challenge":
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"challenge"))
        from relations import verify
    tasks=verify()
    jobs=[(k,t,list(range(i,i+5))) for k in runner.base.h.read(runner.OUT/"programs.json") if k.endswith("/checked")
          for t in tasks for i in range(0,50,5)]
    dst=runner.OUT/("previous.jsonl.gz" if stem=="test" else "challenge_previous.jsonl.gz")
    assert not dst.exists()
    tmp=dst.with_suffix(".tmp")
    with ProcessPoolExecutor(28) as ex,gzip.open(tmp,"wt",compresslevel=3) as f:
        for result in ex.map(play,jobs):
            for r in result:f.write(json.dumps(r,default=runner.base.old.json_value,separators=(",",":"))+"\n")
    tmp.replace(dst)
