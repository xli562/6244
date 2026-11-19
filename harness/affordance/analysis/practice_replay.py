"""Apply the frozen counterfactual scorer to the development cases as well."""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bench

if __name__=="__main__":
    bench.old.register(bench.h.read(bench.PRIOR/"manifest.json"))
    jobs=[]
    with gzip.open(bench.OUT/"practice.jsonl.gz","rt") as f:
        for r in map(json.loads,f):
            for w in bench.witnesses(r):
                jobs.append((r["key"],"new",r["task"],[r["i"]],True,w))
    bench.deploy(jobs,20,"practice_diagnose")
