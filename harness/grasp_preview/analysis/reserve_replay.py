"""Replay last round's short-budget alerts without using their outcomes as inputs."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import benchmark as b

def one(row):
    key,t,i=row["key"],row["task"],row["i"]
    with patch.object(b,"DEV",(*b.DEV,t)):
        rows={m:b.play((key,m,t,[i]))[0] for m in ("old","new","no_release")}
    return dict(key=key,task=t,i=i,old_remaining=[e["remaining"] for e in row["releases"] if e["loss"]],rows=rows)

def main(stem):
    with gzip.open(b.previous.OUT/"test.jsonl.gz","rt") as f:
        selected=[r for r in map(json.loads,f) if r["arm"]=="new" and any(e["loss"] for e in r["releases"])]
    with ProcessPoolExecutor(16) as pool,gzip.open(b.OUT/(stem+".jsonl.gz"),"wt") as f:
        for row in pool.map(one,selected):f.write(json.dumps(row,default=b.base.old.json_value)+"\n")
if __name__=="__main__":main(sys.argv[1])
