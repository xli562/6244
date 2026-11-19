"""Revisit every previous new/receipt episode that raised the null-comparison exception."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from unittest.mock import patch
import attribution as audit
r=audit.runner

def one(x):
    play=r.base.play
    def load_previous_assets(job):
        with patch.object(r.base,"OUT",r.prior.OUT):
            return play(job)
    k,t,i=audit.key(x)
    with patch.object(r.base,"play",load_previous_assets):
        row=r.play((k,"new" if x["arm"]=="early" else "receipt",t,[i]))[0]
    return dict(key=k,task=t,i=i,old_error=x["error"],error=row["error"],filled=row["height_fixes"],stable=row["stable"])

if __name__=="__main__":
    r.freeze()
    jobs=[]
    for stem in ("test","challenge"):
        with gzip.open(r.prior.OUT/(stem+".jsonl.gz"),"rt") as f:
            jobs.extend(x for x in map(json.loads,f) if x["arm"] in ("early","receipt") and "NoneType" in str(x["error"]))
    with ProcessPoolExecutor(22) as pool,gzip.open(r.OUT/"previous_errors.jsonl.gz","wt") as f:
        for row in pool.map(one,jobs):f.write(json.dumps(row)+"\n")
