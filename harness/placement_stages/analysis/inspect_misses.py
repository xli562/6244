"""Observe a fixed sample of unsuccessful early runs to distinguish grip failures from targeting."""
import gzip
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
import audit
import checkpoints

def probe(r):
    obs=[]
    acquire,deliver=checkpoints.acquire,checkpoints.deliver
    def pickup(api,item,attempt):
        sample=dict(phase="grasp",item=item,attempt=attempt,at=api.remaining())
        obs.append(sample)
        try:
            ok=acquire(api,item,attempt)
            sample["lifted"]=bool(ok)
            return ok
        finally:
            sample["remaining"]=api.remaining()
    def carry(api,item,target):
        sample=dict(phase="deliver",target=target,at=api.remaining())
        obs.append(sample)
        try:return deliver(api,item,target)
        finally:
            p=api.pos(item)-api.pos(target)
            sample.update(remaining=api.remaining(),xy=round(float((p[0]**2+p[1]**2)**.5),4),
                          dz=round(float(p[2]),4))
    with patch.object(checkpoints,"acquire",pickup),patch.object(checkpoints,"deliver",carry):
        rr=audit.run.play((r["key"],"early",r["task"],[r["i"]]))[0]
    assert rr["stable"]==r["stable"] and rr["checks"]==r["checks"]
    return dict(key=r["key"],i=r["i"],task=r["task"],group=r["group"],probes=obs,official=rr["official"])
def main():
    data=audit.read("test")
    old={audit.episode_key(r):r for r in data if r["arm"]=="old"}
    groups=defaultdict(list)
    for r in data:
        if r["arm"]!="early" or r["stable"]:continue
        r["group"]="lost" if old[audit.episode_key(r)]["stable"] else "both_failed"
        groups[r["task"],r["key"].split("/")[0],r["group"]].append(r)
    jobs=[]
    for group in groups.values():
        ordered=sorted(group,key=lambda r:(r["i"],r["key"]))
        for i in range(min(6,len(ordered))):
            jobs.append(ordered[i*len(ordered)//min(6,len(ordered))])
    with ProcessPoolExecutor(20) as pool,gzip.open(audit.run.OUT/"miss_samples.jsonl.gz","wt") as f:
        for row in pool.map(probe,jobs):f.write(json.dumps(row)+"\n")

if __name__=="__main__":main()
