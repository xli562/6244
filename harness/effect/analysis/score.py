"""Recount every paired rollout, keeping no-effect, partial effect and losses separate."""
import csv
import gzip
import json
from collections import Counter,defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parents[3]/"runs/effect_01"
def read(stem):
    with gzip.open(OUT/(stem+".jsonl.gz"),"rt") as f:
        yield from map(json.loads,f)
def case(r):
    a,g,_=r["key"].split("/")
    return a,g,r["task"],r["i"]
def score():
    counts=defaultdict(Counter)
    tasks=defaultdict(Counter)
    changed=[]
    native={r["task"] for r in json.loads((OUT/"manifest.json").read_text())["tasks"] if r["origin"]=="LIBERO-Pro"}
    kept=json.loads((OUT/"text_audit.json").read_text())["kept"]
    for stage in ("dev","test"):
        groups=defaultdict(dict)
        for r in read(stage):
            d=groups[case(r)]
            assert r["arm"] not in d
            d[r["arm"]]=r
        assert len(groups)==300*(len(kept) if stage=="test" else 1)
        cert={case(r):r for r in read(stage+"_causal")}
        for k,g in groups.items():
            a,gen,t,i=k
            assert set(g)=={"receipt","old","new"} and all(0<=r["steps"]<=500 for r in g.values())
            new,old=g["new"],g["old"]
            gain=bool(new["stable"] and not old["stable"])
            loss=bool(old["stable"] and not new["stable"])
            log=cert.pop(k,None)
            triggered=any(x["source"]=="observation" for x in new["adjustments"])
            assert bool(log)==bool(triggered and new["stable"])
            valid=bool(gain and log and log["matched"] and log["matched_old"] and log["corrected"])
            reason=log["first_reason"] if log else None
            c=Counter(n=1,gain=int(gain),loss=int(loss),triggered=int(triggered),
                      certified=int(valid),strict_no_effect=int(valid and reason=="no_effect"),
                      wrong_direction=int(valid and reason=="wrong_direction"),shortfall=int(valid and reason=="shortfall"),
                      blind_repeat_success=int(valid and log["matched_repeat"] and log["repeat_stable"]),
                      adjusted_only=int(valid and log["matched_repeat"] and not log["repeat_stable"]),
                      goal_not_confirmed=int(gain and not valid))
            for mode,row in g.items():
                c[mode]=int(row["stable"])
                c[mode+"_official"]=int(row["official"])
                c[mode+"_errors"]=int(bool(row["error"]))
                c[mode+"_steps"]=row["steps"]
                c[mode+"_no_action"]=int(not row["measurements"])
            counts[stage,a].update(c)
            if stage=="test":
                counts["public" if t in native else "derived",a].update(c)
            tasks[stage,a,t].update(c)
            if gain or loss:
                changed.append(dict(stage=stage,agent=a,generation=gen,task=t,index=i,gain=gain,loss=loss,
                                    certified=valid,reason=reason,repeat_stable=log and log["repeat_stable"]))
        assert not cert
    with (OUT/"by_task.csv").open("w") as f:
        fields=sorted(set().union(*(t.keys() for t in tasks.values())))
        w=csv.DictWriter(f,fieldnames=["stage","agent","task",*fields]); w.writeheader()
        for (stage,a,t),v in sorted(tasks.items()):
            w.writerow(dict(stage=stage,agent=a,task=t,**v))
    with (OUT/"paired_changes.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(changed[0])); w.writeheader();w.writerows(changed)
    result={"/".join(k):dict(v) for k,v in counts.items()}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    for k,v in result.items():
        print(k,*[v[x] for x in ("n","receipt","old","new","gain","loss","certified","strict_no_effect","wrong_direction","shortfall","blind_repeat_success","adjusted_only")])
if __name__=="__main__": score()
