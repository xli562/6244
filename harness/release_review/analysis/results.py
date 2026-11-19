import csv
from collections import Counter
import json
import gzip
from pathlib import Path
import sys
import attribution as audit
r=audit.runner

def tabulate(stem):
    data=audit.records(stem)
    arms={a:{} for a in ("new","old","literal","continue","receipt","no_release")}
    tasks=r.DEV if stem=="dev_locked" else tuple(x["task"] for x in r.base.h.read(r.OUT/"manifest.json")["tasks"])
    for row in data:
        k,t,i=audit.key(row)
        key=(k.split("/")[0],int(k.split("/")[1]),t,i)
        assert key not in arms[row["arm"]] and 0<=row["steps"]<=460 and len(row["adjustments"])<=2
        arms[row["arm"]][key]=row
    expected={(m,j,t,i) for m in ("codex","claude") for j in range(1,4) for t in tasks for i in range(50)}
    assert all(set(v)==expected for v in arms.values())
    checked=audit.records(stem+"_audit")
    strict={(v["key"].split("/")[0],int(v["key"].split("/")[1]),v["task"],v["i"]):v for v in checked if v["proven"]}
    out={}
    for m in ("both","codex","claude"):
        out[m]={}
        for t in ["ALL",*tasks]:
            scope=sorted(k for k in expected if (m=="both" or k[0]==m) and (t=="ALL" or k[2]==t))
            counts=Counter()
            for k in scope:
                a={name:v[k] for name,v in arms.items()}
                success={name:x["stable"] for name,x in a.items()}
                for name,x in a.items():
                    counts[name]+=x["stable"]
                    counts[name+"_errors"]+=bool(x["error"])
                v=strict.get(k)
                counts.update(n=1,prior_bad_switch=success["continue"] and not success["old"],
                              review_bad_switch=success["continue"] and not success["no_release"],
                              prior_losses_restored=success["continue"] and not success["old"] and success["no_release"],
                              fresh_losses=success["continue"] and success["old"] and not success["no_release"],
                              gain=success["new"] and not success["old"],loss=success["old"] and not success["new"],
                              release_gain=success["new"] and not success["no_release"],
                              release_loss=success["no_release"] and not success["new"],
                              release_alert=any(e["loss"] for e in a["new"]["releases"]),
                              release_acted=any(e.get("acted",False) for e in a["new"]["releases"]),
                              strict=v is not None,strict_old_fail=bool(v and v["also_old_failed"]),
                              strict_detached=bool(v and not v["episode"]["support_after"]),
                              strict_before_open=bool(v and v["episode"]["before"]),
                              numeric_repairs=a["new"]["height_fixes"])
            counts["loss_reduction"]=counts["prior_bad_switch"]-counts["review_bad_switch"]
            counts["strict_states"]=len({k[2:] for k in scope if k in strict})
            out[m][t]=dict(counts)
    (r.OUT/(stem+"_summary.json")).write_text(json.dumps(out,indent=1)+"\n")
    with (r.OUT/(stem+"_by_task.csv")).open("w") as f:
        names=list(out["both"]["ALL"])
        writer=csv.DictWriter(f,fieldnames=["model","task",*names])
        writer.writeheader()
        for m,tasks in out.items():
            for t,v in tasks.items():writer.writerow(dict(model=m,task=t,**v))
    if stem=="dev_locked":
        with gzip.open(r.prior.OUT/"test.jsonl.gz","rt") as f:
            hist={(x['key'],x['task'],x['i'],x['arm']):x for x in map(json.loads,f) if x['arm'] in ('old','early') and x['task']==r.DEV[0]}
        original_lost=Counter()
        for (m,j,t,i),new in arms["new"].items():
            if t!=r.DEV[0]:continue
            ref=(f"{m}/{j}/checked",t,i)
            if hist[(*ref,"old")]["stable"] and not hist[(*ref,"early")]["stable"]:
                original_lost[m+"_restored"]+=new["stable"]
                original_lost[m+"_n"]+=1
        (r.OUT/"previous_losses.json").write_text(json.dumps(original_lost,indent=1)+"\n")
    return out
if __name__=="__main__":tabulate(sys.argv[1])
