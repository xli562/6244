from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import benchmark as b

def records(name):
    with gzip.open(b.OUT/(name+".jsonl.gz"),"rt") as f:return list(map(json.loads,f))
def key(r):
    return (*r["key"].split("/")[:2],r["task"],r["i"])
def main(stem,tasks):
    arms=defaultdict(dict)
    for r in records(stem):
        k=key(r)
        assert k not in arms[r["arm"]] and 0<=r["steps"]<=460 and len(r["adjustments"])<=2
        arms[r["arm"]][k]=r
    expected={(m,str(j),t,i) for m in ("codex","claude") for j in (1,2,3) for t in tasks for i in range(50)}
    assert set(arms)=={"new","old","receipt","no_preview","continue","no_release"} and all(set(data)==expected for data in arms.values())
    audited={key(r):r for r in records(stem+"_audit") if r["proven"]}
    summary={}
    for m in ("both","codex","claude"):
        summary[m]={}
        for t in ("ALL",*tasks):
            cnt=Counter()
            scope=[k for k in sorted(expected) if (m=="both" or k[0]==m) and (t=="ALL" or k[2]==t)]
            for k in scope:
                ok={a:data[k]["stable"] for a,data in arms.items()}
                r=arms["new"][k]
                # Reference same physical start and program, suppress stage switching.
                for a,v in ok.items():
                    cnt[a]+=v
                    cnt[a+"_errors"]+=bool(arms[a][k]["error"])
                old_loss=ok["continue"] and not ok["old"]
                new_loss=ok["continue"] and not ok["new"]
                cnt.update(n=1,won=ok["new"] and not ok["old"],lost=ok["old"] and not ok["new"],
                   old_interrupt_losses=old_loss,new_interrupt_losses=new_loss,
                   recovered_reference=old_loss and ok["new"],introduced_reference=new_loss and ok["old"],
                   preview_won=ok["new"] and not ok["no_preview"],preview_lost=ok["no_preview"] and not ok["new"],
                   release_won=ok["new"] and not ok["no_release"],release_lost=ok["no_release"] and not ok["new"],
                   alerted=any(e["loss"] for e in r["releases"]),budget_ready=any(e["loss"] and e.get("retries_left",0)>0 and e["remaining"]>=e.get("retry_cost",1000) for e in r["releases"]),
                   attempted=any(e.get("acted",False) for e in r["releases"]),strict=k in audited,
                   detached=k in audited and not audited[k]["episode"]["support_after"],
                   strict_old_failed=k in audited and audited[k]["also_old_failed"],
                   shorter_inspections=sum(e.get("inspected",40)<40 for e in r["releases"]))
            cnt["net"]=cnt["won"]-cnt["lost"]
            cnt["fewer_interrupt_losses"]=cnt["old_interrupt_losses"]-cnt["new_interrupt_losses"]
            cnt["strict_layouts"]=len({k[2:] for k in scope if k in audited})
            summary[m][t]=dict(cnt)
    (b.OUT/(stem+"_summary.json")).write_text(json.dumps(summary,indent=1)+"\n")
    with (b.OUT/(stem+"_scores.csv")).open("w") as f:
        w=csv.DictWriter(f,fieldnames=["model","task",*summary["both"]["ALL"]])
        w.writeheader()
        for m,tbl in summary.items():
            for t,v in tbl.items():w.writerow(dict(model=m,task=t,**v))
    print(stem,summary["both"]["ALL"])
if __name__=="__main__":
    stem=sys.argv[1]
    from holdback import verify
    tasks=list(b.DEV) if stem.startswith("dev") else (["stage_0375814487c77a"] if stem=="ketchup" else verify())
    main(stem,tasks)
