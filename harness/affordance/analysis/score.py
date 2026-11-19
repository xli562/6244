"""Count episode gains, and verify changed choices via physical counterfactual replays."""
import csv
import gzip
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"harness/affordance"))
from evaluate import witnesses
OUT=ROOT/"runs/affordance_01"

def read(stem):
    with gzip.open(OUT/(stem+".jsonl.gz"),"rt") as f:
        yield from map(json.loads,f)

def case(r):
    a,rep,_=r["key"].split("/")
    return a,rep,r["task"],r["i"]

def prefix(r,w):
    for i,e in enumerate(r["attempts"]):
        if (e["at"]==w["at"] and e["step"]["op"]==w["op"]
                and e["evidence"].get(w["role"],{}).get("resolved")==w["new"]):
            return r["attempts"][:i+1]
    return None

def main():
    strict=set(json.loads((OUT/"semantic_protocol.json").read_text())["primary"])
    strict.update(t["task"] for t in json.loads((OUT/"additional_manifest.json").read_text())["tasks"])
    summary=defaultdict(Counter)
    detail=defaultdict(Counter)
    decisions=[]
    for stem,diag in [("test","diagnose"),("additional_test","additional_diagnose"),("practice","practice_diagnose")]:
        data=defaultdict(dict)
        replays={}
        for r in read(stem):
            group=data[case(r)]
            assert r["mode"] not in group
            group[r["mode"]]=r
        for r in read(diag):
            k=(case(r),tuple(sorted(r["force"].items())))
            assert k not in replays
            replays[k]=r
        for k,by in data.items():
            assert set(by)==({"old","new"} if stem=="practice" else {"old","new","receipt"})
            a,rep,task,i=k
            new,old=by["new"],by["old"]
            gain=new["stable"] and not old["stable"]
            counts=Counter(n=1,gains=int(gain),losses=int(old["stable"] and not new["stable"]))
            for mode,r in by.items():
                counts[mode]=int(r["stable"])
                counts[mode+"_errors"]=int(bool(r["error"]))
                counts[mode+"_no_motion"]=int(r["steps"]==0)
            feedback=new["feedback"]
            counts["feedback"]=int(bool(feedback))
            counts["no_proposal_gain"]=int(gain and bool(feedback) and feedback[0]["reason"]=="no_proposal" and not old["attempts"] and not old["actions"])
            counts["blocked_gain"]=int(gain and bool(feedback) and feedback[0]["reason"]=="blocked")
            ws=witnesses(new)
            counts["goal_role_changed"]=int(bool(ws))
            proved=False
            for w in ws:
                replay=replays.pop((k,tuple(sorted(w.items()))))
                before=prefix(new,w)
                checkpoint=bool(before and before==prefix(replay,w) and replay["force_hit"])
                acted=any(e["at"]==w["at"] and e["step"]["op"]==w["op"] and e["bound"].get(w["role"])==w["old"] for e in replay["actions"])
                if checkpoint and acted and not replay["stable"]:
                    proved=True
                counts["replay_events"]+=1
                counts["replay_not_matched"]+=int(not (checkpoint and acted))
                decisions.append(dict(cohort=stem,agent=a,rep=rep,task=task,index=i,**w,matched=checkpoint,submitted=acted,
                                      raw_stable=replay["stable"],incremental_gain=gain))
            counts["proved"]=int(proved)
            counts["proved_gain"]=int(proved and gain)
            counts["proved_also_old_success"]=int(proved and old["stable"])
            cohorts=[stem]
            if stem!="practice":
                cohorts.append("all")
                if task in strict:
                    cohorts.append("strict")
            for c in cohorts:
                summary[c,a].update(counts)
            detail[stem,a,task].update(counts)
        assert not replays
    for a in ("codex","claude"):
        assert summary["test",a]["n"]==1800
        assert summary["strict",a]["n"]==1050
        assert summary["all",a]["n"]==2250
        assert summary["practice",a]["n"]==150
    report={"/".join(k):dict(v) for k,v in sorted(summary.items())}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    with (OUT/"by_task.csv").open("w") as f:
        fields=sorted(set.union(*(set(v) for v in detail.values())))
        writer=csv.DictWriter(f,fieldnames=["cohort","agent","task",*fields])
        writer.writeheader()
        for (c,a,t),s in sorted(detail.items()):
            writer.writerow(dict(cohort=c,agent=a,task=t,**s))
    with (OUT/"changed_choices.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(decisions[0]) if decisions else ["cohort","agent","rep","task","index"])
        writer.writeheader()
        writer.writerows(decisions)
    for k,v in report.items():
        print(k,*[v.get(x,0) for x in ("n","receipt","old","new","gains","losses","proved","proved_gain","no_proposal_gain","replay_not_matched")])
if __name__=="__main__":
    main()
