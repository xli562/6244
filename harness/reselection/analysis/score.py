"""Verify the paired design and count episode-level benefits and remaining vetoes."""
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/reselection_01"


def read_rows(filename):
    with gzip.open(OUT / filename, "rt") as stream:
        yield from map(json.loads, stream)


def group(row):
    agent,rep,_=row["key"].split("/")
    return ("archive" if rep.startswith("old") else "fresh",agent,rep.removeprefix("old"),row["task"],row["i"])


def revisits(row):
    return [e for v in row["reviews"] for e in v["roles"].values() if "revisit" in e]


def tally(stop, recover, receipt, replay):
    rows = [stop,recover,receipt]
    counts=Counter(n=1)
    for r in rows:
        m=r["mode"]
        counts[m]=int(r["stable"])
        counts[m+"_official"]=int(r["official"])
        counts[m+"_error"]=int(bool(r["error"]))
        counts[m+"_no_motion"]=int(r["steps"]==0)
        if r["stable"] or not r["terminal_veto"]:
            continue
        diag=replay[m]
        final=r["reviews"][-1]
        # An identical checkpoint must have actually fired. Audit rounding cannot turn
        # a mismatched checkpoint into a confirmed false veto.
        hit=diag["released"]
        assert (not hit or final in diag["reviews"])
        counts[m+"_veto"]=1
        tied=any(len(final["roles"][role].get("finalists",[]))>1 for role in r["terminal_veto"])
        counts[m+"_tie_veto"]=int(tied)
        counts[m+"_diagnostic_matched"]=int(hit)
        counts[m+"_false_veto"]=int(hit and diag["stable"])
        counts[m+"_false_tie_veto"]=int(hit and diag["stable"] and tied)
    checks=revisits(recover)
    counts["reread"]=int(bool(checks))
    counts["changes_without_review"]=int(not checks and stop["stable"]!=recover["stable"])
    counts["unique"]=int(any(c["resolved"] for c in checks))
    counts["new_pick"]=int(any(c["resolved"] and c["pick"]!=c["resolved"] for c in checks))
    counts["agent_span"]=int(any(c["revisit"].get("via")=="agent" for c in checks))
    counts["fallback_span"]=int(any(c["revisit"].get("via")=="parser" for c in checks))
    counts["rescued"]=int(recover["stable"] and not stop["stable"])
    counts["lost"]=int(stop["stable"] and not recover["stable"])
    if counts["rescued"]:
        counts["rescued_new_pick"]=counts["new_pick"]
        counts["rescued_raw_would_succeed"]=counts["stop_false_veto"]
        counts["rescued_raw_would_fail"]=int("stop" in replay and replay["stop"]["released"] and not replay["stop"]["stable"])
    return counts


def main():
    trios=defaultdict(dict)
    diagnostics=defaultdict(dict)
    for dest,f in [(trios,"test.jsonl.gz"),(diagnostics,"counterfactual.jsonl.gz")]:
        for r in read_rows(f):
            key=group(r)
            assert r["mode"] not in dest[key]
            dest[key][r["mode"]]=r
    manifest=json.loads((OUT/"manifest.json").read_text())
    tasks=[r["task"] for r in manifest["tasks"]]
    expected={(cohort,a,str(rep),t,i) for cohort in ("archive","fresh") for a in ("codex","claude")
              for rep in range(1,4) for t in tasks for i in manifest["indices"]}
    assert set(trios)==expected and set(diagnostics)<=expected
    summary=defaultdict(Counter)
    detail=defaultdict(Counter)
    by_rep=defaultdict(Counter)
    reasons=defaultdict(Counter)
    rejections={}
    for p in OUT.glob("*/*/*/rejection.json"):
        rejections[str(p.relative_to(OUT))]=json.loads(p.read_text())["reason"]
    decision_rows=[]
    for key,m in trios.items():
        cohort,a,rep,t,i=key
        assert set(m)=={"receipt","stop","recover"}
        diag=diagnostics.get(key,{})
        assert set(diag)=={mode for mode in ("stop","recover") if not m[mode]["stable"] and m[mode]["terminal_veto"]}
        counts=tally(m["stop"],m["recover"],m["receipt"],diag)
        for block in (summary[cohort,a], detail[cohort,a,t],by_rep[cohort,a,rep]):
            block.update(counts)
        if counts["rescued"] or counts["lost"] or counts["recover_false_veto"]:
            decision_rows.append({"cohort":cohort,"agent":a,"rep":rep,"task":t,"state":i,
                                  "rescued":counts["rescued"],"lost":counts["lost"],
                                  "still_false_veto":counts["recover_false_veto"],
                                  "stop_raw_success":counts["stop_false_veto"] if "stop" in diag else "",
                                  "remaining_roles":",".join(m["recover"]["terminal_veto"]),
                                  "review_evidence":json.dumps(revisits(m["recover"]),separators=(",",":"))})
        if counts["recover_false_veto"]:
            final=m["recover"]["reviews"][-1]["roles"]
            reasons[cohort,a].update({str([(role,final[role].get("revisit",{}).get("reason") or final[role].get("reason") or "selector empty or tied") for role in m["recover"]["terminal_veto"]]):1})
    with (OUT/"decisions.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=["cohort","agent","rep","task","state","rescued","lost","still_false_veto","stop_raw_success","remaining_roles","review_evidence"])
        w.writeheader()
        w.writerows(decision_rows)
    with (OUT/"by_task.csv").open("w") as f:
        keys=sorted(set().union(*(v.keys() for v in detail.values())))
        w=csv.DictWriter(f,fieldnames=["cohort","agent","task",*keys])
        w.writeheader()
        for (c,a,t),v in sorted(detail.items()):
            w.writerow({"cohort":c,"agent":a,"task":t,**{k:v[k] for k in keys}})
    output={name:{"/".join(k):v for k,v in src.items()} for name,src in
            [("summary",summary),("by_generation",by_rep),("false_veto_reasons",reasons)]}
    output.update(unique_states=len(tasks)*len(manifest["indices"]),primary_and_archive_episodes=3*len(trios),
                  diagnostic_episodes=sum(len(v) for v in diagnostics.values()), rejections=rejections)
    (OUT/"audit.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output["summary"],indent=1))

if __name__=="__main__":
    main()
