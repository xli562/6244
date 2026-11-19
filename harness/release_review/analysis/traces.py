"""Separate tasks that reached the relevant motor skill from plans that never reached it."""
import json
from collections import Counter
import attribution as audit
from fresh import public
r=audit.runner

def initial_choice(new,old):
    for n,o in zip(new["decisions"],old["decisions"]):
        if n["event"]!=o["event"]:break
        if n["decision"]=="continue" and o["parent"] in ("regrasp","align"):
            return n["event"]["reason"]+"_"+n["event"]["item"]
        if n["parent"]!=o["parent"]:break
    return None

def main(stem):
    rows=audit.records(stem)
    keyed={(row["key"],row["task"],row["i"],row["arm"]):row for row in rows}
    counts={}
    for row in rows:
        if row["arm"]!="new":continue
        k,t,i=audit.key(row)
        s=counts.setdefault(t,Counter())
        asset=r.prior.OUT if stem=="dev_locked" else r.OUT
        _,goals=public.description((asset/"assets"/(t+".bddl")).read_bytes())
        def matches(g,e):
            b=e["bound"]
            return e["op"]=="place" and g[1]==b.get("item") and (g[2]==b.get("target") or
                    g[2].startswith(str(b.get("target"))+"_") or str(b.get("target")).startswith(g[2]+"_"))
        all_bound=all(any(matches(g,e) for e in row["measurements"]) for g in goals)
        s.update(all_expected_pairs_attempted=all_bound,n=1,at_limit=row["steps"]==460,
                 failed_before_all_pairs=not all_bound and not row["stable"],no_placement=not row["measurements"],
                 gap_alerts=sum(e["reason"]=="drifted" for e in row["releases"] if e["loss"]),
                 contact_loss_alerts=sum(e["reason"]=="lost_support" and e["previous_support"] for e in row["releases"] if e["loss"]))
        old=keyed[k,t,i,"old"]
        no=keyed[k,t,i,"no_release"]
        ref=keyed[k,t,i,"continue"]
        if old["stable"] and not no["stable"]:
            reason=initial_choice(no,old)
            s["old_win_lost_after_"+str(reason)]+=1
        for event in row["releases"]:
            if event["loss"]:
                s["release_alert_events"]+=1
                s["alert_no_corrections_left"]+=event["corrections_used"]>=2
                s["alert_under_120_steps"]+=event["remaining"]<120
                s["alert_no_prior_contact"]+=not event["previous_support"]
        if not old["stable"] and no["stable"] and ref["stable"]:
            reason=initial_choice(no,old)
            s["restored_with_identical_prefix_"+str(reason)]+=1
    (r.OUT/(stem+"_traces.json")).write_text(json.dumps(counts,indent=1)+"\n")
if __name__=="__main__":
    import sys
    main(sys.argv[1])
