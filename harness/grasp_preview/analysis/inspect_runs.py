"""Readable per-task accounting for geometric decisions and time left when loss was seen."""
import json
from collections import Counter
from pathlib import Path
import sys
from score import records,key,b

def main(stem):
    rows=records(stem)
    groups={arm:{key(r):r for r in rows if r["arm"]==arm} for arm in ("new","old","no_preview","no_release")}
    data={}
    for k,new in groups["new"].items():
        t=new["task"]
        item=data.setdefault(t,dict(events=Counter(),budget=[],example_gain=None,example_loss=None))
        flags=item["events"]
        alt=groups["no_preview"][k]
        old=groups["old"][k]
        flags["unchanged_frozen_old_win"]+=old["stable"] and new["stable"]
        flags["has_preview"]+=bool(new["preview"])
        flags["has_measurements"]+=bool(new["measurements"])
        flags["at_limit"]+=new["steps"]==460
        for e in new["preview"]:
            flags["choose_"+e["action"]]+=1
        for e in new["releases"]:
            if e["loss"]:
                item["budget"].append(dict(remaining=e["remaining"],cost=e.get("retry_cost"),
                        retries_left=e.get("retries_left"),acted=bool(e.get("acted"))))
        for name,qualifies in (("example_gain",new["stable"] and not alt["stable"]),
                              ("example_loss",alt["stable"] and not new["stable"])):
            if qualifies and not item[name] and new["preview"]:
                item[name]=dict(key=new["key"],i=new["i"],first=new["preview"][0],new_steps=new["steps"],alt_steps=alt["steps"])
    (b.OUT/(stem+"_trace_summary.json")).write_text(json.dumps(data,indent=1)+"\n")
if __name__=="__main__":main(sys.argv[1])
