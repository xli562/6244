"""Independent semantic overlap audit; frozen model and harness reused unchanged."""
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bench as b
from select_tasks import upstream,language

DISTINCT_PRIMARY={
    "turn off the stove",
    "put the yellow and white mug to the front of the white mug",
    "put the red mug on the left plate",
    "put the wine bottle in the bottom drawer of the cabinet",
}
ADDITIONAL=[
    "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
    "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
]

def main(stage):
    b.lock()
    primary=b.h.read(b.OUT/"manifest.json")
    selected={r["task"] for r in primary["tasks"] if r["instruction"] in DISTINCT_PRIMARY}
    assert len(selected)==len(DISTINCT_PRIMARY)
    b.h.save_lock(b.OUT/"semantic_protocol.json",dict(primary=sorted(selected),additional=ADDITIONAL,
                    system=b.h.digest((b.OUT/"system.json").read_text()),script=b.h.digest(Path(__file__).read_text()),
                    criterion="Exclude paraphrases, prior complete subgoals, and generic bowl placements even if color differs. Include new directions, required operations or item/destination pairs."))
    manifest=b.OUT/"additional_manifest.json"
    if stage=="select":
        hist={}
        for p in [*sorted((b.ROOT/"sim/data").glob("libero_*.json")),b.ROOT/"testsets/libero_holdout_v1/manifest.json",
                  b.ROOT/"harness/interaction/manifest.json",b.old.OUT/"manifest.json",b.PRIOR/"manifest.json",b.OUT/"manifest.json"]:
            data=b.h.read(p)
            hist.update({r["task"]:r.get("suite",data.get("suite")) for r in data["tasks"]})
        known={language(s,t) for t,s in hist.items()}
        seen=set()
        for t,s in hist.items():
            seen.update(upstream.state_keys((upstream.UP/"init_files"/s/(t+".pruned_init")).read_bytes()))
        with ThreadPoolExecutor(3) as p:
            rows=list(p.map(upstream.collect,[("libero_90",t) for t in ADDITIONAL]))
        tasks=[]
        for r,keys in rows:
            lang=language(r["suite"],r["task"])
            assert lang not in known and r["task"] not in hist and not seen.intersection(keys) and len(set(keys))==50
            known.add(lang)
            seen.update(keys)
            tasks.append(dict(r,instruction=lang))
        b.h.save_lock(manifest,dict(tasks=tasks,indices=list(range(50)),revision=upstream.REV))
        return
    m=b.h.read(manifest)
    b.old.register(m)
    b.h.bench.prepare()
    if stage=="test":
        jobs=[(k,mode,t["task"],list(range(i,i+5)),False,None) for k in b.programs()
              for mode in (("old","new") if k.endswith("checked") else ("receipt",))
              for t in m["tasks"] for i in range(0,50,5)]
    else:
        with gzip.open(b.OUT/"additional_test.jsonl.gz","rt") as f:
            jobs=[(r["key"],"new",r["task"],[r["i"]],False,w) for r in map(json.loads,f) for w in b.witnesses(r)]
    b.deploy(jobs,32,"additional_"+stage)

if __name__=="__main__":
    main(sys.argv[1])
