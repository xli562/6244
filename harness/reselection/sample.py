"""Select only after freezing. Compare task IDs, original English and all prior state hashes."""
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from generate import ROOT, old
from select_tasks import upstream, language

def draw():
    history = {}
    for p in [*sorted((ROOT/"sim/data").glob("libero_*.json")),
              ROOT/"testsets/libero_holdout_v1/manifest.json", ROOT/"harness/interaction/manifest.json",
              old.OUT/"manifest.json"]:
        data = old.h.read(p)
        history.update({r["task"]:r.get("suite",data.get("suite")) for r in data["tasks"]})
    used = {language(s,t) for t,s in history.items()}
    pool = []
    for p in (upstream.UP/"bddl_files/libero_90").glob("*.bddl"):
        t = p.stem
        lang = language("libero_90",t)
        if t in history or lang in used:
            continue
        spatial = re.search(r"\b(left|right|front|back|middle|between|near|next|behind)\b",lang)
        group = "spatial" if spatial else "composed" if " and " in lang or "both " in lang else "other"
        rank = hashlib.sha256(("reread-v1/"+t).encode()).hexdigest()
        pool.append((rank,t,lang,group))
    chosen=[]
    for group,n in [("spatial",6),("composed",3),("other",3),(None,12)]:
        for _,t,lang,g in sorted(pool):
            if lang in used or group and group!=g:
                continue
            chosen.append((t,lang,g))
            used.add(lang)
            n-=1
            if not n or len(chosen)==12:
                break
        if len(chosen)==12:
            break
    prior=set()
    for t,s in history.items():
        prior.update(upstream.state_keys((upstream.UP/"init_files"/s/(t+".pruned_init")).read_bytes()))
    with ThreadPoolExecutor(6) as workers:
        data=list(workers.map(upstream.collect,[("libero_90",t) for t,_,_ in chosen]))
    keys=[v for _,states in data for v in states]
    if len(chosen)!=12 or len(set(keys))!=600 or set(keys)&prior:
        raise ValueError("Task count, state count, or historical independence check failed")
    return {"revision":upstream.REV,"source":upstream.URL,"excluded_tasks":len(history),"excluded_states":len(prior),
            "indices":list(range(50)),"tasks":[dict(r,instruction=lang,stratum=g) for (r,_),(_,lang,g) in zip(data,chosen)]}
