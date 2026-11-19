"""Build held-out placement tasks on versioned public LIBERO physics scenes."""
import gzip
import itertools
import json
import re
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
from programs import base, OUT, ROOT
import holdout as public

ASSETS=OUT/"assets"
UP=public.upstream.UP/"bddl_files"
REV="8f1084e3132a39270c3a13ebe37270a43ece2a01"
DONORS=("alphabet_soup","butter","chocolate_pudding","cream_cheese","ketchup","milk",
        "new_salad_dressing","orange_juice","tomato_sauce","porcelain_mug","akita_black_bowl","moka_pot")
SLOTS=("basket:2","tray:1","bowl:1","plate:1","stove:1","cabinet:1","rack:1","plate:2")

def history():
    tasks,langs,atoms,states=public.history()
    for manifest in (ROOT/"runs").glob("**/*manifest.json"):
        data=json.loads(manifest.read_text())
        if not isinstance(data,dict):continue
        for r in data.get("tasks",[]):
            if not isinstance(r,dict):continue
            t=r.get("task")
            if not t or t in tasks:continue
            paths=list(UP.glob("*/"+t+".bddl"))
            if not paths:continue
            path=paths[0]
            lang,goals=public.description(path.read_bytes())
            tasks[t]=path.parent.name
            langs.add(lang)
            atoms.update(map(public.atom,goals))
            init=UP.parent/"init_files"/path.parent.name/(t+".pruned_init")
            states.update(public.upstream.state_keys(init.read_bytes()))
    def collect(record):
        if isinstance(record,dict):
            for k,v in record.items():
                if k=="bound" and isinstance(v,dict) and v.get("item") and v.get("target"):
                    atoms.add(public.atom(["on",v["item"],v["target"]]))
                elif k=="trace" and isinstance(v,list):
                    for row in v:
                        if isinstance(row,list) and len(row)>=3 and all(isinstance(x,str) for x in row[:3]) and row[2] in ("lift_m","requested_grasp","placed_xy_error_m"):
                            atoms.add(public.atom(["on",row[0],row[1]]))
                else:collect(v)
        elif isinstance(record,list):
            for v in record:collect(v)
    for directory in ("preflight_01","interaction_01","composition_01","libero_holdout_v1"):
        for p in (ROOT/"runs"/directory).rglob("*.json"):
            collect(json.loads(p.read_text()))
    for r in base.h.read(base.OUT/"manifest.json")["tasks"]:
        langs.add(r["instruction"])
        atoms.update(map(tuple,r["atoms"]))
        for state in np.load(base.OUT/"assets"/(r["task"]+".npy")):
            states.add(public.sha(np.asarray(state,dtype="<f8").tobytes()))
    for p in [base.OUT/"test.jsonl.gz",base.OUT/"dev.jsonl.gz",OUT/"dev.jsonl.gz"]:
        with gzip.open(p,"rt") as f:
            for episode in map(json.loads,f):
                for event in episode["measurements"]:
                    b=event["bound"]
                    if b.get("item") and b.get("target"):
                        atoms.add(public.atom(["on",b["item"],b["target"]]))
    return len(tasks)+len(base.h.read(base.OUT/"manifest.json")["tasks"]),langs,atoms,states

def candidates(dst,n):
    result=[]
    for p in sorted(UP.glob("*/*.bddl")):
        if p.parent.name not in ("libero_90","libero_goal","libero_10","libero_object","libero_spatial"): continue
        raw=p.read_bytes()
        _,goals=public.description(raw)
        if len(goals)!=n or any(len(g)!=3 or g[0] not in ("on","in") or public.category(g[2])!=dst for g in goals): continue
        if dst=="cabinet" and any("top_side" not in g[2] for g in goals):continue
        if any("_region" in g[2] and dst=="plate" for g in goals):continue
        src=raw.decode()
        nodes={node[0]:node[1:] for node in public.tree(src) if isinstance(node,list) and node}
        targets=[name for tag in (":objects",":fixtures") for name in nodes[tag]
                 if re.search(r"_\d+$",name) and public.category(name)==dst]
        if len(targets)>1 and not (dst=="plate" and n==2 and len(set(g[2] for g in goals))==2): continue
        ids=[g[1] for g in goals]
        if len(set(ids))!=n:continue
        # Keep the donor spawn and destination geometry of the public task.
        if not all(re.search(r"^\s*"+i+" - "+i.rsplit("_",1)[0]+r"\s*$",src,re.M) for i in ids):continue
        for donors in itertools.permutations(DONORS,n):
            if any(re.search(r"\b"+d+r"_\d+\b",src) for d in donors):continue
            changed=src
            clauses=[]
            for (_,i,t),d in zip(goals,donors):
                changed=re.sub(r"\b"+i+" - "+i.rsplit("_",1)[0]+r"\b",d+"_1 - "+d,changed)
                changed=re.sub(r"\b"+i+r"\b",d+"_1",changed)
                phrase={"rack":"on the rack","tray":"in the tray","bowl":"in the bowl","basket":"in the basket",
                        "cabinet":"on top of the cabinet","stove":"on the stove","plate":"on the plate"}[dst]
                if dst=="plate" and len(set(g[2] for g in goals))>1:
                    orig=public.description(raw)[0]
                    directions=re.findall(r"on the (left|right) plate",orig)
                    if len(directions)!=n:break
                    phrase="on the "+directions[len(clauses)]+" plate"
                clauses.append("put the "+d.replace("new_","").replace("akita_","").replace("_"," ")+" "+phrase)
            if len(clauses)!=n:continue
            lang=" and ".join(clauses)
            changed=re.sub(r"\(:language[^)]*\)","(:language "+lang+")",changed,count=1,flags=re.I)
            blob=changed.encode()
            key=tuple(sorted(map(public.atom,public.description(blob)[1])))
            ref=str(p.relative_to(UP))+"|"+"+".join(donors)
            result.append((public.sha(("stage-holdout-v1/"+ref).encode()),p,blob,key,lang,donors))
    return sorted(result,key=lambda r:r[0])

def select():
    assert subprocess.check_output(["git","-C",str(UP),"rev-parse","HEAD"],text=True).strip()==REV
    ASSETS.mkdir(exist_ok=True)
    shutil.copyfile(base.h.bench.CONFIG/"config.yaml",ASSETS/"config.yaml")
    count,langs,atoms,states=history()
    initial_states=len(states)
    initial_atoms=sorted(a for a in atoms if a)
    used=set()
    selected=[]
    exclusions=[]
    for spec in SLOTS:
        dst,num=spec.split(":")
        pool=candidates(dst,int(num))
        # Prefer previously unused source classes across this test, then fixed hash.
        pool.sort(key=lambda r:(len(set(r[5])&used),r[0]))
        for signature,p,blob,key,lang,donors in pool:
            if set(key)&atoms or len(set(key))!=len(key) or lang in langs:continue
            name="stage_"+signature[:14]
            path=ASSETS/(name+".bddl")
            path.write_bytes(blob)
            try:
                with patch.object(base,"OUT",OUT),patch.object(public,"ASSETS",ASSETS):
                    public.prepare_starts(name,states)
            except Exception as exc:
                exclusions.append((name,str(exc)[:180]))
                path.unlink()
                continue
            selected.append(dict(task=name,source=str(p.relative_to(UP)),source_sha256=public.sha(p.read_bytes()),
                                 donors=donors,instruction=lang,atoms=key,bddl_sha256=public.sha(blob),
                                 states_sha256=public.sha((ASSETS/(name+".npy")).read_bytes())))
            atoms.update(key)
            langs.add(lang)
            used.update(donors)
            break
    if len(selected)<4:raise RuntimeError("insufficient non-overlapping placements")
    base.h.save_lock(OUT/"manifest.json",dict(revision=REV,origin="LIBERO-derived",tasks=selected,
                 historical_tasks=count,historical_states=initial_states,historical_atoms=initial_atoms,reset_failures=exclusions))
    base.h.save_lock(OUT/"selection.json",{p:base.digest(OUT/p) for p in ("manifest.json","freeze.json")})

def verify():
    with patch.object(public,"OUT",OUT),patch.object(public,"ASSETS",ASSETS):
        return public.verify()
