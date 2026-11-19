"""Separate preregistered, two-relation holdout. Reuse the already frozen controllers."""
import argparse
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runner as r
import suite
from suite import public

SPECS=[
    ("libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl",
     {"red_coffee_mug":"butter","porcelain_mug":"milk","chocolate_pudding":"chefmate_8_frypan"},
     "put the butter in the frying pan and put the milk on the plate",
     [("On","butter_1","chefmate_8_frypan_1"),("On","milk_1","plate_1")]),
    ("libero_90/LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.bddl",
     {"cream_cheese":"plate"},
     "put the plate in the tray and put the tomato sauce on the plate",
     [("In","plate_1","wooden_tray_1_contain_region"),("On","tomato_sauce_1","plate_1")])]
def prepare():
    r.freeze()
    old=suite.verify()
    lock=dict(code=r.base.digest(Path(__file__)),harness=r.base.digest(r.OUT/"freeze.json"),specs=json.loads(json.dumps(SPECS)))
    r.base.h.save_lock(r.OUT/"challenge_design.json",lock)
    _,langs,atoms,states=suite.history()
    import numpy as np
    for t in old:
        for state in np.load(suite.ASSETS/(t+".npy")):states.add(public.sha(np.asarray(state,dtype="<f8").tobytes()))
        lang,goals=public.description((suite.ASSETS/(t+".bddl")).read_bytes())
        atoms.update(map(public.atom,goals));langs.add(lang)
    selected=[]
    for path,swap,lang,goals in SPECS:
        src=(suite.UP/path).read_bytes()
        remote=public.get("https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/"+suite.REV+"/libero/libero/bddl_files/"+path)
        assert remote==src
        b=src.decode()
        for orig,new in swap.items():
            assert orig+"_1 - "+orig in b and new+"_1 - "+new not in b
            b=b.replace(orig+"_1 - "+orig,new+"_1 - "+new)
            b=re.sub(r"\b"+orig+"_1"+r"\b",new+"_1",b)
        b=re.sub(r"\(:language[^)]*\)","(:language "+lang+")",b,count=1,flags=re.I)
        b=re.sub(r"\(:obj_of_interest[^)]*\)","(:obj_of_interest "+" ".join(sorted({o for g in goals for o in g[1:]}))+")",b,flags=re.I)
        b=re.sub(r"\(:goal[\s\S]*$","(:goal (And "+" ".join("("+" ".join(g)+")" for g in goals)+")))",b,flags=re.I)
        blob=b.encode()
        got,parsed=public.description(blob)
        key=set(map(public.atom,parsed))
        assert got==lang and len(key)==2 and not key&atoms and lang not in langs and None not in key
        name="relation_"+public.sha(blob)[:14]
        (suite.ASSETS/(name+".bddl")).write_bytes(blob)
        with patch.object(r.base,"OUT",r.OUT),patch.object(public,"ASSETS",suite.ASSETS):
            public.prepare_starts(name,states)
        selected.append(dict(task=name,instruction=lang,source=path,source_sha256=public.sha(src),atoms=sorted(key),
           bddl_sha256=public.sha(blob),states_sha256=r.base.digest(suite.ASSETS/(name+".npy"))))
        atoms.update(key);langs.add(lang)
    r.base.h.save_lock(r.OUT/"challenge_manifest.json",dict(tasks=selected,design=r.base.digest(r.OUT/"challenge_design.json")))

def verify():
    r.freeze()
    design=r.base.h.read(r.OUT/"challenge_design.json")
    assert design["code"]==r.base.digest(Path(__file__)) and design["harness"]==r.base.digest(r.OUT/"freeze.json")
    m=r.base.h.read(r.OUT/"challenge_manifest.json")
    assert m["design"]==r.base.digest(r.OUT/"challenge_design.json")
    for task in m["tasks"]:
        for ext,field in ((".bddl","bddl_sha256"),(".npy","states_sha256")):
            assert r.base.digest(suite.ASSETS/(task["task"]+ext))==task[field]
    return [t["task"] for t in m["tasks"]]

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("prepare","test"))
    p.add_argument("--workers",type=int,default=48)
    args=p.parse_args()
    if args.stage=="prepare":prepare()
    else:
        tasks=verify()
        keys=r.base.h.read(r.OUT/"programs.json")
        jobs=[(k,m,t,list(range(i,i+5))) for k in keys for t in tasks for m in
              (("receipt",) if k.endswith("/receipt") else ("old","early","late")) for i in range(0,50,5)]
        r.deploy(jobs,"challenge",args.workers)
