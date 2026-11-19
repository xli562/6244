"""Keep both atoms of each derived compound out of the entire archived practice history."""
import gzip
import json
import re
import random
import shutil
from unittest.mock import patch

import numpy as np
from policies import base,prior,OUT
import suite
from suite import public

KITCHEN="libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
BASKET="libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl"
TRAY="libero_90/LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.bddl"
PLATE="libero_goal/put_the_cream_cheese_in_the_bowl.bddl"
R="glazed_rim_porcelain_ramekin"
PAN="chefmate_8_frypan"
TASKS=[
 (KITCHEN,{"akita_black_bowl":R,"plate":PAN},"put the ramekin in the frying pan",
  [("On",R+"_1",PAN+"_1")]),
 (BASKET,{"cream_cheese":"popcorn"},"put the popcorn in the basket",
  [("In","popcorn_1","basket_1_contain_region")]),
 (PLATE,{"cream_cheese":"alphabet_soup","wine_bottle":"butter"},"put the alphabet soup in the bowl and put the butter on the plate",
  [("On","alphabet_soup_1","akita_black_bowl_1"),("On","butter_1","plate_1")]),
 (TRAY,{"alphabet_soup":R,"cream_cheese":"chocolate_pudding","tomato_sauce":"orange_juice"},
  "put the chocolate pudding in the ramekin and put the orange juice in the tray",
  [("On","chocolate_pudding_1",R+"_1"),("In","orange_juice_1","wooden_tray_1_contain_region")]),
 (TRAY,{"alphabet_soup":PAN},"put the frying pan in the tray and put the cream cheese in the frying pan",
  [("In",PAN+"_1","wooden_tray_1_contain_region"),("On","cream_cheese_1",PAN+"_1")]),
 (BASKET,{"alphabet_soup":R,"cream_cheese":"butter"},"put the ramekin in the basket and put the butter in the ramekin",
  [("In",R+"_1","basket_1_contain_region"),("On","butter_1",R+"_1")]),
]

def canon(atom):
    if not atom:return None
    alias={"ramekin":"bowl","chefmate":"frypan","new_salad_dressing":"salad_dressing"}
    return tuple(alias.get(x,x) for x in atom)

def past():
    n,langs,atoms,states=suite.history()
    for file in ("manifest.json","challenge_manifest.json"):
        for row in base.h.read(prior.OUT/file)["tasks"]:
            n+=1
            langs.add(row["instruction"])
            atoms.update(map(tuple,row["atoms"]))
            for state in np.load(prior.OUT/"assets"/(row["task"]+".npy")):
                states.add(public.sha(np.asarray(state,dtype="<f8").tobytes()))
    for archive,stem in [(prior.OUT,"dev"),(prior.OUT,"test"),(prior.OUT,"challenge"),(OUT,"dev_locked")]:
        with gzip.open(archive/(stem+".jsonl.gz"),"rt") as f:
            for ep in map(json.loads,f):
                for e in ep["measurements"]:
                    b=e["bound"]
                    if b.get("item") and b.get("target"):
                        atoms.add(public.atom(["on",b["item"],b["target"]]))
    return n,langs,{canon(a) for a in atoms if a},states

def starts(name,seen,assets):
    seed=int(public.sha(name.encode())[:8],16)
    data=[]
    np.random.seed(seed)
    random.seed(seed)
    with patch.object(base,"OUT",OUT),patch.object(base.h.bench,"CONFIG",assets),patch.object(base.h.bench,"paths",base.file_paths):
        env=base.h.bench.environment(name,extra_steps=40)
        try:
            for i in range(300):
                np.random.seed((seed+i)%2**32)
                random.seed(seed+i)
                env.reset()
                state=env.get_sim_state().copy()
                key=public.sha(np.asarray(state,dtype="<f8").tobytes())
                if key in seen or not np.isfinite(state).all():continue
                env.set_init_state(state)
                for _ in range(5):env.step([0,0,0,0,0,0,-1])
                if any(env.env._eval_predicate(g) for g in env.env.parsed_problem["goal_state"]):continue
                seen.add(key)
                data.append(state)
                if len(data)==50:break
        finally:env.close()
    if len(data)!=50:raise ValueError("Fewer than 50 unique, fully unsolved starts")
    np.save(assets/(name+".npy"),np.asarray(data),allow_pickle=False)

def select():
    assets=OUT/"assets"
    assets.mkdir(exist_ok=True)
    shutil.copyfile(base.h.bench.CONFIG/"config.yaml",assets/"config.yaml")
    n,langs,atoms,states=past()
    initial_atoms=sorted(atoms)
    initial_states=len(states)
    selected=[]
    for path,swaps,lang,goals in TASKS:
        url="https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/"+suite.REV+"/libero/libero/bddl_files/"+path
        original=(suite.UP/path).read_bytes()
        assert public.get(url)==original
        src=original.decode()
        for old,new in swaps.items():
            assert old+"_1 - "+old in src and new+"_1 - "+new not in src
            src=src.replace(old+"_1 - "+old,new+"_1 - "+new)
            src=re.sub(r"\b"+old+r"_1\b",new+"_1",src)
        src=re.sub(r"\(:language[^)]*\)","(:language "+lang+")",src,count=1,flags=re.I)
        src=re.sub(r"\(:obj_of_interest[^)]*\)","(:obj_of_interest "+" ".join(sorted({o for g in goals for o in g[1:]}))+")",src,flags=re.I)
        src=re.sub(r"\(:goal[\s\S]*$","(:goal (And "+" ".join("("+" ".join(g)+")" for g in goals)+")))",src,flags=re.I)
        data=src.encode()
        parsed,goals=public.description(data)
        keys=set(canon(public.atom(g)) for g in goals)
        assert parsed==lang and len(keys)==len(goals) and None not in keys
        if keys&atoms or lang in langs:
            continue
        name="review_"+public.sha(data)[:14]
        (assets/(name+".bddl")).write_bytes(data)
        starts(name,states,assets)
        selected.append(dict(task=name,instruction=lang,atoms=sorted(keys),url=url,source_sha256=public.sha(original),
                   bddl_sha256=public.sha(data),states_sha256=base.digest(assets/(name+".npy"))))
        atoms.update(keys)
        langs.add(lang)
    assert len(selected)>=4
    base.h.save_lock(OUT/"manifest.json",dict(tasks=selected,historical_tasks=n,historical_states=initial_states,
        historical_atoms=initial_atoms,revision=suite.REV,freeze=base.digest(OUT/"freeze.json")))

def verify():
    m=base.h.read(OUT/"manifest.json")
    assert m["freeze"]==base.digest(OUT/"freeze.json")
    for row in m["tasks"]:
        for ext,key in ((".bddl","bddl_sha256"),(".npy","states_sha256")):
            assert base.digest(OUT/"assets"/(row["task"]+ext))==row[key]
    return [r["task"] for r in m["tasks"]]
