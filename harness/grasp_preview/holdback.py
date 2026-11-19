"""Preregister new placement graphs on pinned public scenes; exclude every practiced edge."""
import gzip
import json
from unittest.mock import patch
import numpy as np
from generate_candidates import OUT,previous,base
import fresh
from fresh import PAN,R,BASKET,TRAY,PLATE,public

SHELF="libero_90/KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf.bddl"
CABINET="libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl"
S=fresh.suite
TASKS=[
 (PLATE,{"cream_cheese":"red_coffee_mug"},"nest the red coffee mug in the bowl",
  [("On","red_coffee_mug_1","akita_black_bowl_1")]),
 (SHELF,{PAN:"porcelain_mug"},"put the porcelain mug on the cabinet shelf",
  [("In","porcelain_mug_1","wooden_two_layer_shelf_1_top_region")]),
 (CABINET,{"akita_black_bowl":PAN},"put the frying pan on top of the cabinet",
  [("On",PAN+"_1","wooden_cabinet_1_top_side")]),
 (BASKET,{"cream_cheese":PAN,"alphabet_soup":"bbq_sauce"},
  "put the frying pan in the basket and put the bbq sauce in the frying pan",
  [("In",PAN+"_1","basket_1_contain_region"),("On","bbq_sauce_1",PAN+"_1")]),
 (PLATE,{"cream_cheese":"cookies","wine_bottle":"macaroni_and_cheese"},
  "put the cookies in the bowl and put the macaroni and cheese on the plate",
  [("On","cookies_1","akita_black_bowl_1"),("On","macaroni_and_cheese_1","plate_1")]),
 (TRAY,{"alphabet_soup":"porcelain_mug","cream_cheese":"popcorn"},
  "put the porcelain mug and the popcorn in the tray",
  [("In","porcelain_mug_1","wooden_tray_1_contain_region"),("In","popcorn_1","wooden_tray_1_contain_region")]),
 (TRAY,{"alphabet_soup":"moka_pot","cream_cheese":PAN,"tomato_sauce":"orange_juice","ketchup":"cookies"},
  "put the moka pot in the tray and put the cookies in the tray and put the orange juice in the frying pan",
  [("In","moka_pot_1","wooden_tray_1_contain_region"),("In","cookies_1","wooden_tray_1_contain_region"),("On","orange_juice_1",PAN+"_1")]),
]

def select():
    n,langs,atoms,states=fresh.past()
    for row in base.h.read(previous.OUT/"manifest.json")["tasks"]:
        n+=1
        langs.add(row["instruction"])
        atoms.update(tuple(a) for a in row["atoms"])
        for s in np.load(previous.OUT/"assets"/(row["task"]+".npy")):
            states.add(public.sha(np.asarray(s,dtype="<f8").tobytes()))
    for directory,stem in [(previous.OUT,"test"),(OUT,"dev_final")]:
        with gzip.open(directory/(stem+".jsonl.gz"),"rt") as f:
            for ep in map(json.loads,f):
                for event in ep["measurements"]:
                    b=event["bound"]
                    if b.get("item") and b.get("target"):atoms.add(fresh.canon(public.atom(["on",b["item"],b["target"]])))
    with patch.object(fresh,"OUT",OUT),patch.object(fresh,"TASKS",TASKS),patch.object(fresh,"past",lambda:(n,langs,atoms,states)):
        fresh.select()

def verify():
    with patch.object(fresh,"OUT",OUT):return fresh.verify()
