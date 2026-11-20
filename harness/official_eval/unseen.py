"""Generate a fresh batch of unseen placement tasks for the frozen newest system.

A task is unseen when, against the project's entire archived history (every manifest, practice and
test log up to and including runs/preview_01):
  * its goal atom (object category -> destination category, after the project's aliases) is new,
  * its English instruction is new, and
  * every one of its 50 starting states is new and starts unsolved.
Tasks are built the way earlier rounds built theirs (harness/release_review/fresh.py): swap one
movable object in a pinned stock LIBERO scene for a different stock object and set a single goal.
Candidates are ordered by a fixed hash, never by policy performance; at most PER_SCENE per scene.

    python harness/official_eval/unseen.py select
"""
import gzip, hashlib, json, re, shutil, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness/grasp_preview"))
import benchmark as latest                       # loads the full harness chain onto sys.path
import fresh                                     # release_review task builder (scene rewrite + starts)
import holdback                                  # grasp_preview's history extension

base, public = fresh.base, fresh.public
OUT = ROOT / "runs/unseen_eval_01"
ASSETS = OUT / "assets"
SALT = "unseen-eval-01/"
N_TASKS, PER_SCENE = 12, 2

# (stock scene, movable object to replace, destination goal term, relation, destination phrase)
SCENES = [
    ("libero_goal/put_the_cream_cheese_in_the_bowl.bddl", "cream_cheese", "akita_black_bowl_1", "On", "in the bowl"),
    ("libero_goal/put_the_cream_cheese_in_the_bowl.bddl", "cream_cheese", "plate_1", "On", "on the plate"),
    ("libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl",
     "cream_cheese", "basket_1_contain_region", "In", "in the basket"),
    ("libero_90/LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.bddl",
     "alphabet_soup", "wooden_tray_1_contain_region", "In", "in the tray"),
    ("libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl", "akita_black_bowl", "plate_1", "On", "on the plate"),
    ("libero_90/KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf.bddl", "chefmate_8_frypan",
     "wooden_two_layer_shelf_1_top_region", "In", "on the cabinet shelf"),
    ("libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl", "akita_black_bowl", "wooden_cabinet_1_top_side", "On",
     "on top of the cabinet"),
]
# Stock LIBERO objects already loaded successfully by earlier rounds.
DONORS = ["alphabet_soup", "bbq_sauce", "butter", "chocolate_pudding", "cookies", "cream_cheese", "ketchup",
          "macaroni_and_cheese", "milk", "new_salad_dressing", "orange_juice", "popcorn", "tomato_sauce",
          "porcelain_mug", "red_coffee_mug", "moka_pot", "akita_black_bowl", "glazed_rim_porcelain_ramekin",
          "chefmate_8_frypan"]
PHRASE = {"akita_black_bowl": "black bowl", "glazed_rim_porcelain_ramekin": "ramekin",
          "chefmate_8_frypan": "frying pan", "new_salad_dressing": "salad dressing",
          "macaroni_and_cheese": "macaroni and cheese"}


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def history():
    """grasp_preview's history (everything through release_01 + preview dev) plus preview_01's test round."""
    n, langs, atoms, states = fresh.past()
    for directory in (latest.previous.OUT, latest.OUT):           # release_01, preview_01
        for row in base.h.read(directory / "manifest.json")["tasks"]:
            n += 1
            langs.add(row["instruction"])
            atoms.update(tuple(a) for a in row["atoms"])
            for s in np.load(directory / "assets" / (row["task"] + ".npy")):
                states.add(public.sha(np.asarray(s, dtype="<f8").tobytes()))
    for p in [latest.previous.OUT / "test.jsonl.gz", *sorted(latest.OUT.glob("*.jsonl.gz"))]:
        with gzip.open(p, "rt") as f:
            for ep in map(json.loads, f):
                for event in ep.get("measurements", []):
                    b = event.get("bound", {})
                    if b.get("item") and b.get("target"):
                        atoms.add(fresh.canon(public.atom(["on", b["item"], b["target"]])))
    return n, langs, {a for a in atoms if a}, states


def candidates():
    rows = []
    for path, old, dest, rel, where in SCENES:
        src = (fresh.suite.UP / path).read_text()
        for donor in DONORS:
            if donor == old or f"{donor}_1 - {donor}" in src or dest.startswith(donor):
                continue
            lang = f"put the {PHRASE.get(donor, donor.replace('_', ' '))} {where}"
            rows.append((sha(SALT + path + donor + dest), path, {old: donor}, lang, [(rel, donor + "_1", dest)]))
    return sorted(rows)


def select():
    if (OUT / "manifest.json").exists():
        raise FileExistsError(OUT / "manifest.json")
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "runs/official_eval_01/assets/config.yaml", ASSETS / "config.yaml")   # local paths
    n, langs, atoms, states = history()
    hist_atoms, hist_states = sorted(atoms), len(states)
    per_scene, selected, audit = {}, [], {}
    for _, path, swaps, lang, goals in candidates():
        if len(selected) == N_TASKS:
            break
        if per_scene.get(path, 0) >= PER_SCENE:
            continue
        src = (fresh.suite.UP / path).read_text()
        for old, new in swaps.items():
            src = src.replace(old + "_1 - " + old, new + "_1 - " + new)
            src = re.sub(r"\b" + old + r"_1\b", new + "_1", src)
        src = re.sub(r"\(:language[^)]*\)", "(:language " + lang + ")", src, count=1, flags=re.I)
        src = re.sub(r"\(:obj_of_interest[^)]*\)", "(:obj_of_interest " + " ".join(sorted({o for g in goals for o in g[1:]})) + ")", src, flags=re.I)
        src = re.sub(r"\(:goal[\s\S]*$", "(:goal (And " + " ".join("(" + " ".join(g) + ")" for g in goals) + ")))", src, flags=re.I)
        data = src.encode()
        parsed, parsed_goals = public.description(data)
        keys = {fresh.canon(public.atom(g)) for g in parsed_goals}
        tag = f"{path}:{swaps}"
        if None in keys or parsed != lang:
            audit[tag] = "unparsable"; continue
        if keys & atoms:
            audit[tag] = f"seen goal atom {sorted(keys & atoms)}"; continue
        if lang in langs:
            audit[tag] = "seen instruction"; continue
        name = "unseen_" + public.sha(data)[:14]
        (ASSETS / (name + ".bddl")).write_bytes(data)
        try:
            with patch.object(fresh, "OUT", OUT):
                fresh.starts(name, states, ASSETS)
        except Exception as exc:                                  # object fails to load or no unsolved starts
            audit[tag] = "simulator: " + str(exc)[:160]
            (ASSETS / (name + ".bddl")).unlink()
            continue
        selected.append(dict(task=name, instruction=lang, scene=path, swaps=swaps, atoms=sorted(keys),
                             bddl_sha256=public.sha(data), states_sha256=base.digest(ASSETS / (name + ".npy"))))
        print(f"selected {name}: {lang}  [{path.split('/')[-1][:50]}]", flush=True)
        atoms |= keys
        langs.add(lang)
        per_scene[path] = per_scene.get(path, 0) + 1
    base.h.dump(OUT / "audit.json", audit)
    base.h.save_lock(OUT / "manifest.json", dict(tasks=selected, historical_tasks=n, historical_states=hist_states,
                                                historical_atoms=hist_atoms, rule=__doc__.split("\n\n")[0]))
    print(f"{len(selected)} unseen tasks; {len(audit)} candidates rejected")


def tasks():
    m = base.h.read(OUT / "manifest.json")
    for row in m["tasks"]:
        for ext, key in ((".bddl", "bddl_sha256"), (".npy", "states_sha256")):
            assert base.digest(ASSETS / (row["task"] + ext)) == row[key], row["task"]
    return m["tasks"]


if __name__ == "__main__":
    if sys.argv[1:] != ["select"]:
        sys.exit(__doc__)
    select()
