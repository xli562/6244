"""Select independent LIBERO-Pro goals, then create paired starts with the stock simulator."""
import gzip
import hashlib
import json
import random
import re
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import numpy as np

from compile_policies import ROOT,OUT,BASE,old,previous
from select_tasks import upstream

REV="c86fc3b8293185a6f373677018ff3e37f8391602"
REPO="zhouxueyang/LIBERO-Pro"
URL=f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/"
ASSETS=OUT/"assets"
CACHE=OUT/"public_pool"
N_TASKS=8

def sha(b):
    return hashlib.sha256(b).hexdigest()

def get(url):
    for i in range(4):
        try:
            with urllib.request.urlopen(url,timeout=45) as r:
                return r.read()
        except OSError:
            if i==3: raise
            time.sleep(i+1)

def tree(s):
    stack=[[]]
    for tok in re.findall(r"\(|\)|[^\s()]+",re.sub(r";[^\n]*","",s.lower())):
        if tok=="(":
            node=[]
            stack[-1].append(node)
            stack.append(node)
        elif tok==")":
            stack.pop()
        else:
            stack[-1].append(tok)
    if len(stack)!=1: raise ValueError("unbalanced public BDDL")
    return stack[0][0]

def description(b):
    nodes={e[0]:e[1:] for e in tree(b.decode()) if isinstance(e,list) and e}
    lang=" ".join(nodes[":language"])
    goal=nodes[":goal"][0]
    return lang,(goal[1:] if goal[0]=="and" else [goal])

def category(name):
    s=name.lower().split("_init_region")[0]
    s=re.sub(r"^.*?table_", "",s)
    s=re.split(r"_\d+(?:_|$)",s)[0]
    for cls in ("cabinet","drawer","microwave","stove","basket","caddy","shelf","rack","tray","plate","bowl","mug","cup","bottle","frying_pan","frypan","moka_pot","book","ramekin","soup"):
        if cls in s:
            return {"drawer":"cabinet","cup":"mug","frying_pan":"frypan"}.get(cls,cls)
    return re.sub(r"_(region|slot|side)$","",s)

def atom(g):
    if not isinstance(g,list) or not all(isinstance(x,str) for x in g):
        return None
    verb=g[0].lower()
    if verb in ("on","in") and len(g)==3:
        return ("place",category(g[1]),category(g[2]))
    if verb in ("open","close","turnon","turnoff") and len(g)==2:
        return (verb,category(g[1]))
    return None

def history():
    past={}
    for p in [*sorted((ROOT/"sim/data").glob("libero_*.json")),ROOT/"testsets/libero_holdout_v1/manifest.json",
              ROOT/"harness/interaction/manifest.json",old.OUT/"manifest.json",previous.PRIOR/"manifest.json",
              BASE/"manifest.json",BASE/"additional_manifest.json"]:
        d=old.h.read(p)
        past.update({r["task"]:r.get("suite",d.get("suite")) for r in d["tasks"]})
    langs=set()
    atoms=set()
    states=set()
    for task,suite in past.items():
        lang,goals=description((upstream.UP/"bddl_files"/suite/(task+".bddl")).read_bytes())
        langs.add(lang)
        atoms.update(map(atom,goals))
        states.update(upstream.state_keys((upstream.UP/"init_files"/suite/(task+".pruned_init")).read_bytes()))
    for p in [BASE/"practice.jsonl.gz",BASE/"test.jsonl.gz",BASE/"additional_test.jsonl.gz",
              previous.PRIOR/"test.jsonl.gz",previous.PRIOR/"practice.jsonl.gz"]:
        with gzip.open(p,"rt") as f:
            for r in map(json.loads,f):
                for e in r.get("actions",r.get("acts",[])):
                    bound=e.get("bound",{})
                    if bound.get("item") and bound.get("target"):
                        atoms.add(("place",category(bound["item"]),category(bound["target"])))
    return past,langs,atoms,states

def public_pool():
    CACHE.mkdir(exist_ok=True)
    meta=json.loads(get(f"https://huggingface.co/api/datasets/{REPO}/revision/{REV}"))
    assert meta["sha"]==REV
    paths=sorted(x["rfilename"] for x in meta["siblings"] if re.match(r"bddl_files/libero_(?:10|spatial|goal|object)_(?:task|object)/",x["rfilename"]) and x["rfilename"].endswith(".bddl"))
    def fetch(p):
        dst=CACHE/(sha(p.encode())+".bddl")
        if not dst.exists():
            dst.write_bytes(get(URL+urllib.parse.quote(p)))
        return p,dst.read_bytes()
    with ThreadPoolExecutor(6) as ex:
        return list(ex.map(fetch,paths))

def prepare_starts(name,official_history):
    from experiment_runner import file_paths,h
    states=[]
    seed=int(sha(name.encode())[:8],16)
    with patch.object(h.bench,"paths",file_paths),patch.object(h.bench,"CONFIG",ASSETS):
        np.random.seed(seed)
        random.seed(seed)
        env=h.bench.environment(name,extra_steps=40)
        try:
            seen=set()
            for i in range(250):
                np.random.seed((seed+i)%2**32)
                random.seed(seed+i)
                env.reset()
                data=env.get_sim_state()
                key=sha(np.asarray(data,dtype="<f8").tobytes())
                if env.check_success() or key in seen or key in official_history or not np.isfinite(data).all(): continue
                states.append(data.copy())
                seen.add(key)
                if len(states)==50: break
        finally:
            env.close()
    if len(states)!=50: raise ValueError("insufficient unique unsolved starts")
    np.save(ASSETS/(name+".npy"),np.array(states),allow_pickle=False)
    official_history.update(seen)

def novel_substitutions():
    """Fallback only when fewer than eight distinct PRO tasks work in stock LIBERO.
    Use a real receiving surface/region from the public goal and a different co-present object.
    """
    result=[]
    for path in sorted((upstream.UP/"bddl_files/libero_90").glob("*.bddl")):
        data=path.read_bytes()
        _,goals=description(data)
        if len(goals)!=1 or len(goals[0])!=3: continue
        rel,source,target=goals[0]
        if rel not in ("in","on"): continue
        dst=category(target)
        if dst not in ("basket","tray","plate","cabinet"): continue
        src=data.decode()
        objects=re.search(r"\(:objects\s+(.*?)\)",src,re.S|re.I)[1]
        for name in re.findall(r"\b[a-z_]+_\d+\b",objects):
            if name in (source,target) or target.startswith(name+"_"): continue
            obj=category(name)
            if obj in ("basket","tray","cabinet","shelf","rack","plate","stove","microwave"): continue
            if rel=="in" and obj in ("frypan","moka_pot","bottle","book"): continue
            if len(re.findall(r"\b"+re.escape(name.rsplit("_",1)[0])+r"_\d+\b",objects))!=1: continue
            item=name.rsplit("_",1)[0].replace("_"," ").replace("akita ","")
            if dst=="cabinet":
                part=next((x for x in ("top","middle","bottom") if x in target.split("_")),None)
                if not part: continue
                destination="the "+part+" drawer of the cabinet" if rel=="in" else "top of the cabinet"
            else:
                destination="the "+dst
            lang=f"put the {item} {rel} {destination}"
            new=re.sub(r"\(:language[^)]*\)","(:language "+lang+")",src,count=1,flags=re.I)
            new=re.sub(r"\(:goal[\s\S]*$",f"(:goal (And ({rel} {name} {target}))))",new,flags=re.I)
            result.append((f"derived/{path.name}/{name}",new.encode(),lang,frozenset([atom([rel,name,target])])))
    return sorted(result,key=lambda r:sha(("derived-01/"+r[0]).encode()))

def select():
    ASSETS.mkdir(exist_ok=True)
    shutil.copyfile(old.h.bench.CONFIG/"config.yaml",ASSETS/"config.yaml")
    past,langs,seen,states=history()
    initial_states=len(states)
    original_atoms=set(seen)
    pool=public_pool()
    eligible=[]
    audit={}
    for p,b in pool:
        lang,goals=description(b)
        key=frozenset(map(atom,goals))
        if not key or None in key or key&seen or lang in langs:
            audit[p]="historical or unsupported goal"
            continue
        eligible.append((p,b,lang,key))
    # Prefer rotation across source suites, then use a fixed hash, never policy performance.
    eligible.sort(key=lambda r:sha(("post-effect-01/"+r[0]).encode()))
    suites={}
    for row in eligible: suites.setdefault(row[0].split("/")[1],[]).append(row)
    ordered=[rows[i] for i in range(len(eligible)) for rows in suites.values() if i<len(rows)]
    selected=[]
    for p,b,lang,key in ordered+novel_substitutions():
        if key&seen or lang in langs:
            audit[p]="overlap with selected task"
            continue
        name="pro_"+sha(p.encode())[:14]
        bddl=ASSETS/(name+".bddl")
        bddl.write_bytes(b)
        try:
            prepare_starts(name,states)
        except Exception as exc:
            audit[p]="stock simulator: "+str(exc)[:180]
            bddl.unlink()
            continue
        selected.append(dict(task=name,origin="derived" if p.startswith("derived/") else "LIBERO-Pro",source=p,instruction=lang,atoms=sorted(key),bddl_sha256=sha(b),
                             states_sha256=sha((ASSETS/(name+".npy")).read_bytes())))
        langs.add(lang)
        seen.update(key)
        if len(selected)==N_TASKS: break
    old.h.dump(OUT/"eligibility.json",dict(public_entries=len(pool),eligible_before_dedup=len(eligible),excluded=audit))
    if len(selected)<2: raise ValueError("No compatible, atom-disjoint test tasks")
    old.h.save_lock(OUT/"manifest.json",dict(revision=REV,dataset=REPO,tasks=selected,indices=list(range(50)),
                                            historical_tasks=len(past),historical_states=initial_states,
                                            historical_atoms=sorted(x for x in original_atoms if x)))
    old.h.save_lock(OUT/"selection.json",{p:sha((OUT/p).read_bytes()) for p in ["freeze.json","manifest.json"]})

def verify():
    assert old.h.read(OUT/"selection.json")=={p:sha((OUT/p).read_bytes()) for p in ["freeze.json","manifest.json"]}
    rows=old.h.read(OUT/"manifest.json")["tasks"]
    for r in rows:
        for ext,k in [(".npy","states_sha256"),(".bddl","bddl_sha256")]:
            assert sha((ASSETS/(r["task"]+ext)).read_bytes())==r[k]
    return [r["task"] for r in rows]
