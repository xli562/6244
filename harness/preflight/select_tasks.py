"""After policy freeze, draw a fixed sample without consulting policies or scores."""

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "testsets/libero_holdout_v1"))
import collect as upstream


def language(suite, task):
    src = (upstream.UP / "bddl_files" / suite / (task + ".bddl")).read_text()
    raw = re.search(r"\(:language\s+([^)]*)", src).group(1)
    return " ".join(re.findall(r"\w+", raw.lower()))


def select():
    history = {}
    for path in [
        *sorted((ROOT / "sim/data").glob("libero_*.json")),
        ROOT / "testsets/libero_holdout_v1/manifest.json",
        ROOT / "harness/interaction/manifest.json",
    ]:
        spec = json.loads(path.read_text())
        for row in spec["tasks"]:
            history[row["task"]] = row.get("suite", spec.get("suite"))
    used = {language(s, t) for t, s in history.items()}
    available = {}
    for p in (upstream.UP / "bddl_files/libero_90").glob("*.bddl"):
        lang = language("libero_90", p.stem)
        if p.stem in history or lang in used:
            continue
        composed = (
            re.search(r"\band (?:close|open|turn|stack|push)\b", lang)
            or (
                not lang.startswith(("pick ", "pick up "))
                and re.search(r"\band (?:put|place)\b", lang)
            )
            or "both " in lang
        )
        group = (
            "composed"
            if composed
            else "spatial"
            if re.search(r"\b(left|right|middle|front|back|under|between)\b", lang)
            else "other"
        )
        available[p.stem] = (group, lang)

    def rank(name):
        return hashlib.sha256(("preflight-v1/" + name).encode()).hexdigest()

    chosen = []

    # Up to four in each stratum; fill vacancies in hash order. Prefer different scenes within a stratum.
    def take(candidates, quota):
        scenes = set()
        for distinct in [True, False]:
            for t in sorted(candidates, key=rank):
                lang = available[t][1]
                scene = (
                    t.partition("_SCENE")[0]
                    + "_SCENE"
                    + t.partition("_SCENE")[2].partition("_")[0]
                )
                if lang in used or (distinct and scene in scenes):
                    continue
                chosen.append(t)
                used.add(lang)
                scenes.add(scene)
                quota -= 1
                if quota == 0:
                    return

    for g in ("composed", "spatial", "other"):
        take([n for n, (group, _) in available.items() if group == g], 4)
    if len(chosen) < 12:
        take(available, 12 - len(chosen))
    historical = set()
    for t, s in history.items():
        historical.update(
            upstream.state_keys(
                (upstream.UP / "init_files" / s / (t + ".pruned_init")).read_bytes()
            )
        )
    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(upstream.collect, [("libero_90", t) for t in chosen]))
    state_ids = [k for _, keys in rows for k in keys]
    if (
        len(chosen) != 12
        or len(set(state_ids)) != 600
        or historical.intersection(state_ids)
    ):
        raise ValueError("Test coverage or state independence failed")
    return {
        "revision": upstream.REV,
        "historical_tasks": len(history),
        "historical_states": len(historical),
        "tasks": [
            dict(
                row,
                stratum=available[row["task"]][0],
                instruction=available[row["task"]][1],
            )
            for row, _ in rows
        ],
        "indices": list(range(50)),
    }
