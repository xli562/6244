"""Paired deployment of fixed policies; use the existing simulator entry point and scorer."""
import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

from generate import old, ROOT, OUT, SAVED, REPS
from recovery import ReviewLibrary

h = old.h
DEV = {"codex/3/checked": "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
       "claude/3/checked": "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate"}

@lru_cache(None)
def reader(src):
    h.episode.check_policy_source(src)
    ns = {}
    exec(compile(src, "agent_policy.py", "exec"), ns)
    return ns.get("reread")


class StopOnTie(ReviewLibrary):
    def preview(self, step):
        evidence = super().preview(step)
        if (evidence and any(ev["resolved"] is None and len(ev.get("finalists", [])) > 1 for ev in evidence.values())
                and not (getattr(self,"released",False) or (hasattr(self, "expected_review") and self.reviews[-1] == self.expected_review))):
            raise h.bench.EpisodeEnd()
        return evidence


class Diagnostic(StopOnTie):
    """Only release a terminal veto at the same control step; retain all earlier checks."""
    def preview(self, step):
        evidence = super().preview(step)
        if (not self.released and self.remaining() == self.release_at and self.reviews
                and self.reviews[-1] == self.expected_review):
            self.released = True
        if self.released:
            for role, ev in evidence.items():
                if ev["resolved"] is None and step.get(role) in self.catalog():
                    evidence = deepcopy(evidence)
                    evidence[role].update(resolved=step[role], status="confirmed")
        return evidence


def rollout(job):
    key, mode, task, ids, src, thresholds = job
    libraries = []
    def factory(sensors, unused):
        lib = (Diagnostic if thresholds else StopOnTie)(sensors, mode, reader(src))
        if thresholds:
            lib.expected_review = thresholds[len(libraries)]
            lib.release_at = lib.expected_review["at"]
            lib.released = False
        libraries.append(lib)
        return lib
    with patch.object(old, "SkillLibrary", factory):
        rows = old.episodes((key, task, ids, src))
    logs = []
    for r, lib in zip(rows, libraries):
        reviews = lib.reviews
        terminal = reviews[-1] if reviews and reviews[-1]["at"] == 500-r["steps"] else {}
        veto = [k for k,v in terminal.get("roles", {}).items() if not v.get("resolved")]
        logs.append(dict(key=key, mode=mode, task=task, i=r["index"], stable=r["settled"], official=r["success"],
                         steps=r["steps"], error=r["error"], terminal_veto=veto, reviews=reviews, released=getattr(lib,"released",False),
                         acts=[{"intent":ev["intent"],"bound":ev["bound"]} for ev in lib.events],
                         trace=r["trace"]))
    return logs


def deploy(jobs, workers, filename):
    dest = OUT / filename
    if dest.exists():
        raise FileExistsError(dest)
    tmp = dest.with_suffix(".tmp")
    n = 0
    with gzip.open(tmp, "wt", compresslevel=2) as sink, ProcessPoolExecutor(workers) as pool:
        for rows in pool.map(rollout, jobs):
            for row in rows:
                sink.write(json.dumps(row, default=old.json_value, separators=(",",":"))+"\n")
            n += len(rows)
            if n % 1000 == 0:
                print(n, flush=True)
    os.replace(tmp, dest)
    print(filename,n,flush=True)


def frozen_programs():
    hashes = h.read(OUT / "programs.json")
    keys = {f"{a}/{r}/{f}" for a in h.AGENTS for r in REPS for f in ("checked","receipt")}
    assert hashes.keys() == keys
    programs = {}
    for k, sha in hashes.items():
        programs[k] = (OUT / k / "proposal.py").read_text()
        assert h.digest(programs[k]) == sha
        assert len((OUT / k / "calls.jsonl").read_text().splitlines()) == 1
    # Independent robustness check: deploy all six previously compiled pairs, including the two practice policies.
    prior = h.read(SAVED / "programs.json")
    assert prior.keys() == keys
    for k,sha in prior.items():
        src = (SAVED / k / "proposal.py").read_text()
        assert h.digest(src) == sha
        a,r,f = k.split("/")
        programs[f"{a}/old{r}/{f}"] = src
    return programs


def freeze():
    old.lock()
    sources = sorted(Path(__file__).parent.glob("*.py"))
    paths = sources + [old.HERE / "experiment.py", old.HERE / "binding.py", old.HERE / "skill_library.py",
                       old.OUT / "system.json", old.OUT / "manifest.json", SAVED / "programs.json", OUT / "programs.json"]
    frozen_programs()
    h.save_lock(OUT / "freeze.json", {"sha256":{str(p.relative_to(ROOT)):h.digest(p.read_text()) for p in paths},
                "old_system":h.system_spec(), "states":50, "repetitions":3, "budget":"one call per program; no test calls",
                "cohorts":["fresh primary", "prior second batch, all policies, supplemental"], "modes":["receipt","stop","recover"], "selection":"fixed hash sample: 6 spatial, 3 composed, 3 other; fill shortages by hash",
                "diagnostic":"Replay failed stop and recovery vetoes from the identical observed decision, allow original nominations thereafter"})


def main(stage, workers):
    if stage == "dev":
        manifest = h.read(SAVED / "manifest.json")
        old.register(manifest)
        h.bench.prepare()
        jobs = [(key, mode, t, list(range(i,i+5)), (SAVED / key / "proposal.py").read_text(), None)
                for key,t in DEV.items() for mode in ("stop","recover","raw") for i in range(0,50,5)]
        deploy(jobs, workers, "practice.jsonl.gz")
        return
    freeze()
    if stage == "select":
        from sample import draw
        h.save_lock(OUT / "manifest.json", draw())
        h.save_lock(OUT / "selection.json", {p:h.digest((OUT/p).read_text()) for p in ("manifest.json","freeze.json")})
        return
    programs = frozen_programs()
    manifest = h.read(OUT / "manifest.json")
    assert h.read(OUT / "selection.json") == {p:h.digest((OUT/p).read_text()) for p in ("manifest.json","freeze.json")}
    old.register(manifest)
    h.bench.prepare()
    if stage == "test":
        jobs = [(key, mode, t["task"], list(range(i,i+5)), src, None)
                for key,src in programs.items() for t in manifest["tasks"]
                for mode in (("stop","recover") if key.endswith("/checked") else ("receipt",))
                for i in range(0,50,5)]
        deploy(jobs,workers,"test.jsonl.gz")
    if stage == "diagnose":
        with gzip.open(OUT / "test.jsonl.gz","rt") as f:
            rows = [r for r in map(json.loads,f) if r["mode"] in ("recover","stop") and r["terminal_veto"] and not r["stable"]]
        groups = {}
        for r in rows:
            groups.setdefault((r["key"],r["mode"],r["task"],r["i"]//5),[]).append(r)
        jobs = [(key,mode,task,[r["i"] for r in batch],programs[key],[r["reviews"][-1] for r in batch])
                for (key,mode,task,_),batch in groups.items()]
        deploy(jobs,workers,"counterfactual.jsonl.gz")

if __name__ == "__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=["dev","select","test","diagnose"])
    cli.add_argument("--workers",type=int,default=48)
    args=cli.parse_args()
    main(args.stage,args.workers)
