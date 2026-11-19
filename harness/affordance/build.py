"""Compile universal policies before selecting tasks; provide the frozen prior as a callable."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"harness/reselection"))
from generate import old, OUT as PRIOR, REPS

OUT=ROOT/"runs/affordance_01"
GUIDE="""You have one CLI call. A previous universal run(api) below is available as api.original().
Write a small Python module in a fenced Python block. Your new run(api) may simply call api.original()
or improve its general planning. No specific tasks, object IDs, compiled lookup tables, imports except numpy/math,
private names/attributes, getattr, hasattr, re, file access or further model calls.
"""
RETRY="""Also implement retry(api, feedback), invoked by the enhanced deployment at most twice, when no action
was submitted or when the checker rejected a role. Old deployment uses exactly your same run but has no retry.
For comparability run(api) uses the ordinary prior interface ONLY, including original(). retry alone can use:
api.supports(name) -> list of skill roles e.g. 'open.target','close.target','place.item','place.target','stroke.target'.
api.options(op,role) -> list of compatible observed names. api.phrases(op,role) -> scoped role phrases from the original instruction.
api.suggest(op,role,phrase) -> list of {'name': observed name,'selector': legal expect selector} for best compatible
matches to the phrase, empty if no evidence (ties retained). In retry these options are live. Feedback is
{'reason':'no_proposal'|'blocked'|'planner_error','step': last rejected step dict or {}, 'evidence': role checks}.
Rebuild the blocked role or construct a whole new step for a silently skipped instruction, using the original
words, supports/options/suggest. Send a step to api.act with valid expect for EVERY role. Preserve correct roles.
Never default to first of tied candidates. Handle common two-clause instructions with pronouns.
No task-specific cases. Keep retry under ~60 lines. No need to repeat the original program in your output."""
def one(key):
    agent,_,family=key.split("/")
    prior=(PRIOR/key/"proposal.py").read_text()
    prompt=(PRIOR/key/"prompt.txt").read_text()+"\n"+GUIDE+"\nPrior implementation:\n"+prior+"\n"
    prompt+= RETRY if family=="checked" else "Use direct actions with receipts only. Write the strongest general run(api) you can; api.original() is available."
    src=old.previous.generate(agent,OUT/key,prompt)
    return key,old.h.digest(src)
if __name__=="__main__":
    keys=[f"{a}/{r}/{f}" for r in REPS for a in old.h.AGENTS for f in ("checked","receipt")]
    with ThreadPoolExecutor(12) as pool:
        old.h.save_lock(OUT/"programs.json",dict(pool.map(one,keys)))
