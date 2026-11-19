"""Check simulator starts after the same five settling steps used by the scorer."""
from concurrent.futures import ProcessPoolExecutor
import json
from unittest.mock import patch
import numpy as np
import audit
from suite import verify
r=audit.run

def inspect(task):
    states=np.load(r.OUT/"assets"/(task+".npy"))
    done=[]
    partial=[]
    with patch.object(r.base,"OUT",r.OUT),patch.object(r.base.h.bench,"CONFIG",r.OUT/"assets"),patch.object(r.base.h.bench,"paths",r.base.file_paths):
        env=r.base.h.bench.environment(task,extra_steps=40)
        try:
            for i,s in enumerate(states):
                env.env.reset()
                env.set_init_state(s)
                for _ in range(5):env.step([0,0,0,0,0,0,-1])
                g=env.env.parsed_problem["goal_state"]
                if env.check_success():done.append(i)
                if any(env.env._eval_predicate(x) for x in g):partial.append(i)
        finally:env.close()
    return task,dict(complete_at_start=done,any_subgoal_at_start=partial)
if __name__=="__main__":
    import sys
    from pathlib import Path
    prefix=""
    if sys.argv[1:]==["challenge"]:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"challenge"))
        from relations import verify
        prefix="challenge_"
    with ProcessPoolExecutor(8) as pool:
        out=dict(pool.map(inspect,verify()))
    (r.OUT/(prefix+"settled_starts.json")).write_text(json.dumps(out,indent=1)+"\n")
