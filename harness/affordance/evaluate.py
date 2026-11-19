"""Controller-side measurement only. Match a changed nomination to an official task role."""
def witnesses(row):
    if not row["stable"] or row["mode"]!="new":
        return []
    goals=row["goals"]
    result=[]
    for act in row["actions"]:
        op=act["step"]["op"]
        for role,new in act["bound"].items():
            if not new:
                continue
            at=act["at"]
            original=next((p["step"].get(role) for p in row["attempts"]
                           if p["at"]==at and p["step"]["op"]==op),None)
            if not original or original==new or original not in row["names"]:
                continue
            if op in ("open","close"):
                ok={g[1] for g in goals if len(g)==2 and g[0]==op}
            elif op=="place":
                relevant=[g for g in goals if len(g)==3 and g[0] in ("on","in")]
                if role=="item":
                    ok={g[1] for g in relevant if g[2]==act["bound"]["target"]}
                else:
                    ok={g[2] for g in relevant if g[1]==act["bound"]["item"]}
            else:
                continue
            # A change of subgoal at the same physics step can also change the nominated
            # object. Do not count the first choice as wrong if another goal requests it.
            if op=="place" and any(g[1 if role=="item" else 2]==original for g in relevant):
                continue
            if ok=={new}:
                check=dict(at=at,op=op,role=role,old=original,new=new)
                if check not in result:
                    result.append(check)
    return result
