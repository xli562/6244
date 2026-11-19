"""Cross-check all paired episode IDs, the literal previous release and the delayed control."""
from collections import Counter
import json
from statistics import median
import audit

def med(seq):
    return median(seq) if seq else None

def analyze(stem="test"):
    prefix="challenge_" if stem=="challenge" else ""
    rows=audit.read(stem)
    early={audit.episode_key(r):r for r in rows if r["arm"]=="early"}
    old={audit.episode_key(r):r for r in rows if r["arm"]=="old"}
    late={audit.episode_key(r):r for r in rows if r["arm"]=="late"}
    literal={audit.episode_key(r):r for r in audit.read(prefix+"previous")}
    receipt={(r["key"].replace("/receipt","/checked"),r["task"],r["i"]):r for r in rows if r["arm"]=="receipt"}
    assert early.keys()==old.keys()==late.keys()==literal.keys()==receipt.keys()
    judged={audit.episode_key(r):r for r in audit.read(stem+"_audit")}
    manifest=json.loads((audit.run.OUT/(prefix+"manifest.json")).read_text())["tasks"]
    scores={}
    for model in ["both","codex","claude"]:
        scores[model]={}
        for task in ["ALL"]+[r["task"] for r in manifest]:
            keys=[k for k in early if (model=="both" or k[0].startswith(model+"/")) and (task=="ALL" or k[1]==task)]
            gains=[k for k in keys if early[k]["stable"] and not old[k]["stable"]]
            losses=[k for k in keys if not early[k]["stable"] and old[k]["stable"]]
            strict=[k for k in gains if k in judged and judged[k]["credited"] and not literal[k]["stable"] and not literal[k]["error"]]
            delayed=[judged[k]["late_remaining"] for k in strict if judged[k]["late_remaining"] is not None]
            scores[model][task]=dict(n=len(keys),early=sum(early[k]["stable"] for k in keys),
                old=sum(old[k]["stable"] for k in keys),late=sum(late[k]["stable"] for k in keys),
                receipt=sum(receipt[k]["stable"] for k in keys),previous=sum(literal[k]["stable"] for k in keys),
                previous_changed=sum(literal[k]["stable"]!=old[k]["stable"] for k in keys),
                gained=len(gains),lost=len(losses),strict=len(strict),unique_starts=len({k[1:] for k in strict}),
                strict_receipt_failed=sum(not receipt[k]["stable"] and not receipt[k]["error"] for k in strict),
                stage=Counter(judged[k]["phase"]+"/"+judged[k]["reason"] for k in strict),
                median_early=med([judged[k]["early_remaining"] for k in strict]),median_late=med(delayed),
                late_delivered=len(delayed),late_end_without_repair=len(strict)-len(delayed),
                stable_by_455=[sum(a[k]["stable"] and a[k]["steps"]<=455 for k in keys) for a in (early,old,late,receipt)],
                losses_stage=Counter((early[k]["requests"][0]["event"]["phase"]+"/"+early[k]["requests"][0]["event"]["reason"]
                            if early[k]["requests"] else "no request") for k in losses),
                losses_errors=sum(bool(early[k]["error"]) for k in losses),
                early_errors=Counter(str(early[k]["error"]) for k in keys if early[k]["error"]),
                late_errors=Counter(str(late[k]["error"]) for k in keys if late[k]["error"]),
                early_official=sum(early[k]["official"] for k in keys),
                gain_late_succeeded=sum(late[k]["stable"] for k in gains),
                gain_control_error=sum(bool(late[k]["error"] or old[k]["error"]) for k in gains))
    (audit.run.OUT/(prefix+"comparisons.json")).write_text(json.dumps(scores,indent=1)+"\n")
if __name__=="__main__":
    import sys
    analyze(*sys.argv[1:])
