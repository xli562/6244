"""Expose controller capabilities and supply bounded, observation-only retry feedback."""
import re
from copy import deepcopy
import numpy as np
from binding import TOL
from recovery import DIRECTIONS, ReviewLibrary, role_spans, words
from skill_library import ROLES

class PlannerStalled(Exception):
    pass

class ActionLibrary(ReviewLibrary):
    def __init__(self, sensors, mode, reader, original, forced=None):
        super().__init__(sensors,"receipt" if mode=="receipt" else "recover",reader)
        self.variant=mode
        self.original_program=original
        self.initial=None
        self.pending=None
        self.attempts=[]
        self.feedback=[]
        self.query_count=0
        self.phase="plan"
        self.forced=forced
        self.forced_hit=False
        self.last_intents={}
        self.capabilities={}
        self.scene=self.catalog()
        if mode=="new":
            sites=sensors._env.env.object_sites_dict
            model=sensors._env.sim.model
            for n,v in self.scene.items():
                roles=["stroke.target","place.target"]
                if v["kind"]=="item":
                    roles.append("place.item")
                # Existing drawer controller addresses named levels of sliding joints.
                suffix=n.rsplit("_",2)
                if (v["kind"]=="region" and len(suffix)==3 and suffix[2]=="region"
                        and suffix[1] in ("top","middle","bottom") and v["parent"]==suffix[0]
                        and any(model.jnt_type[model.joint_name2id(j)]==2 for j in getattr(sites.get(n),"joints",[]))):
                    roles+=["open.target","close.target"]
                self.capabilities[n]=roles

    def supports(self,name):
        self.query_count+=1
        return list(self.capabilities.get(name,()))

    def options(self,op,role):
        self.query_count+=1
        return [n for n,roles in self.capabilities.items() if f"{op}.{role}" in roles]

    def phrases(self,op,role):
        return role_spans(self.instruction,op,role) if role in ROLES.get(op,()) else []

    def suggest(self,op,role,phrase):
        if not isinstance(phrase,str) or not phrase.strip():
            return []
        tokens=words(phrase)
        # Relative destinations need a spatial reference; an interior with a side label
        # is insufficient evidence for an exterior location.
        if re.search(r"\b(left|right|front|back) of\b",phrase,re.I):
            return []
        pool=self.options(op,role)
        nouns=tokens-DIRECTIONS-{"drawer","shelf"} if op in ("open","close") else tokens-DIRECTIONS
        dirs=tokens&DIRECTIONS
        scores={}
        for n in pool:
            lab=words(n)
            common=nouns&lab
            if not common and not (op in ("open","close") and "drawer" in tokens):
                continue
            if op=="place" and role=="target" and "init" in n.split("_") and not dirs:
                continue
            contained=(op=="place" and role=="target" and re.search(r"\b(?:in|into|inside)\s+"+re.escape(phrase.strip())+r"(?:\b|$)",self.instruction,re.I))
            child="contain" in lab or "inside" in lab
            scores[n]=(len(common),len(dirs&lab),int(bool(contained and child)),
                       int(self.scene[n]["kind"]!="region") if op=="place" and role=="target" else 0)
        if not scores:
            return []
        best=[n for n in scores if scores[n]==max(scores.values())]
        near=None
        if dirs and any(not dirs<=words(n) for n in best):
            if self.scene[best[0]]["kind"]!="item":
                return []
            shared=set.intersection(*(words(n) for n in best)) & nouns
            refs=[n for n,v in self.scene.items() if v["kind"]=="region" and shared and shared|dirs<=words(n)]
            if len(refs)!=1:
                return []
            near=refs[0]
            distances={n:float(np.linalg.norm(self.pos(n)[:2]-self.pos(near)[:2])) for n in best}
            best=[n for n in best if distances[n]<=min(distances.values())+TOL]
        result=[]
        for n in best:
            v=self.scene[n]
            terms=sorted((tokens & words(n)) | ({"contain"} if "contain" in words(n) and role=="target" else set()))
            spec={"kind":v["kind"],"terms":terms}
            if v["parent"]:
                spec["parent"]=v["parent"]
            if near:
                spec["near"]=near
            result.append({"name":n,"selector":spec})
        return result

    def preview(self,step):
        evidence=super().preview(step)
        if evidence is None:
            return None
        valid=self.variant=="new"
        for role,e in evidence.items():
            if valid:
                supported=self.options(step["op"],role)
                before=e["resolved"]
                pool=e.get("finalists",[])
                allowed=[n for n in pool if n in supported]
                if before and before not in supported:
                    e.update(resolved=None,status="unresolved",reason="skill cannot act on nominated entity")
                if not e["resolved"] and len(allowed)==1:
                    n=allowed[0]
                    e.update(resolved=n,status="confirmed" if n==e["pick"] else "corrected", supported=allowed)
            key=(self.remaining(),step["op"],role)
            self.last_intents.setdefault(key,step.get(role))
        record=dict(at=self.remaining(),phase=self.phase,step=deepcopy(step),evidence=deepcopy(evidence))
        if not self.attempts or self.attempts[-1]!=record:
            self.attempts.append(record)
        if self.forced and not self.forced_hit:
            f=self.forced
            if self.remaining()==f["at"] and step["op"]==f["op"] and evidence[f["role"]]["resolved"]==f["new"]:
                evidence=deepcopy(evidence)
                evidence[f["role"]].update(resolved=f["old"],status="confirmed")
                self.forced_hit=True
                self.force_binding=(self.remaining(),step["op"],f["role"],f["old"])
        if getattr(self,"force_binding",None):
            at,op,role,n=self.force_binding
            if at==self.remaining() and op==step["op"]:
                evidence=deepcopy(evidence)
                evidence[role].update(resolved=n,status="confirmed")
        if any(e["resolved"] is None for e in evidence.values()):
            self.pending={"reason":"blocked","step":deepcopy(step),"evidence":deepcopy(evidence)}
            if self.variant=="old" and any(not e["resolved"] and len(e.get("finalists",[]))>1 for e in evidence.values()):
                from build import old
                raise old.h.bench.EpisodeEnd()
            if valid:
                raise PlannerStalled()
        else:
            self.pending=None
        return evidence


class Public:
    allowed={"act","preview","pos","catalog","remaining","instruction"}
    tools={"supports","options","phrases","suggest"}
    def __init__(self,lib):
        self._lib=lib
    def original(self):
        return self._lib.original_program(self)
    def __getattr__(self,name):
        if name in self.allowed or (self._lib.variant=="new" and name in self.tools):
            return getattr(self._lib,name)
        raise AttributeError(name)
