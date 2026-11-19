"""Budgeted grasp arbitration and open-finger support observations."""
from copy import deepcopy

import numpy as np

from policies import prior
from checkpoints import StageLibrary
from manipulation import boxes, finger_contacts, geom_ids, acquire, deliver

ACTIVE=460
UP=np.array([0.,0.,1.])


class Released(Exception):
    """The observed placement completed an open-finger inspection."""


class ReviewLibrary(StageLibrary):
    def configure(self,arm,callback):
        self.mode=arm
        self.decisions=[]
        self.releases=[]
        self.approach=None
        self.opening=False
        self.seating=False
        self.audit_hook=None
        self.peak=0.
        self.height_fixes=0
        self.closures=0
        self.rested=None
        api=self._sensors
        api.remaining=lambda: max(0,ACTIVE-api._steps)
        def motor(action,steps=1):
            command=np.asarray(action,dtype=float)
            if command.shape!=(7,) or not np.isfinite(command).all():
                raise ValueError("Invalid OSC command")
            command=np.clip(command,-1,1)
            for _ in range(steps):
                if not api.remaining():raise prior.base.h.bench.EpisodeEnd()
                api._closed=bool(command[-1]>0)
                _,_,done,_=api._env.step(command)
                api._steps+=1
                a=self.approach
                if a:
                    p=api.pos(a["item"])
                    self.peak=max(self.peak,float(p[2]-a["z0"]))
                    if self.audit_hook:self.audit_hook(self,"step")
                    if not self.opening and not self.seating and self.mode!="literal" and self.peak>.025 and api._closed:
                        gap=np.linalg.norm(p[:2]-api.pos(a["target"])[:2])
                        if self.inside(p) and self.touch():
                            self.inspect_release()
                if done and (not a or self.mode=="literal"):
                    raise prior.base.h.bench.EpisodeEnd()
        api.action=motor
        super().configure("receipt" if arm=="receipt" else "early",callback)
        grip=api.grip
        def observed_grip(closed):
            if closed:
                self.closures+=1
            elif self.approach and not self.opening and self.mode!="literal" and api._closed and self.peak>.025:
                a=self.approach
                if np.linalg.norm((api.pos(a["item"])-api.pos(a["target"]))[:2])<.10:
                    self.inspect_release()
            return grip(closed)
        api.grip=observed_grip
        self.motor=motor

    def inside(self,position):
        a=self.approach
        api=self._sensors
        delta=position-api.pos(a["target"])
        if a["site"] is None:return np.linalg.norm(delta[:2])<.029
        sim=api._env.sim
        idx=a["site"]
        local=sim.data.site_xmat[idx].reshape(3,3).T@delta
        size=sim.model.site_size[idx]
        return np.all(np.abs(local[:2])<=np.maximum(size[:2],.025)) and abs(local[2])<size[2]+.09

    def touch(self):
        a=self.approach
        sim=self._sensors._env.sim
        left,right=a["geoms"]
        return any((c.geom1 in left and c.geom2 in right) or (c.geom2 in left and c.geom1 in right)
                   for c in sim.data.contact[:sim.data.ncon])

    def inspect_release(self):
        api=self._sensors
        a=self.approach
        def relative():return api.pos(a["item"])-api.pos(a["target"])
        start=relative()
        contact=self.touch()
        self.opening=True
        if self.audit_hook:self.audit_hook(self,"open")
        try:
            for vertical,count in [(0.,10),(.4,12),(0.,18)]:
                for _ in range(count):
                    self.motor([0,0,vertical,0,0,0,-1])
                    contact |= self.touch()
        finally:
            self.opening=False
        end=relative()
        support=self.touch()
        gap=float(np.linalg.norm(end[:2]))
        drift=float(np.linalg.norm(end-start))
        off=not self.inside(api.pos(a["item"])) and gap>.034 and gap>np.linalg.norm(start[:2])+.009
        lost=contact and (not support or off)
        event=dict(phase="settled_release",reason="lost_support" if not support else "drifted",
                   item=a["item"],target=a["target"],previous_support=bool(contact),support=support,
                   drift=round(drift,5),xy_gap=round(gap,5),remaining=api.remaining(),
                   loss=bool(lost or (off and drift>.025)),corrections_used=len(self.adjustments),start=start.round(5).tolist(),end=end.round(5).tolist())
        self.rested=None
        if support and self.inside(api.pos(a["item"])):
            self.rested=(a["item"],a["target"],end.copy())
        self.releases.append(event)
        if self.audit_hook:self.audit_hook(self,"settled")
        raise Released

    def arbitrate(self,event):
        a=self.approach
        detail=dict(event,misses=max(1,self.closures),local_options=self.closures<3,
                    wide=a["wide"] if a else False,default="switch",
                    corrections_used=len(self.adjustments),raw_axis_change=event["axis_change"],
                    raw_object_motion=event["object_motion"])
        # Report sub-millimetre settling and negligible rotation as zero. Some
        # compiled policies test these measurements for truthiness.
        if abs(detail["axis_change"])<.02:detail["axis_change"]=0.
        if abs(detail["object_motion"])<.001:detail["object_motion"]=0.
        use_review=self.mode in ("new","no_release")
        # Continue the chosen local search while its physical evidence is unchanged.
        # A different contact, disturbance or phase reopens the decision immediately.
        unchanged=(self.continuation is not None and event["reason"]=="empty_grasp"
                   and event["contacts"]==self.continuation["contacts"]
                   and abs(event["object_motion"]-self.continuation["object_motion"])<.008
                   and abs(event["axis_change"]-self.continuation["axis_change"])<.10)
        decision="continue" if unchanged else (self.policy(deepcopy(detail)) if use_review else "switch")
        self.continuation=detail if decision=="continue" and event["reason"]=="empty_grasp" else None
        # Consulting a legacy policy can consume its own retry counter. Only consult
        # it once the new decision commits to switching.
        parent=(self.parent_handler(deepcopy(event)) if self.parent_handler and decision=="switch"
                and self.mode!="continue" else None)
        self.decisions.append(dict(event=detail,parent=parent,decision=decision,continued_search=unchanged))
        return parent if decision=="switch" and self.mode!="continue" else None

    def normalize(self):
        if self.raw and self.raw["op"]=="place":
            if self.raw.get("max_lift") is None:self.height_fixes+=1
            self.raw["max_lift"]=max(float(self.raw.get("max_lift") or 0),self.peak)
            if self.raw.get("xy_gap") is None:
                b=self.current[1]
                self.raw["xy_gap"]=float(np.linalg.norm((self.pos(b["item"])-self.pos(b["target"]))[:2]))

    def context(self,item,target):
        api=self._sensors
        owner=self.scene.get(target,{}).get("parent") or target
        centers,radii=boxes(api,item)
        wide=bool(len(centers) and min(((centers+radii).max(axis=0)-(centers-radii).min(axis=0))[:2])>.071)
        world=api._env.env
        fixture=world.fixtures_dict.get(owner)
        model=api._env.sim.model
        surface=geom_ids(api,owner) if fixture is None else [model.geom_name2id(n) for n in fixture.contact_geoms]
        site=model.site_name2id(target) if self.scene.get(target,{}).get("parent") else None
        return dict(item=item,target=target,z0=api.pos(item)[2],wide=wide,site=site,
                    geoms=(set(geom_ids(api,item)),set(surface)))

    def adjust(self,strategy,source,requested=None):
        if self.rested and self.current and self.mode!="literal":
            item,target,where=self.rested
            b=self.current[1]
            if item==b.get("item") and target==b.get("target") and np.linalg.norm(self.pos(item)-self.pos(target)-where)<.008:
                return False
        return super().adjust(strategy,source,requested)

    def repair_released(self):
        e=self.releases[-1]
        if not e["loss"] or len(self.adjustments)>=2:return
        # Run the very same handler for the counterfactual, suppress only its release retry.
        if self.mode not in ("new","no_release") or self.policy(deepcopy(e))!="retry_release":return
        e["requested"]=True
        if self.mode=="no_release":return
        e["acted"]=True
        if self.audit_hook:self.audit_hook(self,"retry")
        self.seating=True
        try:
            self.recover("regrasp","post_release",e)
        except Released:
            pass
        finally:
            self.seating=False
        if self.audit_hook:self.audit_hook(self,"retried")

    def perform(self,strategy):
        step,b=self.current
        if step["op"]!="place" or b.get("item") not in self._sensors._env.env.objects_dict:
            return super().perform(strategy)
        self.approach=self.context(b["item"],b["target"])
        self.peak=0.
        self.closures=0
        self.continuation=None
        start=self.remaining()
        try:
            try:
                receipt=super().perform(strategy)
            except Released:
                self.repair_released()
                after=self.sample("place",b,None)
                if self.records:
                    m=self.records[-1]
                    m.update(after=after,remaining=self.remaining(),reason=self.reason(m["desired"],m["before"],after,m["hand_motion"]))
                self.raw.update(xy_gap=after["gap"],remaining=self.remaining(),steps_used=start-self.remaining())
                receipt={"executed":True}
            self.normalize()
            return dict(receipt,**self.raw)
        finally:
            self.normalize()
            self.approach=None

    def recover(self,strategy,source,warning=None):
        # End-of-policy retries also get a measured height, including a measured zero.
        b=self.current[1] if self.current else {}
        own=self.approach is None and b.get("item") in self._sensors._env.env.objects_dict
        if own:
            self.approach=self.context(b["item"],b["target"])
            self.peak=0.
        try:
            try:return super().recover(strategy,source,warning)
            except Released:
                if not own:raise
                self.repair_released()
                return True
        finally:
            self.normalize()
            if own:self.approach=None
