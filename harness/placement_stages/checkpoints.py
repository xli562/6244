"""Observe the unchanged placement trajectory. Late delivery buffers exactly the first request."""
from copy import deepcopy
import numpy as np
from feedback import EffectLibrary
from manipulation import acquire,deliver,finger_contacts

class InterruptPlacement(Exception):
    pass

class StageLibrary(EffectLibrary):
    def configure(self,arm,callback):
        self.timing=arm
        super().configure("receipt" if arm=="receipt" else "new",callback)
        self.checks=[]
        self.requests=[]
        self.repairs=[]
        self.live=None
        self.pending_stage=None
        api=self._sensors
        move,grip,action=api.move,api.grip,api.action
        def guarded_action(*args,**kwargs):
            result=action(*args,**kwargs)
            s=self.live
            if s and not s["closed"] and not s["released"] and not self.pending_stage:
                displacement=np.linalg.norm(api.pos(s["item"])-s["start_xyz"])
                angle=np.linalg.norm(api._body_frame(s["item"])[:,2]-s["start_axis"])
                if np.linalg.norm(api.eef()-api.pos(s["item"]))<.19 and (displacement>.018 or angle>.22):
                    self.check("grasp","slip")
            return result
        api.action=guarded_action
        def guarded_grip(closed):
            s=self.live
            if s and closed:
                s["closed"]=True
            if s and not closed and s["closed"] and s["raised"]:
                self.check("release","off_target",gap_limit=.036)
            result=grip(closed)
            if s:
                if closed:
                    s.update(closed=True,raised=False,z0=api.pos(s["item"])[2],hand0=api.eef().copy(),
                             offset=api.eef()-api.pos(s["item"]),released=False)
                else:
                    s["released"]=s["closed"] and s["raised"]
                    s["closed"]=False
            return result
        def guarded_move(point,steps=30):
            result=move(point,steps)
            s=self.live
            if s and s["closed"]:
                lift=api.pos(s["item"])[2]-s["z0"]
                if not s["raised"] and api.eef()[2]-s["hand0"][2]>.028:
                    if lift<.018:
                        self.check("grasp","empty_grasp")
                    else:
                        s.update(raised=True,offset=api.eef()-api.pos(s["item"]))
                elif s["raised"]:
                    drift=np.linalg.norm(api.eef()-api.pos(s["item"])-s["offset"])
                    if drift>.037 and finger_contacts(api,s["item"])<2:
                        self.check("carry","slip")
                    elif np.linalg.norm(np.asarray(point)[:2]-(api.pos(s["target"])+s["offset"])[:2])<.032:
                        self.check("carry","off_target",gap_limit=.043)
            elif s and s["released"]:
                s["released"]=False
                self.check("release","missed_release",gap_limit=.045)
            return result
        api.move=guarded_move
        api.grip=guarded_grip

    def check(self,phase,reason,gap_limit=None):
        if self.pending_stage or len(self.adjustments)>=2: return
        api=self._sensors
        s=self.live
        item,target=s["item"],s["target"]
        gap=float(np.linalg.norm(api.pos(item)[:2]-api.pos(target)[:2]))
        if gap_limit is not None and gap<=gap_limit:return
        contacts=finger_contacts(api,item)
        event=dict(phase=phase,reason=reason,remaining=self.remaining(),held=bool(s["closed"] and contacts and s["raised"]),
                   lift=round(float(api.pos(item)[2]-s["z0"]),5),xy_gap=round(gap,5),
                   drift=round(float(np.linalg.norm(api.eef()-api.pos(item)-s["offset"])),5),
                   object_motion=round(float(np.linalg.norm(self.pos(item)-s["start_xyz"])),5),
                   axis_change=round(float(np.linalg.norm(api._body_frame(item)[:,2]-s["start_axis"])),5),
                   previous=[x["strategy"] for x in self.adjustments],item=item,target=target,contacts=contacts)
        self.checks.append(event)
        if self.timing not in ("early","late"): return
        strategy=self.handler(deepcopy(event)) if self.handler else None
        if strategy not in ("regrasp","align"):return
        self.requests.append(dict(check=len(self.checks)-1,event=event,strategy=strategy))
        self.pending_stage=(event,strategy)
        if self.timing=="early":raise InterruptPlacement()

    def recover(self,strategy,source,warning=None):
        if self.current is None or self.current[0]["op"]!="place" or self.remaining()<12 or len(self.adjustments)>=2:return False
        api=self._sensors
        _,bound=self.current
        item,target=bound["item"],bound["target"]
        start=self.remaining()
        self.adjustments.append(dict(at=start,source=source,strategy=strategy,event=len(self.records)-1))
        self.repairs.append(dict(at=start,source=source,strategy=strategy,warning=warning))
        # Disable checkpoint re-entry while the correction itself uses the same fingers.
        saved=self.live
        self.live=None
        try:
            held=api._closed and finger_contacts(api,item)==2
            if (strategy=="align" and held) or acquire(api,item,len(self.repairs)-1):
                deliver(api,item,target)
        finally:
            self.live=saved
            gap=np.linalg.norm(api.pos(item)[:2]-api.pos(target)[:2])
            self.raw=dict(op="place",remaining=self.remaining(),steps_used=start-self.remaining(),xy_gap=float(gap),max_lift=None)
        return True

    def adjust(self,strategy,source,requested=None):
        if strategy in ("align","regrasp"):
            return self.recover(strategy,source)
        return super().adjust(strategy,source,requested)

    def perform(self,strategy):
        step,bound=self.current
        if step["op"]!="place" or self.expectation("place",None) is None or not bound.get("item") or bound["item"]==bound["target"]:
            return super().perform(strategy)
        item=bound["item"]
        self.live=dict(item=item,target=bound["target"],start_xyz=self.pos(item).copy(),start_axis=self._sensors._body_frame(item)[:,2].copy(),closed=False,raised=False,released=False,
                       z0=self.pos(item)[2],hand0=self._sensors.eef(),offset=self._sensors.eef()-self.pos(item))
        self.pending_stage=None
        start=self.remaining()
        try:
            try: receipt=super().perform(strategy)
            except InterruptPlacement: receipt={"executed":True}
            if self.pending_stage:
                event,choice=self.pending_stage
                if self.recover(choice,"stage",event):
                    receipt=dict(receipt,**self.raw)
                    receipt["steps_used"]=start-self.remaining()
                    # Parent's end-of-action callback should see the actual final state.
                    rec=self.records[-1]
                    after=self.sample("place",bound,None)
                    rec.update(after=after,remaining=self.remaining(),reason=self.reason(rec["desired"],rec["before"],after,rec["hand_motion"]))
            return receipt
        finally:
            self.live=None
            self.pending_stage=None
