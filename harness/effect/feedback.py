"""Measure observable physical effects and give a frozen policy a bounded opportunity to retry."""
import re
from copy import deepcopy
import numpy as np
from interface import ActionLibrary
from motion import observed_joint, fraction, drive

STRATEGIES={"repeat","joint_high","joint_low","grasp_next","recenter"}

class Controller:
    def __init__(self, lib, original):
        self._lib=lib
        self._original=original

    @property
    def instruction(self):
        return self._lib.instruction

    def original(self):
        return self._original()

    def last(self):
        return deepcopy(self._lib.raw)

    def adjust(self,strategy):
        return self._lib.adjust(strategy,"receipt")


class EffectLibrary(ActionLibrary):
    def configure(self, arm, callback):
        self.arm=arm
        self.callback=callback
        self.raw=None
        self.current=None
        self.records=[]
        self.adjustments=[]
        self.origin=None
        self.excursion=0.
        api=self._sensors
        action=api.action
        def tracked(*args,**kwargs):
            try:
                return action(*args,**kwargs)
            finally:
                if self.origin is not None:
                    self.excursion=max(self.excursion,float(np.linalg.norm(api.eef()-self.origin)))
        api.action=tracked

    def act(self,step):
        ev=self.preview(step)
        roles=["item","target"] if step["op"]=="place" else ["target"]
        bound={r:ev[r]["resolved"] if ev else step.get(r) for r in roles}
        if None in bound.values():
            return super().act(step)
        self.current=(deepcopy(step),bound)
        start=self.remaining()
        initial_adjustments=len(self.adjustments)
        receipt=self.perform("initial")
        while (self.arm in ("new","repeat") and self.records[-1]["reason"] and self.remaining()>0
               and len(self.adjustments)<2 and self.callback):
            entry=self.records[-1]
            decision=self.callback(self.controller,dict(deepcopy(entry),instruction=self.instruction,
                                   previous_strategies=[d["strategy"] for d in self.adjustments]))
            if decision not in STRATEGIES:
                break
            if not self.adjust("repeat" if self.arm=="repeat" else decision,"observation",decision):
                break
        if len(self.adjustments)>initial_adjustments:
            receipt=dict(receipt, **self.raw)
            receipt["steps_used"]=start-self.remaining()
        return receipt

    def adjust(self,strategy,source,requested=None):
        if strategy not in STRATEGIES or self.current is None or len(self.adjustments)>=2 or self.remaining()<16:
            return False
        step,bound=self.current
        if strategy.startswith("joint_") and observed_joint(self._sensors,bound["target"]) is None:
            return False
        if strategy in ("grasp_next","recenter") and step["op"]!="place":
            return False
        self.adjustments.append(dict(at=self.remaining(),source=source,strategy=strategy,
                                     requested=requested or strategy,event=len(self.records)-1))
        self.perform(strategy)
        return True

    def expectation(self,op,j):
        if op=="place":
            if re.search(r"\b(?:beside|between|behind|in front of|to the (?:left|right) of)\b",self.instruction,re.I):
                return None
            return "at_target"
        if j is None:
            return None
        if op=="stroke":
            off=bool(re.search(r"\b(?:off|deactivate)\b",self.instruction,re.I))
            on=bool(re.search(r"\b(?:turn on|switch on|activate)\b",self.instruction,re.I))
            if off==on:
                return None
            active=on
        elif op in ("open","close"):
            active=op=="open"
        else:
            return None
        limits=self._sensors._env.sim.model.jnt_range[j]
        neutral_high=abs(limits[1])<abs(limits[0])
        return "high" if active!=neutral_high else "low"

    def sample(self,op,bound,j):
        api=self._sensors
        if op=="place":
            item=api.pos(bound["item"])
            return dict(item_xyz=item.round(5).tolist(),gap=round(float(np.linalg.norm(item[:2]-api.pos(bound["target"])[:2])),5))
        return {} if j is None else {"joint_fraction":round(fraction(api,j),6)}

    @staticmethod
    def reason(desired,before,after,hand):
        if not desired or hand<.01:
            return None
        if desired=="at_target":
            if after["gap"]<=.045:
                return None
            motion=np.linalg.norm(np.array(after["item_xyz"])-before["item_xyz"])
            if motion<.012:
                return "no_effect"
            return "wrong_direction" if after["gap"]>before["gap"]+.012 else "shortfall"
        goal=float(desired=="high")
        residual=abs(goal-after["joint_fraction"])
        progress=abs(goal-before["joint_fraction"])-residual
        if residual<=.002:
            return None
        return "no_effect" if abs(progress)<.02 else "wrong_direction" if progress<0 else "shortfall"

    def perform(self,strategy):
        api=self._sensors
        step,bound=self.current
        op=step["op"]
        j=observed_joint(api,bound["target"]) if op!="place" else None
        desired=self.expectation(op,j)
        before=self.sample(op,bound,j)
        start=self.remaining()
        self.origin=api.eef().copy()
        self.excursion=0.
        trace_start=len(api._trace)
        receipt={}
        try:
            if strategy in ("initial","repeat"):
                receipt=super().act(step)
            elif strategy.startswith("joint_"):
                drive(api,j,strategy=="joint_high")
            else:
                grasps=step.get("grasps") or ["standard"]
                used=[t[3] for t in api._trace if len(t)>3 and t[0]==bound["item"] and t[2]=="requested_grasp"]
                grasp=next((g for g in ["rim","thin_edge","standard"] if g not in used),grasps[0]) if strategy=="grasp_next" else grasps[0]
                if strategy=="recenter":
                    api.grip(False)
                    api.move(api.pos(bound["item"])+[0,0,.1],18)
                api.place_with(bound["item"],bound["target"],grasp)
            return receipt
        finally:
            after=self.sample(op,bound,j)
            hand=round(self.excursion,5)
            self.origin=None
            self.raw={k:receipt.get(k) for k in ("max_lift","xy_gap")}
            if op=="place" and not receipt:
                lifts=[x[3] for x in api._trace[trace_start:] if len(x)>3 and x[2]=="lift_m"]
                self.raw.update(max_lift=max(lifts,default=None),xy_gap=after["gap"])
            self.raw.update(op=op,steps_used=start-self.remaining(),remaining=self.remaining())
            self.records.append(dict(op=op,bound=bound,strategy=strategy,at=start,remaining=self.remaining(),desired=desired,
                                     before=before,after=after,hand_motion=hand,reason=self.reason(desired,before,after,hand)))
