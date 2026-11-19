from copy import deepcopy
import numpy as np
from generate_candidates import previous
from controller import ReviewLibrary,Released
import preview
import reseat

class PreviewLibrary(ReviewLibrary):
    def configure(self,arm,callback):
        self.experiment_arm=arm
        self.enhanced=arm in ("new","no_release","no_preview","receipt")
        self.reserves=[]
        self.preview_log=[]
        self.local_vote=None
        self.repair_profile=None
        super().configure("receipt" if arm=="receipt" else ("continue" if arm=="continue" else "new"),callback)
        self.disabled_release=arm=="no_release"
        if self.enhanced:
            preview.install(self)
            grip=self._sensors.grip
            def budgeted_grip(closed):
                api=self._sensors
                if not closed and not api._closed and self.approach and self.remaining()<225 and not self.opening:
                    # Only omit a repeated command when both measured fingers are fully open.
                    q=api._env.sim.data.qpos[api._env.robots[0]._ref_gripper_joint_pos_indexes]
                    if len(q)==2 and min(abs(q))>.037:
                        self.reserves.append(dict(remaining=self.remaining(),decision="already_open"))
                        return
                return grip(closed)
            self._sensors.grip=budgeted_grip

    def ledger(self,event):
        used=len(self.adjustments)
        return dict(event,corrections=used,corrections_used=used,retries_left=max(0,2-used))

    def normalize(self):
        super().normalize()
        if self.enhanced and self.raw:self.raw.update(self.ledger(self.raw))

    def arbitrate(self,event):
        inspecting=self.enhanced and self.experiment_arm!="no_preview" and self.approach
        api=self._sensors
        if inspecting and self.local_vote and event["phase"]=="grasp" and event["reason"]=="slip" and not api._closed:
            index,item,xyz,axis=self.local_vote
            if (item==event["item"] and index==self.next_index and np.linalg.norm(api.pos(item)-xyz)<.008
                    and np.linalg.norm(api._body_frame(item)[:,2]-axis)<.10):
                self.decisions.append(dict(event=self.ledger(event),parent=None,decision="continue",continued_search=True))
                return None
        if inspecting and event["phase"]=="carry" and event["reason"]=="off_target" and event["held"]:
            site=preview.receiving_site(self)
            if site is not None:
                sim=api._env.sim
                offset=sim.data.site_xmat[site].reshape(3,3).T@(api.pos(event["item"])-sim.data.site_xpos[site])
                if np.all(np.abs(offset[:2])<=sim.model.site_size[site][:2]+.012):
                    self.decisions.append(dict(event=self.ledger(event),parent=None,decision="continue",continued_search=False))
                    return None
        if inspecting and event["reason"]=="empty_grasp":
            evidence=preview.inspect(self,event)
            action=self.next_choice(self.ledger(dict(event,op="grasp",failed=True,failed_grasp=True,**evidence)))
            self.preview_log.append(dict(at=self.remaining(),item=event["item"],evidence=evidence,action=action))
            if action in ("continue","switch"):
                item=event["item"]
                self.local_vote=(self.next_index,item,api.pos(item).copy(),api._body_frame(item)[:,2].copy()) if action=="continue" else None
                parent=None
                if action=="switch" and self.remaining()>=90 and len(self.adjustments)<2:
                    parent=self.parent_handler(deepcopy(event)) if self.parent_handler else None
                    parent=parent or "regrasp"
                self.decisions.append(dict(event=self.ledger(event),parent=parent,decision=action,continued_search=False))
                return parent
        return super().arbitrate(event)

    def adjust(self,strategy,source,requested=None):
        # Do not spend the last correction on an end-of-action guess while a
        # measured release still has a chance. The first pickup always runs.
        if self.enhanced and self.current and self.current[0]["op"]=="place" and self.releases:
            last=self.releases[-1]
            item=self.current[1].get("item")
            target=self.current[1].get("target")
            if (strategy in ("regrasp","recenter","align") and last["item"]==item and last["target"]==target
                    and last["loss"] and self.remaining()<210 and self.remaining()>=self.retry_cost(last)
                    and len(self.adjustments)<2):
                # A receipt-only program can explicitly request this same short skill.
                own=self.approach is None
                if own:self.approach=self.context(item,target)
                self.seating=True
                try:
                    return self.compact_repair(source,last)
                except Released:return True
                finally:
                    self.seating=False
                    if own:self.approach=None
            if len(self.adjustments)==1 and not last["loss"] and self.remaining()<150:
                self.reserves.append(dict(remaining=self.remaining(),source=source,decision="hold_last_correction"))
                return False
        return super().adjust(strategy,source,requested)

    def inspect_release(self):
        if not self.enhanced:
            return super().inspect_release()
        # Leave time for a physical retry: open first, then shorten only the
        # remaining observation once four consecutive poses agree after retreat.
        api=self._sensors
        a=self.approach
        item,target=a["item"],a["target"]
        def relative():return api.pos(item)-api.pos(target)
        start=relative()
        contact=self.touch()
        self.opening=True
        if self.audit_hook:self.audit_hook(self,"open")
        still=0
        lost_frames=0
        prev=start
        try:
            for i in range(24 if self.seating and self.repair_profile=="short" else 40):
                vertical=.4 if 10<=i<22 else 0.
                self.motor([0,0,vertical,0,0,0,-1])
                now=relative()
                contact |= self.touch()
                still=still+1 if np.linalg.norm(now-prev)<.0015 else 0
                prev=now
                # Save the quiet tail for another placement or a measured retry.
                # An unstable object keeps the complete observation window.
                gap_now=float(np.linalg.norm(now[:2]))
                slipping=contact and (not self.touch() or
                         (not self.inside(api.pos(item)) and gap_now>max(.034,np.linalg.norm(start[:2])+.009)))
                lost_frames=lost_frames+1 if slipping else 0
                needed=reseat.cost({"xy_gap":gap_now,"end":now})
                if i>=14 and lost_frames>=3 and needed<=self.remaining()<=needed+4:
                    self.reserves.append(dict(remaining=self.remaining(),decision="release_retry_deadline"))
                    break
                if i>=17 and still>=4 and (self.remaining()<170 or i>=25):break
        finally:self.opening=False
        support=self.touch()
        gap=float(np.linalg.norm(now[:2]))
        off=not self.inside(api.pos(item)) and gap>.034 and gap>np.linalg.norm(start[:2])+.009
        drift=float(np.linalg.norm(now-start))
        self.rested=(item,target,now.copy()) if support and self.inside(api.pos(item)) else None
        self.releases.append(self.ledger(dict(phase="settled_release",reason="lost_support" if not support else "drifted",
            item=item,target=target,previous_support=bool(contact),support=support,drift=round(drift,5),xy_gap=round(gap,5),
            remaining=api.remaining(),loss=bool((contact and (not support or off)) or (off and drift>.025)),
            inspected=i+1,start=start.round(5).tolist(),end=now.round(5).tolist())))
        if self.audit_hook:self.audit_hook(self,"settled")
        raise Released

    def repair_released(self):
        if not self.enhanced:return super().repair_released()
        event=self.releases[-1]
        event.update(self.ledger(event))
        event.setdefault("inspected",40)
        event["retry_cost"]=self.retry_cost(event)
        if self.experiment_arm=="receipt" or not event["loss"] or not event["retries_left"]:return
        if self.release_choice(deepcopy(event))!="retry_release" or self.remaining()<event["retry_cost"]:return
        event["requested"]=True
        if self.disabled_release:return
        if self.audit_hook:self.audit_hook(self,"retry")
        event["acted"]=True
        self.seating=True
        try:
            self.compact_repair("post_release",event)
        except Released:pass
        finally:self.seating=False
        if self.audit_hook:self.audit_hook(self,"retried")

    def retry_cost(self,event):
        # Bounded budget for a short local reseat, including release inspection.
        return reseat.cost(event)

    def compact_repair(self,source,event):
        b=self.current[1]
        at=self.remaining()
        if len(self.adjustments)>=2 or at<self.retry_cost(event):return False
        self.adjustments.append(dict(at=at,source=source,strategy="reseat",event=len(self.records)-1))
        record=dict(at=at,source=source,strategy="reseat",limit=self.retry_cost(event))
        self.repairs.append(record)
        self.repair_profile="short" if reseat.near(event) else "normal"
        try:
            return reseat.run(self,b["item"],b["target"])
        finally:
            self.repair_profile=None
            record["spent"]=at-self.remaining()
            if record["spent"]>record["limit"]:raise RuntimeError("Local retry exceeded its allocation")
            self.normalize()
