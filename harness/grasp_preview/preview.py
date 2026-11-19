"""Inspect the next native waypoint without moving or branching the simulator."""
import numpy as np
from manipulation import boxes
from drawer_skills import DrawerSkills

class WaypointReader(DrawerSkills):
    """Run the native offset loop against constant positions and a recording hand."""
    def __init__(self,api):
        self.pos=api.pos
        self.mode=api.mode
        self.diagnostic=api.diagnostic
        self._body_frame=api._body_frame
        self._trace=[]
        self.closes=[]
    def transfer(self,item,target,grasp=(0,0,0),drop=(0,0,.08)):
        self.closes.append(self.pos(item)+grasp)
        return 0.
    def move(self,xyz,steps):
        self.point=np.asarray(xyz)
    def grip(self,closed):
        if closed:self.closes.append(self.point.copy())

def install(lib):
    api=lib._sensors
    native_place,native_carry=api._place,api._carry
    move,grip=api.move,api.grip
    lib.offsets=[]
    lib.next_index=0
    lib.command=None
    lib.close_error=np.zeros(3)
    def plan(points):
        lib.local_vote=None
        lib.offsets=[np.asarray(p).copy() for p in points]
        lib.next_index=0
    def place(item,target,slot,count):
        reader=WaypointReader(api)
        DrawerSkills._place(reader,item,target,slot,count)
        plan([p-api.pos(item) for p in reader.closes])
        return native_place(item,target,slot,count)
    def carry(item,target,offsets,*args,**kwargs):
        plan(offsets)
        return native_carry(item,target,offsets,*args,**kwargs)
    def measured_move(point,steps=30):
        lib.command=np.array(point)
        return move(point,steps)
    def measured_grip(closed):
        if closed and lib.live and lib.command is not None:
            lib.close_error=api.eef()-lib.command
            lib.next_index+=1
        return grip(closed)
    api._place,api._carry,api.move,api.grip=place,carry,measured_move,measured_grip

def inspect(lib,event):
    api=lib._sensors
    a=lib.approach
    ix=lib.next_index
    if not a or ix<1 or ix>=len(lib.offsets):return {"available":False}
    item=a["item"]
    centers,radii=boxes(api,item)
    if not len(centers):return {"available":False}
    point=api.pos(item)+lib.offsets[ix]
    direction=api._wrist()[:,0].copy()
    direction[2]=0.
    direction/=max(.001,np.linalg.norm(direction))
    cross=np.array([-direction[1],direction[0],0.])
    relative=centers-point
    # Project the collision primitives intersecting the new finger path.
    section=(abs(relative@cross)<=radii@abs(cross)+.006)&(abs(relative[:,2])<=radii[:,2]+.009)
    if not section.any():section=np.ones(len(centers),dtype=bool)
    offset=relative[section]@direction
    extent=radii[section]@abs(direction)
    left,right=float(min(offset-extent)),float(max(offset+extent))
    width=right-left
    aperture=.078
    changed=float(np.linalg.norm(lib.offsets[ix]-lib.offsets[ix-1]))
    reach=float(np.linalg.norm(lib.close_error))
    above=float(point[2]+max(0.,lib.close_error[2])-max((centers+radii)[:,2]))
    accessible=(width<=aperture-.003 or min(abs(left),abs(right))<=.014) and above<.005
    site=receiving_site(lib)
    sim=api._env.sim
    margin=.029 if site is None else float(min(sim.model.site_size[site][:2]))
    uncertainty=float(np.linalg.norm([.5*width,np.linalg.norm(lib.close_error[:2]),event["object_motion"]]))
    return dict(available=True,point=point.round(5).tolist(),width=round(width,5),aperture=aperture,
                reach_error=round(reach,5),edge_access=bool(accessible),displaced=round(changed,5),
                landing_margin=round(max(.025,margin),5),uncertainty=round(uncertainty,5),
                predicted_above=round(above,5))

def receiving_site(lib):
    a=lib.approach
    api=lib._sensors
    site=a["site"]
    sim=api._env.sim
    if site is None:
        # A planner may bind a receiving body instead of its interior. A single
        # centered, volumetric box on that body describes the very same aperture.
        interior=[]
        for name,entry in lib.scene.items():
            if entry.get("parent")!=a["target"]:continue
            index=sim.model.site_name2id(name)
            half=sim.model.site_size[index]
            if int(sim.model.site_type[index])!=6 or min(half)<.01:continue
            delta=sim.data.site_xpos[index]-api.pos(a["target"])
            if np.linalg.norm(delta[:2])<=min(half[:2]):interior.append(index)
        if len(interior)==1:site=interior[0]
    return site
