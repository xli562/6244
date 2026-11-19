"""Geometry-based recovery using fingertip contacts and OSC motion alone."""
import numpy as np
from scipy.spatial.transform import Rotation

UP=np.array([0.,0.,1.])

def geom_ids(api,name):
    obj=api._env.env.objects_dict.get(name)
    if obj is None:
        return []
    model=api._env.sim.model
    return [model.geom_name2id(n) for n in obj.contact_geoms]

def boxes(api,name):
    """Conservative world bounds of each collision primitive."""
    sim=api._env.sim
    ids=geom_ids(api,name)
    centers=sim.data.geom_xpos[ids].copy()
    radii=[]
    for i in ids:
        sizes=sim.model.geom_size[i].copy()
        kind=sim.model.geom_type[i]
        if kind==2: sizes[:]=sizes[0]
        elif kind in (3,5): sizes=np.array([sizes[0],sizes[0],sizes[1]+(sizes[0] if kind==3 else 0)])
        radii.append(abs(sim.data.geom_xmat[i].reshape(3,3))@sizes)
    return centers,np.array(radii)

def finger_contacts(api, item):
    sim=api._env.sim
    obj=set(geom_ids(api,item))
    fingers=api._env.robots[0].gripper.important_geoms
    left={sim.model.geom_name2id(n) for n in fingers["left_finger"]}
    right={sim.model.geom_name2id(n) for n in fingers["right_finger"]}
    touching=set()
    for c in sim.data.contact[:sim.data.ncon]:
        if c.geom1 in obj: touching.add(c.geom2)
        if c.geom2 in obj: touching.add(c.geom1)
    return int(bool(left&touching))+int(bool(right&touching))

def acquire(api,item,attempt):
    if api._closed: api.grip(False)
    api.move(api.eef()+UP*.09,18)
    centers,radii=boxes(api,item)
    lo=(centers-radii).min(axis=0)
    hi=(centers+radii).max(axis=0)
    height=hi[2]-lo[2]
    z=lo[2]+height*(.48 if attempt==0 else .68)
    mask=np.abs(centers[:,2]-z)<=radii[:,2]+.003
    if not mask.any(): mask=np.abs(centers[:,2]-z)==min(np.abs(centers[:,2]-z))
    bot=(centers[mask]-radii[mask]).min(axis=0)
    top=(centers[mask]+radii[mask]).max(axis=0)
    width=top-bot
    axis=int(np.argmin(width[:2]))
    closing=np.eye(3)[axis]
    point=(bot+top)/2
    point[2]=z
    long_axis=1-axis
    if width[long_axis]>1.8*width[axis] and width[axis]<.072:
        # Favor the thicker end of a tapered object over the narrow neck.
        mass=np.prod(np.maximum(radii[mask],.001),axis=1)
        centroid=np.average(centers[mask],axis=0,weights=mass)
        point[long_axis]=np.clip(centroid[long_axis],bot[long_axis]+.012,top[long_axis]-.012)
    if width[axis]>.071:
        # Wide objects need an edge pinch; choose the edge facing the free hand.
        direction=1 if api.eef()[axis]>=point[axis] else -1
        point[axis]+=direction*(width[axis]/2-.008)
    wrist=api._wrist()[:,0]
    angle=np.arctan2(np.cross(wrist,closing)[2],wrist@closing)
    angle=(angle+np.pi/2)%np.pi-np.pi/2
    api.move(point+UP*.10,22)
    api._orient(Rotation.from_rotvec(UP*angle).as_matrix()@api._wrist())
    api.move(point,24)
    z0=api.pos(item)[2]
    api.grip(True)
    api.move(api.eef()+UP*.065,18)
    return api.pos(item)[2]-z0>.025 and finger_contacts(api,item)>0

def deliver(api,item,target):
    centers,radii=boxes(api,item)
    bottom=float((centers-radii)[:,2].min())
    bottom_offset=api.pos(item)[2]-bottom
    goal=api.pos(target)
    reference=goal[2]
    centers,radii=boxes(api,target)
    if len(centers):
        central=np.all(np.abs(centers[:,:2]-goal[:2])<=radii[:,:2]+.003,axis=1)
        if central.any(): reference=max((centers+radii)[central,2])
    hand=api.eef()
    offset=hand-api.pos(item)
    z=reference+bottom_offset+offset[2]
    hover=max(hand[2],z+.10)
    api.move([*hand[:2],hover],18)
    api.move([*(goal+offset)[:2],hover],35)
    for dz in (.065,.020,.006):
        delta=api.pos(target)[:2]-api.pos(item)[:2]
        api.move([*(api.eef()[:2]+delta),z+dz],18)
    api.grip(False)
    api.move(api.eef()+UP*.09,16)
