"""Read-only joint geometry, used to drive the same contact controller in all arms."""
import numpy as np
from scipy.spatial.transform import Rotation

def observed_joint(api, target):
    sim=api._env.sim
    env=api._env.env
    info=api.catalog().get(target,{})
    owner=info.get("parent") or target
    if owner not in env.objects_dict and owner not in env.fixtures_dict:
        return None
    if target in env.object_sites_dict:
        names=env.object_sites_dict[target].joints
    else:
        names=[]
    if not names:
        names=env.get_object(owner).joints
    ids=[sim.model.joint_name2id(n) for n in names]
    ids=[j for j in ids if sim.model.jnt_limited[j] and sim.model.jnt_type[j] in (2,3)]
    return ids[0] if len(ids)==1 else None

def fraction(api,j):
    m,d=api._env.sim.model,api._env.sim.data
    lo,hi=m.jnt_range[j]
    return float((d.qpos[m.jnt_qposadr[j]]-lo)/(hi-lo))

def drive(api,j,high):
    """Pinch the exposed moving link and follow its measured slide or hinge axis."""
    sim=api._env.sim
    m,d=sim.model,sim.data
    geoms=np.flatnonzero((m.geom_bodyid==m.jnt_bodyid[j]) & (m.geom_group==0))
    if not len(geoms):
        return
    centers=d.geom_xpos[geoms].copy()
    top=centers[centers[:,2]>centers[:,2].max()-.009]
    contact=top.mean(axis=0)
    axis=d.xaxis[j].copy()
    center=d.xanchor[j].copy()
    limit=m.jnt_range[j][int(high)]
    sign=1 if high else -1
    up=np.array([0.,0.,1.])
    api.grip(False)
    api.move(contact+.09*up,32)
    if len(top)>1:
        _,_,basis=np.linalg.svd(top-top.mean(axis=0),full_matrices=False)
        along=basis[0].copy()
        along[2]=0
        if np.linalg.norm(along)>.2:
            closing=np.cross(up,along)
            wrist=api._wrist()[:,0]
            angle=np.arctan2(np.cross(wrist,closing)[2],np.dot(wrist,closing))
            angle=(angle+np.pi/2)%np.pi-np.pi/2
            if m.jnt_type[j]==3 and abs(axis[2])>.8:
                # The jaws are symmetric. Leave wrist travel available in the desired direction.
                swing=(limit-d.qpos[m.jnt_qposadr[j]])*axis[2]
                angle=min((angle-np.pi,angle,angle+np.pi),key=lambda a:max(abs(a),abs(a+swing)))
            api._orient(Rotation.from_rotvec(up*angle).as_matrix()@api._wrist())
    api.move(contact,30)
    api.grip(True)
    q0=d.qpos[m.jnt_qposadr[j]]
    start=api.eef().copy()
    wrist=api._wrist()
    travel=float(limit-q0)
    hinge=m.jnt_type[j]==3
    overshoot=.022 if hinge else .003
    for t in range(1,146):
        # Brake near the stop to reduce rebound when the fingers release.
        u=min(1.,t/95)
        amt=(travel+sign*overshoot)*(1-(1-u)**2)
        if t>95 and sign*(limit-d.qpos[m.jnt_qposadr[j]])>0:
            # Add pressure gradually when contact compliance absorbs the last part of the motion.
            amt+=sign*(.002 if hinge else .00015)*(t-95)
        rot=Rotation.from_rotvec(axis*amt)
        xyz=center+rot.apply(start-center) if hinge else start+axis*amt
        orient=rot.as_matrix()@wrist if hinge else wrist
        err=Rotation.from_matrix(orient@api._wrist().T).as_rotvec()
        api.action([*((xyz-api.eef())*15),*(err*3),1])
    api.grip(False)
    api.move(api.eef()+up*.08,18)
