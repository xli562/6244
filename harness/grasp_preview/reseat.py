"""A local pickup and reseat with a known worst-case control-step cost."""
import numpy as np
from manipulation import boxes,finger_contacts,UP

def near(event):
    return event["xy_gap"]<.105 and abs(event["end"][2])<.09

def cost(event):
    return sum((8,12,8,8,16,10,24)) if near(event) else sum((13,16,8,12,20,14,40))

def run(lib,item,target):
    api=lib._sensors
    centers,extent=boxes(api,item)
    if not len(centers):return False
    low=(centers-extent).min(0)
    high=(centers+extent).max(0)
    point=(low+high)*.5
    d=api._wrist()[:,0].copy()
    d[2]=0.
    d/=max(.001,np.linalg.norm(d))
    width=max((centers-point)@d+extent@abs(d))-min((centers-point)@d-extent@abs(d))
    if width>.073:
        sign=1 if np.dot(api.eef()-point,d)>=0 else -1
        point+=sign*d*(width/2-.008)
    goal=api.pos(target)
    short=lib.repair_profile=="short"
    travel=16 if short else 20
    api.move(point+UP*.077,8 if short else 13)
    api.move(point,12 if short else 16)
    z=api.pos(item)[2]
    lib.motor([0,0,0,0,0,0,1],8)
    api.move(api.eef()+UP*(.055 if short else .075),8 if short else 12)
    if api.pos(item)[2]-z<.017 or not finger_contacts(api,item):return False
    centers,extent=boxes(api,item)
    bottom=min((centers-extent)[:,2])
    offset=api.eef()-api.pos(item)
    surface=goal[2]
    rc,rr=boxes(api,target)
    if len(rc):
        middle=np.all(np.abs(rc[:,:2]-goal[:2])<=rr[:,:2]+.003,axis=1)
        if middle.any():surface=max((rc+rr)[middle,2])
    end=goal+offset
    end[2]=surface+(api.pos(item)[2]-bottom)+offset[2]+.004
    api.move([*end[:2],max(api.eef()[2],end[2]+.065)],travel)
    api.move(end,10 if short else 14)
    lib.inspect_release()
