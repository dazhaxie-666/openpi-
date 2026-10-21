"""Conservative current-frame body/hand checks before observations reach A.

Handedness scores classify left/right; they are not evidence that a patch of
clothing is a hand. No label-only or stale-track fallback authorizes control.
"""
import numpy as np

SIDES=('left','right')


def wrist_tolerance(pose,image_size=(1280,720)):
    wh=np.asarray(image_size,dtype=float)
    shoulder=np.linalg.norm((pose[11,:2]-pose[12,:2])*wh)
    return float(np.clip(.28*shoulder,30.*wh[1]/720.,90.*wh[1]/720.))


def credible_hand(candidate,image_size=(1280,720)):
    norm=np.asarray(candidate.get('norm'),dtype=float)
    world=np.asarray(candidate.get('world'),dtype=float)
    if norm.shape!=(21,3) or world.shape!=(21,3) or not np.isfinite(norm).all() or not np.isfinite(world).all():return False
    if np.any(norm[:,:2]<-.05) or np.any(norm[:,:2]>1.05):return False
    wh=np.asarray(image_size,dtype=float)
    palm_width=np.linalg.norm((norm[5,:2]-norm[17,:2])*wh)
    palm_length=np.linalg.norm((norm[[5,9,13,17],:2].mean(axis=0)-norm[0,:2])*wh)
    metric_width=np.linalg.norm(world[5]-world[17])
    span=np.max(np.ptp(norm[:,:2]*wh,axis=0))
    # A side-on palm may have almost zero projected width. Its long axis and
    # nondegenerate 3D palm can still support identity/orientation; pinch has a
    # separate evidence check and must not disable the entire arm/hand channel.
    return bool(max(palm_width,palm_length)>=6.*wh[1]/720.
                and .015<=metric_width<=.18 and span<=.65*wh[1])


def body_hand_consistency(pose,candidates,image_size=(1280,720)):
    trusted={s:False for s in SIDES};reasons={s:'body missing' for s in SIDES}
    if pose is None or np.shape(pose)!=(33,4):return trusted,reasons
    wh=np.asarray(image_size,dtype=float)
    unique=[]
    for c in candidates:
        if credible_hand(c,image_size) and not any(np.linalg.norm((c['wrist']-d['wrist'])*wh)<25.*wh[1]/720. for d in unique):unique.append(c)
    for j,s in enumerate(SIDES):
        ids=[11,12,13+j,15+j]
        if not np.isfinite(pose[ids]).all() or np.min(pose[ids,3])<.6:
            reasons[s]='shoulder/elbow/wrist confidence low';continue
        if np.any(pose[15+j,:2]<0) or np.any(pose[15+j,:2]>1):
            reasons[s]='body wrist outside frame';continue
        # Two separately located credible hands provide an independent check
        # against body wrists drifting onto the chest. A lone visible hand must
        # not disable the other otherwise valid arm.
        if len(unique)>=2 and min(np.linalg.norm((c['wrist']-pose[15+j,:2])*wh) for c in unique)>wrist_tolerance(pose,image_size):
            reasons[s]='body/hand wrist disagreement';continue
        trusted[s]=True;reasons[s]='tracking'
    return trusted,reasons


class ConfirmedHands:
    """New/reacquired hands need two nearby observations, never a cached image."""
    def __init__(self):
        self.previous={};self.last_time={};self.count={s:0 for s in SIDES}
        self.source_stamp=None

    def update(self,assigned,capture_s,image_size=(1280,720)):
        if self.source_stamp is not None and capture_s<=self.source_stamp:return {}
        self.source_stamp=capture_s
        confirmed={};wh=np.asarray(image_size,dtype=float)
        for s in SIDES:
            candidate=assigned.get(s)
            if candidate is None:
                self.count[s]=0
                continue
            dt=capture_s-self.last_time.get(s,-np.inf)
            close=(0<dt<=.3 and s in self.previous and
                   np.linalg.norm((candidate['wrist']-self.previous[s])*wh)<=max(35.,800.*dt)*wh[1]/720.)
            self.count[s]=self.count[s]+1 if close and self.count[s]>0 else 1
            self.previous[s]=candidate['wrist'].copy();self.last_time[s]=capture_s
            if self.count[s]>=2:confirmed[s]=candidate
        return confirmed


def b_joint_reference(goal,limits):
    """Project only decimal-rounding differences; reject real outliers locally."""
    try:q=np.asarray(goal,dtype=float)
    except (TypeError,ValueError):return None,'malformed A reference'
    if q.shape!=(7,) or not np.isfinite(q).all():return None,'nonfinite/malformed A reference'
    error=np.maximum(limits[:,0]-q,q-limits[:,1])
    if np.max(error)>1e-4:return None,'A reference outside B limits: joint %d by %.6g rad' % (int(np.argmax(error))+1,np.max(error))
    return np.clip(q,limits[:,0],limits[:,1]),None
