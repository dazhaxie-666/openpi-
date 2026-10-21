"""Source-clock and explicit torso-reference contract for live human motion."""
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation


class SourceClock:
    def __init__(self,max_gap=.25):
        self.seq=-1;self.stamp=None;self.max_gap=float(max_gap);self.rejected=0

    def accept(self,seq,stamp):
        if not np.isfinite(stamp) or int(seq)<=self.seq or (self.stamp is not None and stamp<=self.stamp):
            self.rejected+=1;return False,0.,'NON_MONOTONIC_SOURCE'
        dt=0. if self.stamp is None else float(stamp-self.stamp)
        self.seq,self.stamp=int(seq),float(stamp)
        return True,dt,'SOURCE_GAP' if dt>self.max_gap else ''


def unit(vector):
    vector=np.asarray(vector,dtype=float);length=np.linalg.norm(vector)
    if not np.isfinite(vector).all() or length<1e-7:raise ValueError('DEGENERATE_FRAME')
    return vector/length


def torso_basis(shoulder_axis,up):
    x=unit(shoulder_axis);y=unit(up-x*np.dot(up,x));z=unit(np.cross(x,y))
    return np.column_stack((x,np.cross(z,x),z))


class TorsoReference:
    """Reliable hips, bounded short hold, or explicit calibrated upper-body mode.

    Upper-body mode assumes a camera-upright, mostly frontal neutral pose. Its
    virtual hips are a coordinate construction, never detected/metric evidence.
    Large torso turns require recalibration, rather than guessed invisible hips.
    """
    def __init__(self,mode='auto',frames=10):
        if mode not in ('auto','full_body','upper_body','original'):raise ValueError('unknown torso mode')
        self.mode=mode;self.frames=frames;self.samples=[];self.rotation=None
        self.neutral=None;self.width=None;self.last_hip_time=None;self.calibration_id='uncalibrated'
        self.active_mode=None;self.last_time=None

    def process(self,world,stamp,pose_norm=None):
        info=dict(valid=False,state='LOST',reason='TORSO_MISSING',mode=self.active_mode or self.mode,
                  calibration_id=self.calibration_id,virtual_hips=False,rotation=None)
        w=None if world is None else np.asarray(world,dtype=float)
        if w is None or w.shape!=(33,4) or not np.isfinite(w[[11,12],:]).all():return None,info
        if self.mode=='original':
            if not np.isfinite(w[[11,12,23,24]]).all():return None,info
            from retarget import compute_body_frame
            try:_,r=compute_body_frame(*[w[i,:3] for i in (11,12,23,24)])
            except ValueError:return None,info
            return w.copy(),dict(info,valid=True,state='TRACKING',reason='',rotation=r,mode='original')
        shoulder=w[12,:3]-w[11,:3];width=np.linalg.norm(shoulder)
        if min(w[[11,12],3])<.65 or not .12<width<.85:return None,dict(info,reason='SHOULDER_BASELINE_INVALID')
        middle=w[[11,12],:3].mean(axis=0)
        hips=np.isfinite(w[[23,24]]).all() and min(w[[23,24],3])>=.65
        if hips and pose_norm is not None:
            p=np.asarray(pose_norm)
            hips=bool(p.shape==(33,4) and np.isfinite(p[[23,24]]).all()
                      and np.all((p[[23,24],:2]>=0)&(p[[23,24],:2]<=1)))
        spine=middle-w[[23,24],:3].mean(axis=0) if hips else np.array([0.,-1.,0.])
        hips=bool(hips and .18<np.linalg.norm(spine)<.85 and np.linalg.norm(np.cross(unit(shoulder),unit(spine)))>.3)
        if self.neutral is None:
            if self.mode=='full_body' and not hips:return None,dict(info,state='REACQUIRING',reason='VISIBLE_HIPS_REQUIRED')
            up=unit(spine) if hips and self.mode!='upper_body' else np.array([0.,-1.,0.])
            try:rotation=torso_basis(shoulder,up)
            except ValueError:return None,dict(info,reason='TORSO_DEGENERATE')
            if self.samples and Rotation.from_matrix(self.samples[-1][0].T@rotation).magnitude()>.18:
                self.samples.clear()
            self.samples.append((rotation,width,hips))
            if len(self.samples)<self.frames:
                return None,dict(info,state='REACQUIRING',reason=f'TORSO_CALIBRATING_{len(self.samples)}/{self.frames}')
            self.neutral=Rotation.from_matrix(np.stack([s[0] for s in self.samples])).mean().as_matrix()
            self.width=float(np.median([s[1] for s in self.samples]))
            self.active_mode='full_body' if self.mode=='full_body' or (self.mode=='auto' and all(s[2] for s in self.samples)) else 'upper_body_calibrated'
            payload=dict(mode=self.active_mode,width=self.width,neutral=self.neutral.tolist())
            self.calibration_id=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
            self.rotation=self.neutral.copy()
        info.update(mode=self.active_mode,calibration_id=self.calibration_id)
        if not .7<width/self.width<1.4:return None,dict(info,reason='SHOULDER_SCALE_CHANGED')
        if self.active_mode=='full_body':
            if hips:
                up=unit(spine);self.last_hip_time=stamp
            elif self.last_hip_time is not None and stamp-self.last_hip_time<=.3:
                up=self.rotation[:,1];info.update(state='HOLD',reason='TORSO_SHORT_HOLD')
            else:return None,dict(info,reason='TORSO_UNOBSERVABLE')
        else:
            if np.arccos(np.clip(np.dot(unit(shoulder),self.neutral[:,0]),-1,1))>np.deg2rad(30):
                return None,dict(info,reason='UPPER_BODY_TURN_RECALIBRATE')
            up=self.neutral[:,1]
        try:r=torso_basis(shoulder,up)
        except ValueError:return None,dict(info,reason='TORSO_DEGENERATE')
        dt=.05 if self.last_time is None else stamp-self.last_time
        angle=Rotation.from_matrix(self.rotation.T@r).magnitude()
        if dt<=.25 and angle>np.deg2rad(8)+np.deg2rad(120)*max(0.,dt):
            return None,dict(info,reason='TORSO_INNOVATION_REJECTED')
        self.rotation=r;self.last_time=stamp
        accepted=w.copy()
        # A's unchanged torso math consumes these explicit reference points.
        # Preserve the untouched detector output in a separate raw field.
        hip_center=middle-.45*r[:,1]
        accepted[23,:3]=hip_center-.12*r[:,0];accepted[24,:3]=hip_center+.12*r[:,0]
        accepted[[23,24],3]=1.
        return accepted,dict(info,valid=True,state=info['state'] if info['state']=='HOLD' else 'TRACKING',reason=info['reason'] if info['state']=='HOLD' else '',
                             rotation=r.copy(),virtual_hips=True)
