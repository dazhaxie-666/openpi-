"""Relative image-plane palm turn -> tabletop gripper yaw, independently of XYZ.

Only the five palm-base image landmarks are used. This is one deliberate
rotation channel, not monocular metric 3D wrist reconstruction.
"""
import numpy as np


def wrap(angle):return float(np.arctan2(np.sin(angle),np.cos(angle)))


class DirectPalmYaw:
    def __init__(self):
        self.reference=None;self.samples=[];self.stamp=None;self.good_stamp=None
        self.yaw=0.;self.raw_angle=0.;self.state='WAIT';self.reason='show palm briefly'
        self.candidate=None

    def hold(self,reason):
        self.state='HOLD' if self.reference is not None else 'WAIT';self.reason=reason
        return self.yaw

    def update(self,normalized,stamp,image_size=(1280,720)):
        if not np.isfinite(stamp):return self.hold('invalid palm clock')
        if self.stamp is not None and stamp<=self.stamp:return self.yaw
        dt=.05 if self.stamp is None else min(.1,stamp-self.stamp);self.stamp=float(stamp)
        p=None if normalized is None else np.asarray(normalized,float)
        if p is None or p.shape!=(21,3):return self.hold('palm missing; keep yaw')
        p=p[[0,5,9,13,17],:2]
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):return self.hold('palm clipped')
        p=p*np.asarray(image_size);p-=p.mean(0)
        sv=np.linalg.svd(p,compute_uv=False)
        if sv[0]<18.*image_size[1]/720. or sv[1]<.10*sv[0]:return self.hold('palm edge-on / too small')
        if self.reference is None:
            self.samples.append(p/max(np.linalg.norm(p),1e-9));self.samples=self.samples[-5:]
            if len(self.samples)<5:return self.hold('palm neutral calibration')
            if np.max(np.linalg.norm(self.samples-np.mean(self.samples,axis=0),axis=(1,2)))>.08:
                return self.hold('hold palm briefly for neutral')
            self.reference=np.mean(self.samples,axis=0)
        a=self.reference;u,_,v=np.linalg.svd(a.T@p)
        rotation=u@np.diag([1.,np.linalg.det(u@v)])@v
        scale=float(np.sum((a@rotation)*p)/np.sum(a*a))
        error=float(np.linalg.norm(scale*a@rotation-p)/np.linalg.norm(p))
        if scale<=0 or error>.20:return self.hold('palm orientation uncertain')
        angle=wrap(np.arctan2(rotation[0,1],rotation[0,0]))
        if abs(angle)>np.deg2rad(115):return self.hold('palm flip ambiguous')
        gap=self.good_stamp is None or stamp-self.good_stamp>.25 or self.state=='HOLD'
        innovation=abs(wrap(angle-self.raw_angle))
        if gap or innovation>np.deg2rad(12)+np.deg2rad(360)*dt+1e-9:
            consistent=self.candidate is not None and 0<stamp-self.candidate[1]<=.2 and abs(wrap(angle-self.candidate[0]))<np.deg2rad(12)
            if not consistent:
                self.candidate=(angle,stamp);return self.hold('confirm palm orientation')
        self.candidate=None;self.raw_angle=angle;self.good_stamp=float(stamp)
        target=float(np.clip(-angle,-np.deg2rad(70),np.deg2rad(70)))
        if abs(target)<np.deg2rad(1):target=0.
        step=(target-self.yaw)*(1-np.exp(-2*np.pi*4*dt))
        self.yaw+=float(np.clip(step,-np.deg2rad(90)*dt,np.deg2rad(90)*dt))
        self.state='TRACKING';self.reason='relative palm turn'
        return self.yaw
