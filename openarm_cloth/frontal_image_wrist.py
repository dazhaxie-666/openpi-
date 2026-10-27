"""Fixed-camera lateral/vertical control with a fixed observed shoulder scale.

Nominal control units only: neither model Z nor a per-frame scale is used.
"""
import numpy as np


class FrontalImageWrist:
    def __init__(self,nominal_span=.4):
        self.nominal_span=float(nominal_span);self.samples=[];self.scale=None;self.stamp=None
        self.reason='IMAGE_CALIBRATING';self.generation=None
        self.pending_recenter=True

    def take_recenter(self,valid):
        ready=bool(valid and self.pending_recenter)
        if ready:self.pending_recenter=False
        return ready

    def update(self,pose,stamp,image_size=(1280,720)):
        if not np.isfinite(stamp):return None
        p=None if pose is None else np.asarray(pose,float);ids=[11,12,16]
        if p is None or p.shape!=(33,4) or not np.isfinite(p[ids][:,[0,1,3]]).all():
            self.reason='IMAGE_BODY_MISSING';return None
        if np.min(p[ids,3])<.6 or np.any((p[ids,:2]<0)|(p[ids,:2]>1)):
            self.reason='IMAGE_WRIST_OR_SHOULDERS_UNOBSERVED';return None
        wh=np.asarray(image_size);axis=(p[11,:2]-p[12,:2])*wh;span=np.linalg.norm(axis)
        if span<35.*wh[1]/720. or span>.9*wh[0] or abs(axis[1])>abs(axis[0]):
            self.reason='IMAGE_SHOULDERS_UNOBSERVED';return None
        if self.scale is None:
            if self.stamp is None or stamp>self.stamp:self.samples.append(float(span));self.stamp=stamp
            if len(self.samples)<5:self.reason='IMAGE_CALIBRATING';return None
            self.scale=float(np.median(self.samples))
        if not .55*self.scale<=span<=1.8*self.scale:
            self.reason='IMAGE_CAMERA_REFERENCE_CHANGED';return None
        center=p[[11,12],:2].mean(0)
        delta=(p[16,:2]-center)*wh*self.nominal_span/self.scale
        self.reason='IMAGE_TRACKING'
        return np.array([0.,delta[0],-delta[1]])
