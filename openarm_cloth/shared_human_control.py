"""Shared-task continuity using a currently observed, associated right wrist.

No new detector, predicted landmarks, model depth, or anatomical handedness
fallback. Shoulder calibration and the existing stream's hand identity gate
remain prerequisites. Losing that hand for >0.35s requires body reacquisition.
"""
import numpy as np
from human_grasp_task import HumanInput


class SharedWristContinuity:
    def __init__(self):
        self.stamp=None;self.generation=None;self.point=None;self.wrist=None
        self.reason='IMAGE_BODY_REQUIRED';self.fallback=False
        self.body_offset=np.zeros(3)

    def update(self,human,obs,tracker,now_s,source_clock=False,paused=False):
        self.reason='IMAGE_BODY_REQUIRED'
        if paused or obs is None or not source_clock and not 0<=now_s-obs['read_end']/1e9<=.35:
            return human
        generation=obs.get('vision_generation',0);stamp=obs['read_end']/1e9
        if generation!=self.generation:
            self.__init__();self.generation=generation
        arm=obs.get('arms',{}).get('right',{})
        h=None if not arm.get('hand_valid') else np.asarray(arm.get('hand_norm'),float)
        image=obs.get('raw_image');wh=np.asarray(image.shape[1::-1] if image is not None else (1280,720))
        palm=[0,5,9,13,17]
        observed=bool(h is not None and h.shape==(21,3) and np.isfinite(h[palm]).all()
            and np.all((h[palm,:2]>=0)&(h[palm,:2]<=1))
            and np.max(np.ptp(h[palm,:2]*wh,axis=0))>=18.*wh[1]/720.)
        if human.valid:
            if self.fallback:
                expected=self.point.copy()
                if observed and self.stamp is not None and 0<=stamp-self.stamp<=.35:
                    delta=(h[0,:2]-self.wrist)*wh
                    if np.linalg.norm(delta)<=80.*wh[1]/720.:
                        expected+=np.array([0.,delta[0],-delta[1]])*tracker.nominal_span/tracker.scale
                self.body_offset=expected-np.asarray(human.point)
            point=np.asarray(human.point)+self.body_offset
            self.fallback=False;self.reason='IMAGE_BODY_TRACKING'
            if observed:
                self.point=point.copy();self.wrist=h[0,:2].copy();self.stamp=stamp
            return HumanInput(human.seq,point,human.pinch_ratio,True,human.capture_s)
        reason=obs.get('position_quality',{}).get('right','')
        permitted=reason in ('body missing','body shoulders outside frame','shoulder/wrist confidence low')
        if (not permitted or not observed or tracker.scale is None or self.stamp is None
                or not 0<=stamp-self.stamp<=.35):
            return human
        delta=(h[0,:2]-self.wrist)*wh
        if np.linalg.norm(delta)>80.*wh[1]/720.:
            self.stamp=None;self.reason='IMAGE_HAND_JUMP_REJECTED';return human
        point=self.point+np.array([0.,delta[0],-delta[1]])*tracker.nominal_span/tracker.scale
        self.point=point.copy();self.wrist=h[0,:2].copy();self.stamp=stamp;self.fallback=True
        self.reason='IMAGE_ASSOCIATED_HAND_TRACKING'
        tracker.reason=self.reason
        return HumanInput(human.seq,point,human.pinch_ratio,True,human.capture_s)
