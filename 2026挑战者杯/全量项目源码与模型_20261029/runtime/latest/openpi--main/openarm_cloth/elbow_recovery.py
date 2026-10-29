"""Latest-only 2D elbow display repair. Never supplies motor or hand-ROI input."""
import numpy as np
from human_body_2d import ImageBodyDetector,same_person_image
from live_observation_quality import body_hand_consistency


def select_display_pose(pose,cache,stamp,image_size,sides):
    """Repair only image elbows; keep current MP shoulders/wrists and all control inputs."""
    if pose is None:return pose,stamp,False
    if cache is None or not 0<=stamp-cache['capture_s']<=.2:return pose,stamp,False
    other=cache['norm']
    if not same_person_image(pose,other,image_size):return pose,stamp,False
    result=pose.copy();changed=False
    for side in sides:
        elbow=14 if side=='right' else 13
        if not cache['valid'].get(side,False):continue
        p=pose[elbow];q=other[elbow]
        bad=(not np.isfinite(p[[0,1,3]]).all() or p[3]<.6 or np.any((p[:2]<0)|(p[:2]>1)))
        differs=np.linalg.norm((p[:2]-q[:2])*image_size)>40
        if bad or differs:
            result[elbow,:2]=q[:2];result[elbow,2]=np.nan
            # This is a display validity marker, not calibrated MP visibility.
            result[elbow,3]=1.;changed=True
    return result if changed else pose,cache['capture_s'] if changed else stamp,changed


class BackgroundElbowRecovery:
    def __init__(self,pool,detector=None):
        self.pool=pool;self.detector=detector if detector is not None else ImageBodyDetector()
        self.pending=None;self.cache=None;self.generation=None;self.last_submit=-np.inf;self.error=None

    def _detect(self,frame,reference):
        result=self.detector.detect(frame,reference)
        valid,_=body_hand_consistency(result['norm'],[],frame.shape[1::-1],min_confidence=.45)
        if not same_person_image(reference,result['norm'],frame.shape[1::-1]):valid={s:False for s in valid}
        return dict(result,valid=valid)

    def observe(self,frame,pose,stamp,generation,bad_arm):
        if generation!=self.generation:self.cache=None;self.generation=generation
        if self.pending is not None and self.pending[0].done():
            future,source,epoch=self.pending;self.pending=None
            try:
                value=future.result()
                if epoch==generation:self.cache=dict(value,capture_s=source,generation=epoch)
            except Exception as error:
                self.cache=None;self.error=f'{type(error).__name__}: {error}'
        period=.2 if bad_arm else 1.
        if (self.pending is None and pose is not None and stamp-self.last_submit>=period
                and np.isfinite(pose[0,[0,1,3]]).all() and pose[0,3]>=.5):
            self.last_submit=stamp
            self.pending=(self.pool.submit(self._detect,frame,pose.copy()),stamp,generation)
        if self.cache is not None and 0<=stamp-self.cache['capture_s']<=.2:return self.cache
        return None
