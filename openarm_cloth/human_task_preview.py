"""Readable status panel beside the unoccluded human camera preview."""
import textwrap
import cv2
import numpy as np


class PreviewCadence:
    """Limit expensive rendering, never queue old previews or sleep for them."""
    def __init__(self,hz=10.):
        if not np.isfinite(hz) or hz<=0:raise ValueError('preview Hz must be finite and positive')
        self.period=1./hz
        self.next_due=None

    def due(self,now):
        if self.next_due is not None and now<self.next_due-1e-9:return False
        if self.next_due is None:self.next_due=now+self.period
        else:
            # Keep the display phase: scheduling from the late control tick
            # drifts the deadline, then skips every other near-period tick.
            missed=max(1,int((now-self.next_due+1e-9)/self.period)+1)
            self.next_due+=missed*self.period
        return True


def calibration_prompt(summary):
    if summary.get('calibrated'):return None
    return 'GRIPPER NOT CALIBRATED: G pinch / O spread x10 each; hold 1.2s'


def framing_prompt(obs):
    if obs is None:return 'Waiting for camera; include shoulders, right forearm and hand'
    pose=obs.get('pose_norm')
    if pose is None or np.shape(pose)!=(33,4):return 'BODY LOST: show shoulders, right forearm and hand'
    pose=np.asarray(pose)
    shoulders=pose[[11,12]]
    if np.isfinite(shoulders).all() and np.min(shoulders[:,3])>=.6:
        if np.mean(shoulders[:,1])>.78:
            return 'POSE ESTIMATE LOST: R to reacquire; check shoulder / wrist overlay'
    if not obs.get('position_trusted',{}).get('right',True):
        return 'RIGHT WRIST LOST: keep right shoulder, forearm and wrist in view'
    return None


def poll_task_preview(window,key):
    """Keyboard and close events are serviced even between rendered frames."""
    # WIN32 waitKey(1) often waits one Windows timer quantum (~15ms).
    # pollKey still pumps HighGUI input/close events without that forced wait.
    code=cv2.pollKey()&255
    if code!=255:key(code)
    return cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)>=1


def compose_task_preview(camera,lines,stage,paused=False,start_requested=False,direct=True,prompt=None,calibration_warning=None,gripper_status=None,arm_status=None,interaction_hint=None):
    canvas=np.full((720,1440,3),22,np.uint8)
    canvas[:540,:960]=cv2.resize(camera,(960,540))
    cv2.line(canvas,(960,0),(960,720),(80,80,80),2)
    if paused:
        title='PAUSED - SPACE TO RESUME'
    elif stage=='ready':
        title=('START REQUESTED - SHOW WRIST' if direct else 'START REQUESTED - SHOW OPEN HAND') if start_requested else 'READY - SPACE / START TO BEGIN'
    else:title=stage.upper()+' - C TO RECENTER'
    if gripper_status and not paused:title=gripper_status
    if arm_status and not paused:title=arm_status
    cv2.putText(canvas,title,(16,590),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,230,255),2,cv2.LINE_AA)
    for i,line in enumerate(((prompt or calibration_warning or 'Wrist drives arm; missing fingers hold gripper.' if direct else
                             'Assisted mode requires measured hand gestures.'),
                             'SPACE start/pause | C recenter | R restart vision | Q save/exit')):
        color=(0,230,255) if i==0 and calibration_warning and not prompt else (235,235,235)
        cv2.putText(canvas,line,(16,632+i*30),cv2.FONT_HERSHEY_SIMPLEX,.65,color,1,cv2.LINE_AA)
    wrapped=[]
    for line in lines:wrapped.extend(textwrap.wrap(str(line),width=43) or [''])
    # The usual task diagnostics fit at this size. Fit extra wrapped lines
    # inside the panel rather than allowing the calibration status to vanish.
    spacing=min(28.,680/max(1,len(wrapped)))
    scale=min(.62,spacing/43.)
    for i,line in enumerate(wrapped):
        cv2.putText(canvas,line,(977,30+round(i*spacing)),cv2.FONT_HERSHEY_SIMPLEX,
                    scale,(245,245,245),1,cv2.LINE_AA)
    if interaction_hint is not None:
        from shared_interaction_ui import draw_shared_interaction_hint
        draw_shared_interaction_hint(canvas,interaction_hint)
    return canvas
