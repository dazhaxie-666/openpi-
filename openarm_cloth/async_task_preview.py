"""Camera preview owns its GUI thread and reads a bounded latest-frame slot."""
from queue import SimpleQueue,Empty
import threading
import time
import os
import ctypes

import cv2
import numpy as np

from human_task_preview import compose_task_preview


HAND_EDGES=((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
            (15,16),(13,17),(0,17),(17,18),(18,19),(19,20))


def draw_tracking_overlay(camera,overlay,mirror,now_ns):
    """Display-only landmarks on fresh camera; report their age and expire them.

    Camera and inference have independent clocks. Never discard the skeleton
    just because a newer camera frame arrived, or feed display-held points to
    the robot. Green means accepted observation, grey means rejected estimate.
    """
    if not overlay:return 'WAITING FOR TRACKING'
    age_ms=max(0.,(now_ns-overlay['read_end'])/1e6)
    if age_ms>200:
        status=f'OVERLAY STALE {age_ms:.0f} ms - waiting for fresh tracking'
    else:
        pose_age_ms=max(0.,(now_ns-overlay.get('pose_read_end',overlay['read_end']))/1e6)
        pose=overlay.get('pose_norm') if pose_age_ms<=200 else None;statuses=[]
        def point(p):
            x=float(p[0]);y=float(p[1])
            if mirror:x=1-x
            return tuple(np.rint(np.clip([x,y],-2,3)*[960,540]).astype(int))
        def edge(a,b,color):
            cv2.line(camera,point(a),point(b),(0,0,0),6,cv2.LINE_AA)
            cv2.line(camera,point(a),point(b),color,3,cv2.LINE_AA)
        for side in overlay.get('sides',('right',)):
            trusted=overlay.get('position_trusted',{}).get(side,False)
            arm_trusted=overlay.get('arm_trusted',overlay.get('position_trusted',{})).get(side,False)
            arm=overlay.get('arms',{}).get(side,{})
            hand=arm.get('hand_norm') if arm.get('hand_valid') else None
            statuses.append(f'{side.upper()} WRIST {"OK" if trusted else "LOST"} HAND {"OK" if hand is not None else "LOST"}')
            color=(0,255,0) if arm_trusted else (130,130,130)
            indices=(12,14,16) if side=='right' else (11,13,15)
            if pose is not None:
                for a,b in zip(indices,indices[1:]):
                    xy=pose[[a,b],:2]
                    if (np.isfinite(xy).all() and np.all((xy>=0)&(xy<=1))
                            and np.min(pose[[a,b],3])>=overlay.get('min_pose_confidence',.6)):edge(pose[a],pose[b],color)
            if hand is not None:
                for a,b in HAND_EDGES:
                    if np.isfinite(hand[[a,b],:2]).all():edge(hand[a],hand[b],(0,255,0))
                for p in hand:
                    if np.isfinite(p[:2]).all():cv2.circle(camera,point(p),4,(0,220,255),-1,cv2.LINE_AA)
        status=f'OVERLAY {age_ms:.0f} ms | '+ ' | '.join(statuses)
    cv2.rectangle(camera,(6,6),(950,38),(20,20,20),-1)
    cv2.putText(camera,status,(14,28),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1,cv2.LINE_AA)
    return status


class LatestTaskPreview:
    def __init__(self,window,hz=30.,frame_provider=None,render=None):
        if not np.isfinite(hz) or hz<=0:raise ValueError('invalid preview rate')
        self.window=window;self.period=1./hz;self.frame_provider=frame_provider
        self.render=render;self.lock=threading.Lock();self.stop=threading.Event()
        self.payload=None;self.events=SimpleQueue();self.error=None
        self.rendered=0;self.camera_frames=set();self.started=None;self.ended=None
        self.cached_payload=None;self.cached_canvas=None;self.cached_camera_key=None;self.cached_camera=None
        self.work_ms=[]
        self.thread=threading.Thread(target=self._run,daemon=True,name='task-camera-preview');self.thread.start()

    def submit(self,payload):
        # Immutable display description; image buffers are never modified here.
        with self.lock:self.payload=dict(payload)

    def take_events(self):
        events=[]
        while True:
            try:events.append(self.events.get_nowait())
            except Empty:return events

    def _draw(self,payload,raw):
        image=payload.get('image')
        overlay=payload.get('overlay')
        if raw is not None:
            # A newly captured frame is useful to the operator before inference
            # completes. It never becomes a robot observation through this path.
            image=raw[3] if overlay or raw[0]!=payload.get('source_seq') else image
        if image is None:image=np.full((540,960,3),45,np.uint8)
        key=(id(image),payload.get('mirror',True))
        if key!=self.cached_camera_key:
            self.cached_camera=cv2.resize(image,(960,540))
            if key[1]:self.cached_camera=cv2.flip(self.cached_camera,1)
            self.cached_camera_key=key
        if payload is not self.cached_payload:
            self.cached_canvas=compose_task_preview(self.cached_camera,payload['lines'],payload['stage'],
                payload.get('paused',False),payload.get('start_requested',False),direct=True,
                prompt=payload.get('prompt'),calibration_warning=payload.get('calibration_warning'),
                gripper_status=payload.get('gripper_status'),arm_status=payload.get('arm_status'),
                interaction_hint=payload.get('interaction_hint'))
            self.cached_payload=payload
        # Status text changes at control cadence, not at camera cadence. Reuse
        # the panel and resize only genuinely new camera buffers.
        frame=self.cached_canvas.copy();frame[:540,:960]=self.cached_camera
        self.overlay_status=draw_tracking_overlay(frame[:540,:960],overlay,payload.get('mirror',True),time.perf_counter_ns())
        button='START' if payload['stage']=='ready' else 'RESUME' if payload.get('paused') else 'PAUSE'
        cv2.rectangle(frame,(12,488),(235,530),(45,160,45),-1)
        cv2.putText(frame,button,(28,518),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        cv2.imshow(self.window,frame)
        key=cv2.pollKey()&255
        if key!=255:self.events.put(key)
        if cv2.getWindowProperty(self.window,cv2.WND_PROP_VISIBLE)<1:
            self.events.put(ord('q'));self.stop.set()

    def _run(self):
        opened=False;timer=False
        try:
            # Match the existing B live runner: Windows' coarse default timer
            # otherwise turns 33 ms frame waits into visibly uneven intervals.
            if os.name=='nt':timer=ctypes.windll.winmm.timeBeginPeriod(1)==0
            if self.render is None:
                cv2.namedWindow(self.window,cv2.WINDOW_NORMAL);opened=True
                cv2.resizeWindow(self.window,1440,720)
                def mouse(event,x,y,flags,param):
                    if event==cv2.EVENT_LBUTTONUP and 12<=x<=235 and 488<=y<=530:self.events.put(32)
                cv2.setMouseCallback(self.window,mouse)
            self.started=time.perf_counter()
            while not self.stop.is_set():
                start=time.perf_counter()
                with self.lock:payload=self.payload
                if payload is not None:
                    raw=self.frame_provider() if self.frame_provider else None
                    (self.render or self._draw)(payload,raw)
                    self.work_ms.append((time.perf_counter()-start)*1000)
                    self.rendered+=1
                    if raw is not None:self.camera_frames.add(raw[0])
                self.stop.wait(max(0.,self.period-(time.perf_counter()-start)))
        except Exception as error:
            self.error=f'{type(error).__name__}: {error}';self.events.put(ord('q'))
        finally:
            self.ended=time.perf_counter()
            if opened:
                try:cv2.destroyWindow(self.window)
                except cv2.error:pass
            if timer:ctypes.windll.winmm.timeEndPeriod(1)

    def close(self):
        self.stop.set();self.thread.join(timeout=5)
        if self.thread.is_alive():self.error=self.error or 'preview thread did not stop'

    def report(self):
        elapsed=(self.ended or time.perf_counter())-(self.started or time.perf_counter())
        return dict(rendered_frames=self.rendered,distinct_camera_frames=len(self.camera_frames),
            active_wall_s=max(0.,elapsed),render_hz=self.rendered/elapsed if elapsed>0 else 0.,
            distinct_camera_hz=len(self.camera_frames)/elapsed if elapsed>0 else 0.,
            render_work_ms_p50=float(np.median(self.work_ms)) if self.work_ms else None,
            render_work_ms_p95=float(np.percentile(self.work_ms,95)) if self.work_ms else None,
            independent_of_control=True,error=self.error)
