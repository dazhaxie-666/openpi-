"""Two anatomical arms/hands from the same live USB frame; bounded latest slot."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import combinations
import threading
import time
import traceback

import cv2
import numpy as np

from human_teleop_core import ROOT,human_arm_pose,human_wrist
from human_gripper_gestures import fist_features
from hand_association import associate_hands
from member_b_realtime import USBLatestFrame
from smoothing import VectorOneEuroFilter
from live_observation_quality import credible_hand,body_hand_consistency,ConfirmedHands,wrist_tolerance
from motion_observation import SourceClock

SIDES=('left','right')


def assign_candidates(candidates,pose,previous,image_size=(1280,720),predictions=None):
    """Deduplicate overlapping crops, then one-to-one anatomical association.

    One physical detection can never drive both grippers. Ambiguous overlapping
    hands remain missing; they are not assigned just to fill both channels.
    """
    if pose is None or np.shape(pose)!=(33,4):return {}
    # Association requires a fresh, visible body wrist. Labels and previous
    # positions may break ties, but cannot invent an anatomical anchor.
    pose=np.asarray(pose,dtype=float).copy()
    for j,s in enumerate(SIDES):
        prediction=(predictions or {}).get(s)
        if prediction is not None and pose[15+j,3]<.6 and np.isfinite(pose[[11,12]]).all() and np.min(pose[[11,12],3])>=.65:
            point,age=prediction
            if 0<=age<=.12 and np.isfinite(point).all():pose[15+j,:2]=point;pose[15+j,3]=1.
    for wrist in (15,16):
        if not np.isfinite(pose[wrist]).all() or pose[wrist,3]<.6:
            pose[wrist,3]=0.
    if not np.any(pose[[15,16],3]>=.6):return {}
    previous=np.full((2,2),np.nan)  # no missing-body fallback in associate_hands
    wh=np.asarray(image_size,dtype=float)
    tolerance=wrist_tolerance(pose,image_size)
    unique=[]
    for c in sorted(candidates,key=lambda c:c['score'],reverse=True):
        if not any(np.linalg.norm(c['wrist']-d['wrist'])<.035 for d in unique): unique.append(c)
    best={};best_cost=float('inf')
    for subset in combinations(unique,min(2,len(unique))):
        # Pixel-space distance handles the camera's non-square aspect ratio.
        anchors=pose.copy();anchors[:,:2]*=wh
        slots=associate_hands([c['wrist']*wh for c in subset],['unknown']*len(subset),anchors,previous,max_distance=tolerance)
        result={};cost=0.
        for c,slot in zip(subset,slots):
            if slot is None: continue
            anchor=pose[(15,16)[slot],:2] if pose is not None and pose[(15,16)[slot],3]>=.3 else previous[slot]
            cost+=float(np.linalg.norm(c['wrist']-anchor)) if np.isfinite(anchor).all() else .1
            result[SIDES[slot]]=c
        if len(result)>len(best) or (len(result)==len(best) and cost<best_cost):best=result;best_cost=cost
    # Crossing/coincident wrists cannot be disambiguated by a handedness score.
    # Retain a side only if current geometry or a fresh prior distinguishes it.
    for side,candidate in list(best.items()):
        j=SIDES.index(side);other=1-j
        if pose[15+other,3]<.6:continue
        distances=[np.linalg.norm((candidate['wrist']-pose[15+k,:2])*wh) for k in (j,other)]
        if max(distances)<=tolerance and abs(distances[0]-distances[1])<max(8.,.12*tolerance):
            prediction=(predictions or {}).get(side);alternative=(predictions or {}).get(SIDES[other])
            distinguished=prediction is not None and alternative is not None and np.linalg.norm((candidate['wrist']-prediction[0])*wh)+15<np.linalg.norm((candidate['wrist']-alternative[0])*wh)
            if not distinguished:best.pop(side)
    return best


def tracked_hand_roi(hand,width,height,predicted=None):
    """Fresh last-hand crop recovery, independent of a temporarily bad elbow."""
    h=np.asarray(hand,dtype=float)
    if h.shape!=(21,3) or not np.isfinite(h).all():return None
    wh=np.array([width,height]);p=h[:,:2]*wh
    shift=np.zeros(2) if predicted is None else np.clip(np.asarray(predicted)*wh-p[0],-.1*wh,.1*wh)
    center=(p.min(axis=0)+p.max(axis=0))/2+shift
    size=int(np.clip(max(120.,np.max(np.ptp(p,axis=0))*1.9),min(100,min(wh)),min(wh)*.76))
    lo=np.clip(np.round(center-size/2).astype(int),0,wh-size)
    return lo,lo+size


def roi_bounds(pose,side,width,height,hand_norm=None):
    """Causal square crop; a foreshortened forearm must not shrink a large hand.

    The most recently seen complete hand supplies its extent. Shoulder width
    and image height provide context when that hand is missing or a fist opens.
    At an image edge move the square inside, rather than clipping off fingers.
    """
    elbow,wrist=(13,15) if side=='left' else (14,16)
    if pose is None or pose[wrist,3]<.6 or not np.isfinite(pose[[11,12,elbow,wrist]]).all(): return None
    if np.any(pose[wrist,:2]<0) or np.any(pose[wrist,:2]>1): return None
    wh=np.array([width,height]);w=pose[wrist,:2]*wh;e=pose[elbow,:2]*wh
    shoulder=np.linalg.norm((pose[11,:2]-pose[12,:2])*wh)
    context=max(min(wh)*.34,shoulder*.7)
    center=w+.4*(w-e);size=max(np.linalg.norm(w-e)*1.6,context)
    if hand_norm is not None:
        hand=np.asarray(hand_norm,dtype=float)
        if hand.shape==(21,3) and np.isfinite(hand).all():
            points=hand[:,:2]*wh;low=points.min(axis=0);high=points.max(axis=0)
            # Translate the last hand box to the newest available body wrist.
            shift=np.clip(w-points[0],-.12*wh,.12*wh)
            center=(low+high)/2+shift
            size=max(context,float(np.max(high-low))*1.7)
    size=int(np.clip(size,min(100,min(wh)),min(wh)*.76))
    lo=np.clip(np.round(center-size/2).astype(int),0,wh-size);hi=lo+size
    return (lo,hi) if np.min(hi-lo)>=100 else None


def roi_inference(mp,detector,rgb,bounds,timestamp,side):
    start=time.perf_counter_ns();lo,hi=bounds
    crop=np.ascontiguousarray(rgb[lo[1]:hi[1],lo[0]:hi[0]])
    result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop),timestamp)
    candidates=[]
    for i,hand in enumerate(result.hand_landmarks):
        norm=np.array([[(p.x*(hi[0]-lo[0])+lo[0])/rgb.shape[1],
                        (p.y*(hi[1]-lo[1])+lo[1])/rgb.shape[0],p.z*(hi[0]-lo[0])/rgb.shape[1]] for p in hand])
        cat=result.handedness[i][0]
        candidates.append(dict(wrist=norm[0,:2],norm=norm,
                               world=np.array([[p.x,p.y,p.z] for p in result.hand_world_landmarks[i]]),
                               label=cat.category_name,score=cat.score,roi=side))
    return candidates,(time.perf_counter_ns()-start)/1e6


def full_inference(mp,detector,image,timestamp):
    """Independent full-image recovery, scheduled in parallel, never serial UI work."""
    result=detector.detect_for_video(image,timestamp)
    candidates=[]
    for i,hand in enumerate(result.hand_landmarks):
        norm=np.array([[p.x,p.y,p.z] for p in hand]);cat=result.handedness[i][0]
        candidates.append(dict(wrist=norm[0,:2],norm=norm,
                               world=np.array([[p.x,p.y,p.z] for p in result.hand_world_landmarks[i]]),
                               label=cat.category_name,score=cat.score,roi=None))
    return candidates


class BimanualHumanStream:
    def __init__(self,device=0,pose_model='full',resolution=(1280,720),camera_factory=None,hand_confidence=.5):
        self.latest=None;self.error=None;self.camera=None;self.closed=False;self.metadata={}
        self.hand_confidence=float(hand_confidence)
        if not 0<self.hand_confidence<=1:raise ValueError('hand_confidence must be in (0, 1]')
        self.lock=threading.Lock();self.device=device;self.pose_model=pose_model;self.resolution=resolution
        self.camera_factory=camera_factory;self.finished=False
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def snapshot(self):
        with self.lock: return self.latest

    def _run(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            models=ROOT.parent/'pose2arm-main'/'models'
            def options(name): return python.BaseOptions(model_asset_path=str(models/name))
            with ExitStack() as stack:
                pose=stack.enter_context(vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=options(f'pose_landmarker_{self.pose_model}.task'),running_mode=vision.RunningMode.VIDEO,num_poses=1)))
                # MediaPipe's palm detector is less confident when the back of
                # the hand faces the camera. Lower thresholds increase recall;
                # geometry/consistency checks below still reject bad candidates.
                hand_gate=dict(min_hand_detection_confidence=self.hand_confidence,
                               min_hand_presence_confidence=self.hand_confidence,
                               min_tracking_confidence=self.hand_confidence)
                hands=stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                    base_options=options('hand_landmarker.task'),running_mode=vision.RunningMode.VIDEO,num_hands=2,**hand_gate)))
                roi={s:stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                    base_options=options('hand_landmarker.task'),running_mode=vision.RunningMode.VIDEO,num_hands=1,**hand_gate))) for s in SIDES}
                pool=stack.enter_context(ThreadPoolExecutor(max_workers=4))
                self.camera=(self.camera_factory or USBLatestFrame)(self.device,*self.resolution,30)
                self.metadata=dict(self.camera.metadata,pose_model=self.pose_model,arms=list(SIDES),
                                   hand_detection_confidence=self.hand_confidence,
                                   identity='anatomical left -> robot left, right -> right; preview mirror only',
                                   hand_search='square ROI + parallel full-frame; fresh body wrist + hand geometry + two-frame confirmation',
                                   observation_gate='body/hand wrist consistency; missing anchors never use handedness fallback',
                                   filter_clock='camera return timestamps, not variable inference completion')
                print('BIMANUAL CAMERA '+str(self.metadata),flush=True)
                filters={i:VectorOneEuroFilter(3,30,min_cutoff=1.5,beta=4) for i in (11,12,13,14,15,16,23,24)}
                hand_filters={s:VectorOneEuroFilter(63,30,min_cutoff=2,beta=5) for s in SIDES}
                previous=np.full((2,2),np.nan);seen=np.zeros(2);last_hand={s:None for s in SIDES}
                confirmation=ConfirmedHands()
                source_clock=SourceClock();processed=skipped=0;last_source=0
                previous_velocity=np.zeros((2,2));last_previous=np.full((2,2),np.nan)
                normals={s:None for s in SIDES};previous_pose=None;previous_hand={s:None for s in SIDES}
                sequence=0;timestamp=-1;last_infer=None;last_full=0
                while not self.closed:
                    sequence,read_start,read_end,bgr=self.camera.read_new(sequence)
                    valid_source,source_dt,source_reason=source_clock.accept(sequence,read_end/1e9)
                    if not valid_source:continue
                    skipped+=max(0,sequence-last_source-1);last_source=sequence;processed+=1
                    start=time.perf_counter_ns();capture_time=read_end;timestamp=max(timestamp+1,capture_time//1000000)
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    pose_future=pool.submit(pose.detect_for_video,image,timestamp)
                    # Recovery runs on this same input frame alongside body/ROI
                    # models. Do not append a full detector after waiting for all
                    # other models, which creates periodic critical-path spikes.
                    full_future=None
                    if capture_time-last_full>=200000000:
                        last_full=capture_time;full_future=pool.submit(full_inference,mp,hands,image,timestamp)
                    hand_futures=[]
                    for s in SIDES:
                        recent=last_hand[s] is not None and capture_time-last_hand[s]<=300000000
                        bounds=roi_bounds(previous_pose,s,bgr.shape[1],bgr.shape[0],previous_hand[s] if recent else None)
                        if bounds is None and recent and capture_time-last_hand[s]<=200000000:
                            j=SIDES.index(s)
                            prediction=previous[j]+previous_velocity[j]*max(0.,min(.12,capture_time/1e9-seen[j])) if np.isfinite(previous[j]).all() else None
                            bounds=tracked_hand_roi(previous_hand[s],bgr.shape[1],bgr.shape[0],prediction)
                        if bounds is not None: hand_futures.append(pool.submit(roi_inference,mp,roi[s],rgb,bounds,timestamp,s))
                    body=pose_future.result();pose_done=time.perf_counter_ns()
                    world=raw=norm=None
                    if body.pose_landmarks:
                        norm=np.array([[p.x,p.y,p.z,p.visibility] for p in body.pose_landmarks[0]])
                        world=np.array([[p.x,p.y,p.z,p.visibility] for p in body.pose_world_landmarks[0]]);raw=world.copy()
                        hz=30 if last_infer is None else 1/max(.01,min(.2,(capture_time-last_infer)/1e9))
                        for i,f in filters.items():
                            if world[i,3]>=.3:
                                for scalar in f.filters: scalar.freq=hz
                                world[i,:3]=f.filter(world[i,:3])
                    previous_pose=norm;last_infer=capture_time
                    candidates=[];roi_times=[]
                    for future in hand_futures:
                        found,ms=future.result();candidates.extend(found);roi_times.append(ms)
                    if full_future is not None: candidates.extend(full_future.result())
                    previous[(capture_time/1e9-seen)>.2]=np.nan
                    image_size=(bgr.shape[1],bgr.shape[0])
                    candidates=[c for c in candidates if credible_hand(c,image_size)]
                    pose_trusted,pose_quality=body_hand_consistency(norm,candidates,image_size)
                    association_pose=None if norm is None else norm.copy()
                    if association_pose is not None:
                        for j,s in enumerate(SIDES):
                            if not pose_trusted[s]:association_pose[15+j,3]=0.
                    # Do not aim the next hand crop at a body wrist that has
                    # just failed consistency checks. Full-frame recovery runs
                    # independently and can reacquire a real visible hand.
                    previous_pose=association_pose
                    predictions={s:(previous[j]+previous_velocity[j]*min(.12,max(0.,capture_time/1e9-seen[j])),capture_time/1e9-seen[j])
                                 for j,s in enumerate(SIDES) if np.isfinite(previous[j]).all()}
                    associated=assign_candidates(candidates,association_pose,previous,image_size,predictions)
                    assigned=confirmation.update(associated,capture_time/1e9,image_size)
                    hand_done=time.perf_counter_ns();arms={}
                    for index,s in enumerate(SIDES):
                        candidate=assigned.get(s);hraw=hfiltered=hnorm=fist=None;distance=None
                        if candidate is not None:
                            hraw=candidate['world'];hnorm=candidate['norm']
                            fist=fist_features(hraw,hnorm,(bgr.shape[1],bgr.shape[0]))
                            if last_hand[s] is not None and capture_time-last_hand[s]>400000000:
                                hand_filters[s]=VectorOneEuroFilter(63,30,min_cutoff=2,beta=5)
                            hz=30 if last_hand[s] is None else 1/max(.01,min(.2,(capture_time-last_hand[s])/1e9))
                            for scalar in hand_filters[s].filters: scalar.freq=hz
                            hfiltered=hand_filters[s].filter(hraw.reshape(-1)).reshape(21,3)
                            elapsed=capture_time/1e9-seen[index]
                            if np.isfinite(previous[index]).all() and .005<elapsed<.2:
                                previous_velocity[index]=np.clip((candidate['wrist']-previous[index])/elapsed,-2.,2.)
                            else:previous_velocity[index]=0.
                            previous[index]=candidate['wrist'];seen[index]=capture_time/1e9;last_hand[s]=capture_time
                            previous_hand[s]=hnorm.copy()
                            distance=float(np.linalg.norm(hfiltered[4]-hfiltered[8]))
                        arm=human_arm_pose(world,hfiltered,normals[s],side=s) if pose_trusted[s] else None
                        if arm is not None: normals[s]=arm['normal']
                        arms[s]=dict(pose=arm,point=human_wrist(world,s) if pose_trusted[s] else None,hand_valid=candidate is not None,
                                     hand_quality='confirmed' if candidate is not None else ('awaiting second observation' if s in associated else 'no hand at trusted body wrist'),
                                     position_point=human_wrist(world,s,require_elbow=False) if pose_trusted[s] else None,
                                     hand_raw=hraw,hand_filtered=hfiltered,hand_norm=hnorm,fist=fist,distance=distance,
                                     roi_used=candidate is not None and candidate['roi'] is not None)
                    annotated=bgr.copy()
                    if norm is not None:
                        for index,s in enumerate(SIDES):
                            color=(255,180,0) if s=='left' else (0,255,255)
                            if not pose_trusted[s]:color=(110,110,110)
                            for a,b in ((11+index,13+index),(13+index,15+index)):
                                points=[tuple((norm[i,:2]*[bgr.shape[1],bgr.shape[0]]).astype(int)) for i in (a,b)]
                                cv2.line(annotated,*points,color,3)
                            hn=arms[s]['hand_norm']
                            if hn is not None:
                                for p in hn: cv2.circle(annotated,(int(p[0]*bgr.shape[1]),int(p[1]*bgr.shape[0])),3,color,-1)
                    result=dict(seq=sequence,read_start=read_start,read_end=read_end,infer_start=start,
                                done=time.perf_counter_ns(),arms=arms,raw_image=bgr,image=annotated,
                                pose_raw=raw,pose_filtered=world,pose_norm=norm,
                                pose_trusted=pose_trusted,pose_quality=pose_quality,
                                source_dt=source_dt,source_reason=source_reason,
                                captured_sequence=sequence,processed_sources=processed,skipped_capture_sources=skipped,nonmonotonic_sources=source_clock.rejected,
                                pose_ms=(pose_done-start)/1e6,hand_ms=(hand_done-start)/1e6,
                                roi_model_ms=np.asarray(roi_times))
                    with self.lock: self.latest=result
        except EOFError:
            self.finished=True
        except Exception:
            self.error=traceback.format_exc();print(self.error,flush=True)
        finally:
            if self.camera is not None: self.camera.close()

    def close(self):
        self.closed=True;self.thread.join(timeout=5)
