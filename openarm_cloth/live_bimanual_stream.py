"""Two anatomical arms/hands from the same live USB frame; bounded latest slot."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import combinations
import threading
import time
import traceback

import cv2
import numpy as np

from human_teleop_core import ROOT,human_arm_pose,human_wrist,human_upper_body_wrist
from human_gripper_gestures import fist_features
from hand_association import associate_hands
from member_b_realtime import USBLatestFrame
from smoothing import VectorOneEuroFilter
from live_observation_quality import credible_hand,body_hand_consistency,ConfirmedHands,wrist_tolerance
from motion_observation import SourceClock

SIDES=('left','right')


def assign_candidates(candidates,pose,previous,image_size=(1280,720),predictions=None,
                      expected_sides=SIDES):
    """Deduplicate overlapping crops, then one-to-one anatomical association.

    One physical detection can never drive both grippers. Ambiguous overlapping
    hands remain missing; they are not assigned just to fill both channels.

    In a single-arm task ``expected_sides`` has one entry. A lone credible hand
    is then bound to that side without requiring it to sit near a drifting Pose
    wrist. Bimanual callers keep the default two-sided anatomical association.
    """
    expected=(expected_sides,) if isinstance(expected_sides,str) else tuple(expected_sides)
    if not expected or any(s not in SIDES for s in expected):
        raise ValueError('expected_sides must be a non-empty subset of left/right')
    unique=[]
    for c in sorted(candidates,key=lambda c:c['score'],reverse=True):
        if not any(np.linalg.norm(c['wrist']-d['wrist'])<.035 for d in unique): unique.append(c)
    if len(expected)==1 and len(unique)==1 and np.isfinite(unique[0]['wrist']).all():
        # The operator has already selected one controlled arm. A second hand
        # is not available to disambiguate anatomy here, so do not let a noisy
        # Pose wrist discard an otherwise credible hand detection.
        return {expected[0]:unique[0]}
    if pose is None or np.shape(pose)!=(33,4):return {}
    # Association requires a fresh, visible body wrist. Labels and previous
    # positions may break ties, but cannot invent an anatomical anchor.
    pose=np.asarray(pose,dtype=float).copy()
    for j,s in enumerate(SIDES):
        prediction=(predictions or {}).get(s)
        if prediction is not None and pose[15+j,3]<.6 and np.isfinite(pose[[11,12]][:,[0,1,3]]).all() and np.min(pose[[11,12],3])>=.65:
            point,age=prediction
            if 0<=age<=.12 and np.isfinite(point).all():pose[15+j,:2]=point;pose[15+j,3]=1.
    for wrist in (15,16):
        if not np.isfinite(pose[wrist,[0,1,3]]).all() or pose[wrist,3]<.6:
            pose[wrist,3]=0.
    if not np.any(pose[[15,16],3]>=.6):return {}
    previous=np.full((2,2),np.nan)  # no missing-body fallback in associate_hands
    wh=np.asarray(image_size,dtype=float)
    tolerance=wrist_tolerance(pose,image_size)
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
    if len(expected)==1:
        best={side:candidate for side,candidate in best.items() if side in expected}
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


def roi_bounds(pose,side,width,height,hand_norm=None,allow_missing_elbow=False):
    """Causal square crop; a foreshortened forearm must not shrink a large hand.

    The most recently seen complete hand supplies its extent. Shoulder width
    and image height provide context when that hand is missing or a fist opens.
    At an image edge move the square inside, rather than clipping off fingers.
    """
    elbow,wrist=(13,15) if side=='left' else (14,16)
    if pose is None or pose[wrist,3]<.6 or not np.isfinite(pose[[11,12,wrist]][:,[0,1,3]]).all(): return None
    if np.any(pose[wrist,:2]<0) or np.any(pose[wrist,:2]>1): return None
    wh=np.array([width,height]);w=pose[wrist,:2]*wh
    shoulder=np.linalg.norm((pose[11,:2]-pose[12,:2])*wh)
    context=max(min(wh)*.34,shoulder*.7)
    finite_elbow=np.isfinite(pose[elbow,[0,1,3]]).all()
    elbow_ok=finite_elbow and (not allow_missing_elbow or
        pose[elbow,3]>=.6 and np.all((pose[elbow,:2]>=0)&(pose[elbow,:2]<=1)))
    if not elbow_ok and not allow_missing_elbow:return None
    if elbow_ok:
        e=pose[elbow,:2]*wh;center=w+.4*(w-e);size=max(np.linalg.norm(w-e)*1.6,context)
    else:center=w;size=context
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


def single_hand_roi(pose,side,width,height,hand=None,predicted=None):
    """Keep shoulder-scaled context when a foreshortened hand shrinks its box.

    Recent complete hand bounds still supply their centre/extent. Preserve the
    existing forearm-context recovery when no recent hand is available: a
    wrist-centred replacement regressed startup on the recorded comparison.
    This does not change bimanual crops.
    """
    wh=np.array([width,height]);base=tracked_hand_roi(hand,width,height,predicted)
    if base is None:return roi_bounds(pose,side,width,height,allow_missing_elbow=True)
    p=None if pose is None else np.asarray(pose)
    shoulders=(p is not None and p.shape==(33,4) and np.isfinite(p[[11,12]][:,[0,1,3]]).all()
               and np.min(p[[11,12],3])>=.6)
    context=max(120.,float(np.linalg.norm((p[11,:2]-p[12,:2])*wh))) if shoulders else 120.
    center=(base[0]+base[1])/2;size=max(context,float(base[1][0]-base[0][0]))
    size=int(np.clip(np.ceil(size),min(100,min(wh)),min(wh)*.76))
    if size<100:return None
    lo=np.clip(np.rint(center-size/2).astype(int),0,wh-size)
    return lo,lo+size


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


def hand_landmarker_thresholds(confidence=.5,presence=None,tracking=None):
    """Detection score, landmark presence and tracking IoU are distinct gates."""
    values=dict(min_hand_detection_confidence=float(confidence),
                min_hand_presence_confidence=float(confidence if presence is None else presence),
                min_tracking_confidence=float(confidence if tracking is None else tracking))
    if any(not np.isfinite(v) or not 0<v<=1 for v in values.values()):raise ValueError('hand confidences must be finite in (0, 1]')
    return values


def apply_vision_reacquire(flags,stream,mapping=None):
    """Explicit recovery holds execution until the operator resumes with SPACE."""
    if not flags.get('reacquire'):return False
    flags['reacquire']=False
    if stream is None:return False
    flags['paused']=True
    flags['calibrate']=None
    stream.request_reacquire()
    if mapping is not None:mapping.reset()
    return True


class BimanualHumanStream:
    def __init__(self,device=0,pose_model='full',resolution=(1280,720),camera_factory=None,
                 hand_confidence=.5,hand_presence_confidence=None,hand_tracking_confidence=None,
                 expected_sides=SIDES,hand_search='adaptive',position_frame='original',body_position_source='mediapipe',elbow_recovery=False):
        self.latest=None;self.error=None;self.camera=None;self.closed=False;self.metadata={}
        self.hand_thresholds=hand_landmarker_thresholds(hand_confidence,hand_presence_confidence,hand_tracking_confidence)
        self.expected_sides=(expected_sides,) if isinstance(expected_sides,str) else tuple(expected_sides)
        if not self.expected_sides or any(s not in SIDES for s in self.expected_sides):
            raise ValueError('expected_sides must be a non-empty subset of left/right')
        if hand_search not in ('adaptive','full'):raise ValueError('unknown hand search policy')
        self.hand_search=hand_search
        if position_frame not in ('original','upper-body'):raise ValueError('unknown position frame')
        self.position_frame=position_frame
        if body_position_source not in ('mediapipe','rtmpose'):raise ValueError('unknown body position source')
        if body_position_source=='rtmpose' and (position_frame!='upper-body' or len(self.expected_sides)!=1):
            raise ValueError('2D body input requires single-arm upper-body position mode')
        self.body_position_source=body_position_source
        self.elbow_recovery=bool(elbow_recovery and body_position_source=='mediapipe' and len(self.expected_sides)==1)
        self.lock=threading.Lock();self.device=device;self.pose_model=pose_model;self.resolution=resolution
        self.vision_generation=self.requested_generation=0
        self.camera_factory=camera_factory;self.finished=False
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def snapshot(self):
        with self.lock: return self.latest

    def preview_snapshot(self):
        """Raw latest camera image for display only, never an accepted command."""
        camera=self.camera
        if camera is None or not hasattr(camera,'condition'):return None
        with camera.condition:return camera.latest

    def request_reacquire(self):
        """Invalidate old observations immediately; rebuild models in their owner thread.

        Coalesce key repeats while a reset is pending. In-flight inference from
        an earlier generation cannot republish an observation after this call.
        """
        with self.lock:
            if self.requested_generation==self.vision_generation:self.requested_generation+=1
            self.latest=None
            return self.requested_generation

    def vision_status(self):
        with self.lock:
            return dict(generation=self.vision_generation,requested=self.requested_generation,
                        pending=self.vision_generation!=self.requested_generation)

    def _run(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            models=ROOT.parent/'pose2arm-main'/'models'
            def options(name): return python.BaseOptions(model_asset_path=str(models/name))
            with ExitStack() as stack:
                from human_body_2d import ImageBodyDetector,image_plane_wrist,same_person_image
                from live_reference_conditioning import AdaptiveReferenceFilter
                image_body=ImageBodyDetector() if self.body_position_source=='rtmpose' else None
                position_filters={s:AdaptiveReferenceFilter(min_cutoff=3.,beta=3.) for s in SIDES}
                vision_stack=stack.enter_context(ExitStack())
                def create_landmarkers():
                    body=vision_stack.enter_context(vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                        base_options=options(f'pose_landmarker_{self.pose_model}.task'),running_mode=vision.RunningMode.VIDEO,num_poses=1)))
                    full=vision_stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                        base_options=options('hand_landmarker.task'),running_mode=vision.RunningMode.VIDEO,num_hands=2,**self.hand_thresholds)))
                    # A one-hand crop can use VIDEO tracking without searching
                    # for an absent second hand every frame. Keep the two-hand
                    # full-image model for recovery and ambiguity checks.
                    crop_sides=() if len(self.expected_sides)==1 and self.hand_search=='full' else self.expected_sides
                    crops={
                        s:vision_stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                            base_options=options('hand_landmarker.task'),running_mode=vision.RunningMode.VIDEO,num_hands=1,**self.hand_thresholds))) for s in crop_sides}
                    return body,full,crops
                pose,hands,roi=create_landmarkers()
                pool=stack.enter_context(ThreadPoolExecutor(max_workers=4))
                recovery=None;recovery_error=None
                if self.elbow_recovery:
                    from elbow_recovery import BackgroundElbowRecovery,select_display_pose
                    try:recovery=BackgroundElbowRecovery(pool)
                    except Exception as error:recovery_error=f'{type(error).__name__}: {error}'
                self.camera=(self.camera_factory or USBLatestFrame)(self.device,*self.resolution,30)
                self.metadata=dict(self.camera.metadata,pose_model=self.pose_model,arms=list(SIDES),
                                   expected_sides=list(self.expected_sides),hand_thresholds=self.hand_thresholds,
                                   position_frame=self.position_frame,
                                   body_position_source=self.body_position_source,
                                   elbow_recovery_scope='background image elbow overlay only; control and hand crops remain MediaPipe' if recovery else None,
                                   elbow_recovery_error=recovery_error,
                                   body_2d_model_sha256=image_body.sha256 if image_body else None,
                                   body_2d_scope='image shoulder/elbow/wrist; nominal shoulder-relative Y/Z; no model 3D; palm supplies X' if image_body else None,
                                   identity='anatomical left -> robot left, right -> right; preview mirror only',
                                   hand_search=(('single-hand ROI tracking + body-guided recovery; two-hand full-frame audit at 5 Hz and on loss' if self.hand_search=='adaptive'
                                                 else 'full-frame every frame; one expected hand') if len(self.expected_sides)==1
                                                else 'square ROI + parallel full-frame; fresh body wrist + hand geometry + two-frame confirmation'),
                                   observation_gate='body/hand wrist consistency; missing anchors never use handedness fallback',
                                   vision_recovery='R: pause + recreate body/full-hand/ROI VIDEO trackers; no automatic resume',
                                   filter_clock='camera return timestamps, not variable inference completion')
                print('BIMANUAL CAMERA '+str(self.metadata),flush=True)
                filters={i:VectorOneEuroFilter(3,30,min_cutoff=1.5,beta=4) for i in (11,12,13,14,15,16,23,24)}
                hand_filters={s:VectorOneEuroFilter(63,30,min_cutoff=2,beta=5) for s in SIDES}
                previous=np.full((2,2),np.nan);seen=np.zeros(2);last_hand={s:None for s in SIDES}
                confirmation=ConfirmedHands()
                source_clock=SourceClock();processed=skipped=0;last_source=0
                previous_velocity=np.zeros((2,2));last_previous=np.full((2,2),np.nan)
                normals={s:None for s in SIDES};previous_pose=None;previous_hand={s:None for s in SIDES};body_context=None
                sequence=0;timestamp=-1;last_infer=None;last_full=0
                single_missing=True
                generation=0;reacquire_ms=0.;last_display_repair=-np.inf
                while not self.closed:
                    with self.lock:requested=self.requested_generation
                    if requested!=generation:
                        # Every inference from the previous iteration has been
                        # joined before models are closed. No extra model runs
                        # continuously, and old graph resources are released.
                        reset_start=time.perf_counter_ns()
                        vision_stack.close();pose,hands,roi=create_landmarkers()
                        filters={i:VectorOneEuroFilter(3,30,min_cutoff=1.5,beta=4) for i in (11,12,13,14,15,16,23,24)}
                        hand_filters={s:VectorOneEuroFilter(63,30,min_cutoff=2,beta=5) for s in SIDES}
                        previous=np.full((2,2),np.nan);seen=np.zeros(2);last_hand={s:None for s in SIDES}
                        confirmation=ConfirmedHands();previous_velocity=np.zeros((2,2))
                        normals={s:None for s in SIDES};previous_pose=None;previous_hand={s:None for s in SIDES}
                        last_infer=None;last_full=0;generation=requested
                        single_missing=True
                        body_context=None
                        last_display_repair=-np.inf
                        position_filters={s:AdaptiveReferenceFilter(min_cutoff=3.,beta=3.) for s in SIDES}
                        reacquire_ms=(time.perf_counter_ns()-reset_start)/1e6
                    sequence,read_start,read_end,bgr=self.camera.read_new(sequence)
                    valid_source,source_dt,source_reason=source_clock.accept(sequence,read_end/1e9)
                    if not valid_source:continue
                    skipped+=max(0,sequence-last_source-1);last_source=sequence;processed+=1
                    start=time.perf_counter_ns();capture_time=read_end;timestamp=max(timestamp+1,capture_time//1000000)
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    pose_future=pool.submit(pose.detect_for_video,image,timestamp)
                    image_body_future=pool.submit(image_body.detect,bgr,body_context) if image_body is not None else None
                    # Recovery runs on this same input frame alongside body/ROI
                    # models. Do not append a full detector after waiting for all
                    # other models, which creates periodic critical-path spikes.
                    full_future=None
                    single=len(self.expected_sides)==1
                    recent_single=single and last_hand[self.expected_sides[0]] is not None and capture_time-last_hand[self.expected_sides[0]]<=200000000
                    if (capture_time-last_full>=200000000 or
                            single and (self.hand_search=='full' or single_missing or not recent_single)):
                        last_full=capture_time;full_future=pool.submit(full_inference,mp,hands,image,timestamp)
                    hand_futures=[]
                    for s in self.expected_sides:
                        if s not in roi:continue
                        recent=last_hand[s] is not None and capture_time-last_hand[s]<=300000000
                        bounds=None
                        if single:
                            j=SIDES.index(s)
                            prediction=previous[j]+previous_velocity[j]*max(0.,min(.12,capture_time/1e9-seen[j])) if np.isfinite(previous[j]).all() else None
                            bounds=single_hand_roi(previous_pose,s,bgr.shape[1],bgr.shape[0],
                                previous_hand[s] if recent else None,prediction)
                        else:bounds=roi_bounds(previous_pose,s,bgr.shape[1],bgr.shape[0],previous_hand[s] if recent else None)
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
                    image_body_result=image_body_future.result() if image_body_future is not None else None
                    position_norm=image_body_result['norm'] if image_body_result is not None else norm
                    body_context=position_norm
                    previous[(capture_time/1e9-seen)>.2]=np.nan
                    image_size=(bgr.shape[1],bgr.shape[0])
                    candidates=[c for c in candidates if credible_hand(c,image_size)]
                    pose_trusted,pose_quality=body_hand_consistency(norm,candidates,image_size)
                    position_confidence=.45 if image_body is not None else .6
                    position_trusted,position_quality=body_hand_consistency(position_norm,candidates,image_size,require_elbow=False,min_confidence=position_confidence)
                    display_trusted,display_quality=body_hand_consistency(position_norm,candidates,image_size,min_confidence=position_confidence)
                    if image_body is not None and not same_person_image(norm,position_norm,image_size):
                        position_trusted={s:False for s in SIDES};display_trusted={s:False for s in SIDES}
                        position_quality={s:'body identity unconfirmed' for s in SIDES}
                        display_quality=position_quality.copy()
                    association_pose=None if position_norm is None else position_norm.copy()
                    display_pose_norm=position_norm;display_stamp=capture_time/1e9;display_repaired=False
                    display_cache=None
                    if recovery is not None:
                        display_cache=recovery.observe(bgr,norm,capture_time/1e9,generation,
                            any(not display_trusted[s] for s in self.expected_sides) or capture_time/1e9-last_display_repair<1.)
                        display_pose_norm,display_stamp,display_repaired=select_display_pose(
                            norm,display_cache,capture_time/1e9,image_size,self.expected_sides)
                        if display_repaired:
                            last_display_repair=capture_time/1e9
                            display_trusted,display_quality=body_hand_consistency(display_pose_norm,candidates,image_size)
                    if association_pose is not None:
                        for j,s in enumerate(SIDES):
                            if not position_trusted[s]:association_pose[15+j,3]=0.
                    # Do not aim the next hand crop at a body wrist that has
                    # just failed consistency checks. Full-frame recovery runs
                    # independently and can reacquire a real visible hand.
                    previous_pose=association_pose
                    predictions={s:(previous[j]+previous_velocity[j]*min(.12,max(0.,capture_time/1e9-seen[j])),capture_time/1e9-seen[j])
                                 for j,s in enumerate(SIDES) if np.isfinite(previous[j]).all()}
                    associated=assign_candidates(candidates,association_pose,previous,image_size,predictions,
                                                 expected_sides=self.expected_sides)
                    assigned=confirmation.update(associated,capture_time/1e9,image_size)
                    single_missing=single and self.expected_sides[0] not in assigned
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
                        image_point=image_plane_wrist(position_norm,s,image_size,min_confidence=position_confidence) if image_body is not None and position_trusted[s] else None
                        if image_point is not None:image_point=position_filters[s].update(image_point,capture_time/1e9)
                        arms[s]=dict(pose=arm,point=human_wrist(world,s) if pose_trusted[s] else None,hand_valid=candidate is not None,
                                     hand_quality='confirmed' if candidate is not None else ('awaiting second observation' if s in associated else 'no hand at trusted body wrist'),
                                     position_point=(image_point if image_body is not None else
                                         (human_upper_body_wrist(world,s) if self.position_frame=='upper-body' else
                                          human_wrist(world,s,require_elbow=False)) if position_trusted[s] else None),
                                     hand_raw=hraw,hand_filtered=hfiltered,hand_norm=hnorm,fist=fist,distance=distance,
                                     roi_used=candidate is not None and candidate['roi'] is not None)
                    annotated=bgr.copy()
                    if position_norm is not None:
                        for index,s in enumerate(SIDES):
                            color=(255,180,0) if s=='left' else (0,255,255)
                            if not display_trusted[s]:color=(110,110,110)
                            for a,b in ((11+index,13+index),(13+index,15+index)):
                                coords=position_norm[[a,b],:2]
                                if np.isfinite(coords).all() and np.all((coords>=0)&(coords<=1)):
                                    points=[tuple((position_norm[i,:2]*[bgr.shape[1],bgr.shape[0]]).astype(int)) for i in (a,b)]
                                    cv2.line(annotated,*points,color,3)
                            hn=arms[s]['hand_norm']
                            if hn is not None:
                                for p in hn: cv2.circle(annotated,(int(p[0]*bgr.shape[1]),int(p[1]*bgr.shape[0])),3,color,-1)
                    result=dict(seq=sequence,read_start=read_start,read_end=read_end,infer_start=start,
                                vision_generation=generation,vision_reacquire_ms=reacquire_ms,
                                done=time.perf_counter_ns(),arms=arms,raw_image=bgr,image=annotated,
                                pose_raw=raw,pose_filtered=world,pose_norm=norm,
                                position_norm=position_norm,display_trusted=display_trusted,display_quality=display_quality,
                                display_pose_norm=display_pose_norm,display_pose_read_end=int(display_stamp*1e9),
                                display_min_confidence=position_confidence,display_elbow_repaired=display_repaired,
                                display_elbow_ms=display_cache['ms'] if display_cache is not None else 0.,
                                display_elbow_error=recovery.error if recovery is not None else recovery_error,
                                body_position_source=self.body_position_source,
                                position_min_confidence=position_confidence,
                                body_2d_score=image_body_result['score'] if image_body_result is not None else None,
                                body_2d_ms=image_body_result['ms'] if image_body_result is not None else 0.,
                                pose_trusted=pose_trusted,pose_quality=pose_quality,
                                position_trusted=position_trusted,position_quality=position_quality,
                                source_dt=source_dt,source_reason=source_reason,
                                captured_sequence=sequence,processed_sources=processed,skipped_capture_sources=skipped,nonmonotonic_sources=source_clock.rejected,
                                pose_ms=(pose_done-start)/1e6,hand_ms=(hand_done-start)/1e6,
                                roi_model_ms=np.asarray(roi_times))
                    result.update(hand_full_scan=full_future is not None,hand_candidate_count=len(candidates),
                                  hand_search_policy=self.hand_search)
                    with self.lock:
                        if generation==self.requested_generation:
                            self.latest=result;self.vision_generation=generation
        except EOFError:
            self.finished=True
        except Exception:
            self.error=traceback.format_exc();print(self.error,flush=True)
        finally:
            if self.camera is not None: self.camera.close()

    def close(self):
        self.closed=True;self.thread.join(timeout=5)
