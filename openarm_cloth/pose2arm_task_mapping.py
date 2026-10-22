"""A arm references plus explicit torso, relative-palm and pinch adapters.

Only the bounded latest observation crosses the worker boundary. A's original
OpenArmLiveController owns calibration, filters, retargeting and pose IK.
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'pose2arm-main'))
from openarm_live import OpenArmLiveController
from human_teleop_core import human_wrist
from motion_observation import TorsoReference,SourceClock
from human_wrist_control import RelativePalmTracker,BWristRetargeter
from human_pinch_control import HumanPinchControl,PinchCalibration,pinch_features,load_personal_calibration
from retarget import OPENARM_BODY_TO_ROBOT

SIDES = ('left', 'right')


def observed_sides(obs, sides=SIDES):
    world=None if obs is None else obs.get('pose_raw')
    world=None if world is None else np.asarray(world,dtype=float)
    result={}
    for side in sides:
        shoulder,elbow,wrist=(11,13,15) if side=='left' else (12,14,16)
        ids=[11,12,23,24,shoulder,elbow,wrist]
        result[side]=bool(world is not None and world.shape==(33,4) and np.isfinite(world[ids]).all()
                          and np.min(world[[11,12,shoulder,elbow,wrist],3])>=.45
                          and obs.get('pose_trusted',{}).get(side,True))
    return result


class Pose2ArmTaskMapping:
    def __init__(self, fps=20., sides=SIDES,torso_mode='original',wrist_mode='full',
                 gripper_mode='continuous',calibration_path=None,model=None,source_dt=False):
        self.fps, self.sides = float(fps), tuple(sides)
        self.torso_mode,self.wrist_mode,self.gripper_mode=torso_mode,wrist_mode,gripper_mode
        self.model=model;self.source_dt=source_dt
        profiles=load_personal_calibration(calibration_path)
        self.pinch_calibrations={s:PinchCalibration(profiles.get(s)) for s in self.sides}
        self.calibration_capture=None
        self.reset()

    def reset(self):
        held_fingers=getattr(self,'fingers',{s:.044 for s in self.sides}).copy()
        self.controllers = {s: OpenArmLiveController(self.fps, n_calib_frames=10, side=s) for s in self.sides}
        self.grippers = {s: HumanPinchControl(self.gripper_mode,self.pinch_calibrations[s]) for s in self.sides}
        for side in self.sides:
            self.grippers[side].aperture=held_fingers[side]
            self.grippers[side].closed=held_fingers[side]<.022
        self.palms={s:RelativePalmTracker() for s in self.sides}
        self.torso=TorsoReference(self.torso_mode);self.clock=SourceClock();self.wrist_solver=None
        self.last_arm_accept={s:None for s in self.sides};self.reacquiring={s:False for s in self.sides};self.arm_candidate={}
        self.fingers = held_fingers
        self.calibration_capture=None
        self.last_seq = -1
        self.last_capture = -1
        self.latest = None

    def begin_pinch_calibration(self,label):
        if label not in ('closed','open'):raise ValueError('unknown pinch label')
        self.calibration_capture=dict(label=label,start=None,samples={s:[] for s in self.sides})

    def export_calibration(self):
        return {s:c.export() for s,c in self.pinch_calibrations.items()}

    def process(self, obs):
        if obs is None:
            return None
        seq, capture = int(obs['seq']), int(obs['read_end'])
        accepted_source,dt,source_reason=self.clock.accept(seq,capture/1e9)
        if not accepted_source:
            return self.latest
        self.last_seq, self.last_capture = seq, capture
        start = time.perf_counter_ns()
        world = obs.get('pose_raw')
        world = None if world is None else np.asarray(world, dtype=float)
        result = dict(seq=seq, capture_ns=capture, joints={},a_nominal_joints={}, fingers={}, pinch_ratio={}, status={}, scales={}, calibration_frames={},
                      module_state={},module_reason={},palm_relative={},wrist_metrics={},finger_valid={},upper_direction={},fore_direction={},quality={},last_accepted={},
                      pose_raw=None if world is None else world.copy())
        accepted,torso_info=self.torso.process(world,capture/1e9,obs.get('pose_norm'))
        result.update(torso=torso_info,pose_accepted=accepted,source_dt=dt,source_reason=source_reason)
        if self.calibration_capture is not None and self.calibration_capture['start'] is None:
            self.calibration_capture['start']=capture/1e9+.35
        observed=observed_sides(obs,self.sides)
        if source_reason=='SOURCE_GAP':self.reacquiring={s:True for s in self.sides}
        for side in self.sides:
            shoulder, elbow, wrist = (11,13,15) if side=='left' else (12,14,16)
            ids = [11,12,23,24,shoulder,elbow,wrist]
            valid = observed[side] and accepted is not None
            controller = self.controllers[side]
            status = (obs.get('pose_quality',{}).get(side) or 'missing shoulder/elbow/wrist')+': hold this arm'
            if valid:
                segments=np.stack([accepted[elbow,:3]-accepted[shoulder,:3],accepted[wrist,:3]-accepted[elbow,:3]])
                lengths=np.linalg.norm(segments,axis=1)
                if np.any(lengths<[.08,.06]) or np.any(lengths>.65):valid=False;status='ARM_BONE_LENGTH_INVALID'
                if valid and len(controller.calib_frames)>=10:
                    baseline=np.median([[np.linalg.norm(e-s),np.linalg.norm(w-e)] for s,e,w in controller.calib_frames],axis=0)
                    if np.any(lengths/baseline<.5) or np.any(lengths/baseline>1.8):valid=False;status='ARM_BONE_LENGTH_OUTLIER'
                if valid and self.last_arm_accept[side] is not None and capture/1e9-self.last_arm_accept[side]>.25:self.reacquiring[side]=True
                if valid and self.reacquiring[side]:
                    directions=segments/lengths[:,None];previous=self.arm_candidate.get(side)
                    consistent=previous is not None and np.min(np.sum(previous*directions,axis=1))>np.cos(np.deg2rad(25))
                    self.arm_candidate[side]=directions
                    if consistent:self.reacquiring[side]=False
                    else:valid=False;status='ARM_REACQUIRING_SOURCE_GAP'
            if valid:
                if self.source_dt and dt>0:
                    for filt in (controller.root_filter,controller.arm_filter):
                        for scalar in filt.filters:scalar.freq=1/max(dt,.001)
                # The original A code is called, not a new approximation of it.
                q, status = controller.process(accepted)
                if q is not None and status=='tracking':
                    self.last_arm_accept[side]=capture/1e9
                    nominal=np.asarray(q).copy();result['a_nominal_joints'][side]=nominal.copy()
                    transform=OPENARM_BODY_TO_ROBOT@torso_info['rotation'].T
                    upper=transform@(accepted[elbow,:3]-accepted[shoulder,:3])
                    fore=transform@(accepted[wrist,:3]-accepted[elbow,:3])
                    result['upper_direction'][side]=upper/np.linalg.norm(upper);result['fore_direction'][side]=fore/np.linalg.norm(fore)
                    arm=obs.get('arms',{}).get(side,{})
                    hand=arm.get('hand_raw') if arm.get('hand_valid') else None
                    frame,relative=self.palms[side].update(upper,fore,hand,transform,capture/1e9,side)
                    if self.wrist_mode=='full' and relative is not None:
                        if self.wrist_solver is None:
                            if self.model is None:
                                from live_motion_scene import build_motion_scene
                                self.model,_=build_motion_scene()
                            self.wrist_solver=BWristRetargeter(self.model)
                        q=self.wrist_solver.solve(side,nominal,self.palms[side].delta,frame)
                        result['wrist_metrics'][side]=dict(self.wrist_solver.metrics[side])
                    result['joints'][side] = np.asarray(q).copy()
                    if relative is not None:result['palm_relative'][side]=relative.copy()
                arm = obs.get('arms',{}).get(side,{})
                features=pinch_features(arm.get('hand_raw'),arm.get('hand_norm')) if arm.get('hand_valid') else None
                self.fingers[side] = self.grippers[side].update(features, capture/1e9)
                result['fingers'][side] = self.fingers[side]
            else:
                arm=obs.get('arms',{}).get(side,{})
                features=pinch_features(arm.get('hand_raw'),arm.get('hand_norm')) if arm.get('hand_valid') else None
                self.fingers[side]=self.grippers[side].update(features,capture/1e9)
                if features is not None:result['fingers'][side]=self.fingers[side]
                if accepted is None:status=torso_info['reason']
                self.palms[side].state='HOLD' if self.palms[side].relative is not None else 'LOST'
                self.palms[side].reason='ARM_FRAME_MISSING'
                if status!='ARM_REACQUIRING_SOURCE_GAP':self.arm_candidate.pop(side,None)
            if self.calibration_capture is not None:
                elapsed=capture/1e9-self.calibration_capture['start']
                if 0<=elapsed<=.75 and features is not None:self.calibration_capture['samples'][side].append(features)
            grip=self.grippers[side];palm=self.palms[side]
            result['pinch_ratio'][side] = grip.raw if grip.raw is not None else np.nan
            result['finger_valid'][side]=grip.state=='TRACKING'
            arm_state='TRACKING' if side in result['joints'] else ('REACQUIRING' if 'calibrating' in status or 'REACQUIRING' in status else ('HOLD' if self.last_arm_accept[side] is not None and capture/1e9-self.last_arm_accept[side]<=.25 else 'LOST'))
            result['module_state'][side]=dict(torso=torso_info['state'],arm=arm_state,palm=palm.state,pinch=grip.state,association='TRACKING' if obs.get('arms',{}).get(side,{}).get('hand_valid') else 'LOST')
            result['module_reason'][side]=dict(torso=torso_info['reason'],arm=status,palm=palm.reason,pinch=grip.reason)
            result['quality'][side]={module:state=='TRACKING' for module,state in result['module_state'][side].items()}
            result['last_accepted'][side]=dict(torso=capture/1e9 if torso_info['valid'] else self.torso.last_time,
                arm=self.last_arm_accept[side],palm=palm.last_valid,pinch=grip.last_accepted,
                association=capture/1e9 if obs.get('arms',{}).get(side,{}).get('hand_valid') else None)
            result['status'][side] = status
            result['scales'][side] = controller.retargeter.scale
            result['calibration_frames'][side] = len(controller.calib_frames)
        if self.calibration_capture is not None and capture/1e9-self.calibration_capture['start']>.75:
            for s in self.sides:self.pinch_calibrations[s].add_trial(self.calibration_capture['label'],self.calibration_capture['samples'][s])
            self.calibration_capture=None
        result['pinch_calibration']={s:c.summary() for s,c in self.pinch_calibrations.items()}
        result['calibration_collecting']=None if self.calibration_capture is None else self.calibration_capture['label']
        result.update(done_ns=time.perf_counter_ns(), solve_ms=(time.perf_counter_ns()-start)/1e6)
        self.latest = result
        return result


class LatestPose2ArmWorker:
    """One in-flight solve + one replaceable latest slot, no observation FIFO."""
    def __init__(self, sides=SIDES,**mapping_options):
        self.mapping = Pose2ArmTaskMapping(sides=sides,**mapping_options)
        self.condition = threading.Condition()
        self.pending = self.latest = None
        self.closed = False
        self.reset_requested = False
        self.generation = 0
        self.last_submitted = -1
        self.error = None
        self.calibration_request=None
        self.thread = threading.Thread(target=self._run, daemon=True, name='pose2arm-task-reference')
        self.thread.start()

    def submit(self, obs):
        if obs is None: return
        with self.condition:
            if int(obs['seq'])<=self.last_submitted: return
            self.last_submitted=int(obs['seq'])
            # Keep images out of this boundary, and don't share mutable filters.
            self.pending=dict(seq=obs['seq'],read_end=obs['read_end'],
                pose_trusted=dict(obs.get('pose_trusted',{})),pose_quality=dict(obs.get('pose_quality',{})),
                pose_norm=None if obs.get('pose_norm') is None else np.asarray(obs['pose_norm']).copy(),
                pose_raw=None if obs.get('pose_raw') is None else np.asarray(obs['pose_raw']).copy(),
                arms={s:dict(hand_valid=a.get('hand_valid',False),
                    hand_norm=None if a.get('hand_norm') is None else np.asarray(a['hand_norm']).copy(),
                    hand_raw=None if a.get('hand_raw') is None else np.asarray(a['hand_raw']).copy())
                    for s,a in obs.get('arms',{}).items()})
            self.condition.notify()

    def snapshot(self):
        with self.condition: return self.latest

    def reset(self):
        with self.condition:
            self.generation+=1
            self.reset_requested=True
            self.pending=self.latest=None
            self.calibration_request=None
            self.last_submitted=-1
            self.condition.notify()

    def begin_pinch_calibration(self,label):
        with self.condition:
            self.calibration_request=label;self.condition.notify()

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.pending is not None or self.reset_requested or self.calibration_request is not None)
                    if self.closed: return
                    reset=self.reset_requested; self.reset_requested=False
                    obs=self.pending; self.pending=None
                    generation=self.generation
                    calibration=self.calibration_request;self.calibration_request=None
                if reset: self.mapping.reset()
                if calibration:self.mapping.begin_pinch_calibration(calibration)
                if obs is not None:
                    result=self.mapping.process(obs)
                    with self.condition:
                        if generation==self.generation: self.latest=result
        except Exception as error:
            self.error=f'{type(error).__name__}: {error}'

    def close(self):
        with self.condition:
            self.closed=True
            self.condition.notify()
        self.thread.join(timeout=5)


def synthetic_pose_observation(seq, clock):
    """Explicit software landmarks for mapping/recording checks, never a human."""
    world=np.zeros((33,4));world[:,3]=1.
    world[11,:3],world[12,:3]=[-.19,0,0],[.19,0,0]
    world[23,:3],world[24,:3]=[-.15,.5,0],[.15,.5,0]
    motion=.05*np.sin(max(0.,clock-.8)*1.2)
    world[13,:3],world[15,:3]=[-.31,.12,-.10],[-.36,.02+motion,-.30]
    world[14,:3],world[16,:3]=[.31,.12,-.10],[.36,.02-motion,-.30]
    hand=np.zeros((21,3));hand[5,0]=.04;hand[17,0]=-.04
    hand[8,0]=.012 if 1.<clock%10<3. else .096
    arms={s:dict(point=human_wrist(world,s),position_point=human_wrist(world,s,require_elbow=False),
                 hand_valid=True,hand_raw=hand.copy()) for s in SIDES}
    return dict(seq=seq,read_end=int((1000.+clock)*1e9),pose_raw=world,arms=arms)
