"""Original A live mapping -> B dynamic actuators in the robot-only scene."""
from datetime import datetime
import hashlib
import json
import time
import uuid

import cv2
import mujoco
import numpy as np

from a_reference_adapter import verify_kinematics
from live_motion_scene import build_motion_scene
from live_bimanual_stream import BimanualHumanStream,apply_vision_reacquire
from live_test_dataset import TestDatasetRecorder, array_or_nan
from member_b_scene import ROOT
from pose2arm_task_mapping import LatestPose2ArmWorker, Pose2ArmTaskMapping, observed_sides, synthetic_pose_observation
from run_a_reference_on_b import MotionFeedforwardController
from live_reference_conditioning import JointReferenceTrajectory
from live_observation_quality import b_joint_reference
from live_motion_runtime import LivePhysicsWorker,capture_control
from human_pinch_control import DEFAULT_CALIBRATION,save_personal_calibration,calibration_status

SIDES=('left','right')


class AReferenceMotionController:
    """B's existing motion feedforward; no task IK, object gates or replanning."""
    def __init__(self,model,data,responsive=False):
        self.model,self.data=model,data
        self.control=MotionFeedforwardController(model,data)
        self.specs=self.control.specs
        self.joints={s:data.qpos[self.specs[s].qpos].copy() for s in SIDES}
        self.fingers={s:.044 for s in SIDES}
        self.trajectories={s:JointReferenceTrajectory(q,limits=self.specs[s].q_range,frequency=30. if responsive else 20.,filter_enabled=not responsive) for s,q in self.joints.items()}
        self.finger_trajectories={s:JointReferenceTrajectory(np.array([.044]),speed=[.132],acceleration=[.8],limits=[[0.,.044]]) for s in SIDES}
        self.control.set_targets(self.joints,self.fingers)
        self.reference_rejections={}
        self.active_sides={s:False for s in SIDES}
        self.active_fingers={s:False for s in SIDES}

    def step(self,result,visible,paused=False,dt=.05,finger_visible=None):
        active={};finger_active={}
        self.reference_rejections={}
        for side in SIDES:
            valid=not paused and visible.get(side,False) and result is not None and side in result['joints']
            active[side]=valid
            if valid:
                limits=self.specs[side].q_range
                goal,reason=b_joint_reference(result['joints'][side],limits)
                if reason:
                    self.reference_rejections[side]=reason;active[side]=False
                else:
                    stamp=result.get('capture_ns',int(self.data.time*1e9))/1e9
                    self.trajectories[side].observe(goal,stamp)
            finger_active[side]=bool(not paused and result is not None and result.get('finger_valid',{}).get(side,active[side])
                                     and (finger_visible or {}).get(side,True))
            if finger_active[side]:
                finger=result['fingers'].get(side,self.fingers[side])
                if not np.isfinite(finger) or not 0<=finger<=.044:
                    self.reference_rejections[side]='invalid B gripper aperture: hold';finger_active[side]=False
                else:self.finger_trajectories[side].goal[:]=finger
        self.active_sides=active.copy()
        self.active_fingers=finger_active.copy()
        steps=round(dt/self.model.opt.timestep)
        physics_dt=float(self.model.opt.timestep)
        # Reference acceleration is recomputed at 400 Hz. Integrate its
        # constant-acceleration segment at every B physics tick, preserving
        # continuous position/velocity and leaving B torque control at 2 kHz.
        # Re-solving four tiny NumPy profiles at 2 kHz wastes CPU alongside MP.
        stride=max(1,round(.0025/physics_dt))
        for first in range(0,steps,stride):
            count=min(stride,steps-first);duration=count*physics_dt
            start_q={s:self.trajectories[s].position.copy() for s in SIDES}
            start_v={s:self.trajectories[s].velocity.copy() for s in SIDES}
            start_f={s:self.finger_trajectories[s].position[0] for s in SIDES}
            start_fv={s:self.finger_trajectories[s].velocity[0] for s in SIDES}
            for side in SIDES:
                self.trajectories[side].advance(duration,active[side])
                self.finger_trajectories[side].advance(duration,finger_active[side])
            acceleration={s:(self.trajectories[s].velocity-start_v[s])/duration for s in SIDES}
            finger_acceleration={s:(self.finger_trajectories[s].velocity[0]-start_fv[s])/duration for s in SIDES}
            for tick in range(1,count+1):
                elapsed=tick*physics_dt
                for side in SIDES:
                    self.joints[side]=start_q[side]+start_v[side]*elapsed+.5*acceleration[side]*elapsed**2
                    self.fingers[side]=float(start_f[side]+start_fv[side]*elapsed+.5*finger_acceleration[side]*elapsed**2)
                self.control.target_velocity[:]=np.concatenate([start_v[s]+acceleration[s]*elapsed for s in SIDES])
                self.control.set_targets(self.joints,self.fingers)
                self.control.apply();mujoco.mj_step(self.model,self.data)
        if not np.isfinite(self.data.qpos).all():raise RuntimeError('nonfinite B motion state')


def run(args):
    if args.check and not args.scripted_check:raise ValueError('--check requires explicit --scripted-check')
    if args.scripted_check and not args.check:raise ValueError('--scripted-check requires --check')
    cv2.setNumThreads(1)
    out=ROOT/'outputs'/'a_live'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    model,simulation_data=build_motion_scene()
    control=AReferenceMotionController(model,simulation_data,responsive=getattr(args,'execution_profile','responsive')=='responsive')
    data=simulation_data if args.check else mujoco.MjData(model)
    if not args.check:
        data.qpos[:]=simulation_data.qpos;data.qvel[:]=simulation_data.qvel;mujoco.mj_forward(model,data)
    geometry=verify_kinematics(model,control.specs)
    session_id=str(uuid.uuid4());episodes=[]
    recorder=TestDatasetRecorder(out/'dataset',dict(mapping='A calibrated arm directions + B full relative-palm wrist',
        session_id=session_id,protocol_version='motion_v2_development_1',dataset_split='development',
        release_verdict='NOT_EVALUABLE; no independent reference or acceptance episodes',
        torso_mode=getattr(args,'torso_mode','auto'),wrist_mode=args.wrist_mode,gripper_mode=getattr(args,'gripper_mode','continuous'),
        execution_profile=getattr(args,'execution_profile','responsive'),
        alignment='camera source_seq and physics a_source_seq are separate clocks; use explicit reference timestamps',
        calibration='10 fresh valid frames per side; original A median arm length',
        arms=list(SIDES),side_array_order=list(SIDES),execution='B dynamic actuators; causal source filtering + acceleration-bounded physics-clock reference',
        gripper_mapping='personal palm-normalized 3D + observable 2D pinch; uncalibrated fallback is explicitly marked',
        robot_qpos_index={s:control.specs[s].qpos.tolist() for s in SIDES},
        identity='human anatomical left->robot left, right->right; preview mirror only',
        synthetic_input=bool(args.check),human_validation_performed=False,
        a_b_kinematics_error_m=geometry,pose_model=args.pose_model,
        scene='B robot-only scene; no table, cube, task IK or task collision projection',
        a_source_hashes={n:hashlib.sha256((ROOT.parent/'pose2arm-main'/n).read_bytes()).hexdigest()
                         for n in ('openarm_live.py','retarget.py','ik_solver.py','smoothing.py')},
        b_source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                         for n in ('run_pose2arm_live.py','pose2arm_task_mapping.py','motion_observation.py','human_pinch_control.py','human_wrist_control.py','live_motion_runtime.py','live_reference_conditioning.py','live_observation_quality.py','live_bimanual_stream.py','member_b_actuators.py','run_a_reference_on_b.py')},
        source_timing='synthetic virtual capture clock; no camera latency measurement' if args.check else 'host camera return/inference/first physics step'),
        save_video=bool(args.save_video and not args.check))
    mujoco.mj_saveModel(model,str(out/'scene.mjb'))
    stream=worker=physics=None
    mapping_options=dict(torso_mode=getattr(args,'torso_mode','auto'),wrist_mode=args.wrist_mode,
                         gripper_mode=getattr(args,'gripper_mode','continuous'),calibration_path=getattr(args,'calibration',None),model=model,source_dt=True)
    mapper=Pose2ArmTaskMapping(**mapping_options) if args.check else None
    viewer=None;failure=None;last_seq=-1;index=0;timings=[];human_observed=False
    flags=dict(quit=False,paused=False,reset=False,reacquire=False,mirror=True,calibrate=None,episode=False)
    episode_start=None;episode_number=0
    window='POSE2ARM + FULL WRIST | C mapping reset | R vision reacquire | G pinch / O open | Q exit'
    def key(code):
        if code in (81,113,27):flags['quit']=True
        elif code==32:flags['paused']=not flags['paused']
        elif code in (67,99):flags['reset']=True
        elif code in (82,114):flags['reacquire']=True
        elif code in (77,109):flags['mirror']=not flags['mirror']
        elif code in (71,103):flags['calibrate']='closed'
        elif code in (79,111):flags['calibrate']='open'
        elif code in (84,116):flags['episode']=True
    start=time.perf_counter();expected=float(data.time)
    try:
        if not args.check:
            from mujoco import viewer as mjviewer
            stream=BimanualHumanStream(args.camera,args.pose_model,(args.width,args.height),
                hand_confidence=getattr(args,'hand_confidence',.5),hand_presence_confidence=getattr(args,'hand_presence_confidence',None),
                hand_tracking_confidence=getattr(args,'hand_tracking_confidence',None))
            worker=LatestPose2ArmWorker(**mapping_options)
            viewer=mjviewer.launch_passive(model,data,key_callback=key)
            viewer.cam.lookat[:]=[.15,0,.52];viewer.cam.distance=1.55;viewer.cam.azimuth=180;viewer.cam.elevation=-25
            cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,960,540)
            def reference_provider():
                observation=stream.snapshot()
                if stream.error:raise RuntimeError(stream.error)
                if worker.error:raise RuntimeError(worker.error)
                if not flags['paused']:worker.submit(observation)
                visible_now=observed_sides(observation)
                hands_now={s:bool(observation and observation['arms'].get(s,{}).get('hand_valid')) for s in SIDES}
                if observation is None or time.perf_counter_ns()-observation['read_end']>250000000:
                    visible_now={s:False for s in SIDES};hands_now={s:False for s in SIDES}
                return worker.snapshot() if not flags['paused'] else None,visible_now,flags['paused'],hands_now
            physics=LivePhysicsWorker(control,reference_provider=reference_provider)
        print('ORIGINAL POSE2ARM -> B BOTH ARMS | 10 fresh frames calibrate | output '+str(out),flush=True)
        while not flags['quit'] and (viewer is None or viewer.is_running()):
            tick=time.perf_counter();now=time.perf_counter_ns()
            apply_vision_reacquire(flags,stream,mapper or worker)
            if abs(data.time-expected)>1e-7:raise RuntimeError('Viewer reset detected; restart to recalibrate B robot')
            obs=synthetic_pose_observation(index,index*.05) if args.check else stream.snapshot()
            if flags['reset']:
                (mapper or worker).reset();flags['reset']=False
            if flags['calibrate']:
                (mapper or worker).begin_pinch_calibration(flags['calibrate']);flags['calibrate']=None
            if flags['episode']:
                if episode_start is None or now-episode_start>=15000000000:
                    episode_number+=1;episode_start=now
                    episodes.append(dict(episode_id=episode_number,start_ns=now,planned_duration_s=15.,completed=False))
                flags['episode']=False
            episode_active=episode_start is not None and now-episode_start<15000000000
            if episode_start is not None and not episode_active:episodes[-1]['completed']=True
            result=None
            if not flags['paused']:
                if mapper:result=mapper.process(obs)
                else:
                    worker.submit(obs);result=worker.snapshot()
                    if worker.error:raise RuntimeError(worker.error)
            clock=obs['read_end'] if args.check and obs is not None else now
            if result is not None and not 0<=(clock-result['capture_ns'])/1e9<=.35:result=None
            visible=observed_sides(obs)
            if not args.check and (obs is None or now-obs['read_end']>350000000):visible={s:False for s in SIDES}
            if not args.check and any(visible.values()):human_observed=True
            if physics is None:
                first_step=time.perf_counter_ns();control.step(result,visible,flags['paused'])
                physics_end=time.perf_counter_ns()
                state=capture_control(control,result,first_step,physics_end,.05,.05,index)
            else:
                if physics.error:raise RuntimeError(physics.error)
                state=physics.snapshot();result=state.result
                first_step,physics_end=state.first_ns,state.end_ns
                data.qpos[:]=state.qpos;data.qvel[:]=state.qvel;data.ctrl[:]=state.ctrl;data.time=state.sim_time
                mujoco.mj_forward(model,data)
            expected=float(data.time)
            fresh=obs is not None and obs['seq']!=last_seq
            side_valid=[bool(state.active_sides[s]) for s in SIDES]
            side_hand=[bool(obs and obs['arms'].get(s,{}).get('hand_valid')) for s in SIDES]
            row=dict(control_frame=np.asarray(index),source_seq=np.asarray(obs['seq'] if obs else -1),
                capture_read_end_ns=np.asarray(obs['read_end'] if obs else 0,dtype=np.int64),
                first_physics_step_ns=np.asarray(first_step,dtype=np.int64),sim_time=np.asarray(data.time),
                inference_done_ns=np.asarray(obs.get('done',now) if obs else now,dtype=np.int64),
                physics_end_ns=np.asarray(physics_end,dtype=np.int64),startup=np.asarray(result is None),
                fresh_source=np.asarray(fresh),paused=np.asarray(flags['paused']),
                pose_raw=array_or_nan(obs.get('pose_raw') if obs else None,(33,4)),
                pose_filtered=array_or_nan(obs.get('pose_filtered') if obs else None,(33,4)),
                pose_norm=array_or_nan(obs.get('pose_norm') if obs else None,(33,4)),
                vision_generation=np.asarray(obs.get('vision_generation',0) if obs else -1),
                vision_reacquire_ms=np.asarray(obs.get('vision_reacquire_ms',0.) if obs else np.nan),
                action=state.action.copy(),robot_qpos=data.qpos.copy(),robot_qvel=data.qvel.copy(),
                robot_ctrl=data.ctrl.copy(),side_pose_valid=np.asarray(side_valid),side_hand_valid=np.asarray(side_hand),
                pose_valid=np.asarray(any(side_valid)),hand_valid=np.asarray(any(side_hand)),
                side_closed=np.asarray([state.fingers[s]<.022 for s in SIDES]),
                gripper_closed=np.asarray(any(state.fingers[s]<.022 for s in SIDES)),
                read_to_step_ms=np.asarray((first_step-result['capture_ns'])/1e6 if result and not args.check else np.nan),
                inference_ms=np.asarray((obs.get('done',now)-obs['read_end'])/1e6 if obs and not args.check else np.nan),
                a_reference_joint=np.concatenate([result['joints'].get(s,np.full(7,np.nan)) if result else np.full(7,np.nan) for s in SIDES]),
                a_source_seq=np.asarray(result['seq'] if result else -1),
                reference_rejection=np.asarray([state.rejections.get(s,'') for s in SIDES]),
                observation_quality=np.asarray([obs.get('pose_quality',{}).get(s,'') if obs else '' for s in SIDES]),
                side_hand_quality=np.asarray([obs.get('arms',{}).get(s,{}).get('hand_quality','') if obs else '' for s in SIDES]),
                conditioned_joint_goal=state.goal.copy(),
                applied_joint_velocity=state.velocity.copy(),
                physics_tick=np.asarray(state.tick),physics_wall_dt_s=np.asarray(state.wall_dt),physics_sim_dt_s=np.asarray(state.sim_dt),
                physics_reference_capture_ns=np.asarray(result['capture_ns'] if result else 0,dtype=np.int64),
                physics_reference_done_ns=np.asarray(result['done_ns'] if result else 0,dtype=np.int64),
                wall_time_ns=np.asarray(now,dtype=np.int64),episode_id=np.asarray(episode_number),episode_active=np.asarray(episode_active),
                captured_sequence=np.asarray(obs.get('captured_sequence',obs['seq']) if obs else -1),
                skipped_capture_sources=np.asarray(obs.get('skipped_capture_sources',0) if obs else 0),
                accepted_pose=array_or_nan(result.get('pose_accepted') if result else None,(33,4)),
                a_nominal_joint=np.concatenate([result.get('a_nominal_joints',{}).get(s,np.full(7,np.nan)) if result else np.full(7,np.nan) for s in SIDES]),
                module_state=np.asarray([[result.get('module_state',{}).get(s,{}).get(module,'LOST') if result else 'LOST' for module in ('torso','arm','palm','pinch','association')] for s in SIDES]),
                module_reason=np.asarray([[result.get('module_reason',{}).get(s,{}).get(module,'') if result else 'NO_FRESH_REFERENCE' for module in ('torso','arm','palm','pinch')] for s in SIDES]),
                palm_rotation_local=np.stack([result.get('palm_relative',{}).get(s,np.full((3,3),np.nan)) if result else np.full((3,3),np.nan) for s in SIDES]),
                upper_direction=np.stack([result.get('upper_direction',{}).get(s,np.full(3,np.nan)) if result else np.full(3,np.nan) for s in SIDES]),
                fore_direction=np.stack([result.get('fore_direction',{}).get(s,np.full(3,np.nan)) if result else np.full(3,np.nan) for s in SIDES]),
                module_last_accepted_s=np.asarray([[result.get('last_accepted',{}).get(s,{}).get(m) if result and result.get('last_accepted',{}).get(s,{}).get(m) is not None else np.nan for m in ('torso','arm','palm','pinch','association')] for s in SIDES]),
                pinch_calibration_id=np.asarray([result['pinch_calibration'][s]['calibration_id'] if result else 'uncalibrated' for s in SIDES]),
                wrist_rotation_residual_rad=np.asarray([result.get('wrist_metrics',{}).get(s,{}).get('rotation_residual_rad',np.nan) if result else np.nan for s in SIDES]),
                torso_mode=np.asarray(result.get('torso',{}).get('mode','unknown') if result else 'unknown'),
                torso_virtual_hips=np.asarray(bool(result and result.get('torso',{}).get('virtual_hips'))),
                torso_calibration_id=np.asarray(result.get('torso',{}).get('calibration_id','uncalibrated') if result else 'uncalibrated'),
                pinch_ratio=np.asarray([result.get('pinch_ratio',{}).get(s,np.nan) if result else np.nan for s in SIDES]),
                gripper_reference_m=np.asarray([result['fingers'].get(s,np.nan) if result else np.nan for s in SIDES]),
                a_reference_pose=array_or_nan(result.get('pose_raw') if result else None,(33,4)),
                a_capture_s=np.asarray(result['capture_ns']/1e9 if result else np.nan),
                a_calibration_scale=np.asarray([result['scales'].get(s) if result and result['scales'].get(s) is not None else np.nan for s in SIDES]),
                a_solve_ms=np.asarray(result['solve_ms'] if result else np.nan))
            for side in SIDES:
                arm=obs.get('arms',{}).get(side,{}) if obs else {}
                for field in ('raw','filtered','norm'):
                    row[f'{side}_hand_{field}']=array_or_nan(arm.get('hand_'+field),(21,3))
            recorder.submit(row,
                obs.get('raw_image') if fresh and obs is not None else None)
            if fresh:last_seq=obs['seq']
            if viewer is not None:
                frame=cv2.resize(obs['image'],(960,540)) if obs and obs.get('image') is not None else np.full((540,960,3),40,np.uint8)
                if flags['mirror']:frame=cv2.flip(frame,1)
                statuses=[s.upper()+': '+(state.rejections.get(s) or (obs.get('pose_quality',{}).get(s) if not visible[s] and obs else None) or (result['status'][s] if result else 'waiting for fresh body')) for s in SIDES]
                lines=['PAUSED - SPACE TO RESUME' if flags['paused'] else 'ORIGINAL A CALIBRATION + SHOULDER/ELBOW/WRIST -> B',
                       'YOUR LEFT -> LEFT ARM | YOUR RIGHT -> RIGHT ARM',*statuses,
                       'PINCH APERTURE: L %.0f%% | R %.0f%%' % tuple(state.fingers[s]/.044*100 for s in SIDES),
                       'HANDS: '+ ' | '.join(s.upper()+': '+('confirmed' if side_hand[j] else 'HOLD - not confirmed') for j,s in enumerate(SIDES)),
                       'PALM / PINCH: '+' | '.join(s[0].upper()+': '+('/'.join(result['module_state'][s][m] for m in ('palm','pinch')) if result else 'LOST') for s in SIDES),
                       'CALIB G pinch / O open: '+ ('COLLECTING '+str(result.get('calibration_collecting')) if result and result.get('calibration_collecting') else ' | '.join(s[0].upper()+': '+str(result['pinch_calibration'][s]['closed_trials'])+'/'+str(result['pinch_calibration'][s]['open_trials'])+' trials' if result else s+': 0/0' for s in SIDES)),
                       ('15s DEVELOPMENT EPISODE %.1fs' % ((now-episode_start)/1e9) if episode_active else 'T starts 15s episode; prompts are not ground truth'),
                       'C mapping reset | R vision reacquire | SPACE resume/pause | Q exit']
                vs=stream.vision_status()
                lines.append('VISION REACQUIRING - WAIT' if vs['pending'] else 'VISION generation '+str(vs['generation']))
                if result:
                    lines.extend(s.upper()+' CALIB '+calibration_status(result['pinch_calibration'][s]) for s in SIDES)
                for i,line in enumerate(lines):
                    cv2.putText(frame,line,(12,28+30*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3)
                    cv2.putText(frame,line,(12,28+30*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
                cv2.imshow(window,frame);code=cv2.waitKey(1)&255
                if code!=255:key(code)
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:flags['quit']=True
                viewer.sync()
                if stream.error:raise RuntimeError(stream.error)
            timings.append((time.perf_counter()-tick)*1000)
            index+=1
            if args.check and index>=args.check_steps:break
            if args.seconds and time.perf_counter()-start>=args.seconds:break
            if not args.check:time.sleep(max(0.,.05-(time.perf_counter()-tick)))
    except KeyboardInterrupt:pass
    except Exception as error:
        failure=f'{type(error).__name__}: {error}'
        raise
    finally:
        if physics is not None:
            physics.close();physics.save_trace(out/'physics_trace.npz')
        if viewer is not None:viewer.close()
        if stream is not None:stream.close()
        if worker is not None:worker.close()
        if not args.check:cv2.destroyAllWindows()
        recorder.update_metadata(run_error=failure,camera=stream.metadata if stream else None,
            human_validation_performed=human_observed,physics_runtime=physics.report() if physics else None,episodes=episodes)
        calibration_source=mapper or (worker.mapping if worker is not None else None)
        calibration=calibration_source.export_calibration() if calibration_source is not None else {}
        save_personal_calibration(out/'personal_calibration.json',calibration)
        if not args.check and any(c.get('profile') for c in calibration.values()):
            save_personal_calibration(getattr(args,'calibration',None) or DEFAULT_CALIBRATION,calibration)
        recorder.close()
        from validate_live_test_dataset import validate
        quality=validate(out/'dataset',check_video=True)
        (out/'quality_report.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
        report=dict(frames=index,run_error=failure,recording_error=recorder.error,dropped_rows=recorder.dropped_rows,
                    synthetic_input=bool(args.check),human_validation_performed=human_observed,recording_valid=quality['valid'],
                    task_success='not assessed; motion-only',mapping='original A online controller',
                    control_work_ms_p95=physics.report()['control_work_ms_p95'] if physics else (float(np.percentile(timings,95)) if timings else None),
                    ui_work_ms_p95=float(np.percentile(timings,95)) if timings else None,
                    physics_runtime=physics.report() if physics else None,release_verdict='NOT_EVALUABLE; independent reference/acceptance missing')
        (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        from evaluate_motion_v2 import diagnostics,reference_template
        (out/'motion_diagnostics.json').write_text(json.dumps(diagnostics(out),indent=2),encoding='utf-8')
        (out/'independent_reference_template.json').write_text(json.dumps(reference_template(out),indent=2),encoding='utf-8')
        print(json.dumps(dict(report,output=str(out)),indent=2),flush=True)
