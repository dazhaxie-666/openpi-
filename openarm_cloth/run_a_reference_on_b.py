"""A's existing motion reference -> B robot only; no grasp or avoidance planner."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from a_reference_adapter import DEFAULT_REFERENCE,SIDES,file_hash,load_reference,verify_kinematics
from handeye_grasp_red_cube import finger_joint_indices
from live_motion_scene import build_motion_scene
from member_b_scene import ROOT
from member_b_actuators import ActuatorController
from task_workspace import CONFIG


def tracking_metrics(model,specs,actual_qpos,nominal):
    actual=mujoco.MjData(model);target=mujoco.MjData(model);errors=[]
    for state,reference in zip(actual_qpos,nominal):
        actual.qpos[:]=state;target.qpos[:]=state
        for j,side in enumerate(SIDES): target.qpos[specs[side].qpos]=reference[7*j:7*(j+1)]
        mujoco.mj_forward(model,actual);mujoco.mj_forward(model,target)
        row=[]
        for side in SIDES:
            ids=[model.joint(f'openarm_{side}_joint{i}').id for i in (1,4,7)]
            a,b=actual.xanchor[ids],target.xanchor[ids]
            def angle(u,v):
                return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/max(1e-12,np.linalg.norm(u)*np.linalg.norm(v)),-1,1))))
            row.append([np.linalg.norm(actual.xpos[specs[side].body_id]-target.xpos[specs[side].body_id]),
                        angle(a[1]-a[0],b[1]-b[0]),angle(a[2]-a[1],b[2]-b[1])])
        errors.append(row)
    return np.array(errors)


def apply_kinematic_frame(model,data,specs,q,closure):
    # Exactly A's qpos + mj_forward replay, with B's METRE slide fingers.
    for j,side in enumerate(SIDES):
        data.qpos[specs[side].qpos]=q[7*j:7*(j+1)]
        data.qpos[finger_joint_indices(model,side)[0]]=.044*(1-closure[j])
    data.qvel[:]=0.
    mujoco.mj_forward(model,data)


class MotionFeedforwardController(ActuatorController):
    """Motion-only velocity feedforward; B's original grasp controller is untouched."""
    def __init__(self,model,data):
        super().__init__(model,data)
        self.target_velocity=np.zeros(14)

    def apply(self):
        data=self.data
        torque=(data.qfrc_bias[self.dof]+self.kp*(self.target-data.qpos[self.qpos])
                +self.kd*(self.target_velocity-data.qvel[self.dof]))
        data.ctrl[self.actuators]=np.clip(torque,self.limits[:,0],self.limits[:,1])
        data.ctrl[self.finger_actuators]=self.finger_target


def simulate_reference(model,data,reference,execution='kinematic',tracking='continuous'):
    if execution not in ('kinematic','dynamic'): raise ValueError('Unknown motion execution mode')
    if tracking not in ('continuous','step'):raise ValueError('Unknown dynamic tracking mode')
    control=MotionFeedforwardController(model,data) if tracking=='continuous' else ActuatorController(model,data)
    specs=control.specs
    addresses=np.concatenate([specs[s].qpos for s in SIDES])
    limits=np.vstack([specs[s].q_range for s in SIDES])
    if np.any(reference.q<limits[:,0]-1e-7) or np.any(reference.q>limits[:,1]+1e-7):
        raise ValueError('A joint reference exceeds B joint limits; no silent clipping allowed')
    apply_kinematic_frame(model,data,specs,reference.q[0],reference.closure[0])
    if execution=='dynamic':
        initial={s:reference.q[0,7*j:7*(j+1)] for j,s in enumerate(SIDES)}
        fingers={s:.044*(1-reference.closure[0,j]) for j,s in enumerate(SIDES)}
        control.set_targets(initial,fingers)
        for _ in range(round(.6/model.opt.timestep)):
            control.apply();mujoco.mj_step(model,data)
    rows=[]
    hz=reference.source_fps if execution=='kinematic' else 20.
    curve=PchipInterpolator(reference.source_time,reference.q,axis=0) if execution=='dynamic' and len(reference.q)>1 else None
    for i,(q,closure) in enumerate(zip(reference.q,reference.closure)):
        fingers={s:float(.044*(1-closure[j])) for j,s in enumerate(SIDES)}
        if execution=='kinematic':
            apply_kinematic_frame(model,data,specs,q,closure)
            data.time=i/hz
        else:
            steps=round(1/hz/model.opt.timestep)
            for substep in range(steps):
                if tracking=='continuous' and curve is not None and i>0:
                    clock=reference.source_time[i-1]+(substep+1)/steps*(reference.source_time[i]-reference.source_time[i-1])
                    position=curve(clock);control.target_velocity[:]=curve(clock,1)
                else:
                    position=q
                    if hasattr(control,'target_velocity'):control.target_velocity[:]=0.
                control.set_targets({s:position[7*j:7*(j+1)] for j,s in enumerate(SIDES)},fingers)
                control.apply();mujoco.mj_step(model,data)
        rows.append(dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy(),
                         joint_target=q.copy(),a_reference_joint=q.copy(),
                         finger_target=np.array([fingers[s] for s in SIDES]),
                         source_frame=int(reference.source_frame[i]),source_time_s=float(reference.source_time[i]),
                         execution_time_s=i/hz,sim_time_s=float(data.time),
                         phase=f'a_reference_{execution}',avoidance_mode='disabled_motion_only',
                         source_arm_trusted=reference.trusted[i]))
    arrays={k:np.asarray([row[k] for row in rows]) for k in rows[0]}
    arrays['fps']=np.array(hz)
    arrays['a_reference_tracking_error']=tracking_metrics(model,specs,arrays['qpos'],reference.q)
    return arrays,dict(max_joint_error_rad=float(np.max(np.abs(arrays['qpos'][:,addresses]-reference.q))),
                       mean_joint_error_deg=float(np.degrees(np.mean(np.abs(arrays['qpos'][:,addresses]-reference.q)))))


def run(args):
    original=load_reference(args.reference,args.video,args.reference_bindings,hz=None)
    base=ROOT/'outputs/a_motion'/f'{args.video.stem}_{args.reference_mode}_{args.execution}'
    out=args.outdir.resolve() if args.outdir else base
    if args.outdir is None and (out/'manifest.json').exists():
        out=out.with_name(out.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if (out/'manifest.json').exists(): raise FileExistsError(f'Choose a new --outdir to preserve {out}')
    out.mkdir(parents=True,exist_ok=True)
    print(f'Recording: {out}',flush=True)
    try:
        model,data=build_motion_scene();control=ActuatorController(model,data)
        compatibility=verify_kinematics(model,control.specs)
        repair_report=None
        cache=ROOT/'outputs/ab_motion/cache'/f'{args.video.stem}_full.npz'
        if args.reference_mode=='clean':
            from full_video_source import load_full_landmarks
            from clean_motion_reference import rebuild_motion
            hand_cache=load_full_landmarks(args.video,cache)
            if str(hand_cache['source_sha256'].item())!=original.video_hash:
                raise ValueError('Hand cache does not match the source video')
            if abs(float(hand_cache['fps'])-original.source_fps)>1e-4 or original.source_frame[-1]>=len(hand_cache['pose_world']):
                raise ValueError('Human cache and A reference video clocks differ')
            rtmw=None;rtmw_path=ROOT.parent/'pose2arm-main/out/pose_ab/rtmw3d.npz'
            bindings=json.loads(Path(args.reference_bindings or ROOT/'a_reference_bindings.json').read_text(encoding='utf-8'))
            if rtmw_path.is_file():
                digest=file_hash(rtmw_path)
                if any(entry.get('rtmw_sha256')==digest and entry['source_sha256']==original.video_hash for entry in bindings['references']):
                    with np.load(rtmw_path,allow_pickle=False) as z:rtmw={k:z[k] for k in z.files}
                    if abs(float(rtmw['fps'])-original.source_fps)>1e-4:raise ValueError('RTMW source clock differs')
            with np.load(original.reference_path,allow_pickle=False) as z:
                clean,repair_report=rebuild_motion({k:z[k] for k in z.files},model,hand_cache,rtmw)
            generated=out/'clean_reference.npz'
            np.savez_compressed(generated,**clean)
            binding=out/'clean_reference_bindings.json'
            binding.write_text(json.dumps(dict(schema_version=1,references=[dict(
                source_sha256=original.video_hash,reference_sha256=file_hash(generated))]),indent=2),encoding='utf-8')
            reference=load_reference(generated,args.video,binding,hz=None if args.execution=='kinematic' else 20.)
            (out/'repair_report.json').write_text(json.dumps(repair_report,indent=2),encoding='utf-8')
        else:
            reference=load_reference(args.reference,args.video,args.reference_bindings,
                                     hz=None if args.execution=='kinematic' else 20.)
        arrays,joint_metrics=simulate_reference(model,data,reference,args.execution,args.tracking)
        if repair_report is not None:
            arrays['source_arm_input_mode']=clean['source_arm_input_mode'][reference.source_frame]
        np.savez_compressed(out/'episode.npz',**arrays)
        metrics=arrays['a_reference_tracking_error'];early=arrays['source_time_s']<4.
        addresses=np.concatenate([control.specs[s].qpos for s in SIDES])
        report=dict(scope='A observations rebuilt as pure imitation on B robot-only model; no grasp scene or avoidance' if repair_report else 'Legacy A reference on B robot-only model',
                    verdict='COMPLETED',execution_mode=args.execution,scene_kind='motion_only',
                    physical_actuator_validation=args.execution=='dynamic',safety_validation=False,
                    simulation_only=True,real_robot_validation=False,
                    reference_mode=args.reference_mode,dynamic_tracking=args.tracking,repair=repair_report,
                    grasp_success_claim=False,frames=len(reference.q),fps=float(arrays['fps']),
                    source_frame_range=[int(arrays['source_frame'][0]),int(arrays['source_frame'][-1])],
                    source_duration_s=float(reference.source_time[-1]+1/float(arrays['fps'])),
                    canonical_kinematics_max_error_m=compatibility,**joint_metrics,
                    mean_a_reference_tcp_error_m=np.mean(metrics[:,:,0],axis=0).tolist(),
                    p95_a_reference_tcp_error_m=np.percentile(metrics[:,:,0],95,axis=0).tolist(),
                    mean_a_reference_direction_error_deg=np.mean(metrics[:,:,1:],axis=0).tolist(),
                    first4s_mean_a_reference_tcp_error_m=np.mean(metrics[early,:,0],axis=0).tolist(),
                    first4s_joint_range_deg=np.degrees(np.ptp(arrays['qpos'][early][:,addresses],axis=0)).reshape(2,7).tolist(),
                    source_arm_untrusted_frames=np.sum(~reference.trusted,axis=0).tolist(),
                    tracking_note='Relative to saved A robot reference, not ground-truth human pose accuracy',
                    warnings=['Simulation only; collision/safety/grasp acceptance intentionally out of scope',
                              'Kinematic replay does not validate physical actuators or real robot deployment'])
        baseline=ROOT/'outputs/ab_motion/p2a_test1/episode.npz'
        if baseline.is_file():
            # The old full-scene model has different qpos shape. Transfer ONLY
            # its named arm joints into the robot-only model for a fair FK metric.
            from task_workspace import load_task_model
            from member_b_scene import SCENE
            from teleop_towel_dataset import build_arm_specs
            oldmodel=load_task_model(SCENE);oldspecs=build_arm_specs(oldmodel)
            with np.load(baseline,allow_pickle=False) as old:
                states=np.tile(model.qpos0,(len(old['qpos']),1))
                for side in SIDES:states[:,control.specs[side].qpos]=old['qpos'][:,oldspecs[side].qpos]
                nominal=np.column_stack([np.interp(old['source_time_s'],reference.source_time,reference.q[:,j]) for j in range(14)])
                before=tracking_metrics(model,control.specs,states,nominal)
                report['previous_full_scene_comparison']=dict(mean_tcp_error_m=np.mean(before[:,:,0],axis=0).tolist(),
                    first4s_tcp_error_m=np.mean(before[old['source_time_s']<4,:,0],axis=0).tolist(),
                    note='Different scene/mode by explicit user request; compares motion, not safety or grasp')
        (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        manifest=dict(schema_version=1,source_video=str(args.video.resolve()),source_sha256=reference.video_hash,
                      a_reference_path=str(reference.reference_path),a_reference_sha256=reference.reference_hash,
                      workspace=CONFIG,scene_kind='motion_only',execution_mode=args.execution,
                      model_nq=model.nq,scene_hashes={name:file_hash(ROOT/name) for name in (
                          'simulation/v1/droid_cube_scene_perception.xml','simulation/v1/droid_cube_scene.xml','live_motion_scene.py')},
                      recording='episode.npz',report='result.json',frames=len(reference.q),verdict='COMPLETED',
                      scope=report['scope'],side_order=list(SIDES),retimed=False)
        if cache.is_file():manifest['human_landmark_cache']=str(cache)
        manifest['reference_mode']=args.reference_mode
        manifest['upstream_reference_path']=str(original.reference_path)
        manifest['upstream_reference_sha256']=original.reference_hash
        manifest['dynamic_tracking']=args.tracking
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (out/'failure.json').unlink(missing_ok=True)
        if not args.no_set_default:
            latest=ROOT/'outputs/a_motion/latest.json';latest.parent.mkdir(parents=True,exist_ok=True)
            latest.write_text(json.dumps(dict(recording=str(out)),indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)),indent=2),encoding='utf-8')
        raise
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    p.add_argument('--reference-bindings',type=Path)
    p.add_argument('--video',type=Path,default=ROOT.parent/'pose2arm-main/videos/p2a_test1.mp4')
    p.add_argument('--outdir',type=Path)
    p.add_argument('--execution',choices=('kinematic','dynamic'),default='kinematic')
    p.add_argument('--reference-mode',choices=('clean','legacy'),default='clean',help='Clean pure imitation is default; legacy retains old historical avoidance')
    p.add_argument('--tracking',choices=('continuous','step'),default='continuous',help='Motion-only dynamic tracking; does not change B grasp control')
    p.add_argument('--check',action='store_true')
    p.add_argument('--no-set-default',action='store_true',help='Do not replace the latest-replay pointer')
    args=p.parse_args();out=run(args)
    if not args.check:
        from replay_video_fusion import main as replay
        replay(['--recording',str(out)])


if __name__=='__main__':main()
