"""Four-layer local diagnostics. RGB features are not calibrated depth truth.

No camera/controller is opened when exporting an episode. The injection test
uses B's simulated actuators only and explicitly bypasses human perception.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
AXES=('x','y','z')


def pose_depth_features(pose,world,image_size):
    result={f'wrist_{kind}_{axis}':np.nan for kind in ('img','world') for axis in AXES}
    result.update(shoulder_px=np.nan,wrist_visibility=np.nan,pose_feature_visible=False)
    p=None if pose is None else np.asarray(pose)
    w=None if world is None else np.asarray(world)
    if p is not None and p.shape==(33,4):
        result.update({f'wrist_img_{a}':float(p[16,j]) for j,a in enumerate(AXES)})
        result['wrist_visibility']=float(p[16,3])
        wh=np.asarray(image_size,float)
        result['shoulder_px']=float(np.linalg.norm((p[11,:2]-p[12,:2])*wh))
        result['pose_feature_visible']=bool(np.isfinite(p[[11,12,16]]).all() and np.min(p[[11,12,16],3])>=.6)
    if w is not None and w.shape==(33,4):
        result.update({f'wrist_world_{a}':float(w[16,j]) for j,a in enumerate(AXES)})
    return result


def _range(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return float(np.ptp(a)) if len(a) else None


def _percentiles(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return dict(p50=float(np.median(a)),p95=float(np.percentile(a,95))) if len(a) else None


def _counts(values):
    from collections import Counter
    return dict(Counter(map(str,values)))


def export_episode(episode,output,plot=True):
    episode=Path(episode).resolve();output=Path(output).resolve()
    if output==episode or episode in output.parents:
        raise ValueError('Keep diagnostic reports outside the original episode')
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((episode/'manifest.json').read_text(encoding='utf-8'))
    with np.load(episode/'episode.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    n=len(d['time_s'])
    if not n:raise ValueError('Empty episode')
    def value(k,i,default=np.nan):return d[k][i] if k in d else default
    camera=manifest.get('camera_metadata',{}) or {}
    size=(camera.get('reported_width',1280),camera.get('reported_height',720))
    rows=[];last_source=None
    for i in range(n):
        seq=int(value('source_seq',i,-1));capture=float(value('capture_s',i))
        identity=(seq,capture)
        fresh=bool(seq>=0 and np.isfinite(capture) and identity!=last_source)
        if fresh:last_source=identity
        f=pose_depth_features(value('human_pose_norm',i,None),value('human_pose',i,None),size)
        row=dict(control_row=i,time_s=float(d['time_s'][i]),source_seq=seq,
            capture_s=capture,source_read_end_ns=int(value('source_read_end_ns',i,0)),fresh_source=fresh,
            stage=str(value('stage',i,'unknown')),phase=str(value('diagnostic_phase',i,'unlabelled')),**f,
            depth_valid=bool(value('human_depth_valid',i,False)),depth_reason=str(value('depth_reason',i,'unknown')),
            pose_valid=bool(value('human_pose_valid',i,False)),
            nominal_palm_z=float(value('palm_camera_z_nominal',i)),
            reprojection_error_px=float(value('palm_reprojection_error_px',i)),
            ik_success=(str(bool(value('ik_accepted',i))) if bool(value('ik_attempted',i,False)) else 'not_attempted') if 'ik_attempted' in d else 'unknown',
            ik_solver_success=(str(bool(value('ik_solver_success',i))) if bool(value('ik_attempted',i,False)) else 'not_attempted') if 'ik_solver_success' in d else 'unknown',
            ik_nfev=float(value('ik_nfev',i)),ik_position_error_m=float(value('ik_position_error_m',i)),
            ik_wrist_delta_rad=float(value('ik_wrist_delta_rad',i)),
            ik_initial_wrist_delta_rad=float(value('ik_initial_wrist_delta_rad',i)),
            motion_warning=str(value('motion_warning',i,'')),
            controller_ack='unavailable_simulation',
            command_stage=str(value('command_stage',i,'unknown')),
            body_position_source=manifest.get('body_position_source','mediapipe'),body_2d_ms=float(value('body_2d_ms',i)),
            control_work_ms=float(value('control_work_ms',i)),observation_age_s=float(value('observation_age_s',i)))
        position_pose=value('human_position_pose_norm',i,None)
        raw_pose=value('human_pose_norm',i,None)
        for label,p in (('position',position_pose),('raw_body',raw_pose)):
            for name,index in (('wrist',16),('elbow',14)):
                for column,j in (('u',0),('v',1),('confidence',3)):
                    row[f'{label}_{name}_{column}']=float(p[index,j]) if p is not None else np.nan
        for prefix,key in (('human_input','human_point'),('body_input','human_body_point'),
                           ('mapped_robot','command_target'),('workspace_goal','workspace_goal'),
                           ('unclipped_goal','unclipped_goal'),('lead_goal','lead_goal'),('actual_tcp','tcp')):
            v=np.asarray(value(key,i,np.full(3,np.nan)))
            row.update({f'{prefix}_{a}':float(v[j]) for j,a in enumerate(AXES)})
        q=value('ik_goal_q',i,np.full(7,np.nan))
        row.update({f'ik_q{j+1}':float(q[j]) for j in range(7)})
        rows.append(row)
    # FK is an offline check of the recorded joint reference, not new dynamics.
    model_path=episode/'scene.mjb'
    fk_available=model_path.exists() and 'qpos' in d and 'action_target' in d
    if fk_available:
        expected=manifest.get('model_sha256')
        if expected and hashlib.sha256(model_path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Model snapshot hash mismatch; cannot trust offline FK')
        import mujoco
        from teleop_towel_dataset import build_arm_specs
        from handeye_grasp_red_cube import finger_joint_indices
        from human_task_robot import pad_points
        model=mujoco.MjModel.from_binary_path(str(model_path));state=mujoco.MjData(model)
        spec=build_arm_specs(model)['right'];fingers=finger_joint_indices(model,'right')[0]
        for i,row in enumerate(rows):
            actual=d['qpos'][i,spec.qpos];command=d['action_target'][i,9:16]
            row.update({f'actual_q{j+1}':float(actual[j]) for j in range(7)})
            row.update({f'command_q{j+1}':float(command[j]) for j in range(7)})
            state.qpos[:]=d['qpos'][i];state.qpos[spec.qpos]=command
            state.qpos[fingers]=d['action_target'][i,16:18]
            mujoco.mj_kinematics(model,state)
            tcp=pad_points(model,state,'right').mean(0)
            row.update({f'command_fk_{a}':float(tcp[j]) for j,a in enumerate(AXES)})
    else:
        for row in rows:
            row.update({f'{p}{j+1}':np.nan for p in ('actual_q','command_q') for j in range(7)})
            row.update({f'command_fk_{a}':np.nan for a in AXES})
    with (output/'chain.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    active=[r for r in rows if r['stage']=='direct']
    fresh=[r for r in active if r['fresh_source']]
    attempts=[r for r in active if r['ik_success'] in ('True','False')]
    phases={}
    for phase in sorted({r['phase'] for r in fresh}):
        group=[r for r in fresh if r['phase']==phase]
        valid=[r for r in group if r['depth_valid'] and np.isfinite(r['human_input_x'])]
        noise=None
        if phase in ('depth_rest','depth_rest_end') and len(valid)>=5:
            x=np.array([r['human_input_x'] for r in valid]);noise=float(np.percentile(abs(x-np.median(x)),95)*1000)
        phases[phase]=dict(fresh_sources=len(group),depth_valid_fraction=float(np.mean([r['depth_valid'] for r in group])),
            static_p95_abs_deviation_mm=noise,
            human_input_x_range_m=_range([r['human_input_x'] for r in valid]))
    report=dict(scope='Layer observations and simulated robot telemetry; no absolute depth truth, hardware acknowledgement or gesture accuracy claim',
        episode=str(episode),episode_sha256=hashlib.sha256((episode/'episode.npz').read_bytes()).hexdigest(),
        axis_convention=dict(image='+u right, +v down; raw image z is relative',
            robot='+X forward, +Y left, +Z up',palm='approaching camera gives positive relative robot X; nominal log projection',
            mirror='display only; no second control mirror',shoulder='pixel-scale diagnostic only; not a substitute for hand reach'),
        active_control_rows=len(active),fresh_active_sources=len(fresh),
        depth_valid_fraction=float(np.mean([r['depth_valid'] for r in fresh])) if fresh else None,
        depth_hold_reasons=_counts([r['depth_reason'] for r in fresh if not r['depth_valid']]),
        motion_warnings=_counts([r['motion_warning'] for r in active if r['motion_warning']]),
        availability={k:k in d for k in ('command_target','ik_attempted','ik_goal_q','workspace_goal')},
        fk_available=bool(fk_available),controller_ack='unavailable_simulation',
        ik=dict(attempts=len(attempts),accepted_fraction=float(np.mean([r['ik_success']=='True' for r in attempts])) if attempts else None,
            position_error_m=_percentiles([r['ik_position_error_m'] for r in attempts])),
        ranges_m={p:{a:_range([r[f'{p}_{a}'] for r in active]) for a in AXES}
                  for p in ('human_input','mapped_robot','command_fk','actual_tcp')},
        control_work_ms=_percentiles([r['control_work_ms'] for r in active]),
        observation_age_s=_percentiles([r['observation_age_s'] for r in fresh]),phases=phases)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if plot:_plot(rows,output)
    return report


def _plot(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    active=[r for r in rows if r['stage']=='direct']
    if not active:return
    t=np.array([r['time_s'] for r in active]);t-=t[0]
    get=lambda k:np.array([r[k] for r in active],float)
    fig,axes=plt.subplots(5,1,figsize=(12,12),sharex=True,layout='constrained')
    axes[0].plot(t,get('shoulder_px'),label='Shoulder apparent width');axes[0].set_ylabel('pixels')
    body=get('body_input_x');finite=np.flatnonzero(np.isfinite(body))
    if len(finite):body=body-body[finite[0]]
    axes[1].plot(t,body*1000,label='Body model X change (comparison)')
    axes[1].plot(t,get('human_input_x')*1000,label='Actual relative hand input X');axes[1].set_ylabel('relative mm')
    for key,label in (('mapped_robot_x','IK position request'),('command_fk_x','Limited joint-command FK'),('actual_tcp_x','Actual simulated TCP')):
        axes[2].plot(t,get(key)*1000,label=label)
    axes[2].set_ylabel('robot X / mm')
    axes[3].step(t,get('depth_valid'),where='post',label='Depth accepted')
    axes[3].step(t,get('pose_valid')*.8,where='post',label='Body accepted (0.8)');axes[3].set_ylabel('valid / hold')
    axes[4].plot(t,get('control_work_ms'),label='Control work')
    axes[4].plot(t,get('observation_age_s')*1000,label='Capture-return to control age')
    axes[4].axhline(50,color='gray',linestyle='--',label='20 Hz cycle budget');axes[4].set_ylabel('ms')
    for ax in axes:ax.legend(loc='upper left',fontsize=8);ax.grid(alpha=.25)
    axes[-1].set_xlabel('Active session time / seconds')
    fig.suptitle('Recorded layer diagnostics; missing historical IK targets remain gaps; no calibrated depth truth')
    fig.savefig(output/'chain.png',dpi=150);plt.close(fig)


def check_actuation(output):
    """Bypass RGB, then IK. Same B model, PD, torque limits and collision checks."""
    from unittest.mock import patch
    import mujoco
    from member_b_scene import build_scene
    from human_task_robot import HumanTaskRobot
    from human_grasp_task import TaskCommand
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model,data=build_scene()
    with patch('human_task_robot.BVisualObserver'):
        robot=HumanTaskRobot(model,data)
    rows=[];checks=[]
    try:
        robot.initialize();robot.direct_wrist_policy='position'
        origin=robot.feedback(0.).tcp.copy();clock=0.
        for offset in (0.,.02,0.,-.02,0.):
            goal=origin+np.array([offset,0.,0.]);warnings=[]
            for _ in range(30):
                # Explicit synthetic fixture geometry, never a human observation.
                robot.visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=clock)
                robot.execute(TaskCommand('direct',goal,.044,'synthetic_axis_injection'),clock,.05)
                if robot.error:raise RuntimeError(robot.error)
                if robot.warning:warnings.append(robot.warning)
                robot.step(.05);clock+=.05
                rows.append(np.r_[clock,offset,goal,robot.feedback(clock).tcp,data.qpos[robot.spec.qpos],robot.joints['right']])
            actual=robot.feedback(clock).tcp
            checks.append(dict(offset_m=offset,target=goal.tolist(),actual=actual.tolist(),
                position_error_m=float(np.linalg.norm(actual-goal)),warnings=sorted(set(warnings)),
                pass_check=bool(not warnings and np.linalg.norm(actual-goal)<.008)))
        # A separate tiny joint-position request bypasses IK entirely. Collision
        # validation uses B's existing segment checker before applying the goal.
        seed=data.qpos[robot.spec.qpos].copy();goal=seed.copy();goal[0]+=.01
        goal=np.clip(goal,robot.spec.q_range[:,0]+1e-7,robot.spec.q_range[:,1]-1e-7)
        robot._check_segment(goal,.044,'direct_move')
        before=data.qpos[robot.spec.qpos].copy()
        robot.joints['right']=goal;robot.control.set_targets(robot.joints,robot.finger_targets)
        robot.step(1.5)
        after=data.qpos[robot.spec.qpos].copy()
        joint_check=dict(request_delta_rad=(goal-before).tolist(),actual_delta_rad=(after-before).tolist(),
            max_error_rad=float(np.max(abs(after-goal))),pass_check=bool(after[0]-before[0]>.004 and np.max(abs(after-goal))<.015))
        report=dict(verdict='PASS' if all(c['pass_check'] for c in checks) and joint_check['pass_check'] else 'FAIL',
            synthetic_input=True,human_validation_performed=False,hardware_connected=False,
            scope='B simulation only; known fixture geometry for collision queries; RGB bypassed, then IK bypassed; unchanged PD/torque limits',
            cartesian_checks=checks,joint_check=joint_check)
    finally:robot.close()
    np.savetxt(output/'axis_injection.csv',rows,delimiter=',',header='time,offset,target_x,target_y,target_z,actual_x,actual_y,actual_z,'+
        ','.join(f'actual_q{i+1}' for i in range(7))+','+','.join(f'command_q{i+1}' for i in range(7)),comments='')
    (output/'actuation_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def camera_only(output,device,seconds):
    """Use installed production Tasks API, without loading robot or simulator."""
    import cv2
    from live_bimanual_stream import BimanualHumanStream
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];baseline=None;last=-1;stream=None;start=time.perf_counter();error=None
    try:
        stream=BimanualHumanStream(device,expected_sides=('right',),position_frame='upper-body')
        while time.perf_counter()-start<seconds:
            obs=stream.snapshot()
            if stream.error:raise RuntimeError(stream.error)
            if obs is not None:
                f=pose_depth_features(obs.get('pose_norm'),obs.get('pose_raw'),obs['raw_image'].shape[1::-1])
                if obs['seq']!=last:
                    rows.append(dict(source_seq=obs['seq'],capture_read_end_ns=obs['read_end'],**f));last=obs['seq']
                if baseline is None and f['pose_feature_visible']:baseline=f.copy()
                im=cv2.resize(obs['raw_image'],(960,540))
                lines=['CAMERA ONLY / NO ROBOT | R baseline | Q save',
                    f'Raw image Z {f["wrist_img_z"]:+.3f} / world Z {f["wrist_world_z"]:+.3f}',
                    f'Shoulder {f["shoulder_px"]:.1f}px / visibility {f["wrist_visibility"]:.2f}',
                    'Relative model features; NOT calibrated camera range']
                if baseline is not None:
                    lines.append(f'Delta image Z {f["wrist_img_z"]-baseline["wrist_img_z"]:+.3f} / shoulder {f["shoulder_px"]-baseline["shoulder_px"]:+.1f}px')
                for j,line in enumerate(lines):cv2.putText(im,line,(10,25+27*j),0,.6,(0,255,0),2)
                cv2.imshow('Depth visual diagnosis',im)
            key=cv2.waitKey(1)&255
            if key==ord('q'):break
            if key==ord('r') and obs is not None and f['pose_feature_visible']:baseline=f.copy()
            time.sleep(.03)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if stream is not None:stream.close()
        cv2.destroyAllWindows()
        if rows:
            with (output/'camera_features.csv').open('w',newline='',encoding='utf-8-sig') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (output/'camera_summary.json').write_text(json.dumps(dict(source='live_usb',robot_loaded=False,
            frames=len(rows),error=error,scope='Raw perception diagnostic, no depth ground truth or robot execution',
            camera_metadata=stream.metadata if stream is not None else {}),indent=2),encoding='utf-8')
    return dict(camera_frames=len(rows),robot_loaded=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--episode',type=Path)
    modes.add_argument('--check-actuation',action='store_true')
    modes.add_argument('--camera-only',action='store_true')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--seconds',type=float,default=35.)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=0:parser.error('--seconds must be positive')
    output=(args.output or ROOT/'outputs/depth_diagnostics'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if args.episode:report=export_episode(args.episode,output,not args.no_plot)
    elif args.check_actuation:report=check_actuation(output)
    else:report=camera_only(output,args.camera,args.seconds)
    print(json.dumps(dict(output=str(output),**report),indent=2))
    return 1 if report.get('verdict')=='FAIL' else 0


if __name__=='__main__':raise SystemExit(main())
