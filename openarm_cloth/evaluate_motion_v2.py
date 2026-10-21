"""Development diagnostics and independent-reference evaluation; fail closed.

Detector coverage is never accuracy. Missing references are NOT_EVALUABLE.
Declared independent input eligibility is never inferred from tracker LOST.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta

PROTOCOL=dict(version='motion_v2_1',episode_s=15.,direction_p95_deg=10.,direction_bad_deg=20.,
              wrist_p95_deg=15.,wrist_bad_deg=30.,bad_duration_s=.1,lag_s=.15,
              gripper_onset_s=.15,gripper_complete_s=.3,availability=.99,max_missing_s=.2,
              fixed_samples=400,alpha=.05)


def lower_success_bound(successes,trials,alpha=.05):
    if not (0<=successes<=trials and trials>0 and 0<alpha<1):raise ValueError('invalid counts or alpha')
    return 0. if successes==0 else float(beta.ppf(alpha,successes,trials-successes+1))


def longest_run(mask,times):
    longest=start=0.
    for i,flag in enumerate(mask):
        if flag and (i==0 or not mask[i-1]):start=float(times[i])
        if flag:
            end=float(times[i+1]) if i+1<len(times) else float(times[i])
            longest=max(longest,end-start)
    return longest


def load_chunks(dataset):
    chunks=[]
    for path in sorted(Path(dataset).glob('chunk_*.npz')):
        with np.load(path,allow_pickle=False) as z:chunks.append({k:z[k].copy() for k in z.files})
    if not chunks:raise ValueError('no completed dataset chunks')
    return {k:np.concatenate([z[k] for z in chunks]) for k in chunks[0]}


def diagnostics(recording):
    root=Path(recording);dataset=root/'dataset' if (root/'dataset').exists() else root
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    rows=load_chunks(dataset)
    raw_times=rows.get('wall_time_ns',rows['physics_end_ns'])/1e9
    times=np.maximum.accumulate(raw_times)
    dt=np.r_[np.maximum(0.,np.diff(times)),0.]
    duration=float(times[-1]-times[0]);weights=dt/max(duration,1e-9)
    result=dict(schema='motion_v2_diagnostics_1',release_verdict='NOT_EVALUABLE',
                missing_evidence=['independent 3D reference','reviewed input eligibility','new independent acceptance episodes'],
                session_id=manifest.get('session_id'),dataset_split=manifest.get('dataset_split','development'),
                rows=len(times),wall_duration_s=duration,sim_duration_s=float(rows['sim_time'][-1]-rows['sim_time'][0]),
                coverage_scope='all saved wall-clock intervals, including LOST/rejected/startup; not independently audited eligibility',
                unique_camera_sources=len(set(int(v) for v in rows['source_seq'] if v>=0)),
                unique_mapping_sources=len(set(int(v) for v in rows['a_source_seq'] if v>=0)),
                per_side={})
    result['realtime_factor']=result['sim_duration_s']/duration if duration else None
    for j,side in enumerate(('left','right')):
        arm=rows['side_pose_valid'][:,j];hand=rows['side_hand_valid'][:,j]
        result['per_side'][side]=dict(command_arm_available_time_fraction=float(np.sum(weights*arm)),
            detected_hand_time_fraction=float(np.sum(weights*hand)),longest_arm_unavailable_s=longest_run(~arm,times),
            longest_hand_unavailable_s=longest_run(~hand,times))
        if 'module_state' in rows:
            result['per_side'][side]['modules']={name:dict(tracking_time_fraction=float(np.sum(weights*(rows['module_state'][:,j,k]=='TRACKING'))),
                longest_nontracking_s=longest_run(rows['module_state'][:,j,k]!='TRACKING',times))
                for k,name in enumerate(('torso','arm','palm','pinch','association'))}
    for field in ('inference_ms','a_solve_ms','read_to_step_ms'):
        values=rows[field];values=values[np.isfinite(values)]
        result[field]=None if not len(values) else dict(p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)))
    result['timing_scope']='host-read/inference/reference-to-physics; no exposure timestamp or independent motion-onset reference'
    return result


def reference_template(recording):
    root=Path(recording);dataset=root/'dataset' if (root/'dataset').exists() else root
    m=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    return dict(schema='motion_independent_reference_1',session_id=m.get('session_id'),
        coordinate_frame='B_model_world',time_basis='host_monotonic_ns',
        independent_measurement=False,reviewers=[],reference_uncertainty_deg=None,time_uncertainty_ms=None,
        fixed_lag_compensation_s=0.,episodes=[dict(episode_id=e['episode_id'],start_ns=e['start_ns'],duration_s=15.,
            input_eligible=None,eligibility_reason='',independent_session_id='',samples=[],events=[],static_intervals=[],
            execution_integrity_review=None) for e in m.get('episodes',[])],
        instructions='Annotate original images without model overlays. Samples require time_ns, upper[2,3], fore[2,3], palm[2,3,3]. Events require time_ns, side, closed. Do not copy detector estimates as truth.')


def evaluate_reference(recording,reference):
    """Compare independent B-world directions/rotations with actual physics.

    No input from this reference file enters the controller. Execution/contact
    review and stable-interval references are mandatory before episode PASS.
    """
    import mujoco
    from scipy.spatial.transform import Rotation
    from teleop_towel_dataset import build_arm_specs
    from handeye_grasp_red_cube import finger_joint_indices
    root=Path(recording)
    if not (root/'physics_trace.npz').exists():return dict(verdict='NOT_EVALUABLE',reason='MISSING_DENSE_PHYSICS_TRACE')
    manifest=json.loads((root/'dataset'/'manifest.json').read_text(encoding='utf-8'))
    if (reference.get('session_id')!=manifest.get('session_id') or not reference.get('independent_measurement')
        or len(set(reference.get('reviewers',[])))<2 or reference.get('coordinate_frame')!='B_model_world'
        or reference.get('time_basis')!='host_monotonic_ns'):
        return dict(verdict='NOT_EVALUABLE',reason='INDEPENDENT_REFERENCE_NOT_VERIFIED')
    if reference.get('reference_uncertainty_deg',999) is None or reference.get('time_uncertainty_ms',999) is None:
        return dict(verdict='NOT_EVALUABLE',reason='REFERENCE_UNCERTAINTY_MISSING')
    uncertainties=np.asarray([reference.get('reference_uncertainty_deg',999),reference.get('time_uncertainty_ms',999)],dtype=float)
    if not np.isfinite(uncertainties).all() or np.any(uncertainties<0) or uncertainties[0]>3.3 or uncertainties[1]>33:
        return dict(verdict='NOT_EVALUABLE',reason='REFERENCE_UNCERTAINTY_TOO_LARGE')
    shift=float(reference.get('fixed_lag_compensation_s',0.))
    if not 0<=shift<=.15:raise ValueError('fixed lag must be preregistered within 0..0.15 s')
    with np.load(root/'physics_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    model=mujoco.MjModel.from_binary_path(str(root/'scene.mjb'));data=mujoco.MjData(model);specs=build_arm_specs(model)
    times=trace['end_ns']/1e9;reports=[]
    for episode in reference.get('episodes',[]):
        report=dict(episode_id=episode['episode_id'],verdict='NOT_EVALUABLE',reasons=[])
        reports.append(report)
        if episode.get('input_eligible') is not True:
            report['reasons'].append('INPUT_ELIGIBILITY_NOT_INDEPENDENTLY_CONFIRMED');continue
        samples=episode.get('samples',[])
        if len(samples)<30 or len(episode.get('events',[]))!=8 or not episode.get('static_intervals'):
            report['reasons'].append('MISSING_REFERENCE_SAMPLES_EVENTS_OR_STATIC_INTERVALS');continue
        sample_times=np.array([s['time_ns']/1e9 for s in samples])
        start=episode['start_ns']/1e9;end=start+15.
        if (not np.isfinite(sample_times).all() or not np.isfinite(start) or np.any(np.diff(sample_times)<=0) or sample_times[0]>start+.05 or sample_times[-1]<end-.05
            or np.max(np.diff(sample_times))>.1):report['reasons'].append('REFERENCE_TIME_COVERAGE_GAP');continue
        if times[0]>start or times[-1]<end+shift:report['reasons'].append('PHYSICS_TIME_COVERAGE_GAP');continue
        raw=[];compensated=[];joint_samples=[];actual_rot=[]
        for s,t in zip(samples,sample_times):
            errors=[];rotations=[]
            for offset in (0.,shift):
                q=np.array([np.interp(t+offset,times,trace['qpos'][:,k]) for k in range(model.nq)])
                data.qpos[:]=q;mujoco.mj_forward(model,data);side_errors=[];side_rot=[]
                for j,side in enumerate(('left','right')):
                    points=[data.xanchor[model.joint(f'openarm_{side}_joint{k}').id] for k in (1,4,7)]
                    vectors=[points[1]-points[0],points[2]-points[1]]
                    direction_errors=[]
                    for key,v in zip(('upper','fore'),vectors):
                        wanted=np.asarray(s[key][j],dtype=float)
                        if wanted.shape!=(3,) or not np.isfinite(wanted).all() or abs(np.linalg.norm(wanted)-1.)>.02:raise ValueError('reference direction must be a finite unit vector')
                        direction_errors.append(float(np.degrees(np.arccos(np.clip(np.dot(v,wanted)/(np.linalg.norm(v)*np.linalg.norm(wanted)),-1,1)))))
                    desired=np.asarray(s['palm'][j],dtype=float)
                    if desired.shape!=(3,3) or not np.allclose(desired.T@desired,np.eye(3),atol=1e-4) or np.linalg.det(desired)<.999:raise ValueError('reference palm is not SO(3)')
                    current=data.xmat[specs[side].body_id].reshape(3,3).copy()
                    side_errors.append(direction_errors+[float(np.degrees(Rotation.from_matrix(desired.T@current).magnitude()))]);side_rot.append(current)
                errors.append(side_errors);rotations.append(side_rot)
            raw.append(errors[0]);compensated.append(errors[1]);joint_samples.append(q);actual_rot.append(rotations[1])
        raw=np.asarray(raw);errors=np.asarray(compensated);q=np.asarray(joint_samples)
        report.update(raw_p95_deg=np.percentile(raw,95,axis=0).tolist(),compensated_p95_deg=np.percentile(errors,95,axis=0).tolist())
        for j,side in enumerate(('left','right')):
            if np.any(np.percentile(errors[:,j,:2],95,axis=0)>10) or np.percentile(errors[:,j,2],95)>15:report['reasons'].append(side+':DIRECTION_OR_WRIST_ERROR')
            if longest_run(np.max(errors[:,j,:2],axis=1)>20,sample_times)>.1 or longest_run(errors[:,j,2]>30,sample_times)>.1:report['reasons'].append(side+':SUSTAINED_POSE_ERROR')
        # Availability uses every physics interval, not only successful callbacks.
        mask=(times>=start)&(times<=end);local=times[mask]
        availability=np.all(trace['module_state'][mask]=='TRACKING',axis=(1,2))
        weights=np.r_[np.diff(local),0.]
        report['joint_module_availability']=float(np.sum(weights*availability)/15.)
        if report['joint_module_availability']<.99 or longest_run(~availability,local)>.2:report['reasons'].append('OBSERVATION_UNAVAILABLE')
        # Events match actual aperture crossing with direction/side preserved.
        aperture={side:trace['qpos'][:,finger_joint_indices(model,side)[0]].mean(axis=1) for side in ('left','right')}
        actual_events=[]
        for side,values in aperture.items():
            state=None
            for i in np.flatnonzero(mask):
                observed=True if values[i]<=.0088 else False if values[i]>=.0352 else None
                if observed is not None:
                    if state is not None and state!=observed:actual_events.append(dict(time=times[i],side=side,closed=observed))
                    state=observed
        used=set()
        for event in episode['events']:
            t=event['time_ns']/1e9;side=event['side'];closed=bool(event['closed'])
            matches=[(k,e) for k,e in enumerate(actual_events) if k not in used and e['side']==side and e['closed']==closed and -.033<=e['time']-t<=.3]
            if not matches:report['reasons'].append('GRIPPER_EDGE_MISSING_OR_LATE');continue
            k,e=min(matches,key=lambda item:abs(item[1]['time']-t));used.add(k)
            values=aperture[side];velocity=np.diff(values)/np.diff(times)
            onset=np.flatnonzero((times[1:]>=t-.033)&(times[1:]<=t+.15)&((velocity<-.002) if closed else (velocity>.002)))
            if not len(onset):report['reasons'].append('GRIPPER_ONSET_LATE')
        if len(used)!=len(actual_events):report['reasons'].append('EXTRA_GRIPPER_EDGE')
        for interval in episode['static_intervals']:
            sm=(sample_times>=interval['start_ns']/1e9)&(sample_times<=interval['end_ns']/1e9)
            if sm.sum()<5:report['reasons'].append('STATIC_REFERENCE_TOO_SHORT');continue
            for j,side in enumerate(('left','right')):
                elbow=q[sm,specs[side].qpos[3]]
                rotations=Rotation.from_matrix(np.asarray(actual_rot)[sm,j]);mean=rotations.mean()
                if np.std(elbow)>np.deg2rad(3) or np.sqrt(np.mean((mean.inv()*rotations).magnitude()**2))>np.deg2rad(5):report['reasons'].append(side+':STATIC_JITTER')
        # A reviewer must inspect physical contact/identity evidence. Do not turn
        # an unknown collision audit or unspecified lag measurement into PASS.
        if episode.get('execution_integrity_review') is not True:report['reasons'].append('EXECUTION_INTEGRITY_NOT_VERIFIED')
        if episode.get('measured_lag_s') is None:report['reasons'].append('INDEPENDENT_LAG_MEASUREMENT_MISSING')
        elif not 0<=episode['measured_lag_s']<=.15:report['reasons'].append('FOLLOWING_LAG_EXCEEDED')
        if episode.get('identity_review') is not True:report['reasons'].append('ANATOMICAL_IDENTITY_NOT_VERIFIED')
        unknown={'EXECUTION_INTEGRITY_NOT_VERIFIED','INDEPENDENT_LAG_MEASUREMENT_MISSING','ANATOMICAL_IDENTITY_NOT_VERIFIED'}
        report['verdict']='NOT_EVALUABLE' if unknown.intersection(report['reasons']) else ('FAIL' if report['reasons'] else 'PASS')
    return dict(verdict='PASS' if reports and all(r['verdict']=='PASS' for r in reports) else 'NOT_EVALUABLE' if any(r['verdict']=='NOT_EVALUABLE' for r in reports) or not reports else 'FAIL',episodes=reports,
                release_verdict='NOT_EVALUABLE; a single development recording is not independent preregistered acceptance')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('recording',type=Path)
    parser.add_argument('--reference',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--write-reference-template',type=Path)
    args=parser.parse_args();report=diagnostics(args.recording)
    if args.write_reference_template:args.write_reference_template.write_text(json.dumps(reference_template(args.recording),indent=2),encoding='utf-8')
    if args.reference:report['independent_evaluation']=evaluate_reference(args.recording,json.loads(args.reference.read_text(encoding='utf-8')))
    if args.output:args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
