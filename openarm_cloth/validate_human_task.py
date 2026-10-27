"""Read-only structural validation; synthetic fixtures never qualify as human data."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from single_arm_runtime import recorded_step_durations


def validate(path):
    path = Path(path)
    errors = []
    warnings = []
    result = dict(recording_valid=False, errors=errors, warnings=warnings,
                  candidate_for_human_bc_review=False, synthetic_input=None)
    try:
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        report = json.loads((path / 'result.json').read_text(encoding='utf-8'))
        events = json.loads((path / 'events.json').read_text(encoding='utf-8'))
        if manifest['schema'] != 'openarm_human_task_v1':
            errors.append('unknown schema')
        with np.load(path / 'episode.npz', allow_pickle=False) as blob:
            rows = {key: blob[key].copy() for key in blob}
        n = len(rows['time_s'])
        if n == 0 or n != manifest['frames'] or n != report['frames']:
            errors.append('frame count mismatch/empty')
        if any(value.ndim == 0 or len(value) != n for value in rows.values()):
            errors.append('inconsistent array lengths')
        for key, width in (('qpos', manifest['nq']), ('qvel', manifest['nv']), ('ctrl', manifest['nu']),
                           ('action_target', 18), ('physics_state', manifest['physics_state_size'])):
            if rows[key].shape != (n, width) or not np.isfinite(rows[key]).all():
                errors.append(key + ': invalid shape or nonfinite state')
        for key in ('time_s', 'sim_time_s'):
            if not np.isfinite(rows[key]).all() or np.any(np.diff(rows[key]) < 0):
                errors.append(key + ': invalid clock order')
        durations=recorded_step_durations(rows,manifest,n)
        if n > 1 and not np.allclose(np.diff(rows['sim_time_s']), durations[:-1], atol=1e-7):
            errors.append('simulation step gaps')
        fingers = rows['action_target'][:, [7, 8, 16, 17]]
        if np.any(fingers < 0) or np.any(fingers > .044 + 1e-9):
            errors.append('finger targets outside metre limits')
        if not np.isfinite(rows['human_point'][rows['human_valid']]).all():
            errors.append('valid human observation contains invalid point')
        if not np.isfinite(rows['pinch_ratio'][rows['human_valid']]).all():
            errors.append('valid human observation contains invalid gesture')
        model = path / manifest['model_file']
        if hashlib.sha256(model.read_bytes()).hexdigest() != manifest['model_sha256']:
            errors.append('model snapshot hash mismatch')
        image_count = 0
        for camera in manifest['cameras']:
            expected = {f'{index:06d}.png' for index in range(n)}
            directory = path / 'images' / camera
            actual = {file.name for file in directory.glob('*.png')}
            if actual != expected:
                errors.append(camera + ': missing/extra images')
            for name in sorted(actual & expected):
                image = cv2.imread(str(directory / name))
                if image is None or list(image.shape[1::-1]) != manifest['image_size']:
                    errors.append(camera + '/' + name + ': unreadable/wrong size')
            image_count += len(actual)
        synthetic = bool(manifest['synthetic_input'])
        shared_observations=manifest.get('shared_observations')
        if shared_observations:
            if not shared_observations.get('valid',False):errors.append('shared RGB-D recording incomplete/dropped/error')
            recorded=0
            for line in (path/shared_observations['index']).read_text(encoding='utf-8').splitlines():
                sample=json.loads(line);index=int(sample['control_frame'])
                if not 0<=index<n or abs(float(rows['time_s'][index])-sample['time_s'])>1e-7:
                    errors.append('shared image/control timestamp mismatch');continue
                rgb=cv2.imread(str(path/sample['rgb']))
                with np.load(path/sample['depth'],allow_pickle=False) as depth_file:depth=depth_file['depth_m']
                if rgb is None or list(rgb.shape[1::-1])!=sample['size_px'] or depth.shape!=rgb.shape[:2] or not np.isfinite(depth).all():
                    errors.append('invalid shared RGB-D observation')
                recorded+=1
            if recorded!=shared_observations['saved']:errors.append('shared RGB-D count mismatch')
            image_count+=recorded
        if synthetic != bool(report['synthetic_input']) or synthetic != (manifest['source'] == 'synthetic_software_fixture'):
            errors.append('source/synthetic labels disagree')
        if synthetic and report['human_validation_performed']:
            errors.append('synthetic input incorrectly labelled as human validation')
        if manifest.get('control_mode', 'assisted') != report.get('control_mode', 'assisted'):
            errors.append('control mode labels disagree')
        reasons = {event['reason'] for event in events}
        if report['verdict'] == 'PASS':
            if not report['evaluation']['physical_task_success']:
                errors.append('PASS without independent physics success')
            required_intents = (('human_direct_start','shared_phase_plan','shared_phase_lower') if manifest.get('shared_grasp_workflow') else
                                ('human_direct_start', 'human_direct_close', 'human_direct_open')
                                if manifest.get('control_mode') == 'direct' else
                                ('human_close_near_camera_target', 'human_open_in_goal_zone: assisted_lower_before_release'))
            for reason in required_intents:
                if reason not in reasons:
                    errors.append('PASS without required human intent gate: ' + reason)
        if not manifest['finished']:
            warnings.append('recording not finalized')
        if synthetic:
            warnings.append('software fixture; exclude from human-demonstration training and success statistics')
        if not manifest['cameras'] and not shared_observations:
            warnings.append('no robot RGB observations; cannot train a visual policy from this recording')
        if np.any(rows['paused']):
            warnings.append('paused rows must be removed explicitly before training')
        if manifest.get('control_mode') == 'direct':
            warnings.append('continuous direct session may include retries; review/segment successful demonstrations before training')
        if manifest.get('shared_grasp_workflow'):
            warnings.append('shared autonomy: use eligible per-trial segments, not the complete session or unassisted-human labels')
            result['eligible_shared_segments']=sum(bool(t.get('eligible_success_segment')) for t in manifest.get('shared_trials',[]))
            full_segments=0;last_end=0
            if manifest.get('shared_segment_schema')=='full_task_v2':
                for trial in manifest.get('shared_trials',[]):
                    start=int(trial['start_frame']);end=int(trial['end_frame_exclusive']);request=trial.get('request_frame')
                    if not 0<=start<end<=n or start<last_end:
                        errors.append('shared full-task segments overlap/outside saved rows');continue
                    last_end=end
                    if not np.all(rows['shared_episode_id'][start:end]==trial['trial_id']):
                        errors.append('shared episode ID does not match full-task segment')
                    if request is not None and not start<=request<end:errors.append('shared grasp request outside full-task segment')
                    if trial.get('eligible_success_segment'):
                        before=rows['shared_phase'][start:request] if request is not None else []
                        phases=set(rows['shared_phase'][start:end])
                        complete=bool('manual' in before and {'close','lift','transport','release','verify'}<=phases)
                        if not complete or not trial.get('complete_task_segment'):errors.append('eligible shared segment misses approach or task phases')
                        else:full_segments+=1
            else:warnings.append('historical shared boundaries start at grasp request; not certified as full-task demonstrations')
            result['eligible_full_task_segments']=full_segments
        result.update(recording_valid=not errors, frames=n, image_count=image_count,
                      task_success=report['verdict'] == 'PASS', synthetic_input=synthetic,
                      candidate_for_human_bc_review=bool(not manifest.get('shared_grasp_workflow') and not errors and not synthetic and report['verdict'] == 'PASS'
                                                         and manifest['finished']),
                      candidate_for_shared_bc_review=bool(manifest.get('shared_grasp_workflow') and not errors and not synthetic
                          and manifest.get('shared_observations',{}).get('valid',False) and manifest['finished']
                          and result.get('eligible_shared_segments',0)>0),
                      candidate_for_full_task_bc_review=bool(manifest.get('shared_grasp_workflow') and not errors and not synthetic
                          and manifest.get('shared_observations',{}).get('valid',False) and manifest['finished']
                          and result.get('eligible_full_task_segments',0)>0),
                      scope='structural consistency and task gates; not human perception ground truth or a training benchmark')
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        errors.append(f'{type(error).__name__}: {error}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.recording)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['recording_valid'] else 1)


if __name__ == '__main__':
    main()
