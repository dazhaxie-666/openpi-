"""Read-only episode review, including timing and manual-target jitter signals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CAMERAS = ('chest_d435i_preview', 'v22_right_d405_rgb')


def recording_digest(session):
    """Pin review/replay to the recorded indices, arrays, manifest and trials."""
    digest = hashlib.sha256()
    for name in ('manifest.json', 'episode.npz', 'shared_trials.json', 'shared_frames.jsonl'):
        digest.update(name.encode())
        with (Path(session) / name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def robot_arrays(rows, manifest):
    """Extract robot-only inputs. Object/free-joint state is never exported."""
    joints = manifest['robot_qpos_index']['right']
    fingers = manifest['finger_qpos_index']['right']
    raw = np.asarray(rows['action_target'][:, 9:18], dtype=np.float64)
    actual = np.asarray(rows['qpos'], dtype=np.float64)
    if len(joints) != 7 or len(fingers) != 2 or len(set(joints + fingers)) != 9:
        raise ValueError('robot/finger index layout is invalid')
    if raw.shape[1] != 9 or not np.isfinite(raw).all():
        raise ValueError('right target must be seven radians plus two metre fingers')
    if not np.allclose(raw[:, 7], raw[:, 8], atol=1e-7, rtol=0):
        raise ValueError('independent finger targets cannot be compressed to one command')
    if np.any(raw[:, 7:] < -1e-8) or np.any(raw[:, 7:] > .044 + 1e-8):
        raise ValueError('finger target is outside metre range')
    state = np.column_stack((actual[:, joints], np.clip(1 - actual[:, fingers].mean(axis=1) / .044, 0, 1)))
    action = np.column_stack((raw[:, :7], np.clip(1 - raw[:, 7:].mean(axis=1) / .044, 0, 1)))
    if not np.isfinite(state).all():
        raise ValueError('robot state contains nonfinite values')
    return state.astype(np.float32), action.astype(np.float32)


def manual_motion_metrics(rows, start, end, settings):
    times = np.asarray(rows['time_s'][start:end], float)
    phase = np.asarray(rows['shared_phase'][start:end]).astype(str)
    target = np.asarray(rows['action_target'][start:end, 9:16], float)
    manual = np.isin(phase, ['manual', 'transport'])
    dt = np.diff(times)
    pair = manual[1:] & manual[:-1] & (phase[1:] == phase[:-1]) & (dt > 1e-6)
    delta = np.diff(target, axis=0)
    speed = np.abs(delta[pair] / dt[pair, None])
    duration = float(dt[pair].sum())
    # Ignore tiny numerical reversals and phase handoffs; this is an audit
    # heuristic, not a perceptual-jitter classifier or a smoothing algorithm.
    direction = np.where(np.abs(delta) >= .001, np.sign(delta), 0)
    previous = np.zeros(7)
    reversals = 0
    for index, valid in enumerate(pair):
        if not valid:
            previous[:] = 0
            continue
        current = direction[index]
        if np.any(current * previous < 0):
            reversals += 1
        moving = current != 0
        previous[moving] = current[moving]
    warnings = np.asarray(rows.get('motion_warning', np.full(len(rows['time_s']), ''))[start:end]).astype(str)
    rejection_fraction = float(np.mean(warnings[manual] != '')) if np.any(manual) else 0.
    rate = reversals * 60 / duration if duration > 0 else 0.
    p95 = float(np.percentile(speed, 95)) if speed.size else 0.
    maximum = float(speed.max()) if speed.size else 0.
    flags = []
    if maximum > settings['manual_joint_speed_warning_rad_s']:
        flags.append('manual target speed spike; inspect wrist and arm video')
    if duration >= 2 and rate > settings['manual_target_reversal_warning_per_minute']:
        flags.append('frequent manual target reversals; inspect before accepting')
    if rejection_fraction > settings['max_manual_rejection_fraction']:
        flags.append('frequent B motion warnings/holds; inspect collision and tracking')
    return dict(manual_duration_s=duration, target_joint_speed_p95_rad_s=p95,
                target_joint_speed_max_rad_s=maximum, reversal_frames=reversals,
                reversal_frames_per_minute=rate, manual_warning_fraction=rejection_fraction,
                review_flags=flags, scope='manual/transport accepted-target heuristic; not ground-truth human tremor')


def camera_metrics(samples, times, start, end, settings):
    metrics, issues = {}, []
    first, last = float(times[start]), float(times[end - 1])
    for camera in CAMERAS:
        chosen = [s for s in samples if s['camera'] == camera and start <= int(s['control_frame']) < end]
        stamps = np.array([float(s['time_s']) for s in chosen])
        gaps = np.diff(stamps)
        hz = float((len(stamps) - 1) / (stamps[-1] - stamps[0])) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.
        gap = float(max(gaps.max(initial=0), stamps[0] - first, last - stamps[-1])) if len(stamps) else None
        if len(stamps) < 2 or np.any(gaps <= 0):
            issues.append(camera + ': missing/duplicate/nonmonotonic observations')
        if gap is None or gap > settings['max_camera_gap_s']:
            issues.append(camera + ': image gap exceeds capture contract')
        if last - first >= 1 and hz < settings['min_camera_hz']:
            issues.append(camera + ': achieved new-image rate below contract')
        metrics[camera] = dict(frames=len(stamps), achieved_hz=hz, max_gap_s=gap)
    return metrics, issues


def audit_recording(session, settings, validate_fn=None):
    session = Path(session)
    if validate_fn is None:
        from validate_human_task import validate
        validate_fn = validate
    structural = validate_fn(session)
    manifest = json.loads((session / 'manifest.json').read_text(encoding='utf-8'))
    trials = json.loads((session / 'shared_trials.json').read_text(encoding='utf-8'))
    collection_file = session / 'collection.json'
    collection = json.loads(collection_file.read_text(encoding='utf-8')) if collection_file.exists() else {}
    with np.load(session / 'episode.npz', allow_pickle=False) as archive:
        rows = {k: archive[k] for k in archive.files}
    samples = [json.loads(line) for line in (session / 'shared_frames.jsonl').read_text(encoding='utf-8').splitlines()]
    mapping_error = None
    try:
        robot_arrays(rows, manifest)
    except (ValueError, IndexError, KeyError) as error:
        mapping_error = str(error)
    times = rows['time_s']
    digest = recording_digest(session)
    contract_digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    replay_path = session / 'raw_action_replay.json'
    replay = json.loads(replay_path.read_text(encoding='utf-8')) if replay_path.exists() else {}
    replay_verified = bool(replay.get('recording_digest') == digest and replay.get('verdict') == 'PASS'
                           and replay.get('execution') == 'actions' and replay.get('physical_actuator_replay'))
    reports = []
    synthetic = bool(manifest.get('synthetic_input', True))
    training_allowed = bool(collection.get('training_allowed', False))
    reviews_path = session / 'review.json'
    reviews = json.loads(reviews_path.read_text(encoding='utf-8')) if reviews_path.exists() else {}
    for trial in trials:
        start, end = int(trial['start_frame']), int(trial['end_frame_exclusive'])
        issues = []
        if not 0 <= start < end <= len(times):
            reports.append(dict(trial_id=trial['trial_id'], issues=['invalid trial bounds'], candidate=False, training_approved=False))
            continue
        if not structural['recording_valid']:
            issues.append('structural validation failed')
        if not manifest.get('finished'):
            issues.append('recording was not finalized')
        if manifest.get('shared_segment_schema') != 'full_task_v2' or not trial.get('complete_task_segment'):
            issues.append('not a complete full_task_v2 demonstration')
        if not trial.get('eligible_success_segment'):
            issues.append('full physical task was not successful')
        if mapping_error:
            issues.append(mapping_error)
        if np.count_nonzero(rows['paused'][start:end]) > settings['max_active_pause_rows']:
            issues.append('pause inside episode; do not concatenate across it')
        dt = np.diff(times[start:end])
        max_gap = float(dt.max(initial=0))
        if np.any(dt <= 0) or max_gap > settings['max_control_gap_s']:
            issues.append('invalid/long control time gap')
        cameras, camera_issues = camera_metrics(samples, times, start, end, settings)
        issues.extend(camera_issues)
        # Independent evaluation only. It is never returned by robot_arrays.
        source_distance = None
        evaluation = rows.get('evaluation_next')
        if evaluation is not None and np.isfinite(evaluation[start, :2]).all():
            source_distance = float(np.linalg.norm(evaluation[start, :2] - np.asarray(manifest['goal']['center_xy_m'])))
        if source_distance is None or source_distance < settings['min_source_goal_distance_m']:
            issues.append('initial object is already at/too near goal; needs a real transfer')
        motion = manual_motion_metrics(rows, start, end, settings)
        candidate = not issues and not synthetic
        review = reviews.get(str(trial['trial_id']), {})
        reviewed = bool(review.get('decision') == 'accept' and review.get('recording_digest') == digest
                        and review.get('quality_contract_digest') == contract_digest)
        reports.append(dict(trial_id=trial['trial_id'], status=trial['status'], start_frame=start,
                            end_frame_exclusive=end, duration_s=float(times[end - 1] - times[start]),
                            physical_success=bool(trial.get('physics', {}).get('physical_task_success')),
                            issues=issues, motion=motion, cameras=cameras, max_control_gap_s=max_gap,
                            initial_goal_distance_m=source_distance, candidate=candidate,
                            needs_visual_review=bool(candidate and not reviewed),
                            training_approved=bool(candidate and reviewed and training_allowed and replay_verified)))
    return dict(schema='openarm_dataset_audit_v1', session=str(session.resolve()),
                dataset_name=collection.get('dataset_name'), stage=collection.get('stage'),
                batch_id=collection.get('batch_id'), batch_name=collection.get('batch_name'),
                synthetic_input=synthetic, training_allowed=training_allowed,
                recording_digest=digest, quality_contract_digest=contract_digest,
                raw_action_replay_verified=replay_verified,
                recording_valid=structural['recording_valid'], structural_errors=structural['errors'],
                structural_warnings=structural['warnings'], trials=reports,
                physical_success_segments=sum(bool(t.get('eligible_success_segment')) for t in trials),
                candidate_successes=sum(t['candidate'] for t in reports),
                training_approved_successes=sum(t['training_approved'] for t in reports),
                scope='candidates require visual review; synthetic/pilot/evaluation excluded from training')
