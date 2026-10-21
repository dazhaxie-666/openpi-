"""Read-only structural validation; synthetic fixtures never qualify as human data."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


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
        if n > 1 and not np.allclose(np.diff(rows['sim_time_s']), 1 / manifest['control_hz'], atol=1e-7):
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
        if synthetic != bool(report['synthetic_input']) or synthetic != (manifest['source'] == 'synthetic_software_fixture'):
            errors.append('source/synthetic labels disagree')
        if synthetic and report['human_validation_performed']:
            errors.append('synthetic input incorrectly labelled as human validation')
        reasons = {event['reason'] for event in events}
        if report['verdict'] == 'PASS':
            if not report['evaluation']['physical_task_success']:
                errors.append('PASS without independent physics success')
            for reason in ('human_close_near_camera_target', 'human_open_in_goal_zone: assisted_lower_before_release'):
                if reason not in reasons:
                    errors.append('PASS without required human intent gate: ' + reason)
        if not manifest['finished']:
            warnings.append('recording not finalized')
        if synthetic:
            warnings.append('software fixture; exclude from human-demonstration training and success statistics')
        if not manifest['cameras']:
            warnings.append('no robot RGB observations; cannot train a visual policy from this recording')
        if np.any(rows['paused']):
            warnings.append('paused rows must be removed explicitly before training')
        result.update(recording_valid=not errors, frames=n, image_count=image_count,
                      task_success=report['verdict'] == 'PASS', synthetic_input=synthetic,
                      candidate_for_human_bc_review=bool(not errors and not synthetic and report['verdict'] == 'PASS'
                                                         and manifest['finished']),
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
