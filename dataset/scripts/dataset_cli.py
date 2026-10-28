"""Local collection workspace. Reuses the production B simulator and controller."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / 'latest' / 'openpi--main'
if not (PROJECT / 'openarm_cloth').is_dir():
    PROJECT = ROOT.parent
B_ROOT = PROJECT / 'openarm_cloth'
sys.dont_write_bytecode = True
sys.path.insert(0, str(B_ROOT))
for env in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(env, '1')


def stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def configuration():
    return json.loads((ROOT / 'config' / 'stages.json').read_text(encoding='utf-8'))


def setup():
    for name in ('raw', 'reports', 'logs', 'snapshots', 'exports', 'tmp', 'calibration'):
        (ROOT / name).mkdir(exist_ok=True)
    for stage in configuration()['stages'].values():
        for name in ('raw', 'reports', 'snapshots'):
            (ROOT / name / stage['name']).mkdir(parents=True, exist_ok=True)
    personal = ROOT / 'calibration/pinch.json'
    old_personal = B_ROOT / 'calibration/personal_motion/pinch.json'
    if not personal.exists() and old_personal.is_file():
        shutil.copy2(old_personal, personal)
    os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
    tempfile.tempdir = str(ROOT / 'tmp')


def source_hashes():
    paths = [B_ROOT / name for name in ('run_human_task.py', 'shared_grasp.py', 'shared_grasp_robot.py',
             'human_task_robot.py', 'member_b_actuators.py', 'human_palm_depth.py', 'frontal_image_wrist.py')]
    paths.extend(sorted((ROOT / 'scripts').glob('*.py')))
    paths.extend([ROOT / 'config' / 'stages.json', B_ROOT / 'simulation/v1/task_workspace.json',
                  B_ROOT / 'simulation/v1/droid_cube_scene_perception.xml'])
    return {str(p.relative_to(ROOT.parent)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.file.flush()


def preflight(camera=None):
    import numpy as np
    import mujoco
    from member_b_scene import build_scene
    from handeye_grasp_red_cube import finger_joint_indices
    from b_scene_fusion import BVisualObserver
    from capture_extension import DatasetCapture
    from run_human_task import build_parser
    config = configuration()
    errors, warnings = [], []
    model, data = build_scene()
    from member_b_actuators import ActuatorController
    control = ActuatorController(model, data)
    for side in ('right',):
        if len(control.specs[side].qpos) != 7 or len(finger_joint_indices(model, side)[0]) != 2:
            errors.append('unexpected right arm/finger layout')
    assets = PROJECT / 'pose2arm-main' / 'models'
    for name in ('hand_landmarker.task', 'pose_landmarker_full.task'):
        if not (assets / name).is_file():
            errors.append('missing model asset: ' + name)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    for name, options_cls, create in (
        ('hand_landmarker.task', vision.HandLandmarkerOptions, vision.HandLandmarker.create_from_options),
        ('pose_landmarker_full.task', vision.PoseLandmarkerOptions, vision.PoseLandmarker.create_from_options),
    ):
        with create(options_cls(base_options=mp_python.BaseOptions(model_asset_path=str(assets / name),
                                delegate=mp_python.BaseOptions.Delegate.CPU))):
            pass
    from types import SimpleNamespace
    observer = BVisualObserver(model, control.specs, camera_sides=(None, 'right'))
    try:
        sample = DatasetCapture(10).collect_frames(SimpleNamespace(observer=observer, data=data), 0.)
        for name, frame in sample.items():
            if frame['rgb'].shape[:2] != frame['depth'].shape or not np.isfinite(frame['depth']).all():
                errors.append('invalid RGB-D render: ' + name)
    finally:
        observer.close()
    build_parser().parse_args(['--shared-grasp', '--collect-shared-dataset'])
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < 10:
        errors.append('less than 10 GiB free on dataset drive')
    elif free < 50:
        warnings.append('less than 50 GiB free; measure storage use before scaling')
    calibration = ROOT / 'calibration/pinch.json'
    if not calibration.is_file():
        warnings.append('no personal pinch calibration; calibrate before live collection')
    camera_result = None
    if camera is not None:
        import time
        import cv2
        capture = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        try:
            if not capture.isOpened():
                errors.append('human camera could not open; close other camera applications')
            else:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                start = time.perf_counter()
                reads, shape = 0, None
                while time.perf_counter() - start < 3:
                    ok, frame = capture.read()
                    if not ok:
                        errors.append('human camera read failed')
                        break
                    reads += 1
                    shape = list(frame.shape[1::-1])
                camera_result = dict(device=camera, frames_read=reads, size_px=shape,
                                     read_rate_hz=reads / (time.perf_counter() - start),
                                     images_saved=False, scope='read throughput, not verified unique exposure rate')
                if reads < 5:
                    errors.append('insufficient human camera frames')
        finally:
            capture.release()
    report = dict(schema='openarm_dataset_preflight_v1', verdict='FAIL' if errors else 'PASS',
                  errors=errors, warnings=warnings, python=sys.executable,
                  packages={n: importlib.metadata.version(n) for n in ('numpy', 'mujoco', 'mediapipe', 'opencv-python')},
                  free_gib=free, camera=camera_result, source_hashes=source_hashes(),
                  capture_hz=config['capture_hz'], control_hz=config['control_hz'],
                  scene_simulation_only=True, control_behavior_changed=False)
    output = ROOT / 'reports/preflight' / (stamp() + '.json')
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Preflight report:', output)
    return 0 if not errors else 1


def batch_label(batch):
    if isinstance(batch, bool) or not isinstance(batch, int) or not 1 <= batch <= 999999:
        raise ValueError('batch must be an integer in 1..999999')
    return f'batch_{batch:03d}'


def workspace_dir(kind, context):
    path = ROOT / kind / context['dataset_name']
    if context.get('batch_id') is not None:
        path = path / batch_label(context['batch_id'])
    return path


def stage_output(stage, batch=1):
    config = configuration()
    name = config['stages'][stage]['name']
    return name, ROOT / 'raw' / name / batch_label(batch) / (stamp() + '_' + uuid.uuid4().hex[:6])


def audit(session):
    from dataset_quality import audit_recording
    report = audit_recording(session, configuration()['quality'])
    name = report['dataset_name'] or 'external_review'
    context_path = Path(session) / 'collection.json'
    context = json.loads(context_path.read_text(encoding='utf-8')) if context_path.exists() else dict(dataset_name=name)
    output = workspace_dir('reports', context) / (Path(session).name + '.audit.json')
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Audit report:', output)
    return report


def collect(args, smoke=False):
    from capture_extension import DatasetCapture
    from run_human_task import build_parser, run
    config = configuration()
    stage = 'smoke' if smoke else args.stage
    if stage == 'smoke' and not smoke:
        raise ValueError('software fixtures use smoke mode, never live collection')
    if args.hz != config['capture_hz']:
        raise ValueError('this v1 contract fixes capture at 10 Hz; change/version config before changing rate')
    name, output = stage_output(stage, args.batch)
    pose_key = args.pose
    if pose_key == 'random':
        pose_key = random.Random(args.seed if args.seed is not None else uuid.uuid4().int).choice(list(config['initial_poses']))
    pose = config['initial_poses'][pose_key]
    argv = ['--control', 'direct', '--mapping', 'wrist', '--shared-grasp', '--collect-shared-dataset',
            '--arms', 'right', '--gripper-mode', 'continuous', '--direct-wrist-policy', 'palm',
            '--no-images', '--control-hz', str(config['control_hz']), '--preview-hz', '30',
            '--robot-preview-hz', '10', '--close-on-finish', '--output', str(output),
            '--calibration', str(ROOT / 'calibration/pinch.json'),
            '--camera', str(args.camera), '--cube-test-pose', *map(str, pose)]
    if smoke:
        argv += ['--check', '--scripted-check', '--check-steps', str(args.steps)]
    else:
        argv += ['--realtime', '--position-frame', 'fixed-image', '--body-position-source', 'mediapipe',
                 '--depth-source', 'palm-planar', '--depth-gain', '1.0']
    capture = DatasetCapture(args.hz, max_trials=args.trials,
                             max_seconds=0 if smoke else args.seconds,
                             reset_seconds=args.reset_seconds)
    metadata = dict(schema='openarm_collection_context_v1', stage=stage, dataset_name=name,
                    batch_id=args.batch, batch_name=batch_label(args.batch),
                    purpose=config['stages'][stage]['purpose'],
                    training_allowed=config['stages'][stage]['training_allowed'] and not smoke,
                    operator=args.operator, seed=args.seed, initial_pose_key=pose_key,
                    initial_fixture_pose=pose, fixture_scope='initialization only; never policy/perception input',
                    simulation_only=True, synthetic_input=smoke, arguments=argv,
                    source_hashes=source_hashes(), created=datetime.now().isoformat(),
                    training_status='raw source; requires review, conversion and action replay')
    capture.set_collection_context(metadata)
    snapshot = workspace_dir('snapshots', metadata) / (output.name + '.json')
    write_json(snapshot, dict(context=metadata, contract=config))
    output.parent.mkdir(parents=True, exist_ok=True)
    print('Dataset:', name)
    print('Batch:', args.batch, '|', batch_label(args.batch))
    print('Session:', output)
    print('Initial pose:', pose_key, pose)
    result = 1
    try:
        result = run(build_parser().parse_args(argv), dataset_capture=capture)
    finally:
        if output.is_dir():
            write_json(output / 'collection.json', metadata)
    report = audit(output)
    if smoke:
        from replay_human_task import replay
        from dataset_quality import recording_digest
        replay_report = replay(output, execution='actions', check=True)
        replay_report['recording_digest'] = recording_digest(output)
        write_json(output / 'raw_action_replay.json', replay_report)
        write_json(workspace_dir('reports', metadata) / (output.name + '.replay.json'), replay_report)
        report = audit(output)
        valid_fixture = any(t.get('physical_success') and not t['issues'] for t in report['trials'])
        passed = report['recording_valid'] and valid_fixture and replay_report['verdict'] == 'PASS'
        write_json(workspace_dir('reports', metadata) / (output.name + '.smoke.json'),
                   dict(verdict='PASS' if passed else 'FAIL', synthetic_input=True, training_allowed=False,
                        replay=replay_report, physical_success_segments=report['physical_success_segments'],
                        scope='software fixture; not human usability or live-camera throughput validation'))
        print('SOFTWARE SMOKE:', 'PASS' if passed else 'FAIL', '| data:', output)
        return 0 if passed else 1
    print('SAVED:', output, '| candidates:', report['candidate_successes'],
          '| approved for training:', report['training_approved_successes'])
    return result if not report['recording_valid'] else 0


def resolve_session(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT / 'raw') or not (path / 'collection.json').is_file():
        raise ValueError('choose a dataset/raw session created by this workspace')
    return path


def replay_session(session, visible=False):
    from replay_human_task import replay
    from dataset_quality import recording_digest
    context = json.loads((session / 'collection.json').read_text(encoding='utf-8'))
    report = replay(session, execution='actions', check=not visible)
    report['recording_digest'] = recording_digest(session)
    write_json(session / 'raw_action_replay.json', report)
    path = workspace_dir('reports', context) / (session.name + '.replay_' + stamp() + '.json')
    write_json(path, report)
    print(json.dumps(report, indent=2))
    audit(session)
    return 0 if report['verdict'] == 'PASS' else 1


def review_session(args):
    session = resolve_session(args.session)
    report = audit(session)
    selected = next((t for t in report['trials'] if t['trial_id'] == args.trial), None)
    if selected is None:
        raise ValueError('unknown trial ID')
    if args.decision == 'accept' and not selected['candidate']:
        raise ValueError('hard quality failure or synthetic source cannot be approved as a human demonstration')
    if args.decision == 'accept' and not report['raw_action_replay_verified']:
        raise ValueError('run Replay first; accepted training demonstrations require matching raw-action replay')
    path = session / 'review.json'
    reviews = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    reviews[str(args.trial)] = dict(decision=args.decision, note=args.note, time=datetime.now().isoformat(),
                                   recording_digest=report['recording_digest'],
                                   quality_contract_digest=report['quality_contract_digest'],
                                   reviewer=args.operator, scope='explicit visual review; original labels unchanged')
    write_json(path, reviews)
    audit(session)
    return 0


def summary():
    rows = []
    for stage in configuration()['stages'].values():
        sessions = sorted((ROOT / 'raw' / stage['name']).rglob('collection.json'))
        candidates = approved = attempted = 0
        batches = {}
        invalid = []
        for context in sessions:
            metadata = json.loads(context.read_text(encoding='utf-8'))
            label = batch_label(metadata['batch_id']) if metadata.get('batch_id') is not None else 'legacy_unbatched'
            batch = batches.setdefault(label, dict(sessions=0, trials=0, candidate_successes=0, training_approved_successes=0))
            batch['sessions'] += 1
            report_path = workspace_dir('reports', metadata) / (context.parent.name + '.audit.json')
            if not report_path.exists():
                invalid.append(context.parent.name)
                continue
            report = json.loads(report_path.read_text(encoding='utf-8'))
            attempted += len(report['trials'])
            candidates += report['candidate_successes']
            approved += report['training_approved_successes']
            batch['trials'] += len(report['trials'])
            batch['candidate_successes'] += report['candidate_successes']
            batch['training_approved_successes'] += report['training_approved_successes']
        rows.append(dict(dataset_name=stage['name'], sessions=len(sessions), trials=attempted,
                         candidate_successes=candidates, training_approved_successes=approved,
                         target_successes=stage['target_successes'], training_allowed=stage['training_allowed'],
                         missing_audits=invalid, batches=batches))
    write_json(ROOT / 'reports' / ('summary_' + stamp() + '.json'), rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'camera', 'smoke', 'collect', 'audit', 'replay', 'review', 'summary'])
    parser.add_argument('--stage', choices=list(configuration()['stages']), default='pilot')
    parser.add_argument('--session')
    parser.add_argument('--pose', choices=['random', *configuration()['initial_poses']], default='center')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--operator', default='operator01')
    parser.add_argument('--batch', type=int, default=1, help='numbered collection batch, e.g. 1, 2, 3')
    parser.add_argument('--hz', type=float, default=10)
    parser.add_argument('--trials', type=int, default=1, help='stop after attempts, including failures; 0 means Q saves')
    parser.add_argument('--reset-seconds', type=float, default=8,
                        help='protected operator reset time between attempts; 0 disables it')
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--visible', action='store_true')
    parser.add_argument('--trial', type=int)
    parser.add_argument('--decision', choices=['accept', 'reject'])
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    batch_label(args.batch)
    if args.trials < 0 or args.seconds < 0 or args.steps < 1:
        parser.error('invalid stop bounds')
    setup()
    log = ROOT / 'logs' / (args.mode + '_' + stamp() + '.log')
    with log.open('w', encoding='utf-8', buffering=1) as stream:
        with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            print('Log:', log)
            try:
                if args.mode in ('check', 'camera'):
                    return preflight(args.camera if args.mode == 'camera' else None)
                if args.mode in ('smoke', 'collect'):
                    return collect(args, smoke=args.mode == 'smoke')
                if args.mode == 'summary':
                    return summary()
                if not args.session:
                    parser.error('--session is required')
                if args.mode == 'review':
                    if args.trial is None or args.decision is None or not args.note.strip():
                        parser.error('review requires --trial, --decision and a nonempty --note')
                    return review_session(args)
                session = resolve_session(args.session)
                if args.mode == 'audit':
                    return 0 if audit(session)['recording_valid'] else 1
                return replay_session(session, args.visible)
            except Exception:
                traceback.print_exc()
                return 1


if __name__ == '__main__':
    raise SystemExit(main())
