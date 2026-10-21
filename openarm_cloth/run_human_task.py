"""Human-controlled single-arm cube task. Live USB, measured video, or explicit fixture test.

Free-motion and autonomous grasp entry points are unchanged. No learning here.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import mujoco
import numpy as np

from full_video_source import FullVideoSource, load_full_landmarks
from human_grasp_task import HumanGraspTask, HumanInput, TaskConfig, normalized_pinch
from human_task_robot import HumanTaskRobot, TaskEvaluator
from member_b_scene import ROOT, TARGET_ZONE, build_scene
from handeye_grasp_red_cube import finger_joint_indices
from task_workspace import TABLE_TOP, CONFIG
from demo_grasp_viewer import place_cube_for_blind_test


def human_input(obs, now_s, source_clock=False):
    if obs is None:
        return HumanInput(-1, None, None, False)
    arm = obs['arms']['right']
    age = now_s - obs['read_end'] / 1e9
    ratio = normalized_pinch(arm.get('hand_raw')) if arm['hand_valid'] else None
    # This task translates a constant-attitude gripper. It needs a reliable
    # torso-relative wrist, not a visible palm or an anatomical arm rotation.
    valid = (arm['point'] is not None and np.asarray(arm['point']).shape == (3,)
             and np.isfinite(arm['point']).all() and (source_clock or 0 <= age <= .35))
    return HumanInput(int(obs['seq']), arm['point'], ratio, bool(valid), obs['read_end'] / 1e9)


def scripted_input(task, feedback, clock, index):
    """Closed-loop SOFTWARE FIXTURE. Never a human-demonstration success claim."""
    point = np.zeros(3)
    ratio = 1.2 if task.stage == 'ready' else .15
    if task.stage == 'follow' and feedback.visual is not None:
        target = feedback.visual['center'] + [0, 0, .055]
        point = task.human_anchor + (target - task.robot_anchor) / task.config.gain
        ratio = .15 if np.linalg.norm(feedback.tcp - feedback.visual['center']) < .085 else 1.2
    elif task.stage == 'transport':
        target = np.r_[task.config.goal_xy, task.robot_anchor[2]]
        point = task.human_anchor + (target - task.robot_anchor) / task.config.gain
        ratio = 1.2 if np.linalg.norm(feedback.tcp[:2] - task.config.goal_xy) < .012 else .15
    elif task.stage in ('lower', 'release', 'retract', 'verify'):
        ratio = 1.2
    return HumanInput(index, point, ratio, True, clock)


def run(args):
    if args.check and not (args.video or args.scripted_check):
        raise ValueError('--check requires --video or --scripted-check; live input uses visible windows')
    if args.scripted_check and not args.check:
        raise ValueError('--scripted-check must be explicitly combined with --check')
    if args.video and args.scripted_check:
        raise ValueError('Choose measured video OR software fixture')
    if args.save_human_video and (args.video or args.scripted_check):
        raise ValueError('--save-human-video is only for live USB')
    if not np.isfinite(args.max_wait) or args.max_wait <= 0:
        raise ValueError('--max-wait must be positive')
    cv2.setNumThreads(1)
    config = TaskConfig(timeout_s=args.timeout, gain=args.gain,
                        goal_xy=np.array(TARGET_ZONE['center_xy_m']), goal_radius_m=TARGET_ZONE['radius_m'],
                        close_ratio=args.close_ratio, open_ratio=args.open_ratio,
                        lower=np.array([.32, -.18, TABLE_TOP + CONFIG['cube_half_size_m']]))
    task = HumanGraspTask(config)
    output = args.output or ROOT / 'outputs' / 'human_task' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_kind = 'synthetic_software_fixture' if args.scripted_check else 'measured_video' if args.video else 'live_usb'
    metadata = dict(schema='openarm_human_task_v1', source=source_kind, simulation_only=True,
                    human_validation_performed=False, training_performed=False,
                    synthetic_input=bool(args.scripted_check), side='right',
                    goal=TARGET_ZONE, timeout_s=args.timeout, control_hz=20,
                    clocks='active wall time' if not args.check else 'simulation control ticks (offline test)',
                    startup='B pad IK fixed reset posture before episode; all episode execution dynamic',
                    source_timing='video pauses during assisted phases' if args.video else source_kind,
                    action_layout='left joint7+finger2,right joint7+finger2; absolute radians/metres',
                    alignment='images and qpos/action_target sampled before the same physical step',
                    human_video_saved=bool(args.save_human_video),
                    preview_mirrored=bool(args.mirror_preview),
                    human_validity='human_valid = pose AND gesture; separate human_pose_valid and human_hand_valid recorded',
                    cameras=[] if args.no_images else ['chest_d435i_preview', 'v22_right_d405_rgb'],
                    source_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                        'human_grasp_task.py', 'human_task_robot.py', 'run_human_task.py',
                        'member_b_actuators.py', 'b_scene_fusion.py')})
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    rows = []
    model = data = robot = evaluator = stream = source = None
    video_writer = None
    flags = dict(start=False, paused=False, quit=False, recenter=False, mirror=bool(args.mirror_preview))
    failure = None
    clock = cursor = 0.
    dt = .05
    last_human_video_seq = None
    window = 'HUMAN TASK | SPACE start/pause | C recenter | Q exit'

    def key(code):
        if code in (81, 113, 27):
            flags['quit'] = True
        elif code == 32:
            if task.stage == 'ready':
                flags['start'] = True
            else:
                flags['paused'] = not flags['paused']
        elif code in (67, 99):
            flags['recenter'] = True
        elif code in (77, 109):
            flags['mirror'] = not flags['mirror']

    def mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONUP and 12 <= x <= 235 and 488 <= y <= 530:
            key(32)

    try:
        model, data = build_scene()
        # Initial fixture only, before camera input. Never used by task planning.
        place_cube_for_blind_test(model, data, args.cube_test_pose)
        robot = HumanTaskRobot(model, data, lift_m=config.lift_m)
        robot.initialize()
        mujoco.mj_saveModel(model, str(output / 'scene.mjb'))
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state_size = mujoco.mj_stateSize(model, state_spec)
        metadata.update(mujoco_version=mujoco.__version__, model_file='scene.mjb',
                        model_sha256=hashlib.sha256((output / 'scene.mjb').read_bytes()).hexdigest(),
                        physics_state_spec=int(state_spec), physics_state_size=state_size,
                        evaluation_hz=1 / float(model.opt.timestep),
                        gain=config.gain, deadband_m=config.deadband_m,
                        max_target_lead_m=config.max_target_lead_m, image_size=[320, 240],
                        close_ratio=config.close_ratio, open_ratio=config.open_ratio)
        evaluator = TaskEvaluator(model, data, config)
        if args.video:
            path = args.video.resolve()
            cache = ROOT / 'outputs' / 'human_task' / 'cache' / (path.stem + '_landmarks.npz')
            source = FullVideoSource(load_full_landmarks(path, cache, False))
            with path.open('rb') as video_file:
                metadata.update(source_video=str(path), source_sha256=hashlib.file_digest(video_file, 'sha256').hexdigest())
        elif not args.scripted_check:
            from live_bimanual_stream import BimanualHumanStream
            stream = BimanualHumanStream(args.camera, args.pose_model, (1280, 720))
        metadata.update(nq=model.nq, nv=model.nv, nu=model.nu, timestep_s=float(model.opt.timestep),
                        robot_joint_names=[model.joint(i).name for i in range(model.njnt)],
                        robot_qpos_index={s: spec.qpos.tolist() for s, spec in robot.control.specs.items()},
                        finger_qpos_index={s: finger_joint_indices(model, s)[0].tolist()
                                           for s in robot.control.specs})
        with ExitStack() as stack:
            viewer = None
            source_preview = None
            if not args.check:
                from mujoco import viewer as mjviewer
                viewer = stack.enter_context(mjviewer.launch_passive(model, data, key_callback=key))
                viewer.cam.lookat[:] = [.43, 0, .46]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.2, 150, -30
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, 960, 540)
                cv2.setMouseCallback(window, mouse)
                if source is not None:
                    from source_video_reader import SourceVideoReader
                    source_preview = SourceVideoReader(args.video)
                    stack.callback(source_preview.close)
            render = None
            if not args.no_images:
                render = stack.enter_context(mujoco.Renderer(model, height=240, width=320))
                for camera in metadata['cameras']:
                    (output / 'images' / camera).mkdir(parents=True)
            wall_origin = last_tick = time.perf_counter()
            next_vision = 0.
            print(f'HUMAN TASK | {source_kind} | output {output}', flush=True)
            print('Right hand: move to cube, PINCH and HOLD; after lift move over GREEN goal, then OPEN.', flush=True)
            print('Open hand for 0.2 s, SPACE or click START. Request waits for tracking readiness. C recenter, M mirror, Q exit.', flush=True)
            index = 0
            while not flags['quit'] and task.stage not in task.terminal:
                tick = time.perf_counter()
                if viewer is not None and not viewer.is_running():
                    break
                if args.check:
                    clock = index * dt
                elif not flags['paused']:
                    clock += tick - last_tick
                last_tick = tick
                if tick - wall_origin > args.max_wait and task.stage == 'ready':
                    task.fail('start_wait_timeout', clock)
                    break
                if clock >= next_vision:
                    robot.observe(clock)
                    next_vision = clock + .2
                feedback = robot.feedback(clock)
                obs = None
                if args.scripted_check:
                    human = scripted_input(task, feedback, clock, index)
                else:
                    obs = source.at(min(cursor, source.duration - 1 / source.fps)) if source else stream.snapshot()
                    human = human_input(obs, time.perf_counter(), source_clock=source is not None)
                    if stream is not None and stream.error:
                        raise RuntimeError('camera_or_recognition_failed: ' + stream.error)
                command = task.update(human, feedback, clock,
                                      start=flags['start'] or args.check or source is not None,
                                      paused=flags['paused'], recenter=flags['recenter'])
                flags['start'] = flags['recenter'] = False
                robot.execute(command, clock, dt)
                if robot.error:
                    task.fail(robot.error, clock)
                robot.control.apply()
                physics_state = np.empty(state_size)
                mujoco.mj_getState(model, data, physics_state, state_spec)
                pose_raw = None if obs is None else obs.get('pose_raw')
                if source is not None and obs is not None:
                    pose_raw = source.blob['pose_world'][obs['seq']]
                row = dict(time_s=np.asarray(clock), sim_time_s=np.asarray(data.time),
                           capture_s=np.asarray(human.capture_s), source_seq=np.asarray(human.seq),
                           human_point=np.full(3, np.nan) if human.point is None else np.asarray(human.point).copy(),
                           pinch_ratio=np.asarray(np.nan if human.pinch_ratio is None else human.pinch_ratio),
                           human_valid=np.asarray(human.valid and human.gesture_valid), stage=np.asarray(task.stage),
                           human_pose_valid=np.asarray(human.valid), human_hand_valid=np.asarray(human.gesture_valid),
                           start_requested=np.asarray(task.start_requested), readiness_message=np.asarray(task.message),
                           preview_mirrored=np.asarray(flags['mirror']),
                           command_stage=np.asarray(command.stage), control_owner=np.asarray(command.owner),
                           paused=np.asarray(flags['paused']), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                           physics_state=physics_state,
                           ctrl=data.ctrl.copy(), action_target=robot.control.action.copy(), tcp=feedback.tcp.copy(),
                           visual_center=np.full(3, np.nan) if feedback.visual is None else feedback.visual['center'].copy(),
                           visual_time_s=np.asarray(np.nan if feedback.visual is None else feedback.visual['time_s']),
                           video_cursor_s=np.asarray(cursor),
                           human_pose=np.full((33, 4), np.nan) if pose_raw is None else pose_raw.copy(),
                           human_hand=np.full((21, 3), np.nan) if obs is None or obs['arms']['right'].get('hand_raw') is None else obs['arms']['right']['hand_raw'].copy())
                if render is not None:
                    for camera in metadata['cameras']:
                        render.update_scene(data, camera=camera)
                        rgb = render.render()
                        if not cv2.imwrite(str(output / 'images' / camera / f'{index:06d}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                            raise OSError('Cannot save synchronized robot camera image')
                if args.save_human_video and obs is not None and human.seq != last_human_video_seq:
                    frame = obs['raw_image']
                    if video_writer is None:
                        video_writer = cv2.VideoWriter(str(output / 'human_camera.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                       20, (frame.shape[1], frame.shape[0]))
                        if not video_writer.isOpened():
                            raise RuntimeError('Human video writer failed')
                    video_writer.write(frame)
                    last_human_video_seq = human.seq
                    row['human_video_frame'] = np.asarray(metadata.get('human_video_frames', 0))
                    metadata['human_video_frames'] = metadata.get('human_video_frames', 0) + 1
                else:
                    row['human_video_frame'] = np.asarray(-1)
                robot.step(dt, evaluator=evaluator, stage=task.stage)
                row['evaluation_next'] = evaluator.sample(data, task.stage, 0.)
                rows.append(row)
                if source and not flags['paused'] and task.stage in ('ready', 'follow', 'transport'):
                    cursor += dt
                    if cursor >= source.duration:
                        task.fail('video_ended_before_task_complete', clock)
                if viewer is not None:
                    frame = np.full((540, 960, 3), 45, np.uint8) if obs is None or 'image' not in obs else cv2.resize(obs['image'], (960, 540))
                    if source_preview is not None and obs is not None:
                        frame = cv2.resize(source_preview.frame(obs['seq']), (960, 540))
                    if flags['mirror']:
                        frame = cv2.flip(frame, 1)
                    lines = [f'RIGHT ARM TASK: {task.stage.upper()} | {clock - (task.started or clock):.1f}s',
                             task.message, 'Move to cube -> PINCH and HOLD -> move to GREEN -> OPEN',
                             f'WRIST {"OK" if human.valid else "MISSING"} | FINGERS {"OK" if human.gesture_valid else "MISSING"}',
                             f'Preview MIRROR {"ON" if flags["mirror"] else "OFF"} | ratio {human.pinch_ratio}',
                             'SPACE/click START | C recenter | M mirror | Q cancel']
                    for j, line in enumerate(lines):
                        cv2.putText(frame, line, (12, 28 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
                        cv2.putText(frame, line, (12, 28 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                    button = 'START' if task.stage == 'ready' else 'RESUME' if flags['paused'] else 'PAUSE'
                    cv2.rectangle(frame, (12, 488), (235, 530), (45, 160, 45), -1)
                    cv2.putText(frame, button, (28, 518), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
                    cv2.imshow(window, frame)
                    code = cv2.waitKey(1) & 255
                    if code != 255:
                        key(code)
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        flags['quit'] = True
                    viewer.sync()
                if index % 20 == 0:
                    print(f'TASK {task.stage} | source {human.seq} | {task.message}', flush=True)
                index += 1
                if not args.check:
                    time.sleep(max(0., dt - (time.perf_counter() - tick)))
            if task.stage not in task.terminal:
                task.cancel(clock)
    except KeyboardInterrupt:
        task.cancel(clock)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        task.fail(failure, clock)
    finally:
        if stream is not None:
            stream.close()
        if robot is not None:
            robot.close()
        if video_writer is not None:
            video_writer.release()
        if not args.check:
            cv2.destroyAllWindows()
        if rows:
            np.savez_compressed(output / 'episode.npz', **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
        evaluation = evaluator.report(task) if evaluator is not None else dict(physical_task_success=False)
        success = evaluation['physical_task_success']
        report = dict(verdict='PASS' if success else 'CANCELLED' if task.stage == 'cancelled' else 'FAIL',
                      controller_stage=task.stage, reason=task.reason, error=failure,
                      input_source=source_kind, synthetic_input=bool(args.scripted_check),
                      human_validation_performed=bool(not args.scripted_check and any(bool(r['human_valid']) for r in rows)),
                      human_task_success=bool(success and not args.scripted_check),
                      scope='shared human intent + B visual docking in simulation; no trained policy',
                      elapsed_task_s=None if task.started is None else clock - task.started,
                      task_clock=metadata['clocks'],
                      meets_15s_limit=bool(success and task.started is not None and clock - task.started <= 15.),
                      frames=len(rows), evaluation=evaluation, events=task.events,
                      grasp_plans=[] if robot is None else robot.plan_log,
                      placement_plans=[] if robot is None else robot.placement_log)
        if task.stage == 'success' and not success:
            report['reason'] = 'independent_physics_evaluation_failed'
        metadata.update(frames=len(rows), finished=True, result='result.json', events='events.json',
                        human_validation_performed=report['human_validation_performed'],
                        recording_present=bool(rows), task_success=bool(success))
        (output / 'events.json').write_text(json.dumps(task.events, indent=2), encoding='utf-8')
        (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        from validate_human_task import validate
        quality = validate(output)
        (output / 'quality_report.json').write_text(json.dumps(quality, indent=2), encoding='utf-8')
        report['recording_valid'] = quality['recording_valid']
        if not quality['recording_valid'] and report['verdict'] == 'PASS':
            report.update(verdict='FAIL', reason='recording_integrity_failed')
            metadata['task_success'] = False
            (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['verdict'] in ('PASS', 'CANCELLED') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--pose-model', choices=('lite', 'full', 'heavy'), default='lite')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--check', action='store_true', help='No GUI; requires measured video or explicit fixture')
    parser.add_argument('--scripted-check', action='store_true', help='Synthetic software input, NEVER a human validation')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=15.)
    parser.add_argument('--max-wait', type=float, default=120.)
    parser.add_argument('--gain', type=float, default=.7)
    parser.add_argument('--cube-test-pose', type=float, nargs=3, default=[.43, 0., 30.],
                        metavar=('X', 'Y', 'YAW_DEG'), help='Initial fixture only; never a perception input')
    parser.add_argument('--close-ratio', type=float, default=.35)
    parser.add_argument('--open-ratio', type=float, default=.8)
    parser.add_argument('--no-images', action='store_true')
    parser.add_argument('--save-human-video', action='store_true', help='Opt in to LOCAL raw human video')
    parser.add_argument('--mirror-preview', action=argparse.BooleanOptionalAction, default=True,
                        help='Mirror preview only; detection, anatomical identity and raw recordings stay unchanged. M toggles.')
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
