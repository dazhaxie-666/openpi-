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
import textwrap

import cv2
import mujoco
import numpy as np

from full_video_source import FullVideoSource, load_full_landmarks
from human_grasp_task import DirectHumanTask, HumanGraspTask, HumanInput, TaskConfig, TaskCommand, normalized_pinch
from human_task_robot import HumanTaskRobot, TaskEvaluator
from member_b_scene import ROOT, TARGET_ZONE, build_scene
from handeye_grasp_red_cube import finger_joint_indices
from task_workspace import TABLE_TOP, CONFIG
from demo_grasp_viewer import place_cube_for_blind_test


def pose2arm_reference_command(result, finger_targets):
    """Task intent may start/score, but never quantizes A's measured aperture."""
    if result is None or (not result['joints'] and not any(result.get('finger_valid',{}).values())):
        return TaskCommand('hold',finger=finger_targets['right'])
    fingers=dict(result['fingers'])
    return TaskCommand('pose2arm_reference',finger=fingers.get('right',finger_targets['right']),
                       owner='pose2arm_original',joint_targets=result['joints'],finger_targets=fingers,
                       reference_capture_s=result['capture_ns']/1e9)


def human_input(obs, now_s, source_clock=False, position_only=False):
    if obs is None:
        return HumanInput(-1, None, None, False)
    arm = obs['arms']['right']
    age = now_s - obs['read_end'] / 1e9
    ratio = normalized_pinch(arm.get('hand_raw')) if arm['hand_valid'] else None
    # This task translates a constant-attitude gripper. It needs a reliable
    # torso-relative wrist, not a visible palm or an anatomical arm rotation.
    point = arm.get('position_point', arm['point']) if position_only else arm['point']
    valid = (point is not None and np.asarray(point).shape == (3,)
             and np.isfinite(point).all() and (source_clock or 0 <= age <= .35))
    return HumanInput(int(obs['seq']), point, ratio, bool(valid), obs['read_end'] / 1e9)


def tracking_diagnostic(obs, human, now_s, source_clock=False):
    """Expose why the input was held, without guessing missing coordinates."""
    age = np.nan if obs is None else now_s - obs['read_end'] / 1e9
    pose = None if obs is None else obs.get('pose_filtered')
    if pose is None and obs is not None:
        pose = obs.get('pose_raw')
    pose = None if pose is None else np.asarray(pose)
    wrist = float(pose[16, 3]) if pose is not None and pose.shape == (33, 4) else np.nan
    if obs is None:
        reason = 'WAITING_FOR_CAMERA'
    elif not source_clock and not 0 <= age <= .35:
        reason = 'STALE_CAMERA_FRAME'
    elif human.valid:
        reason = 'WRIST_OK'
    elif pose is None or pose.shape != (33, 4) or not np.isfinite(pose[[11,12,16,23,24]]).all():
        reason = 'BODY_NOT_DETECTED'
    elif np.min(pose[[11,12],3]) < .45:
        reason = 'SHOULDERS_LOW_CONFIDENCE'
    elif wrist < .45:
        reason = 'WRIST_LOW_CONFIDENCE'
    else:
        reason = 'BODY_FRAME_UNAVAILABLE'
    return reason, age, wrist


def check_simulation_clock(actual, expected):
    """A viewer reset invalidates the episode and its calibrated robot state."""
    if abs(float(actual) - float(expected)) > 1e-7:
        raise RuntimeError(f'external_simulation_reset: expected time {expected:.6f}, got {actual:.6f}; restart this run')


def label_control_hand(frame, obs, mirrored, active_sides=('right',)):
    pose = None if obs is None else obs.get('pose_norm')
    if pose is None:
        return
    height, width = frame.shape[:2]
    for index, name, color in ((15, 'left', (255, 180, 0)), (16, 'right', (0, 255, 255))):
        point = np.asarray(pose[index])
        if not np.isfinite(point).all() or point[3] < .3 or np.any(point[:2] < 0) or np.any(point[:2] > 1):
            continue
        x = (1 - point[0]) if mirrored else point[0]
        at = (int(np.clip(x * width, 5, max(5, width-240))), int(np.clip(point[1] * height + 22, 20, height-12)))
        label='YOUR '+name.upper()+(' (CONTROL)' if name in active_sides else ' (idle)')
        cv2.putText(frame, label, at, cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
        cv2.putText(frame, label, at, cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)


def scripted_input(task, feedback, clock, index):
    """Closed-loop SOFTWARE FIXTURE. Never a human-demonstration success claim."""
    if getattr(task, 'control_mode', '') == 'direct':
        # Deliberate empty closure, reopening and motion reversal. This tests
        # continuous retry, not an automatically aligned successful grasp.
        phase = clock % 10.
        point = np.zeros(3)
        if 4. <= phase < 7.:
            point = np.array([.10, .06, -.025]) * min(1., (phase - 4.) / 2.)
        elif 7. <= phase < 9.:
            point = np.array([.10, .06, -.025]) * max(0., 1 - (phase - 7.) / 2.)
        return HumanInput(index, point, .15 if 1. <= phase < 3. else 1.2, True, clock)
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


def show_task_result(report, model, data):
    """Inspect a saved terminal result. No camera acquisition or physics steps.

    A failed safety/IK check must not look like a crashed application. This
    viewer deliberately freezes the final pose rather than resuming control.
    """
    from mujoco import viewer as mjviewer
    window = 'HUMAN TASK RESULT | Q or close to exit'
    stopped = dict(quit=False)

    def key(code):
        if code in (81, 113, 27):
            stopped['quit'] = True

    reason = report.get('reason') or report.get('error') or 'Task completed'
    explanation = reason
    if reason.startswith('task_motion_collision_rejected:'):
        explanation = 'Safety stop: proposed motion intersects the estimated scene'
    elif reason.startswith('cartesian_target_unreachable:'):
        explanation = 'Requested position/pose cannot be reached accurately'
    elif reason == 'human_input_lost':
        explanation = 'Right wrist tracking was lost for too long'
    elif reason == 'human_hand_lost_during_assist':
        explanation = 'Finger tracking was lost during assisted motion'
    elif reason == 'task_timeout':
        explanation = 'Task exceeded its configured time limit'
    elif reason == 'start_wait_timeout':
        explanation = 'Wait for task start exceeded its configured time limit'
    elif reason.startswith('external_simulation_reset:'):
        explanation = 'Viewer reset detected. Exit and restart to restore the robot work pose.'
    lines = [f'TASK RESULT: {report["verdict"]}', explanation,
             'Final simulation pose is FROZEN. No further robot commands.', '']
    lines += textwrap.wrap(reason, width=100)[:7]
    lines += ['', 'Recording and result.json have already been saved.',
              'Q / ESC / close window: exit. Start a new run to retry.']
    canvas = np.full((540, 960, 3), 35, np.uint8)
    for index, line in enumerate(lines):
        color = (80, 220, 80) if report['verdict'] == 'PASS' else (70, 150, 255)
        cv2.putText(canvas, line, (15, 30 + 29 * index), cv2.FONT_HERSHEY_SIMPLEX, .52,
                    color if index < 2 else (240, 240, 240), 1)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    try:
        with mjviewer.launch_passive(model, data, key_callback=key) as view:
            view.cam.lookat[:] = [.43, 0, .46]
            view.cam.distance, view.cam.azimuth, view.cam.elevation = 1.2, 150, -30
            while view.is_running() and not stopped['quit']:
                cv2.imshow(window, canvas)
                key(cv2.waitKey(30) & 255)
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                view.sync()
                time.sleep(.02)
    finally:
        cv2.destroyWindow(window)


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
    if args.check_steps < 1:
        raise ValueError('--check-steps must be positive')
    cv2.setNumThreads(1)
    direct = args.control == 'direct'
    use_a = direct and args.mapping == 'pose2arm'
    active_sides = ('left','right') if use_a and args.arms=='both' else ('right',)
    timeout = args.timeout if args.timeout is not None else (0. if direct else 15.)
    config = TaskConfig(timeout_s=timeout, gain=args.gain,
                        goal_xy=np.array(TARGET_ZONE['center_xy_m']), goal_radius_m=TARGET_ZONE['radius_m'],
                        close_ratio=args.close_ratio, open_ratio=args.open_ratio,
                        lower=np.array([.32, -.18, TABLE_TOP + CONFIG['cube_half_size_m']]))
    task = DirectHumanTask(config) if direct else HumanGraspTask(config)
    output = args.output or ROOT / 'outputs' / 'human_task' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_kind = 'synthetic_software_fixture' if args.scripted_check else 'measured_video' if args.video else 'live_usb'
    metadata = dict(schema='openarm_human_task_v1', source=source_kind, simulation_only=True,
                    human_validation_performed=False, training_performed=False,
                    synthetic_input=bool(args.scripted_check), side='right',
                    human_side='right', robot_side='right', pose_model=args.pose_model, camera_device=args.camera,
                    motion_mapping='pose2arm_original_online' if use_a else 'relative_wrist',
                    controlled_arms=list(active_sides), a_reference_order='left-first',
                    a_reference_execution='B dynamic actuator targets; no second Cartesian IK' if use_a else None,
                    hand_mapping='; '.join('human '+s+' -> robot '+s for s in active_sides)+'; preview mirror only',
                    direct_wrist_policy='original A shoulder-elbow-wrist pose mapping' if use_a else 'fixed when reachable; local position priority otherwise',
                    goal=TARGET_ZONE, timeout_s=timeout, control_hz=20, control_mode=args.control,
                    timeout_policy='advisory_only; Q ends continuous practice' if direct else 'terminal',
                    clocks='active wall time' if not args.check else 'simulation control ticks (offline test)',
                    startup='B pad IK fixed reset posture before episode; all episode execution dynamic',
                    source_timing=('video continuous' if direct else 'video pauses during assisted phases') if args.video else source_kind,
                    action_layout='left joint7+finger2,right joint7+finger2; absolute radians/metres',
                    alignment='images and qpos/action_target sampled before the same physical step',
                    human_video_saved=bool(args.save_human_video),
                    preview_mirrored=bool(args.mirror_preview),
                    result_window_enabled=not args.check and not args.close_on_finish,
                    human_validity='human_valid = pose AND gesture; separate human_pose_valid and human_hand_valid recorded',
                    cameras=[] if args.no_images else ['chest_d435i_preview', 'v22_right_d405_rgb'],
                    source_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                        'human_grasp_task.py', 'human_task_robot.py', 'run_human_task.py',
                        'member_b_actuators.py', 'b_scene_fusion.py', 'human_teleop_core.py',
                        'live_bimanual_stream.py', 'full_video_source.py', 'pose2arm_task_mapping.py',
                        'live_reference_conditioning.py','live_observation_quality.py',
                        'motion_observation.py','human_wrist_control.py','human_pinch_control.py')})
    if use_a:
        metadata['a_source_hashes']={name:hashlib.sha256((ROOT.parent/'pose2arm-main'/name).read_bytes()).hexdigest()
                                    for name in ('openarm_live.py','retarget.py','ik_solver.py','smoothing.py','openarm_common.py')}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    rows = []
    model = data = robot = evaluator = stream = source = None
    a_mapper = a_worker = None
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
        if use_a:
            from a_reference_adapter import verify_kinematics
            from pose2arm_task_mapping import Pose2ArmTaskMapping, LatestPose2ArmWorker, observed_sides
            metadata['a_b_kinematics_error_m']=verify_kinematics(model,robot.control.specs)
            robot.reference_sides=active_sides
            if args.check:
                a_mapper=Pose2ArmTaskMapping(sides=active_sides,model=model,torso_mode=args.torso_mode,
                    wrist_mode=args.wrist_mode,gripper_mode=args.gripper_mode,calibration_path=args.calibration,source_dt=True)
            else:
                a_worker=LatestPose2ArmWorker(sides=active_sides,model=model,torso_mode=args.torso_mode,
                    wrist_mode=args.wrist_mode,gripper_mode=args.gripper_mode,calibration_path=args.calibration,source_dt=True)
        mujoco.mj_saveModel(model, str(output / 'scene.mjb'))
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state_size = mujoco.mj_stateSize(model, state_spec)
        metadata.update(mujoco_version=mujoco.__version__, model_file='scene.mjb',
                        model_sha256=hashlib.sha256((output / 'scene.mjb').read_bytes()).hexdigest(),
                        physics_state_spec=int(state_spec), physics_state_size=state_size,
                        evaluation_hz=1 / float(model.opt.timestep),
                        gain=config.gain, deadband_m=config.deadband_m,
                        max_target_lead_m=config.max_target_lead_m, image_size=[320, 240],
                        close_ratio=config.close_ratio, open_ratio=config.open_ratio,
                        gripper_mapping='continuous normalized pinch -> 0..0.044 m' if use_a else 'gesture hysteresis',
                        gesture_threshold_role='start/intent only; never overrides continuous A aperture' if use_a else 'aperture switching')
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
            print('POSE2ARM ORIGINAL: 10 fresh frames calibrate each arm; original A joints drive B. C recalibrates.'
                  if use_a else 'DIRECT: wrist moves arm; PINCH closes, OPEN releases anywhere. Retry freely; Q saves and ends.'
                  if direct else 'ASSISTED: move to cube, PINCH and HOLD; after lift move over GREEN goal, then OPEN.', flush=True)
            print('Open hand for 0.2 s, SPACE or click START. Request waits for tracking readiness. C recenter, M mirror, Q exit.', flush=True)
            index = 0
            expected_sim_time = float(data.time)
            while not flags['quit'] and task.stage not in task.terminal:
                tick = time.perf_counter()
                if viewer is not None and not viewer.is_running():
                    flags['quit'] = True
                    break
                check_simulation_clock(data.time, expected_sim_time)
                if args.check:
                    clock = index * dt
                elif not flags['paused']:
                    clock += tick - last_tick
                last_tick = tick
                if not direct and tick - wall_origin > args.max_wait and task.stage == 'ready':
                    task.fail('start_wait_timeout', clock)
                    break
                if clock >= next_vision:
                    robot.observe(clock)
                    next_vision = clock + .2
                feedback = robot.feedback(clock)
                obs = None
                if args.scripted_check:
                    if use_a:
                        from pose2arm_task_mapping import synthetic_pose_observation
                        obs=synthetic_pose_observation(index,clock)
                        human=human_input(obs,time.perf_counter(),source_clock=True,position_only=True)
                    else:
                        human = scripted_input(task, feedback, clock, index)
                else:
                    obs = source.at(min(cursor, source.duration - 1 / source.fps)) if source else stream.snapshot()
                    if source is not None and obs is not None:
                        obs=dict(obs,pose_raw=source.blob['pose_world'][obs['seq']])
                    human = human_input(obs, time.perf_counter(), source_clock=source is not None, position_only=direct)
                    if stream is not None and stream.error:
                        raise RuntimeError('camera_or_recognition_failed: ' + stream.error)
                command = task.update(human, feedback, clock,
                                      start=flags['start'] or args.check or source is not None,
                                      paused=flags['paused'], recenter=flags['recenter'])
                a_result=None
                if use_a:
                    if flags['recenter']:
                        (a_worker or a_mapper).reset()
                    if not flags['paused']:
                        if a_worker:
                            a_worker.submit(obs)
                            if a_worker.error: raise RuntimeError('pose2arm_worker_failed: '+a_worker.error)
                            a_result=a_worker.snapshot()
                        else:
                            a_result=a_mapper.process(obs)
                    if a_result is not None:
                        a_now=(obs['read_end'] if obs is not None else 0) if args.check or source is not None else time.perf_counter_ns()
                        if not 0 <= (a_now-a_result['capture_ns'])/1e9 <= .35:
                            a_result=None
                        else:
                            visible=observed_sides(obs,active_sides)
                            a_result=dict(a_result,joints={s:q for s,q in a_result['joints'].items() if visible[s]},
                                          status={s:a_result['status'][s] if visible[s] else 'current observation missing: HOLD' for s in active_sides})
                    if task.stage=='direct' and not flags['paused']:
                        command=pose2arm_reference_command(a_result,robot.finger_targets)
                    if command.stage=='hold':
                        command.finger=robot.finger_targets['right']
                    if a_result is not None:
                        task.message=' | '.join(s.upper()+': '+a_result['status'][s] for s in active_sides)
                    elif not flags['paused']:
                        task.message='A mapping waiting for fresh body observations (no queued old motions)'
                tracking_status, observation_age, wrist_visibility = tracking_diagnostic(
                    obs, human, time.perf_counter(), source_clock=source is not None)
                if args.scripted_check:
                    tracking_status = 'SYNTHETIC_INPUT'
                elif direct and not use_a and not human.valid and not flags['paused']:
                    task.message = tracking_status + (f' ({wrist_visibility:.2f} / need 0.45)' if tracking_status == 'WRIST_LOW_CONFIDENCE' else '')
                flags['start'] = flags['recenter'] = False
                robot.execute(command, clock, dt)
                if direct and robot.warning:
                    task.reject_motion(robot.warning, human, feedback, clock)
                    if use_a:
                        task.message='B collision hold: '+','.join(robot.reference_blocked_sides)+'; other arm and fingers remain controlled'
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
                           motion_warning=np.asarray(robot.warning or ''),
                           direct_ik_mode=np.asarray(robot.direct_ik_mode),
                           a_reference_joint=np.concatenate([a_result['joints'].get(s,np.full(7,np.nan)) if a_result else np.full(7,np.nan) for s in ('left','right')]),
                           a_reference_valid=np.asarray([bool(a_result and s in a_result['joints']) for s in ('left','right')]),
                           a_nominal_joint=np.concatenate([a_result.get('a_nominal_joints',{}).get(s,np.full(7,np.nan)) if a_result else np.full(7,np.nan) for s in ('left','right')]),
                           accepted_pose=np.asarray(a_result['pose_accepted']) if a_result is not None and a_result.get('pose_accepted') is not None else np.full((33,4),np.nan),
                           module_state=np.asarray([[a_result.get('module_state',{}).get(s,{}).get(k,'LOST') if a_result else 'LOST' for k in ('torso','arm','palm','pinch','association')] for s in ('left','right')]),
                           a_calibration_scale=np.asarray([a_result['scales'].get(s) if a_result and a_result['scales'].get(s) is not None else np.nan for s in ('left','right')]),
                           a_solve_ms=np.asarray(a_result['solve_ms'] if a_result else np.nan),
                           a_source_seq=np.asarray(a_result['seq'] if a_result else -1),
                           a_capture_s=np.asarray(a_result['capture_ns']/1e9 if a_result else np.nan),
                           a_reference_pose=np.asarray(a_result['pose_raw']) if a_result is not None and a_result.get('pose_raw') is not None else np.full((33,4),np.nan),
                           a_applied_step_fraction=np.asarray([robot.reference_step_fraction[s] for s in ('left','right')]),
                           a_blocked_sides=np.asarray([s in robot.reference_blocked_sides for s in ('left','right')]),
                           control_work_ms=np.asarray(np.nan),
                           tracking_status=np.asarray(tracking_status),
                           observation_age_s=np.asarray(observation_age),
                           right_wrist_visibility=np.asarray(wrist_visibility),
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
                expected_sim_time = float(data.time)
                row['evaluation_next'] = evaluator.sample(data, task.stage, 0.)
                rows.append(row)
                if source and not flags['paused'] and task.stage in ('ready', 'follow', 'transport', 'direct'):
                    cursor += dt
                    if cursor >= source.duration:
                        if direct:
                            task.finish(clock, 'video_finished')
                        else:
                            task.fail('video_ended_before_task_complete', clock)
                if direct and args.check and index + 1 >= args.check_steps:
                    task.finish(clock, 'synthetic_check_step_budget_reached')
                if viewer is not None:
                    frame = np.full((540, 960, 3), 45, np.uint8) if obs is None or 'image' not in obs else cv2.resize(obs['image'], (960, 540))
                    if source_preview is not None and obs is not None:
                        frame = cv2.resize(source_preview.frame(obs['seq']), (960, 540))
                    if flags['mirror']:
                        frame = cv2.flip(frame, 1)
                    label_control_hand(frame, obs, flags['mirror'], active_sides)
                    elapsed = clock - task.started if task.started is not None else 0.
                    scored = evaluator.report(task)['physical_task_success']
                    lines = [f'{"POSE2ARM" if use_a else "RIGHT ARM"} {args.control.upper()}: {task.stage.upper()} | {elapsed:.1f}s',
                             task.message, ('YOUR LEFT -> ROBOT LEFT | YOUR RIGHT -> ROBOT RIGHT' if len(active_sides)==2 else
                                            'YOUR RIGHT HAND -> ROBOT RIGHT ARM | LEFT ARM IDLE' if direct
                                            else 'Move to cube -> PINCH and HOLD -> move to GREEN -> OPEN'),
                             f'{tracking_status} | FINGERS {"OK" if human.gesture_valid else "MISSING"} | age {observation_age:.2f}s',
                             f'MIRROR {"ON" if flags["mirror"] else "OFF"} | IK {robot.direct_ik_mode} | ratio {human.pinch_ratio}',
                             ('Task scored PASS; control stays active. Q saves.' if direct and scored else
                              'Time reminder; continue practice or Q to save.' if direct and timeout > 0 and elapsed > timeout else
                              'SPACE/click START | C recenter | M mirror | Q save/exit')]
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
                row['control_work_ms']=np.asarray((time.perf_counter()-tick)*1000)
                index += 1
                if not args.check:
                    time.sleep(max(0., dt - (time.perf_counter() - tick)))
            if task.stage not in task.terminal:
                if direct:
                    task.finish(clock)
                else:
                    task.cancel(clock)
    except KeyboardInterrupt:
        task.cancel(clock)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        task.fail(failure, clock)
    finally:
        if stream is not None:
            stream.close()
        if a_worker is not None:
            a_worker.close()
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
        verdict = ('PASS' if success else 'FAIL' if task.stage == 'failed' else 'INCOMPLETE') if direct else (
            'PASS' if success else 'CANCELLED' if task.stage == 'cancelled' else 'FAIL')
        report = dict(verdict=verdict, control_mode=args.control,
                      controller_stage=task.stage, reason=task.reason, error=failure,
                      input_source=source_kind, synthetic_input=bool(args.scripted_check),
                      human_validation_performed=bool(not args.scripted_check and any(bool(r['human_valid']) for r in rows)),
                      human_task_success=bool(success and not args.scripted_check),
                      scope=('original pose2arm calibrated shoulder-elbow-wrist mapping to B dynamic actuators; no autonomous docking or trained policy'
                             if use_a else 'direct human wrist + gripper control through B dynamic actuators; no autonomous docking or trained policy'
                             if direct else 'shared human intent + B visual docking in simulation; no trained policy'),
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
    if (not args.check and not args.close_on_finish and report['verdict'] != 'CANCELLED'
            and not flags['quit'] and model is not None and data is not None):
        print(f'TASK STOPPED: {report["reason"] or report["verdict"]}. Result saved in {output}. Q closes result windows.', flush=True)
        try:
            show_task_result(report, model, data)
        except Exception as error:
            print(f'Result viewer unavailable: {error}. Saved task result remains at {output}', flush=True)
    return 0 if report['verdict'] in ('PASS', 'CANCELLED', 'INCOMPLETE') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--pose-model', choices=('lite', 'full', 'heavy'), default='full')
    parser.add_argument('--torso-mode',choices=('auto','full_body','upper_body','original'),default='auto')
    parser.add_argument('--wrist-mode',choices=('full','twist','off'),default='full')
    parser.add_argument('--gripper-mode',choices=('continuous','binary','hold'),default='continuous')
    parser.add_argument('--calibration',type=Path,default=ROOT/'calibration'/'personal_motion'/'pinch.json')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--check', action='store_true', help='No GUI; requires measured video or explicit fixture')
    parser.add_argument('--scripted-check', action='store_true', help='Synthetic software input, NEVER a human validation')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--control', choices=('direct', 'assisted'), default='direct',
                        help='Direct wrist/gripper control (default) or previous visual docking task')
    parser.add_argument('--mapping',choices=('pose2arm','wrist'),default='pose2arm',
                        help='Direct mode: original A calibrated full-arm mapping (default), or previous wrist-only control')
    parser.add_argument('--arms',choices=('both','right'),default='both',help='Arms driven by original A mapping; task scoring remains right-arm')
    parser.add_argument('--timeout', type=float, default=None,
                        help='Direct: advisory only, default unlimited. Assisted: terminal, default 15 s; 0 disables')
    parser.add_argument('--check-steps', type=int, default=400, help='Bound headless direct fixture length (20 Hz)')
    parser.add_argument('--max-wait', type=float, default=120.)
    parser.add_argument('--gain', type=float, default=.7,help='Previous --mapping wrist only; original A uses its own arm-length calibration')
    parser.add_argument('--cube-test-pose', type=float, nargs=3, default=[.43, 0., 30.],
                        metavar=('X', 'Y', 'YAW_DEG'), help='Initial fixture only; never a perception input')
    parser.add_argument('--close-ratio', type=float, default=.35)
    parser.add_argument('--open-ratio', type=float, default=.8)
    parser.add_argument('--no-images', action='store_true')
    parser.add_argument('--save-human-video', action='store_true', help='Opt in to LOCAL raw human video')
    parser.add_argument('--mirror-preview', action=argparse.BooleanOptionalAction, default=True,
                        help='Mirror preview only; detection, anatomical identity and raw recordings stay unchanged. M toggles.')
    parser.add_argument('--close-on-finish', action='store_true',
                        help='Close GUI immediately on success/failure; default keeps a frozen result viewer open')
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
