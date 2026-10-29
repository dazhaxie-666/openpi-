"""Direct control never hands arm motion to a grasp planner; retries stay live."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import mujoco

from human_grasp_task import DirectHumanTask, Feedback, HumanInput, TaskConfig, TaskCommand
from human_task_robot import HumanTaskRobot, TaskEvaluator
from member_b_scene import build_scene
from human_teleop_core import human_wrist
from run_human_task import human_input, tracking_diagnostic, check_simulation_clock


class PositionInputTests(unittest.TestCase):
    def test_anatomical_left_hand_does_not_drive_the_right_task(self):
        obs = self.obs(self.pose())
        obs['arms']['right']['point'] = obs['arms']['right']['position_point'] = None
        obs['arms']['left'] = dict(point=np.ones(3), position_point=np.ones(3), hand_valid=False)
        self.assertFalse(human_input(obs, 1.1, position_only=True).valid)

    def test_viewer_reset_is_detected_before_accepting_further_commands(self):
        check_simulation_clock(24.8, 24.8)
        with self.assertRaisesRegex(RuntimeError, 'external_simulation_reset'):
            check_simulation_clock(0., 24.8)
    def pose(self):
        w = np.zeros((33, 4))
        w[:, 3] = 1.
        w[11, :3], w[12, :3] = [-.2, 0, 0], [.2, 0, 0]
        w[23, :3], w[24, :3] = [-.15, .5, 0], [.15, .5, 0]
        w[16, :3] = [.3, -.2, -.1]
        return w

    def obs(self, w):
        return dict(seq=1, read_end=1_000_000_000, pose_filtered=w,
                    arms=dict(right=dict(point=human_wrist(w),
                                         position_point=human_wrist(w, require_elbow=False),
                                         hand_valid=False)))

    def test_position_only_control_does_not_require_unused_elbow(self):
        w = self.pose();w[14, 3] = .1
        obs = self.obs(w)
        self.assertFalse(human_input(obs, 1.1).valid)
        self.assertTrue(human_input(obs, 1.1, position_only=True).valid)

    def test_low_confidence_wrist_is_still_held_and_reason_is_visible(self):
        w = self.pose();w[16, 3] = .1
        obs = self.obs(w)
        human = human_input(obs, 1.1, position_only=True)
        self.assertFalse(human.valid)
        reason, age, confidence = tracking_diagnostic(obs, human, 1.1)
        self.assertEqual(reason, 'WRIST_LOW_CONFIDENCE')
        self.assertAlmostEqual(confidence, .1)

    def test_stale_frames_are_distinguished_from_failed_detection(self):
        obs = self.obs(self.pose())
        human = human_input(obs, 2., position_only=True)
        self.assertFalse(human.valid)
        self.assertEqual(tracking_diagnostic(obs, human, 2.)[0], 'STALE_CAMERA_FRAME')

    def test_missing_fingers_do_not_mislabel_a_valid_wrist(self):
        obs = self.obs(self.pose())
        human = human_input(obs, 1.1, position_only=True)
        self.assertTrue(human.valid)
        self.assertFalse(human.gesture_valid)
        self.assertEqual(tracking_diagnostic(obs, human, 1.1)[0], 'WRIST_OK')


class DirectTaskTests(unittest.TestCase):
    def setUp(self):
        self.task = DirectHumanTask(TaskConfig())
        self.feedback = Feedback(np.array([.43, 0., .5]),
                                 visual=dict(center=np.array([.43, 0., .429]), yaw=0., time_s=0.))
        self.point = np.zeros(3)
        self.seq = 0
        self.clock = 0.

    def tick(self, ratio=1.2, valid=True, **kw):
        human = HumanInput(self.seq, self.point.copy(), ratio, valid, self.clock)
        result = self.task.update(human, self.feedback, self.clock, **kw)
        self.seq += 1
        self.clock += .05
        return result

    def stable(self, ratio=1.2, count=6, **kw):
        for _ in range(count):
            result = self.tick(ratio, **kw)
        return result

    def start(self):
        self.stable(start=True)
        self.assertEqual(self.task.stage, 'direct')

    def test_start_without_object_localization_can_wait_without_ending(self):
        self.feedback.visual = None
        self.stable(start=True)
        self.assertEqual(self.task.stage, 'direct')
        self.assertEqual(self.tick().stage, 'hold')

    def test_explicit_start_with_visible_wrist_does_not_require_fingers(self):
        command=self.tick(ratio=None,start=True)
        self.assertEqual(self.task.stage,'direct')
        self.assertEqual(command.stage,'hold')
        self.point[0]+=.02
        self.assertEqual(self.tick(ratio=None).stage,'direct')

    def test_start_request_waits_for_wrist_but_not_open_gesture(self):
        self.tick(ratio=None,valid=False,start=True)
        self.assertEqual(self.task.stage,'ready')
        self.tick(ratio=None)
        self.assertEqual(self.task.stage,'direct')

    def test_visible_wrist_without_start_stays_ready(self):
        self.tick(ratio=None)
        self.assertEqual(self.task.stage,'ready')

    def test_pinch_anywhere_closes_without_plan_or_automatic_lift(self):
        self.start()
        command = self.stable(.1, count=70)
        self.assertEqual(command.stage, 'direct')
        self.assertEqual(command.finger, 0.)
        self.assertEqual(command.owner, 'human_direct')
        np.testing.assert_allclose(command.target, self.feedback.tcp)
        self.assertEqual(self.task.stage, 'direct')

    def test_open_anywhere_releases_without_automatic_lowering(self):
        self.start()
        self.stable(.1)
        self.feedback.tcp[:2] = [.35, .15]
        command = self.stable(1.2)
        self.assertEqual(command.finger, .044)
        self.assertEqual(self.task.stage, 'direct')

    def test_empty_grasp_and_lost_grip_do_not_end_session(self):
        self.start()
        self.feedback.gripping = False
        self.stable(.1, count=400)
        self.assertEqual(self.task.stage, 'direct')
        self.stable(1.2)
        self.assertEqual(self.stable(.1).finger, 0.)
        self.assertGreaterEqual(sum(e['reason'] == 'human_direct_close' for e in self.task.events), 2)

    def test_tracking_loss_holds_grip_then_rebases_on_recovery(self):
        self.start()
        self.stable(.1)
        self.stable(.1, count=60, valid=False)
        self.assertEqual(self.task.stage, 'direct')
        self.point += 1
        command = self.tick(.1)
        np.testing.assert_allclose(command.target, self.feedback.tcp)
        self.assertEqual(command.finger, 0.)

    def test_short_wrist_dropout_preserves_relative_motion_instead_of_rebasing_every_frame(self):
        self.start()
        for _ in range(5):
            self.tick(valid=False)
            self.point[0] += .006
            command = self.tick()
        self.assertGreater(command.target[0], self.feedback.tcp[0] + .015)
        self.assertLessEqual(np.linalg.norm(command.target - self.feedback.tcp), .040001)

    def test_obstacle_rejection_discards_target_debt_and_allows_retreat(self):
        self.start()
        self.point[0] += .1
        self.tick()
        h = HumanInput(self.seq, self.point.copy(), 1.2, True, self.clock)
        self.task.reject_motion('task_motion_collision_rejected:direct_move:example', h,
                                self.feedback, self.clock)
        self.point[0] -= .02
        command = self.tick()
        self.assertLess(command.target[0], self.feedback.tcp[0])
        self.assertEqual(self.task.stage, 'direct')
        self.assertIsNone(self.task.reason)

    def test_finger_loss_preserves_last_command_and_position_follow(self):
        self.start()
        self.stable(.1)
        self.point[0] += .02
        command = self.tick(None)
        self.assertEqual(command.finger, 0.)
        self.assertGreater(command.target[0], self.feedback.tcp[0])

    def test_success_scoring_does_not_take_control_or_stop_motion(self):
        self.start()
        self.feedback.gripping = True
        self.stable(.1)
        self.stable(1.2)
        self.point[0] += .02
        self.assertEqual(self.tick().stage, 'direct')

    def test_real_program_failure_cannot_be_reported_as_success_after_earlier_score(self):
        evaluator = TaskEvaluator.__new__(TaskEvaluator)
        evaluator.max_lift = .06
        evaluator.best_lift = evaluator.best_placed = .6
        self.start()
        self.assertTrue(evaluator.report(self.task)['physical_task_success'])
        self.task.fail('broken_internal_state', self.clock)
        self.assertFalse(evaluator.report(self.task)['physical_task_success'])

    def test_hold_return_profile_requires_timed_hold_and_stable_source_release(self):
        evaluator=TaskEvaluator.__new__(TaskEvaluator)
        evaluator.max_lift=.06;evaluator.best_lift=3.5;evaluator.best_hold=3.1
        evaluator.best_placed=0.;evaluator.best_source_placed=.6
        evaluator.min_source_return_error=.01;evaluator.source_tolerance_m=.03
        self.start();self.task.collection_task_profile='hold';self.task.collection_hold_seconds=3.
        self.assertTrue(evaluator.report(self.task)['physical_task_success'])
        evaluator.best_hold=2.9
        self.assertFalse(evaluator.report(self.task)['physical_task_success'])


class DirectRobotTests(unittest.TestCase):
    def test_a_reference_blocked_right_arm_does_not_block_left_or_pinch(self):
        model,data=build_scene()
        with patch('human_task_robot.BVisualObserver'):
            robot=HumanTaskRobot(model,data)
        self.addCleanup(robot.close)
        robot.initialize();robot.reference_sides=('left','right')
        robot.visual=dict(center=np.array([.43,0,.429]),yaw=0.,time_s=0.)
        old={s:data.qpos[robot.control.specs[s].qpos].copy() for s in ('left','right')}
        def check(targets,fingers,allow_grasp=False):
            if np.linalg.norm(targets['right']-old['right'])>1e-8:
                raise RuntimeError('task_motion_collision_rejected:pose2arm_reference:right')
        command=TaskCommand('pose2arm_reference',finger=0.,owner='pose2arm_original',
                            joint_targets={s:q+.01 for s,q in old.items()},
                            finger_targets=dict(left=.044,right=0.))
        with patch.object(robot,'_check_reference_path',side_effect=check):robot.execute(command,0.)
        np.testing.assert_allclose(robot.joints['right'],old['right'])
        self.assertGreater(np.linalg.norm(robot.joints['left']-old['left']),.005)
        self.assertLess(robot.finger_targets['right'],.044)
        self.assertIsNone(robot.error)

    def test_position_priority_can_retreat_from_latest_fixed_wrist_lock(self):
        model, data = build_scene()
        with patch('human_task_robot.BVisualObserver'):
            robot = HumanTaskRobot(model, data)
        self.addCleanup(robot.close)
        robot.initialize()
        # Rounded robot encoders from the 18:57 session, before external reset.
        q = np.array([1.449, -.167, -.880, .828, .957, -.518, -1.225])
        data.qpos[robot.spec.qpos] = q
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        robot.joints['right'] = q.copy()
        robot.visual = dict(center=np.array([.43, 0., .429]), yaw=np.pi/6, time_s=0.)
        tcp = robot.feedback(0.).tcp.copy()
        robot.execute(TaskCommand('direct', tcp + [0., -.003, -.003], .044, 'human_direct'), 0.)
        self.assertIsNone(robot.warning)
        self.assertIsNone(robot.error)
        self.assertIsNotNone(robot.goal_xyz)

    def test_recorded_freeze_postures_accept_a_two_mm_request_without_relaxing_ik_limits(self):
        model, data = build_scene()
        with patch('human_task_robot.BVisualObserver'):
            robot = HumanTaskRobot(model, data)
        self.addCleanup(robot.close)
        robot.initialize()
        # Robot encoder postures from two real sessions, not human landmarks.
        postures = [
            [1.56695348025352, .3412137605612123, -1.1918963700745298, 1.1942653872882656,
             1.2473127390351997, -.6961069161280198, -1.4983192962612066],
            [1.276511921784768, .5332169087323914, -1.075989714039169, 1.29014592291191,
             1.1428102389876915, -.7124457213077878, -1.4344984979561843],
            [1.237643592540608, .24083094136961153, -1.0696301581881091, .7600335832309685,
             1.1731346637356288, -.25564636549163877, -1.072907574282581],
        ]
        for q in postures:
            with self.subTest(q=q):
                data.qpos[robot.spec.qpos] = q
                data.qpos[robot.fingers[0]] = .044
                data.qvel[:] = 0.
                mujoco.mj_forward(model, data)
                robot.joints['right'] = np.array(q)
                robot.visual = dict(center=np.array([.43, 0., .429]), yaw=np.pi/6, time_s=0.)
                target = robot.feedback(0.).tcp + [0., 0., .002]
                robot.execute(TaskCommand('direct', target, .044, 'human_direct'), 0.)
                self.assertIsNone(robot.warning)
                self.assertIsNone(robot.error)

    def test_existing_contact_can_retreat_but_cannot_deepen(self):
        robot = HumanTaskRobot.__new__(HumanTaskRobot)
        robot.data = Mock(qpos=np.array([0.]))
        robot.spec = Mock(qpos=np.array([0]))
        robot.fingers = (np.array([0]), np.array([0]))
        robot.object_geom = 1
        robot.attached = False
        robot.model = Mock()
        robot.model.geom.return_value.name = 'fixture_geom'
        def contacts(q, *_):
            return {(1, 2): max(0., .002 + float(q[0]))}
        robot._contacts = contacts
        robot._check_segment(np.array([-.001]), 0., 'direct_move')
        with self.assertRaisesRegex(RuntimeError, 'task_motion_collision_rejected:direct_move'):
            robot._check_segment(np.array([.001]), 0., 'direct_move')
        # Repeated small commands must not creep further into an old contact.
        with self.assertRaisesRegex(RuntimeError, 'task_motion_collision_rejected:direct_move'):
            robot._check_segment(np.array([.0001]), 0., 'direct_move')

    def test_real_b_ik_executes_only_direct_targets_and_recovers_from_unreachable(self):
        model, data = build_scene()
        with patch('human_task_robot.BVisualObserver'):
            robot = HumanTaskRobot(model, data)
        self.addCleanup(robot.close)
        robot.initialize()
        robot.visual = dict(center=np.array([.43, 0., .429]), yaw=0., time_s=0.)
        tcp = robot.feedback(0.).tcp.copy()
        with patch.object(robot, '_make_plan', side_effect=AssertionError('No autonomous plan allowed')):
            robot.execute(TaskCommand('direct', np.array([2., 2., 2.]), .044, 'human_direct'), 0.)
            self.assertIsNone(robot.error)
            self.assertIn('cartesian_target_unreachable', robot.warning)
            robot.execute(TaskCommand('direct', tcp + [0., 0., .005], .044, 'human_direct'), .05)
            self.assertIsNone(robot.error)
            self.assertIsNone(robot.warning)
            robot.step()
        self.assertEqual(robot.command_stage, 'direct')
        self.assertEqual(robot.plan_log, [])
        self.assertFalse(robot.attached)
        self.assertTrue(np.isfinite(data.qpos).all())

    def test_rejected_direct_command_is_recoverable_and_next_command_can_run(self):
        robot = HumanTaskRobot.__new__(HumanTaskRobot)
        robot.side = 'right'
        robot.spec = Mock(qpos=np.array([0]))
        robot.data = Mock(qpos=np.array([.12]))
        robot.joints = dict(right=np.array([.2]))
        robot.finger_targets = dict(right=0.)
        robot.control = Mock()
        robot.error = None
        robot.warning = None
        robot.goal_q = robot.goal_xyz = np.ones(1)
        with patch.object(robot, '_execute', side_effect=[RuntimeError('cartesian_target_unreachable:test'), None]), \
             patch.object(robot,'_execute_gripper_reference'):
            robot.execute(TaskCommand('direct', np.zeros(3), 0., 'human_direct'), 0.)
            self.assertIsNone(robot.error)
            self.assertIn('cartesian_target_unreachable', robot.warning)
            np.testing.assert_allclose(robot.joints['right'], [.12])
            robot.execute(TaskCommand('direct', np.zeros(3), 0., 'human_direct'), .05)
            self.assertIsNone(robot.warning)

    def test_unexpected_errors_are_not_hidden_as_retry(self):
        robot = HumanTaskRobot.__new__(HumanTaskRobot)
        robot.side = 'right'
        robot.spec = Mock(qpos=np.array([0]))
        robot.data = Mock(qpos=np.array([.12]))
        robot.joints = dict(right=np.array([.2]))
        robot.finger_targets = dict(right=0.)
        robot.control = Mock()
        with patch.object(robot, '_execute', side_effect=RuntimeError('broken_internal_state')):
            robot.execute(TaskCommand('direct', np.zeros(3), 0., 'human_direct'), 0.)
        self.assertEqual(robot.error, 'broken_internal_state')


if __name__ == '__main__':
    unittest.main()
