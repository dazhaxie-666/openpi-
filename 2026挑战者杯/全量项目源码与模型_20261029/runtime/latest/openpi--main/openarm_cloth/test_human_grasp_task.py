"""Task intent, loss/recovery, release gates and separation from simulator truth."""
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from human_grasp_task import HumanGraspTask, HumanInput, Feedback, TaskConfig, normalized_pinch
from human_task_robot import HumanTaskRobot, TaskEvaluator, supported_tcp_height
from member_b_scene import build_scene
from run_human_task import human_input


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.task = HumanGraspTask()
        self.clock = 0.
        self.seq = 0
        self.point = np.zeros(3)
        self.feedback = Feedback(np.array([.43, 0., .49]),
                                 visual=dict(center=np.array([.43, 0., .429]), yaw=0., time_s=0.))

    def tick(self, ratio=1.2, valid=True, start=False, **kwargs):
        h = HumanInput(self.seq, self.point.copy(), ratio, valid, self.clock)
        result = self.task.update(h, self.feedback, self.clock, start=start, **kwargs)
        self.clock += .05
        self.seq += 1
        return result

    def stable(self, ratio=1.2, count=5, **kwargs):
        result = None
        for _ in range(count):
            result = self.tick(ratio, **kwargs)
        return result

    def start(self):
        self.stable(start=True)
        self.assertEqual(self.task.stage, 'follow')

    def transport(self):
        self.start()
        self.task.object_anchor = np.array([.43, 0., .429])
        self.task._enter('transport', self.clock, 'test fixture')
        self.task.holding = True
        self.task.rebase(HumanInput(0, self.point, .1, True), self.feedback)
        self.feedback.gripping = True

    def test_no_input_and_no_start_never_triggers_autonomous_grasp(self):
        self.stable(valid=False)
        self.stable()
        self.assertEqual(self.task.stage, 'ready')
        self.assertEqual(self.task.events, [])

    def test_start_requires_measured_open_hand_and_camera(self):
        self.stable(.1, start=True)
        self.assertEqual(self.task.stage, 'ready')
        self.feedback.visual = None
        self.stable(start=True)
        self.assertEqual(self.task.stage, 'ready')

    def test_start_before_tracking_ready_is_remembered_until_open_hand_is_observed(self):
        self.tick(valid=False, start=True)
        self.stable(start=False)
        self.assertEqual(self.task.stage, 'follow')

    def test_position_follow_does_not_require_visible_finger_landmarks(self):
        self.start()
        self.point += [.02, 0, 0]
        command = self.tick(ratio=None)
        self.assertEqual(command.stage, 'follow')
        self.assertGreater(command.target[0], self.feedback.tcp[0])
        self.assertEqual(command.finger, .044)

    def test_hand_loss_clears_old_close_intent_without_stopping_wrist_follow(self):
        self.start()
        self.task.gesture = 'closed'
        command = self.tick(ratio=None)
        self.assertEqual(command.stage, 'follow')
        self.assertIsNone(self.task.gesture)
        self.assertFalse(any(event['stage'] == 'plan' for event in self.task.events))

    def test_carrying_with_wrist_visible_but_fingers_missing_keeps_grip(self):
        self.transport()
        self.point += [.02, 0, 0]
        command = self.tick(ratio=None)
        self.assertEqual(command.stage, 'transport')
        self.assertEqual(command.finger, 0.)
        self.assertFalse(self.task.release_requested)

    def test_assisted_close_waits_for_finger_observations_without_dropping(self):
        self.start()
        self.task._enter('close', self.clock, 'fixture')
        self.feedback.gripping = True
        command = self.stable(ratio=None, count=10)
        self.assertEqual(command.stage, 'hold')
        self.assertEqual(command.finger, 0.)
        self.assertEqual(self.task.stage, 'close')

    def test_repeated_frame_cannot_satisfy_gesture_dwell(self):
        h = HumanInput(1, self.point, .1, True, 0.)
        for t in np.arange(20) * .05:
            self.task.update(h, self.feedback, float(t), start=True)
        self.assertIsNone(self.task.gesture)

    def test_out_of_order_frames_cannot_confirm_a_gesture(self):
        self.task.update(HumanInput(10, self.point, .1, True, 1.), self.feedback, 0.)
        for i in range(6):
            self.task.update(HumanInput(9, self.point, .1, True, 1. + i * .05), self.feedback, (i + 1) * .05)
        self.assertIsNone(self.task.gesture)

    def test_close_before_reaching_object_is_retained_without_far_autopilot(self):
        self.start()
        self.feedback.tcp = np.array([.35, -.15, .6])
        self.stable(.1)
        self.assertEqual(self.task.stage, 'follow')
        self.feedback.tcp = np.array([.43, 0., .49])
        self.assertEqual(self.tick(.1).stage, 'plan')
        self.assertEqual(self.task.events[-1]['reason'], 'human_close_near_camera_target')

    def test_motion_is_relative_and_speed_limited(self):
        self.start()
        before = self.task.target.copy()
        self.point += [1., 0., 0.]
        command = self.tick()
        self.assertEqual(command.owner, 'human_delta')
        self.assertLessEqual(np.linalg.norm(command.target - before), .0060001)

    def test_small_hand_tremor_is_ignored(self):
        self.start()
        before = self.task.target.copy()
        self.point += [.001, 0., 0.]
        np.testing.assert_allclose(self.tick().target, before)

    def test_target_does_not_accumulate_far_ahead_of_blocked_robot(self):
        self.start()
        self.point += [1., 0., 0.]
        for _ in range(40):
            command = self.tick()
            self.assertLessEqual(np.linalg.norm(command.target - self.feedback.tcp), .0400001)

    def test_missing_camera_holds_then_rebases_without_a_jump(self):
        self.start()
        visual = self.feedback.visual
        self.feedback.visual = None
        self.point += [.2, 0., 0.]
        self.assertEqual(self.tick().stage, 'hold')
        self.feedback.visual = visual
        np.testing.assert_allclose(self.tick().target, self.feedback.tcp)

    def test_reached_flag_from_previous_stage_cannot_advance_docking(self):
        self.start()
        self.stable(.1)
        self.feedback.plan_ready = True
        self.tick(.1)
        self.feedback.reached = True
        self.feedback.command_stage = 'follow'
        self.stable(.1)
        self.assertEqual(self.task.stage, 'transit')

    def test_open_cancels_alignment_without_closing(self):
        self.start()
        self.stable(.1)
        self.feedback.plan_ready = True
        self.tick(.1)
        command = self.stable(1.2)
        self.assertEqual(self.task.reason, 'human_cancelled_grasp')
        self.assertEqual(command.finger, .044)

    def test_carrying_input_loss_and_timeout_do_not_drop_payload(self):
        self.transport()
        command = self.tick(valid=False)
        self.assertEqual(command.stage, 'hold')
        self.assertEqual(command.finger, 0.)
        self.stable(valid=False, count=42)
        self.assertEqual(self.task.reason, 'human_input_lost')
        self.assertEqual(self.tick().finger, 0.)

    def test_recovery_rebases_relative_motion(self):
        self.transport()
        self.tick(valid=False)
        self.point += [1., 1., 1.]
        self.feedback.tcp = np.array([.43, -.05, .5])
        command = self.tick(.1)
        np.testing.assert_allclose(command.target, self.feedback.tcp)

    def test_open_far_from_goal_cannot_release(self):
        self.transport()
        command = self.stable(1.2)
        self.assertEqual(self.task.stage, 'transport')
        self.assertEqual(command.finger, 0.)
        self.assertFalse(self.task.release_requested)

    def test_open_at_goal_lowers_first_then_releases_and_retracts(self):
        self.transport()
        self.feedback.tcp = np.array([.43, -.10, .50])
        command = self.stable(1.2)
        self.assertEqual(self.task.stage, 'lower')
        self.assertEqual(command.finger, 0.)
        self.assertTrue(self.task.release_requested)
        self.feedback.command_stage = 'lower'
        self.feedback.reached = True
        self.stable(1.2)
        self.assertEqual(self.task.stage, 'release')
        self.feedback.opened = True
        self.stable(1.2, count=7)
        self.assertEqual(self.task.stage, 'retract')
        self.feedback.command_stage = 'retract'
        self.stable(1.2)
        self.assertEqual(self.task.stage, 'verify')
        self.feedback.visual['center'] = np.array([.43, -.10, .429])
        self.stable(1.2, count=12)
        self.assertEqual(self.task.stage, 'success')

    def test_timeout_during_carry_keeps_grip(self):
        self.transport()
        command = self.task.update(HumanInput(99, self.point, .1, True, 20.), self.feedback, 20.)
        self.assertEqual(self.task.reason, 'task_timeout')
        self.assertEqual(command.finger, 0.)

    def test_pause_does_not_count_as_stable_grip_or_release(self):
        self.transport()
        self.task.stable_since = self.clock
        self.tick(.1, paused=True)
        self.assertIsNone(self.task.stable_since)


class InputTests(unittest.TestCase):
    def test_tracked_wrist_survives_missing_hand_and_unused_arm_orientation(self):
        obs = dict(seq=1, read_end=1_000_000_000,
                   arms=dict(right=dict(point=np.zeros(3), pose=None, hand_raw=None, hand_valid=False)))
        human = human_input(obs, 1.1)
        self.assertTrue(human.valid)
        self.assertIsNone(human.pinch_ratio)

    def test_pinch_is_scale_and_translation_invariant(self):
        hand = np.zeros((21, 3))
        hand[5] = [.04, 0, 0]
        hand[17] = [-.04, 0, 0]
        hand[8] = [.016, 0, 0]
        self.assertAlmostEqual(normalized_pinch(hand), .2)
        self.assertAlmostEqual(normalized_pinch(hand * 2 + 10), .2)
        self.assertIsNone(normalized_pinch(np.zeros((21, 3))))

    def test_live_stale_or_future_frame_is_not_accepted(self):
        hand = np.zeros((21, 3));hand[5, 0] = .04;hand[17, 0] = -.04
        obs = dict(seq=1, read_end=1_000_000_000,
                   arms=dict(right=dict(point=np.zeros(3), pose={}, hand_raw=hand, hand_valid=True)))
        self.assertTrue(human_input(obs, 1.1).valid)
        self.assertFalse(human_input(obs, 2.).valid)
        self.assertFalse(human_input(obs, .5).valid)

    def test_invalid_task_config_is_rejected(self):
        for changes in (dict(timeout_s=-1), dict(gain=float('nan')), dict(close_ratio=.8, open_ratio=.3)):
            with self.assertRaises(ValueError):
                TaskConfig(**changes)


class OracleSeparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.initial = build_scene()

    def test_backend_query_uses_camera_and_robot_only(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.initial.qpos
        with patch('human_task_robot.BVisualObserver'):
            robot = HumanTaskRobot(self.model, data)
        robot.visual = dict(center=np.array([.43, 0., .429]), yaw=.3, time_s=0.)
        before = robot._query_state().qpos.copy()
        address = robot.object_address
        data.qpos[address:address + 3] = [1., 1., 1.]
        np.testing.assert_allclose(robot._query_state().qpos, before)
        robot.visual['center'][0] += .02
        self.assertAlmostEqual(robot._query_state().qpos[address] - before[address], .02)

    def test_controller_success_alone_cannot_pass_independent_evaluation(self):
        task = HumanGraspTask()
        task.stage = 'success'
        evaluation = TaskEvaluator(self.model, self.initial, task.config)
        for _ in range(20):
            evaluation.sample(self.initial, 'success', .05)
        self.assertFalse(evaluation.report(task)['physical_task_success'])

    def test_tilted_cube_corners_clear_support_without_relaxing_collisions(self):
        from itertools import product
        from scipy.spatial.transform import Rotation
        from demo_grasp_viewer import CUBE_HALF
        from task_workspace import TABLE_TOP
        rotation = Rotation.from_euler('xyz', [4., -3., 30.], degrees=True).as_matrix()
        offset = np.array([.002, -.001, -.003])
        height = supported_tcp_height(rotation, offset)
        corners = np.array(list(product((-CUBE_HALF, CUBE_HALF), repeat=3))) @ rotation.T
        corners += np.array([0, 0, height]) + offset
        self.assertGreaterEqual(np.min(corners[:, 2]), TABLE_TOP + .00049999)
        self.assertGreater(height, supported_tcp_height(np.eye(3), offset))


if __name__ == '__main__':
    unittest.main()
