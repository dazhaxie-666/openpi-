"""Direct-mode gripper regressions at the actual task/robot boundary."""
import unittest
from unittest.mock import patch
import numpy as np
from human_grasp_task import TaskCommand
from human_task_robot import HumanTaskRobot
from member_b_scene import build_scene
from human_pinch_control import TaskHandControl
from live_bimanual_stream import hand_landmarker_thresholds
from run_human_task import direct_hand_command
from run_human_task import save_task_calibration
from pathlib import Path


class IndependentFingerExecutionTests(unittest.TestCase):
    def setUp(self):
        self.model,self.data=build_scene()
        self.robot=HumanTaskRobot(self.model,self.data)
        self.addCleanup(self.robot.close)
        self.robot._direct_payload=lambda *a,**kw:None
        self.robot.fresh_visual=lambda *a:dict(time_s=0.)
        self.robot._check_segment=lambda *a,**kw:None

    def test_fingers_can_move_without_a_body_or_cartesian_arm_target(self):
        self.robot.execute(TaskCommand('gripper_reference',finger=.015,owner='human_pinch'),0.)
        self.assertIsNone(self.robot.error)
        self.assertLess(self.robot.finger_targets['right'],.044)

    def test_rejected_arm_target_does_not_discard_valid_finger_command(self):
        with patch.object(self.robot,'_execute_direct',side_effect=RuntimeError('cartesian_target_unreachable:fixture')):
            self.robot.execute(TaskCommand('direct',target=np.ones(3),finger=.015),0.)
        self.assertIsNone(self.robot.error)
        self.assertLess(self.robot.finger_targets['right'],.044)

    def test_gripper_collision_rejection_is_recoverable_and_keeps_aperture(self):
        self.robot._check_segment=lambda *a:(_ for _ in ()).throw(RuntimeError('task_motion_collision_rejected:fixture'))
        self.robot.execute(TaskCommand('gripper_reference',finger=0.),0.)
        self.assertIsNone(self.robot.error);self.assertIsNotNone(self.robot.warning)
        self.assertEqual(self.robot.finger_targets['right'],.044)

    def test_unknown_gripper_error_after_arm_rejection_is_not_hidden(self):
        with patch.object(self.robot,'_execute_direct',side_effect=RuntimeError('cartesian_target_unreachable:fixture')), \
             patch.object(self.robot,'_execute_gripper_reference',side_effect=RuntimeError('broken_gripper_internal_state')):
            self.robot.execute(TaskCommand('direct',np.zeros(3),.015),0.)
        self.assertEqual(self.robot.error,'broken_gripper_internal_state')

    def test_accepted_continuous_pinch_reaches_actual_finger_actuators(self):
        hand=TaskHandControl();held=TaskCommand('hold',finger=.044)
        before=self.data.qpos[self.robot.fingers[0]].copy()
        for i in range(24):
            result=hand.process_features(i,i*.05,dict(r3=.65,r2=None))
            command=direct_hand_command(held,result,'direct',self.robot.finger_targets['right'])
            self.robot.execute(command,i*.05);self.robot.step(.05)
        self.assertIsNone(self.robot.error)
        self.assertGreater(self.robot.finger_targets['right'],.005)
        self.assertLess(self.robot.finger_targets['right'],.035)
        self.assertLess(float(np.mean(self.data.qpos[self.robot.fingers[0]])),float(np.mean(before))-.003)


class HandChannelTests(unittest.TestCase):
    def setUp(self):
        # Synthetic endpoint fixtures must not absorb this user's saved trials.
        loader=patch('human_pinch_control.load_personal_calibration',return_value={})
        loader.start();self.addCleanup(loader.stop)

    def result(self,ratio=.65):
        channel=TaskHandControl()
        for i in range(5):result=channel.process_features(i,i*.05,dict(r3=ratio,r2=None))
        return channel,result

    def test_wrist_command_keeps_a_measured_middle_aperture(self):
        _,hand=self.result()
        original=TaskCommand('direct',np.zeros(3),.044)
        command=direct_hand_command(original,hand,'direct',.044)
        self.assertGreater(command.finger,.005);self.assertLess(command.finger,.035)

    def test_missing_body_does_not_block_an_associated_hand(self):
        channel=TaskHandControl()
        h=np.zeros((21,3));h[5]=[-.04,.03,0];h[17]=[.04,.03,0];h[9]=[0,.06,0]
        h[4]=[-.01,.07,0];h[8]=[.02,.07,0]
        for i in range(5):
            obs=dict(seq=i,read_end=int((1+i*.05)*1e9),pose_raw=None,arms=dict(right=dict(hand_valid=True,hand_raw=h)))
            result=channel.process(obs,1+i*.05)
        self.assertTrue(result['valid'])
        self.assertEqual(direct_hand_command(TaskCommand('hold'),result,'direct',.044).stage,'gripper_reference')

    def test_stale_hand_and_pause_preserve_current_aperture(self):
        channel,hand=self.result()
        stale=channel.process(dict(seq=20,read_end=1_000_000_000,arms={}),2.)
        self.assertFalse(stale['valid']);self.assertEqual(stale['reason'],'STALE_HAND_SOURCE')
        for result,paused in ((stale,False),(hand,True)):
            command=direct_hand_command(TaskCommand('direct',np.zeros(3),.044),result,'direct',.012,paused)
            self.assertAlmostEqual(command.finger,.012)

    def test_duplicates_do_not_complete_confirmation(self):
        channel=TaskHandControl()
        for _ in range(8):result=channel.process_features(1,1.,dict(r3=.1,r2=None))
        self.assertFalse(result['valid']);self.assertEqual(result['aperture'],.044)

    def test_terminal_task_does_not_release_a_partial_aperture(self):
        _,hand=self.result()
        command=direct_hand_command(TaskCommand('hold',finger=.044),hand,'failed',.012)
        self.assertEqual(command.finger,.012)

    def test_personal_endpoints_are_used_for_continuous_opening(self):
        from human_pinch_control import PinchCalibration
        channel=TaskHandControl();calibration=channel.calibration
        for _ in range(10):
            calibration.add_trial('closed',[dict(r3=.2,r2=None) for _ in range(6)])
            calibration.add_trial('open',[dict(r3=.6,r2=None) for _ in range(6)])
        for i in range(5):result=channel.process_features(i,i*.05,dict(r3=.6,r2=None))
        self.assertTrue(result['calibration']['calibrated'])
        self.assertAlmostEqual(result['aperture'],.044)
        self.assertGreater(result['intent_ratio'],.8)

    def test_hold_mode_allows_start_intent_but_never_moves_fingers(self):
        from human_grasp_task import DirectHumanTask,HumanInput,Feedback
        channel=TaskHandControl(mode='hold');task=DirectHumanTask()
        feedback=Feedback(tcp=np.array([.43,0.,.5]),visual=dict(time_s=0.))
        for i in range(20):
            hand=channel.process_features(i,i*.05,dict(r3=1.2,r2=None))
            task.update(HumanInput(i,np.zeros(3),hand['intent_ratio'],True,i*.05),feedback,i*.05,start=True)
        self.assertEqual(task.stage,'direct');self.assertFalse(hand['valid'])
        self.assertEqual(hand['aperture'],.044)

    def test_long_source_gap_requires_fresh_confirmation(self):
        channel,hand=self.result(.1);closed=hand['aperture']
        for seq,stamp in ((10,1.),(11,1.05)):
            result=channel.process_features(seq,stamp,dict(r3=1.2,r2=None))
            self.assertFalse(result['valid']);self.assertEqual(result['aperture'],closed)
        result=channel.process_features(12,1.1,dict(r3=1.2,r2=None))
        self.assertTrue(result['valid']);self.assertGreater(result['aperture'],closed)

    def test_nonfinite_finger_data_never_authorizes_motion(self):
        channel,_=self.result()
        result=channel.process_features(20,1.1,dict(r3=float('nan'),r2=None))
        self.assertFalse(result['valid']);self.assertTrue(np.isfinite(result['aperture']))

    def test_thresholds_are_independent_and_reject_nan(self):
        self.assertEqual(hand_landmarker_thresholds(.35,.5,.6),dict(min_hand_detection_confidence=.35,
            min_hand_presence_confidence=.5,min_tracking_confidence=.6))
        for args in ((float('nan'),None,None),(.5,0,None),(.5,None,1.2)):
            with self.assertRaises(ValueError):hand_landmarker_thresholds(*args)

    def test_optional_calibration_io_failure_is_reported_without_raising(self):
        with patch('run_human_task.save_personal_calibration',side_effect=OSError('fixture write denied')):
            error=save_task_calibration(Path('.'),TaskHandControl(),Path('ignored.json'),synthetic=True)
        self.assertIn('fixture write denied',error)


if __name__=='__main__':unittest.main()
