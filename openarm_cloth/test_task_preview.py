"""Preview cadence sheds render work while preserving operator input."""
import unittest
from unittest.mock import patch
from human_task_preview import PreviewCadence,poll_task_preview,calibration_prompt,framing_prompt
import numpy as np


class TaskPreviewTests(unittest.TestCase):
    def test_small_control_jitter_does_not_halve_robot_preview_rate(self):
        cadence=PreviewCadence(20.)
        for stamp in (0.,.051,.1005,.1515,.201,.2515):
            self.assertTrue(cadence.due(stamp),stamp)

    def test_delayed_control_loop_does_not_replay_a_queue_of_old_previews(self):
        cadence=PreviewCadence(10.)
        self.assertTrue(cadence.due(0.))
        self.assertFalse(cadence.due(.05))
        self.assertTrue(cadence.due(.1))
        self.assertTrue(cadence.due(1.))
        self.assertFalse(cadence.due(1.001))
        self.assertTrue(cadence.due(1.1))

    def test_pause_key_and_closed_window_are_serviced_between_previews(self):
        cadence=PreviewCadence(10.);cadence.due(0.)
        keys=[]
        with patch('human_task_preview.cv2.pollKey',side_effect=[32,-1]),patch(
                'human_task_preview.cv2.getWindowProperty',side_effect=[1.,0.]):
            self.assertFalse(cadence.due(.01))
            self.assertTrue(poll_task_preview('fixture',keys.append))
            self.assertFalse(cadence.due(.02))
            self.assertFalse(poll_task_preview('fixture',keys.append))
        self.assertEqual(keys,[32])

    def test_operator_poll_does_not_wait_for_windows_timer_quantum(self):
        with patch('human_task_preview.cv2.pollKey',return_value=ord('h')),patch(
                'human_task_preview.cv2.waitKey',side_effect=AssertionError('blocking input poll')),patch(
                'human_task_preview.cv2.getWindowProperty',return_value=1.):
            keys=[]
            self.assertTrue(poll_task_preview('fixture',keys.append))
            self.assertEqual(keys,[ord('h')])

    def test_invalid_preview_rate_cannot_disable_operator_window(self):
        for hz in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):PreviewCadence(hz)

    def test_uncalibrated_profile_has_actionable_banner(self):
        self.assertIn('G pinch / O spread',calibration_prompt(dict(calibrated=False)))
        self.assertIsNone(calibration_prompt(dict(calibrated=True)))

    def test_latest_failed_session_gets_actionable_camera_framing_prompt(self):
        pose=np.zeros((33,4));pose[:,3]=1.
        pose[[11,12],1]=[.96,.85];pose[16,1]=1.6;pose[16,3]=.1
        prompt=framing_prompt(dict(pose_norm=pose,position_trusted=dict(right=False)))
        self.assertIn('R to reacquire',prompt)
        pose[[11,12],1]=.4;pose[16,1]=.6
        self.assertIsNone(framing_prompt(dict(pose_norm=pose,position_trusted=dict(right=True))))

    def test_pause_and_arm_hold_are_visible_in_the_primary_title(self):
        from human_task_preview import compose_task_preview
        camera=np.zeros((540,960,3),np.uint8)
        with patch('human_task_preview.cv2.putText') as text:
            compose_task_preview(camera,[],'direct',paused=True,arm_status='ARM TRACKING')
            self.assertEqual(text.call_args_list[0].args[1],'PAUSED - SPACE TO RESUME')
        with patch('human_task_preview.cv2.putText') as text:
            compose_task_preview(camera,[],'direct',arm_status='ARM HOLD - BODY WRIST NOT TRACKED')
            self.assertIn('ARM HOLD',text.call_args_list[0].args[1])


if __name__=='__main__':unittest.main()
