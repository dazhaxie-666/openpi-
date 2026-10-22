"""Preview cadence sheds render work while preserving operator input."""
import unittest
from unittest.mock import patch
from human_task_preview import PreviewCadence,poll_task_preview,calibration_prompt


class TaskPreviewTests(unittest.TestCase):
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
        with patch('human_task_preview.cv2.waitKey',side_effect=[32,255]),patch(
                'human_task_preview.cv2.getWindowProperty',side_effect=[1.,0.]):
            self.assertFalse(cadence.due(.01))
            self.assertTrue(poll_task_preview('fixture',keys.append))
            self.assertFalse(cadence.due(.02))
            self.assertFalse(poll_task_preview('fixture',keys.append))
        self.assertEqual(keys,[32])

    def test_invalid_preview_rate_cannot_disable_operator_window(self):
        for hz in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):PreviewCadence(hz)

    def test_uncalibrated_profile_has_actionable_banner(self):
        self.assertIn('G pinch / O spread',calibration_prompt(dict(calibrated=False)))
        self.assertIsNone(calibration_prompt(dict(calibrated=True)))


if __name__=='__main__':unittest.main()
