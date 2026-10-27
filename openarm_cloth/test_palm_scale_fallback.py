"""Observed image similarity may bridge brief model-fit failures, not missing hands."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from test_human_palm_depth import sample
from human_palm_depth import RelativePalmDepth


class PalmScaleFallbackTests(unittest.TestCase):
    def ready(self):
        tracker=RelativePalmDepth();t=0.
        for _ in range(10):t+=.05;tracker.update(*sample(),t)
        return tracker,t

    def test_brief_bad_world_projection_still_uses_observed_uniform_image_growth(self):
        tracker,t=self.ready()
        with patch('human_palm_depth.palm_translation',return_value=(.6,30.)):
            for depth in np.linspace(.7,.55,10):
                t+=.05;tracker.update(*sample(depth),t)
        self.assertTrue(tracker.valid)
        self.assertEqual(tracker.reason,'DEPTH_SCALE_TRACKING')
        self.assertGreater(tracker.position,.025)

    def test_lateral_motion_or_image_rotation_does_not_create_depth_during_fallback(self):
        tracker,t=self.ready()
        with patch('human_palm_depth.palm_translation',return_value=(.5,30.)):
            for angle,x in zip(np.linspace(0,40,10),np.linspace(0,.1,10)):
                h,n=sample(rotation=Rotation.from_euler('z',angle,degrees=True).as_matrix(),shift=(x,0))
                t+=.05;tracker.update(h,n,t)
                self.assertTrue(tracker.valid)
        self.assertLess(abs(tracker.position),.002)

    def test_foreshortening_missing_hand_or_expired_reference_cannot_authorize_scale_depth(self):
        tracker,t=self.ready();held=tracker.position
        with patch('human_palm_depth.palm_translation',return_value=(.4,30.)):
            t+=.05;tracker.update(*sample(rotation=Rotation.from_euler('y',60,degrees=True).as_matrix()),t)
            self.assertFalse(tracker.valid);self.assertEqual(tracker.position,held)
            for _ in range(25):t+=.05;tracker.update(*sample(.65),t)
            self.assertFalse(tracker.valid)
        t+=.05;held=tracker.position;tracker.update(None,None,t)
        self.assertFalse(tracker.valid);self.assertEqual(tracker.position,held)

    def test_large_scale_step_requires_two_fresh_observations_then_can_recover(self):
        tracker,t=self.ready();held=tracker.position
        with patch('human_palm_depth.palm_translation',return_value=(.44,30.)):
            t+=.05;tracker.update(*sample(.44),t)
            self.assertFalse(tracker.valid);self.assertEqual(tracker.position,held)
            self.assertEqual(tracker.reason,'DEPTH_JUMP_CONFIRMING')
            t+=.05;tracker.update(*sample(.44),t)
            self.assertTrue(tracker.valid)
            self.assertGreater(tracker.position,.035)


if __name__=='__main__':unittest.main()
