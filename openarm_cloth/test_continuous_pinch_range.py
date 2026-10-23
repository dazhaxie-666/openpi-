"""Continuous aperture must preserve visible small gaps and intermediate poses."""
import unittest
import numpy as np
from human_pinch_control import PinchCalibration,HumanPinchControl


class ContinuousPinchRangeTests(unittest.TestCase):
    def calibration(self):
        c=PinchCalibration()
        for closed in np.linspace(.08,.32,10):
            c.add_trial('closed',[dict(r3=float(closed+v),r2=.1+float(v)) for v in np.linspace(-.01,.01,8)])
            c.add_trial('open',[dict(r3=1.2+float(v),r2=1.5+float(v)) for v in np.linspace(-.01,.01,8)])
        return c

    def settle(self,c,features):
        for i in range(30):c.update(features,i*.05)
        return c.aperture/.044

    def test_continuous_endpoints_are_trial_centers_not_classification_tails(self):
        c=self.calibration()
        self.assertAlmostEqual(c.profile['r3']['continuous_closed'],.2)
        self.assertAlmostEqual(c.profile['r3']['continuous_open'],1.2)
        self.assertGreater(c.profile['r3']['closed'],.3)

    def test_visible_image_gap_prevents_zero_when_depth_underestimates_distance(self):
        c=HumanPinchControl('continuous',self.calibration())
        aperture=self.settle(c,dict(r3=.30,r2=.50))
        self.assertGreater(aperture,.20)
        self.assertLess(aperture,.40)

    def test_measured_middle_is_continuous_and_true_endpoints_remain_reachable(self):
        for fraction in (0.,.25,.5,.75,1.):
            with self.subTest(fraction=fraction):
                c=HumanPinchControl('continuous',self.calibration())
                measured=dict(r3=.2+fraction,r2=.1+1.4*fraction)
                self.assertAlmostEqual(self.settle(c,measured),fraction,places=3)

    def test_conflicting_measurements_hold_last_aperture(self):
        c=HumanPinchControl('continuous',self.calibration())
        held=self.settle(c,dict(r3=.7,r2=.8))
        for i in range(30,40):c.update(dict(r3=.2,r2=1.5),i*.05)
        self.assertEqual(c.aperture/.044,held)
        self.assertEqual(c.reason,'PINCH_2D_3D_CONFLICT')

    def test_unobservable_2d_uses_calibrated_3d_without_inventing_a_gap(self):
        c=HumanPinchControl('continuous',self.calibration())
        self.assertAlmostEqual(self.settle(c,dict(r3=.7,r2=None)),.5,places=3)

    def test_binary_mode_keeps_existing_classification_bounds(self):
        c=HumanPinchControl('binary',self.calibration())
        self.assertEqual(self.settle(c,dict(r3=.30,r2=.50)),0.)


if __name__=='__main__':unittest.main()
