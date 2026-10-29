"""Separate endpoint variation across trials from noise within one held pose."""
import unittest
import numpy as np
from human_pinch_control import PinchCalibration


class PinchCalibrationNoiseTests(unittest.TestCase):
    def fixture(self,opens):
        trials={'closed':[],'open':[]}
        for label,centers in (('closed',np.linspace(.1,.3,10)),('open',opens)):
            for center in centers:
                trials[label].append([dict(r3=float(center+offset),r2=None) for offset in np.linspace(-.02,.02,10)])
        return dict(trials=trials)

    def test_separated_stable_trials_with_different_open_extents_are_usable(self):
        c=PinchCalibration(self.fixture(np.linspace(.72,1.3,10)))
        self.assertIsNotNone(c.profile)
        self.assertLess(c.profile['r3']['closed'],c.profile['r3']['open'])

    def test_closed_pose_labeled_open_still_prevents_fitting(self):
        c=PinchCalibration(self.fixture([1.2]*8+[.16,1.3]))
        self.assertIsNone(c.profile)
        self.assertEqual(c.reason,'OPEN_CLOSED_DISTRIBUTIONS_OVERLAP')

    def test_removing_mislabeled_trial_requires_a_replacement(self):
        c=PinchCalibration(self.fixture([1.2]*9))
        self.assertIsNone(c.profile)
        self.assertEqual(c.reason,'NEED_10_CLOSED_AND_10_OPEN_TRIALS')
        c.add_trial('open',[dict(r3=1.2+float(v),r2=None) for v in np.linspace(-.02,.02,10)])
        self.assertIsNotNone(c.profile)

    def test_actual_held_pose_noise_requires_a_clear_margin(self):
        payload=self.fixture([.45]*10)
        for label,center in (('closed',.2),('open',.55)):
            payload['trials'][label]=[[dict(r3=float(center+v),r2=None)
                for v in np.linspace(-.12,.12,10)] for _ in range(10)]
        c=PinchCalibration(payload)
        self.assertIsNone(c.profile)
        self.assertEqual(c.reason,'OPEN_CLOSED_MARGIN_TOO_SMALL')


if __name__=='__main__':unittest.main()
