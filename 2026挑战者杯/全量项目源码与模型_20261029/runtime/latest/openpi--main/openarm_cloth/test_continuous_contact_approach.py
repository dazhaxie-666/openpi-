"""A gradual measured approach must not stop again at an arbitrary boundary."""
import unittest
import test_pinch_contact_evidence as fixtures
from human_pinch_control import HumanPinchControl,PinchCalibration


class ContinuousContactApproachTests(unittest.TestCase):
    setUp=fixtures.PinchContactEvidenceTests.setUp
    sample=fixtures.PinchContactEvidenceTests.sample
    half=fixtures.PinchContactEvidenceTests.half
    def test_gradual_closure_collects_contact_evidence_before_last_fifteen_percent(self):
        self.half()
        for fraction in (.28,.24,.20,.16,.12,.08,.04,0.):
            result=self.sample(.2+fraction,.1+1.4*fraction)
            self.assertTrue(result['valid'],(fraction,result['reason']))

    def test_small_boundary_oscillation_does_not_repeatedly_reconfirm(self):
        self.half()
        for fraction in (.28,.24,.20,.16):self.sample(.2+fraction,.1+1.4*fraction)
        for fraction in (.14,.16,.14,.16,.13):
            result=self.sample(.2+fraction,.1+1.4*fraction)
            self.assertTrue(result['valid'],result['reason'])

    def test_warmed_nearby_pose_cannot_authorize_a_sudden_contact_collapse(self):
        self.half()
        for _ in range(8):self.sample(.45,.45)
        held=self.channel.control.aperture
        result=self.sample(.2,.1)
        self.assertFalse(result['valid'])
        self.assertAlmostEqual(result['aperture'],held)

    def test_both_channels_inside_measured_closed_class_can_finish_contact(self):
        calibration=PinchCalibration();calibration.profile={
            'r3':dict(closed=.35,open=.9,continuous_closed=.21,continuous_open=1.21),
            'r2':dict(closed=.17,open=1.3,continuous_closed=.13,continuous_open=1.83)}
        control=HumanPinchControl(calibration=calibration,contact_guard=True)
        for i in range(25):control.update(dict(r3=.33,r2=.15),i*.05)
        self.assertEqual(control.state,'TRACKING')
        self.assertLess(control.aperture,.0002)

    def test_visible_projected_gap_prevents_closed_class_completion(self):
        calibration=PinchCalibration();calibration.profile={
            'r3':dict(closed=.35,open=.9,continuous_closed=.21,continuous_open=1.21),
            'r2':dict(closed=.17,open=1.3,continuous_closed=.13,continuous_open=1.83)}
        control=HumanPinchControl(calibration=calibration,contact_guard=True)
        for i in range(25):control.update(dict(r3=.33,r2=.6),i*.05)
        self.assertGreater(control.aperture,.01)
