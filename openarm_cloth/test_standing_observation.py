"""A hidden elbow must not disable an observed Cartesian control wrist."""
import unittest
import numpy as np
from live_observation_quality import body_hand_consistency
from human_pinch_control import PinchCalibration
from human_pinch_control import calibration_status
from human_teleop_core import human_wrist
from run_human_task import human_input,tracking_diagnostic


class StandingObservationTests(unittest.TestCase):
    def pose(self):
        pose=np.zeros((33,4));pose[:,3]=1.
        pose[11,:2]=[.65,.15];pose[12,:2]=[.35,.15]
        pose[13,:2]=[.7,.55];pose[14,:2]=[.3,1.1]
        pose[15,:2]=[.75,.7];pose[16,:2]=[.25,.7]
        pose[14,3]=.1
        return pose

    def test_visible_wrist_survives_unobserved_elbow_in_position_mode(self):
        pose=self.pose()
        arm,_=body_hand_consistency(pose,[])
        position,_=body_hand_consistency(pose,[],require_elbow=False)
        self.assertFalse(arm['right']);self.assertTrue(position['right'])

    def test_out_of_frame_wrist_still_cannot_authorize_position_control(self):
        pose=self.pose();pose[16,1]=1.05
        trusted,reasons=body_hand_consistency(pose,[],require_elbow=False)
        self.assertFalse(trusted['right']);self.assertIn('outside frame',reasons['right'])

    def test_out_of_frame_shoulders_are_not_invented_as_anchors(self):
        pose=self.pose();pose[12,1]=-.1
        trusted,reasons=body_hand_consistency(pose,[],require_elbow=False)
        self.assertFalse(trusted['right']);self.assertIn('shoulders outside',reasons['right'])

    def test_position_consumer_accepts_visible_wrist_with_low_elbow(self):
        norm=self.pose();trusted,reasons=body_hand_consistency(norm,[],require_elbow=False)
        world=np.zeros((33,4));world[:,3]=1.
        world[11,:3]=[-.2,0,0];world[12,:3]=[.2,0,0]
        world[23,:3]=[-.15,.5,0];world[24,:3]=[.15,.5,0];world[16,:3]=[.3,-.2,-.1];world[14,3]=.1
        point=human_wrist(world,require_elbow=False)
        obs=dict(seq=1,read_end=1_000_000_000,pose_filtered=world,pose_norm=norm,
            position_trusted=trusted,position_quality=reasons,arms=dict(right=dict(
                point=None,position_point=point,hand_valid=False)))
        self.assertFalse(human_input(obs,1.1).valid)
        self.assertTrue(human_input(obs,1.1,position_only=True).valid)
        translated=world.copy();translated[:,:3]+=[0.,-.8,.1]
        np.testing.assert_allclose(human_wrist(translated,require_elbow=False),point,atol=1e-12)

    def test_image_out_of_frame_diagnostic_is_specific(self):
        from human_grasp_task import HumanInput
        norm=self.pose();norm[16,1]=1.1
        world=np.ones((33,4))
        obs=dict(read_end=1_000_000_000,pose_norm=norm,pose_filtered=world,position_trusted={'right':False})
        reason,_,_=tracking_diagnostic(obs,HumanInput(1,None,None,False),1.1)
        self.assertEqual(reason,'WRIST_OUT_OF_FRAME')

    def test_reversed_personal_labels_have_an_explicit_failure_reason(self):
        calibration=PinchCalibration()
        for _ in range(10):
            calibration.add_trial('closed',[dict(r3=.9,r2=.9) for _ in range(6)])
            calibration.add_trial('open',[dict(r3=.25,r2=.25) for _ in range(6)])
        self.assertIsNone(calibration.profile)
        self.assertEqual(calibration.reason,'OPEN_CLOSED_LABELS_REVERSED')
        self.assertIn('NOT READY',calibration_status(calibration.summary()))
        self.assertIn('REVERSED',calibration_status(calibration.summary()))


if __name__=='__main__':unittest.main()
