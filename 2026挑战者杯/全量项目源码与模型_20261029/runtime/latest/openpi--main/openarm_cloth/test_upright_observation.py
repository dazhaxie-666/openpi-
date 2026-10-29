"""Synthetic upright-teleoperation geometry, not personal landmark fixtures."""
import unittest
import numpy as np
from live_observation_quality import body_hand_consistency


class UprightObservationTests(unittest.TestCase):
    def pose(self):
        p=np.zeros((33,4));p[:,3]=1.
        p[11,:2]=[.65,.4];p[12,:2]=[.35,.4]
        p[13,:2]=[.7,.5];p[14,:2]=[.3,.5]
        p[15,:2]=[.75,.7];p[16,:2]=[.25,.7]
        p[0,:2]=[.5,.2];p[7,:2]=[.56,.2];p[8,:2]=[.44,.2]
        p[23,:2]=[.6,.8];p[24,:2]=[.4,.8]
        return p

    def test_inverted_high_confidence_skeleton_never_authorizes_either_arm(self):
        p=self.pose();p[0,1]=.75;p[[7,8],1]=.7;p[[23,24],1]=.1
        for elbow in (True,False):
            trusted,reasons=body_hand_consistency(p,[],require_elbow=elbow)
            self.assertFalse(any(trusted.values()))
            self.assertEqual(reasons['right'],'body torso inverted')

    def test_upright_sitting_and_standing_scale_are_accepted(self):
        for scale in (1.,.6):
            p=self.pose();p[:,:2]=(p[:,:2]-.5)*scale+.5
            self.assertTrue(all(body_hand_consistency(p,[])[0].values()))

    def test_missing_face_is_not_invented_or_used_to_disable_visible_wrists(self):
        p=self.pose();p[[0,7,8],3]=.1
        self.assertTrue(all(body_hand_consistency(p,[])[0].values()))

    def test_bending_head_down_with_hips_below_shoulders_is_not_inversion(self):
        p=self.pose();p[0,1]=.55;p[[7,8],1]=.5
        self.assertTrue(all(body_hand_consistency(p,[])[0].values()))


if __name__=='__main__':unittest.main()
