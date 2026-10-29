"""Cartesian single-arm axes must not depend on unseen inferred hips."""
import unittest
import numpy as np
from human_teleop_core import human_wrist,human_upper_body_wrist


class UpperBodyWristTests(unittest.TestCase):
    def body(self):
        w=np.zeros((33,4));w[:,3]=1.
        w[11,:3]=[.2,0,0];w[12,:3]=[-.2,0,0]
        w[23,:3]=[.15,.5,0];w[24,:3]=[-.15,.5,0]
        w[16,:3]=[-.3,.1,-.3]
        return w

    def test_unseen_hips_cannot_rotate_depth_into_height(self):
        w=self.body();before=human_upper_body_wrist(w)
        w[[23,24],:3]=np.nan;w[[23,24],3]=0
        np.testing.assert_allclose(human_upper_body_wrist(w),before)
        self.assertIsNone(human_wrist(w,require_elbow=False))

    def test_camera_forward_changes_only_robot_forward(self):
        w=self.body();before=human_upper_body_wrist(w)
        w[16,2]-=.1
        np.testing.assert_allclose(human_upper_body_wrist(w)-before,[.1,0,0],atol=1e-12)

    def test_pure_vertical_and_lateral_axes_preserve_original_frontal_convention(self):
        w=self.body();np.testing.assert_allclose(human_upper_body_wrist(w),human_wrist(w,require_elbow=False))
        before=human_upper_body_wrist(w);w[16,1]-=.1;w[16,0]+=.05
        np.testing.assert_allclose(human_upper_body_wrist(w)-before,[0,.05,.1],atol=1e-12)

    def test_missing_wrist_or_degenerate_shoulders_hold(self):
        for index in (11,12,16):
            w=self.body();w[index,3]=.1
            self.assertIsNone(human_upper_body_wrist(w))
        w=self.body();w[12,:3]=w[11,:3];self.assertIsNone(human_upper_body_wrist(w))


if __name__=='__main__':unittest.main()
