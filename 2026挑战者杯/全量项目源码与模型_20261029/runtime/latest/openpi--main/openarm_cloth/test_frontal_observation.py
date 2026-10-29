"""Real failure seams: wrong model palm rotation and changing body depth."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from human_palm_depth import RelativePalmDepth
from frontal_image_wrist import FrontalImageWrist
from test_human_palm_depth import sample


class FrontalObservationTests(unittest.TestCase):
    def ready_depth(self):
        tracker=RelativePalmDepth(solver='planar');t=0.
        for _ in range(10):
            t+=.05;tracker.update(*sample(),t,identity_continuous=True)
        self.assertTrue(tracker.valid)
        return tracker,t

    def test_bad_model_palm_rotation_cannot_freeze_visible_forward_and_return(self):
        tracker,t=self.ready_depth()
        for depth in np.linspace(.7,.5,30):
            h,n=sample(depth);h=h@Rotation.from_euler('y',100,degrees=True).as_matrix().T
            t+=.05;tracker.update(h,n,t,identity_continuous=True)
        self.assertTrue(tracker.valid);self.assertGreater(tracker.position,.04)
        for depth in np.linspace(.5,.85,35):
            h,n=sample(depth);t+=.05;tracker.update(h*.5,n,t,identity_continuous=True)
        self.assertTrue(tracker.valid);self.assertLess(tracker.position,-.02)

    def test_rotation_fist_and_sideways_motion_at_fixed_range_cannot_drive_depth(self):
        tracker,t=self.ready_depth()
        for angle in np.linspace(0,55,40):
            h,n=sample(rotation=Rotation.from_euler('y',angle,degrees=True).as_matrix(),shift=(.12,0),curl=angle/55)
            t+=.05;tracker.update(h,n,t,identity_continuous=True)
        self.assertTrue(tracker.valid);self.assertLess(abs(tracker.position),.008)

    def test_no_hand_duplicate_or_clipped_palm_can_create_movement(self):
        tracker,t=self.ready_depth();held=tracker.position
        tracker.update(*sample(.4),t);self.assertEqual(tracker.position,held)
        tracker.update(None,None,t+.05);self.assertFalse(tracker.valid)
        h,n=sample();n[:,0]-=1
        tracker.update(h,n,t+.1);self.assertFalse(tracker.valid);self.assertEqual(tracker.position,held)

    def test_actual_adapter_cannot_bypass_stale_or_rejected_body_observations(self):
        from run_human_task import frontal_image_input
        from human_grasp_task import HumanInput
        tracker=FrontalImageWrist();p=self.pose()
        human=HumanInput(1,np.array([20.,20.,20.]),None,True,1.)
        obs=dict(read_end=1_000_000_000,position_norm=p,position_trusted={'right':True})
        for i in range(5):
            obs['read_end']=int((1.+i*.05)*1e9)
            result=frontal_image_input(human,obs,tracker,1.+i*.05)
        self.assertTrue(result.valid);self.assertEqual(result.point[0],0.)
        self.assertFalse(frontal_image_input(human,obs,tracker,2.).valid)
        obs['position_trusted']['right']=False
        self.assertFalse(frontal_image_input(human,obs,tracker,1.2).valid)

    def test_recenter_waits_for_calibration_instead_of_losing_the_request(self):
        tracker=FrontalImageWrist();p=self.pose()
        for i in range(4):
            point=tracker.update(p,(i+1)*.05)
            self.assertIsNone(point);self.assertFalse(tracker.take_recenter(point is not None))
        point=tracker.update(p,.25)
        self.assertTrue(tracker.take_recenter(point is not None))
        self.assertFalse(tracker.take_recenter(True))

    def pose(self):
        pose=np.zeros((33,4));pose[:,3]=1.
        pose[[11,12],:2]=[[.6,.3],[.4,.3]];pose[16,:2]=[.3,.6]
        return pose

    def test_image_mapping_uses_fixed_calibration_not_model_z_or_current_shoulder_scale(self):
        tracker=FrontalImageWrist();p=self.pose()
        for i in range(5):point=tracker.update(p,(i+1)*.05,(1280,720))
        before=point.copy();p[:,2]=np.linspace(-20,20,33)
        p[[11,12],0]=[.65,.35];p[16,0]+=.05
        point=tracker.update(p,.4,(1280,720))
        np.testing.assert_allclose(point-before,[0,.1,0],atol=1e-10)
        p[16,3]=.1;self.assertIsNone(tracker.update(p,.5,(1280,720)))


if __name__=='__main__':unittest.main()
