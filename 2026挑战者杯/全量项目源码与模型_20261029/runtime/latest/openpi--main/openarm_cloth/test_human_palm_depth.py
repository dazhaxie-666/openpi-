"""Independent projection fixtures cover depth, aperture and missing-source behavior."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from human_palm_depth import RelativePalmDepth
from human_grasp_task import HumanInput
from run_human_task import palm_depth_input


def sample(depth=.7,rotation=None,shift=(0.,0.),curl=0.):
    h=np.zeros((21,3));h[[0,5,9,13,17]]=[[0,0,0],[-.035,-.06,.003],[0,-.08,0],[.022,-.07,-.003],[.04,-.05,.002]]
    for base in (5,9,13,17):
        for step in (1,2,3):h[base+step]=h[base]+[0,-.018*step*(1-curl),-.018*step*curl]
    h[1:5]=[[-.025,-.015,0],[-.04,-.025,0],[-.055,-.035,0],[-.065*(1-curl),-.04,0]]
    if rotation is not None:h=h@rotation.T
    xyz=h+np.array([*shift,depth]);n=np.zeros((21,3));n[:,:2]=(xyz[:,:2]/xyz[:,2,None]*1280+[640,360])/[1280,720]
    return h,n


class RelativePalmDepthTests(unittest.TestCase):
    def ready(self):
        tracker=RelativePalmDepth(sensitivity=.35);stamp=0.
        for _ in range(10):
            stamp+=.05;tracker.update(*sample(),stamp)
        self.assertTrue(tracker.valid)
        return tracker,stamp

    def test_near_and_far_motion_are_opposite_and_lateral_is_independent(self):
        tracker,t=self.ready()
        for depth in np.linspace(.7,.55,20):t+=.05;tracker.update(*sample(depth),t)
        self.assertGreater(tracker.position,.055)
        for depth in np.linspace(.55,.85,35):t+=.05;tracker.update(*sample(depth),t)
        self.assertLess(tracker.position,-.045)
        held=tracker.position
        for x in np.linspace(0,.12,20):t+=.05;tracker.update(*sample(.85,shift=(x,0)),t)
        self.assertLess(abs(tracker.position-held),.008)

    def test_palm_rotation_and_model_scale_change_do_not_create_forward_motion(self):
        tracker,t=self.ready()
        for angle in np.linspace(0,50,30):
            h,n=sample(rotation=Rotation.from_euler('y',angle,degrees=True).as_matrix())
            t+=.05;tracker.update(h*.75,n,t)
        self.assertLess(abs(tracker.position),.008)

    def test_finger_closure_does_not_change_depth(self):
        tracker,t=self.ready()
        for curl in np.linspace(0,1,30):t+=.05;tracker.update(*sample(curl=curl),t)
        for _ in range(10):t+=.05;tracker.update(*sample(curl=1),t)
        self.assertLess(abs(tracker.position),.004)

    def test_missing_and_reacquired_hand_holds_without_a_depth_jump(self):
        tracker,t=self.ready();held=tracker.position
        for _ in range(10):t+=.05;tracker.update(None,None,t)
        for _ in range(10):t+=.05;tracker.update(*sample(.4),t)
        self.assertAlmostEqual(tracker.position,held,places=6)
        self.assertTrue(tracker.valid)

    def test_duplicate_source_cannot_change_depth_or_complete_calibration(self):
        tracker=RelativePalmDepth()
        for _ in range(30):tracker.update(*sample(),1.)
        self.assertFalse(tracker.valid)
        tracker,t=self.ready();held=tracker.position
        for _ in range(20):tracker.update(*sample(.4),t)
        self.assertEqual(tracker.position,held)

    def test_clipped_or_tiny_hand_cannot_authorize_depth(self):
        tracker,t=self.ready();held=tracker.position
        h,n=sample();n[:,0]-=1.
        tracker.update(h,n,t+.05);self.assertFalse(tracker.valid)
        self.assertEqual(tracker.position,held)
        h,n=sample();n[:,:2]=(n[:,:2]-.5)*.03+.5
        tracker.update(h,n,t+.1);self.assertFalse(tracker.valid)

    def test_large_unconfirmed_projection_jump_cannot_move_arm(self):
        tracker,t=self.ready();held=tracker.position
        tracker.update(*sample(.3),t+.05)
        self.assertFalse(tracker.valid);self.assertEqual(tracker.position,held)

    def test_simultaneous_gripper_change_cannot_disable_observable_forward_motion(self):
        tracker,t=self.ready();tracker.update(*sample(),t+.05,gripper=.044);t+=.05
        accepted=0
        for depth,grip,curl in zip(np.linspace(.7,.5,30),np.linspace(.044,.002,30),np.linspace(0,1,30)):
            t+=.05;tracker.update(*sample(depth,curl=curl),t,gripper=grip);accepted+=tracker.valid
        self.assertGreater(accepted,25)
        self.assertGreater(tracker.position,.09)

    def test_gripper_jitter_with_static_palm_cannot_create_or_invalidate_depth(self):
        tracker,t=self.ready()
        for grip in (.020,.024,.019,.025,.020,.024)*5:
            t+=.05;tracker.update(*sample(),t,gripper=grip)
            self.assertTrue(tracker.valid)
            self.assertLess(abs(tracker.position),.004)

    def test_brief_quality_rejections_do_not_recenter_away_every_forward_step(self):
        tracker,t=self.ready()
        for i,depth in enumerate(np.linspace(.7,.5,30)):
            h,n=sample(depth);t+=.05
            if i%3==1:n[:,0]-=1.
            tracker.update(h,n,t)
        self.assertGreater(tracker.position,.085)

    def test_default_relative_gain_fits_the_small_single_arm_workspace(self):
        tracker=RelativePalmDepth();t=0.
        for _ in range(10):t+=.05;tracker.update(*sample(.7),t)
        for depth in np.linspace(.7,.35,30):t+=.05;tracker.update(*sample(depth),t)
        self.assertGreater(tracker.position,.09)
        self.assertLess(tracker.position,.11)

    def test_visible_hand_recovers_depth_after_projection_failure_without_losing_travel(self):
        tracker,t=self.ready()
        with patch('human_palm_depth.palm_translation',return_value=None):
            for depth in np.linspace(.7,.5,10):t+=.05;tracker.update(*sample(depth),t)
        for _ in range(10):t+=.05;tracker.update(*sample(.5),t)
        self.assertGreater(tracker.position,.10)

    def test_template_recovery_ignores_model_scale_and_retains_real_translation(self):
        tracker,t=self.ready()
        with patch('human_palm_depth.palm_translation',return_value=(.7,50.)):
            for depth in np.linspace(.7,.5,8):
                h,n=sample(depth);t+=.05;tracker.update(h*.75,n,t)
        for _ in range(10):
            h,n=sample(.5);t+=.05;tracker.update(h*.75,n,t)
        self.assertGreater(tracker.position,.10)

    def test_brief_hand_loss_with_continuous_body_identity_preserves_translation(self):
        tracker,t=self.ready()
        for _ in range(10):t+=.05;tracker.update(None,None,t,identity_continuous=True)
        for _ in range(10):t+=.05;tracker.update(*sample(.5),t,identity_continuous=True)
        self.assertGreater(tracker.position,.10)

    def test_one_body_glitch_does_not_reset_depth_but_long_body_loss_does(self):
        tracker,t=self.ready();t+=.05;tracker.update(*sample(),t,identity_continuous=True)
        tracker.hold('DEPTH_BODY_HOLD',rebase=False,preserve_reference=True,capture_s=t+.1)
        for _ in range(10):t+=.05;tracker.update(*sample(.5),t+.1,identity_continuous=True)
        self.assertGreater(tracker.position,.10)
        held=tracker.position
        tracker.hold('DEPTH_BODY_HOLD',capture_s=t+1.)
        for _ in range(10):t+=.05;tracker.update(*sample(.9),t+1.,identity_continuous=True)
        self.assertAlmostEqual(tracker.position,held,places=6)

    def test_input_adapter_changes_only_depth_and_cannot_fall_back_to_drifting_body_depth(self):
        tracker,t=self.ready()
        body=HumanInput(11,np.array([.4,.12,.08]),None,True,t+.05)
        h,n=sample();obs=dict(read_end=(t+.05)*1e9,arms={'right':dict(hand_valid=True,hand_raw=h,hand_norm=n)})
        hybrid,result=palm_depth_input(body,obs,tracker)
        np.testing.assert_allclose(hybrid.point,[0.,.12,.08],atol=1e-8)
        self.assertEqual(body.point[0],.4)
        body.point[0]=1.;obs['read_end']+=50_000_000;obs['arms']['right']['hand_valid']=False
        hybrid,result=palm_depth_input(body,obs,tracker)
        self.assertFalse(result['valid']);self.assertTrue(hybrid.valid)
        self.assertAlmostEqual(hybrid.point[0],0.,places=8)
        np.testing.assert_allclose(hybrid.point[1:],body.point[1:])

    def test_stale_body_or_pause_invalidates_depth_even_if_source_is_duplicated(self):
        tracker,t=self.ready();h,n=sample()
        obs=dict(read_end=t*1e9,arms={'right':dict(hand_valid=True,hand_raw=h,hand_norm=n)})
        body=HumanInput(10,np.zeros(3),None,False,t)
        hybrid,result=palm_depth_input(body,obs,tracker)
        self.assertFalse(hybrid.valid);self.assertFalse(result['valid'])
        self.assertEqual(result['reason'],'DEPTH_BODY_HOLD')
        body.valid=True
        hybrid,result=palm_depth_input(body,obs,tracker,paused=True)
        self.assertFalse(result['valid']);self.assertEqual(result['reason'],'DEPTH_PAUSED')


if __name__=='__main__':unittest.main()
