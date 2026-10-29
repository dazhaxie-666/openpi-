"""User authority at hidden-target motion and fresh near-target grasp entry."""
import unittest
import numpy as np
from human_grasp_task import HumanInput,Feedback
from shared_grasp import SharedGraspTask


class ManualViewIntentTests(unittest.TestCase):
    def setUp(self):
        self.task=SharedGraspTask();self.task.hand_intent_valid=True
        self.fb=Feedback(np.array([.43,0.,.54]),visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.))
        self.seq=0
        for _ in range(6):self.tick(1.2,start=True)

    def tick(self,ratio,point=None,request_ratio=None,valid=True,**kwargs):
        self.seq+=1;stamp=self.seq*.05
        self.task.request_ratio=ratio if request_ratio is None else request_ratio
        return self.task.update(HumanInput(self.seq,np.zeros(3) if point is None and valid else point,
            ratio,valid,stamp),self.fb,stamp,**kwargs)

    def test_closing_outside_range_cannot_trigger_when_entering_range(self):
        for _ in range(8):self.tick(.15)
        self.fb.tcp[2]=.49
        for _ in range(8):cmd=self.tick(.15)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(self.task.trial_id,0)
        self.assertEqual(cmd.finger,.044)

    def test_latched_closed_label_cannot_trigger_with_current_half_open_hand(self):
        for _ in range(8):self.tick(.15)
        self.fb.tcp[2]=.49
        for _ in range(8):self.tick(.65)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(self.task.trial_id,0)

    def test_closed_3d_channel_with_visible_2d_gap_does_not_request_grasp(self):
        self.fb.tcp[2]=.49
        for _ in range(6):self.tick(1.2)
        for _ in range(8):self.tick(.15,request_ratio=.65)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(self.task.trial_id,0)

    def test_near_open_then_new_sustained_both_channel_pinch_requests_once(self):
        self.fb.tcp[2]=.49
        for _ in range(6):self.tick(1.2)
        for _ in range(3):self.tick(.15)
        self.assertEqual(self.task.phase,'manual')
        for _ in range(4):self.tick(.15)
        self.assertEqual(self.task.phase,'plan');self.assertEqual(self.task.trial_id,1)

    def test_open_while_approaching_then_pinch_immediately_in_range_needs_no_extra_open(self):
        self.assertFalse(self.task.grasp_readiness(self.fb,self.seq*.05)['in_range'])
        for _ in range(10):self.tick(1.2)
        self.fb.tcp[2]=.49
        # First in-range input is already the start of the user's pinch.
        for _ in range(6):self.tick(.15)
        self.assertEqual(self.task.phase,'plan');self.assertEqual(self.task.trial_id,1)

    def test_range_boundary_jitter_does_not_discard_open_approach(self):
        for height in (.54,.532,.535,.533,.531):
            self.fb.tcp[2]=height;self.tick(1.2)
        self.fb.tcp[2]=.49
        for _ in range(6):self.tick(.15)
        self.assertEqual(self.task.phase,'plan')

    def test_natural_open_observation_has_no_preparation_dwell(self):
        self.task.reset_grasp_entry()
        self.tick(1.2)
        self.fb.tcp[2]=.49
        for _ in range(6):self.tick(.15)
        self.assertEqual(self.task.phase,'plan')

    def test_hidden_view_keeps_full_measured_hand_control_not_fixed_recovery_motion(self):
        self.fb.tcp=np.array([.435,.005,.438]);self.tick(1.2)
        self.fb.visual=None
        self.tick(1.2,point=np.zeros(3),recenter=True)
        cmd=self.tick(1.2,point=np.array([.02,.03,.04]))
        self.assertEqual((cmd.stage,cmd.owner),('direct','human_target_hidden'))
        self.assertFalse(self.task.intent_latched);self.assertEqual(cmd.finger,.044)
        self.assertTrue(np.all(cmd.target>self.fb.tcp))
        first=cmd.target.copy()
        for _ in range(10):cmd=self.tick(1.2,point=np.array([.02,.03,.04]))
        self.assertTrue(np.all(cmd.target>first));target=cmd.target.copy()
        cmd=self.tick(1.2,point=np.array([.02,.03,.04]))
        np.testing.assert_allclose(cmd.target,target)
        cmd=self.tick(None,valid=False)
        self.assertEqual(cmd.stage,'hold');self.assertEqual(cmd.finger,.044)

    def test_hidden_target_pinch_never_starts_automatic_view_recovery_or_trial(self):
        self.fb.tcp[2]=.49
        for _ in range(6):self.tick(1.2)
        self.fb.visual=None
        for _ in range(8):cmd=self.tick(.15)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(self.task.trial_id,0)
        self.assertEqual((cmd.stage,cmd.finger),('direct',.044))

    def test_missing_frames_and_pause_discard_grasp_confirmation(self):
        self.fb.tcp[2]=.49
        for _ in range(6):self.tick(1.2)
        for _ in range(3):self.tick(.15)
        self.task.hand_intent_valid=False
        for _ in range(9):self.tick(None,valid=False)
        self.task.hand_intent_valid=True
        for _ in range(8):self.tick(.15)
        self.assertEqual(self.task.phase,'manual')
        for _ in range(6):self.tick(1.2)
        self.tick(.15,paused=True)
        for _ in range(8):self.tick(.15)
        self.assertEqual(self.task.phase,'manual')

    def test_repeated_near_pinch_source_does_not_accumulate_confirmation(self):
        self.fb.tcp[2]=.49
        for _ in range(6):self.tick(1.2)
        self.seq+=1;stamp=self.seq*.05
        self.task.request_ratio=.15
        human=HumanInput(self.seq,np.zeros(3),.15,True,stamp)
        for i in range(30):self.task.update(human,self.fb,stamp+i*.033)
        self.assertEqual(self.task.phase,'manual')


class BothChannelRequestTests(unittest.TestCase):
    profile=dict(r3=dict(closed=.3493185898942841,open=.8880780840079845),
        r2=dict(closed=.1732494036961057,open=1.3350587123612578))

    def test_saved_3d_closed_but_2d_separated_sample_is_not_a_pinch(self):
        from shared_grasp_intent import grasp_request_ratio
        ratio=grasp_request_ratio(dict(r3=.2961503196943812,r2=.5221044962472308),self.profile)
        self.assertGreater(ratio,.35)
        self.assertLessEqual(grasp_request_ratio(dict(r3=.20,r2=.15),self.profile),.35)

    def test_missing_2d_measurement_or_calibration_cannot_confirm_grasp(self):
        from shared_grasp_intent import grasp_request_ratio
        self.assertIsNone(grasp_request_ratio(dict(r3=.2,r2=None),self.profile))
        self.assertIsNone(grasp_request_ratio(dict(r3=.2,r2=.2),dict(r3=self.profile['r3'])))


if __name__=='__main__':unittest.main()
