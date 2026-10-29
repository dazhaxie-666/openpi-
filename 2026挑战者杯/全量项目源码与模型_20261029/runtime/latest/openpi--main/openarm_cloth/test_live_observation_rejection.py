"""Real failure patterns: A/B rounded limits and unanchored hand detections."""
import unittest
import numpy as np
from unittest.mock import patch
from live_bimanual_stream import assign_candidates
from live_motion_scene import build_motion_scene
from run_pose2arm_live import AReferenceMotionController
from pose2arm_task_mapping import observed_sides, synthetic_pose_observation
from live_observation_quality import credible_hand,body_hand_consistency,ConfirmedHands
from human_task_robot import HumanTaskRobot
from human_grasp_task import TaskCommand
from member_b_scene import build_scene


class ObservationRejectionTests(unittest.TestCase):
    def setUp(self):
        self.pose=np.zeros((33,4));self.pose[:,3]=1.
        self.pose[11,:2]=[.65,.2];self.pose[12,:2]=[.35,.2]
        self.pose[13,:2]=[.7,.4];self.pose[14,:2]=[.3,.4]
        self.pose[15,:2]=[.75,.6];self.pose[16,:2]=[.25,.6]

    def test_missing_body_cannot_assign_a_torso_detection_using_its_label(self):
        candidate=dict(wrist=np.array([.5,.5]),score=.99,label='Left')
        self.assertEqual(assign_candidates([candidate],None,np.full((2,2),np.nan)),{})

    def test_unreliable_wrist_cannot_fall_back_to_old_hand_on_the_body(self):
        self.pose[16,3]=.29
        candidate=dict(wrist=np.array([.5,.5]),score=.99,label='Left')
        previous=np.array([[np.nan,np.nan],[.5,.5]])
        self.assertEqual(assign_candidates([candidate],self.pose,previous),{})

    def test_high_handedness_score_does_not_allow_a_170_pixel_wrist_error(self):
        candidate=dict(wrist=np.array([.62,.6]),score=.999,label='Left')
        self.assertEqual(assign_candidates([candidate],self.pose,np.full((2,2),np.nan)),{})

    def test_body_hand_disagreement_blocks_only_the_affected_body_side(self):
        obs=synthetic_pose_observation(1,.05)
        obs['pose_trusted']=dict(left=False,right=True)
        self.assertEqual(observed_sides(obs),dict(left=False,right=True))

    def hand(self,wrist):
        world=np.array([[0.,0.,0.]]+[[.02*((i%4)-1.5),.025*(i//4+1),.003*(i%3)] for i in range(20)])
        # Explicit nondegenerate palm landmarks; this is geometry, not an image.
        world[5]=[-.025,.04,0.];world[17]=[.025,.04,0.]
        norm=world.copy();norm[:,:2]+=wrist
        return dict(wrist=np.array(wrist),norm=norm,world=world,score=.99,label='Left')

    def test_collapsed_skin_patch_geometry_is_not_a_valid_hand(self):
        c=self.hand((.75,.6))
        self.assertTrue(credible_hand(c))
        c['norm'][:]=0.
        self.assertFalse(credible_hand(c))

    def test_two_measured_hands_reject_body_wrists_drifting_onto_chest(self):
        candidates=[self.hand((.75,.6)),self.hand((.25,.6))]
        trusted,_=body_hand_consistency(self.pose,candidates)
        self.assertEqual(trusted,dict(left=True,right=True))
        self.pose[15,:2]=[.5,.4]
        trusted,reasons=body_hand_consistency(self.pose,candidates)
        self.assertEqual(trusted,dict(left=False,right=True))
        self.assertIn('disagreement',reasons['left'])

    def test_new_hand_needs_two_distinct_frames_and_reacquisition_is_confirmed(self):
        gate=ConfirmedHands();c=self.hand((.75,.6))
        self.assertEqual(gate.update(dict(left=c),1.),{})
        self.assertEqual(gate.update(dict(left=c),1.),{})
        self.assertEqual(set(gate.update(dict(left=c),1.05)),{'left'})
        self.assertEqual(gate.update({},1.5),{})
        self.assertEqual(gate.update(dict(left=c),1.55),{})
        self.assertEqual(set(gate.update(dict(left=c),1.6)),{'left'})

    def test_original_a_rounded_boundary_does_not_crash_b(self):
        model,data=build_motion_scene();control=AReferenceMotionController(model,data)
        # Actual left joint 3 from source 219 of the user's failed recording.
        q=np.array([-.4758766304908949,.08043768678387249,1.5707999999999505,
                    2.3063312762118726,1.2233502101946518,.0868940942011501,0.])
        control.step(dict(joints=dict(left=q),fingers={}),dict(left=True,right=True))
        self.assertLessEqual(control.trajectories['left'].goal[2],control.specs['left'].q_range[2,1])

    def test_reference_brakes_before_b_joint_boundary_without_velocity_jump(self):
        model,data=build_motion_scene();control=AReferenceMotionController(model,data)
        trajectory=control.trajectories['left']
        limit=control.specs['left'].q_range[2,1]
        goal=trajectory.position.copy();goal[2]=limit
        trajectory.observe(goal,0.)
        highest=-np.inf;acceleration=0.;last=trajectory.velocity.copy()
        for _ in range(6000):
            q,v=trajectory.advance(.0005)
            highest=max(highest,q[2]);acceleration=max(acceleration,float(np.max(np.abs(v-last)/.0005)));last=v
        self.assertLessEqual(highest,limit+1e-10)
        self.assertLessEqual(acceleration,24.01)

    def test_nonfinite_or_gross_outlier_holds_bad_side_and_other_side_continues(self):
        for bad in (np.full(7,np.nan),np.full(7,99.)):
            model,data=build_motion_scene();control=AReferenceMotionController(model,data)
            old={s:q.copy() for s,q in control.joints.items()}
            control.step(dict(joints=dict(left=bad,right=old['right']+.05),fingers={}),dict(left=True,right=True))
            np.testing.assert_allclose(control.joints['left'],old['left'])
            self.assertGreater(np.linalg.norm(control.joints['right']-old['right']),.01)

    def test_task_reference_uses_same_rounded_boundary_conversion(self):
        model,data=build_scene()
        with patch('human_task_robot.BVisualObserver'):robot=HumanTaskRobot(model,data)
        self.addCleanup(robot.close)
        robot.reference_sides=('left','right');robot.visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.)
        q=robot.joints['left'].copy();q[2]=1.5708
        with patch.object(robot,'_check_reference_path'):
            robot.execute(TaskCommand('pose2arm_reference',joint_targets=dict(left=q),finger_targets={}),0.)
        self.assertIsNone(robot.error)
        self.assertLessEqual(robot.reference_trajectories['left'].goal[2],robot.control.specs['left'].q_range[2,1])


if __name__=='__main__':unittest.main()
