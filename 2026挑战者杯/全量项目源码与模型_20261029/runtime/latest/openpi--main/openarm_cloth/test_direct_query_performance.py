"""Direct queries may omit dynamics, never geometry or collision decisions."""
import unittest
from unittest.mock import patch
import numpy as np
import mujoco
from member_b_scene import build_scene
from human_task_robot import HumanTaskRobot,pad_points
from handeye_grasp_red_cube import solve_gripper_contact_proxy_q
from demo_grasp_viewer import PAD_CONTACT_PROXY_LOCAL
from demonstration_transfer import visual_planning_state


class DirectQueryTests(unittest.TestCase):
    def setUp(self):
        self.model,self.data=build_scene()
        with patch('human_task_robot.BVisualObserver'):
            self.robot=HumanTaskRobot(self.model,self.data)
        self.addCleanup(self.robot.close)
        self.robot.initialize()
        self.robot.visual=dict(center=np.array([.43,0.,.429]),yaw=.3,time_s=0.)

    def test_direct_collision_query_avoids_force_solver_and_matches_full_contacts(self):
        r=self.robot;q=self.data.qpos[r.spec.qpos].copy()
        r.fast_query_mode=False;expected=r._contacts(q,.01,False,False,False)
        r.fast_query_mode=True
        with patch('human_task_robot.mujoco.mj_forward',side_effect=AssertionError('query ran dynamics')):
            actual=r._contacts(q,.01,False,False,False)
        self.assertEqual(actual,expected)

    def test_fast_contacts_match_baseline_including_penetrations_and_payload(self):
        r=self.robot;base=self.data.qpos[r.spec.qpos].copy();hits=0
        for attached in (False,True):
            r.attachment=np.array([0.,0.,-.03]);r.attachment_rotation=np.eye(3)
            for delta in (0.,.2,-.4):
                q=base.copy();q[1]+=delta
                for finger in (0.,.02,.044):
                    r.fast_query_mode=False;a=r._contacts(q,finger,attached,False,False)
                    r.fast_query_mode=True;b=r._contacts(q,finger,attached,False,False)
                    self.assertEqual(a.keys(),b.keys());hits+=len(a)
                    for key in a:self.assertAlmostEqual(a[key],b[key],places=10)
        self.assertGreater(hits,0,'Fixture must exercise contact depths')

    def test_reused_ik_state_refreshes_encoders_and_uses_only_camera_object_pose(self):
        r=self.robot;r.fast_query_mode=True
        first=r._query_state()
        self.data.qpos[r.spec.qpos[0]]+=.03
        self.data.qpos[r.object_address:r.object_address+3]=[2.,2.,2.]
        second=r._query_state()
        self.assertIs(first,second)
        reference=visual_planning_state(self.model,self.data,r.visual['center'],r.visual['yaw'])
        np.testing.assert_allclose(second.qpos,reference.qpos)
        np.testing.assert_allclose(second.geom_xpos,reference.geom_xpos,atol=1e-12)

    def test_kinematic_residual_matches_full_solver_and_restores_state(self):
        r=self.robot;state=r._query_state();saved=state.qpos.copy()
        points=pad_points(self.model,self.data,'right');direction=points[0]-points[1]
        common=dict(model=self.model,data=state,spec=r.spec,finger_geom_ids=r.geoms,
            finger_q_indices=r.fingers,contact_proxy_local=PAD_CONTACT_PROXY_LOCAL,
            target_center=points.mean(0)+[0.,0.,.002],axis_index=1,axis_sign=1.,
            required_span=float(np.linalg.norm(direction)),finger_q=.044,seed=r.joints['right'],
            reference_rotation=r.rotation,multi_start=False,closing_direction=direction)
        a=solve_gripper_contact_proxy_q(**common)
        b=solve_gripper_contact_proxy_q(**common,kinematics_only=True)
        np.testing.assert_allclose(a[0],b[0],atol=1e-8)
        for key in a[3]:self.assertAlmostEqual(a[3][key],b[3][key],places=9)
        np.testing.assert_array_equal(state.qpos,saved)


if __name__=='__main__':unittest.main()
