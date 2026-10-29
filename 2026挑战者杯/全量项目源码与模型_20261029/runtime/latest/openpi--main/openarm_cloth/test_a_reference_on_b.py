"""Reference reuse, B motion-only model and shared A native replay controls."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from a_reference_adapter import AReference,load_reference,resample_reference,verify_kinematics
from live_motion_scene import build_motion_scene
from teleop_towel_dataset import build_arm_specs
from handeye_grasp_red_cube import finger_joint_indices
from replay_controls import ReplayControls
from replay_video_fusion import add_reference_skeletons
from run_a_reference_on_b import apply_kinematic_frame,simulate_reference,tracking_metrics


def blob():
    return dict(joint_angles=np.vstack([np.zeros((3,14)),np.ones((3,14))]),
                target_obs=np.zeros((6,18)),frame_indices=np.tile(np.arange(3),2),fps=np.array(30.),
                gripper_position=np.tile([.2,.8],(6,1)),gripper_order=np.array('right-first'),
                source_arm_trusted=np.array([[False,True],[True,True],[True,True]]))


class ReferenceContractTests(unittest.TestCase):
    def test_original_clip_never_runs_into_mirror_and_grip_is_left_right(self):
        sampled,q,closure,trusted=resample_reference(blob())
        self.assertLessEqual(sampled['source_frame_indices'].max(),2)
        self.assertTrue(np.all(q==0));np.testing.assert_allclose(closure[0],[.8,.2])
        self.assertFalse(trusted[0,0])

    def test_native_clock_preserves_every_A_source_frame(self):
        sampled,q,closure,trusted=resample_reference(blob(),hz=None)
        np.testing.assert_array_equal(sampled['source_frame_indices'],[0,1,2])

    def test_different_video_is_rejected_before_loading_reference(self):
        with patch('a_reference_adapter.file_hash',side_effect=['wrong-reference','wrong-video']):
            with self.assertRaisesRegex(ValueError,'unbound'):load_reference('other.npz','other.mp4')

    def test_step_pause_restart_loop_and_speed_limits(self):
        c=ReplayControls(5);c.key(262);self.assertEqual(c.index,1);self.assertTrue(c.paused)
        c.advance();self.assertEqual(c.index,1);c.key(263);self.assertEqual(c.index,0)
        c.key(32);c.index=4;c.advance();self.assertEqual(c.index,0)
        for _ in range(20):c.key(ord('+'))
        self.assertEqual(c.speed,4.)
        for _ in range(20):c.key(ord('-'))
        self.assertEqual(c.speed,.125);c.key(ord('g'));self.assertFalse(c.ghosts)
        c.key(ord('q'));self.assertTrue(c.quit)


class BMotionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model,cls.data=build_motion_scene();cls.specs=build_arm_specs(cls.model)

    def test_motion_only_model_contains_neither_table_nor_object(self):
        for name in ('folding_table','droid_object_geom'):
            self.assertEqual(mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name),-1)

    def test_A_canonical_geometry_matches_current_B_robot(self):
        self.assertLess(max(verify_kinematics(self.model,self.specs).values()),1e-5)

    def test_changed_B_geometry_cannot_silently_reuse_A_joint_values(self):
        model=copy.copy(self.model);model.body('openarm_right_link5').pos[0]+=.02
        with self.assertRaisesRegex(ValueError,'geometry differs'):verify_kinematics(model,build_arm_specs(model))

    def test_frame_zero_and_independent_B_slide_fingers_are_not_overridden(self):
        data=mujoco.MjData(self.model);data.qpos[:]=self.data.qpos
        q=np.r_[data.qpos[self.specs['left'].qpos],data.qpos[self.specs['right'].qpos]]
        apply_kinematic_frame(self.model,data,self.specs,q,[.25,.75])
        for j,side in enumerate(('left','right')):
            np.testing.assert_array_equal(data.qpos[self.specs[side].qpos],q[7*j:7*(j+1)])
            np.testing.assert_allclose(data.qpos[finger_joint_indices(self.model,side)[0]],.044*(1-[.25,.75][j]))

    def test_missing_source_confidence_does_not_drop_A_existing_reference(self):
        data=mujoco.MjData(self.model);data.qpos[:]=self.data.qpos
        q=np.r_[data.qpos[self.specs['left'].qpos],data.qpos[self.specs['right'].qpos]]
        targets=np.tile(q,(3,1));targets[:,0]+=[0.,.01,.02]
        ref=AReference(targets,np.tile([.2,.8],(3,1)),np.arange(3)/30,np.arange(3),
                       np.zeros((3,2),bool),30.,Path('x'),'', '')
        arrays,report=simulate_reference(self.model,data,ref)
        addresses=np.r_[self.specs['left'].qpos,self.specs['right'].qpos]
        np.testing.assert_array_equal(arrays['qpos'][:,addresses],targets)
        self.assertEqual(float(arrays['fps']),30.)
        self.assertEqual(report['max_joint_error_rad'],0.)
        self.assertTrue(np.all(arrays['avoidance_mode']=='disabled_motion_only'))

    def test_reference_overlay_is_display_only(self):
        state=self.data.qpos.copy();q=np.r_[state[self.specs['left'].qpos],state[self.specs['right'].qpos]]
        scene=mujoco.MjvScene(self.model,maxgeom=100)
        add_reference_skeletons(scene,self.model,state,dict(a_reference_joint=q[None],joint_target=q[None]),0)
        self.assertEqual(scene.ngeom,8);np.testing.assert_array_equal(state,self.data.qpos)
        np.testing.assert_allclose(tracking_metrics(self.model,self.specs,state[None],q[None]),0.,atol=1e-5)


if __name__=='__main__':unittest.main()
