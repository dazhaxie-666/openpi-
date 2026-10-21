"""Human-source, object-relative transfer and B-scene isolation regressions."""
from pathlib import Path
import unittest
import numpy as np

from video_demonstration import VideoDemonstration
from demonstration_transfer import visual_planning_state, sample_joint_path
from member_b_scene import build_scene
from teleop_towel_dataset import build_arm_specs


def demonstration(curvature=.02, close=True):
    time = np.arange(80) / 30
    wrist = np.column_stack([.15*time, curvature*np.sin(time*4), .04*time])
    closure = np.r_[np.zeros(35), np.full(45, .94 if close else 0.)]
    return VideoDemonstration(time, wrist, closure, np.ones(80, bool),
                              np.ones(80, bool), np.full(80, .6), {})


class VideoFusionTests(unittest.TestCase):
    def test_missing_close_never_runs_autonomous_grasp_as_fusion(self):
        with self.assertRaisesRegex(ValueError, 'open.*close'):
            demonstration(close=False)

    def test_missing_hand_cannot_authorize_closure(self):
        demo = demonstration()
        with self.assertRaises(ValueError):
            VideoDemonstration(demo.time, demo.wrist, demo.closure, demo.pose_valid,
                               np.zeros(80, bool), demo.arm_length, {})

    def test_invalid_timestamp_and_nonfinite_valid_pose_rejected(self):
        demo = demonstration()
        for time, wrist in ((demo.time[::-1], demo.wrist),
                            (demo.time, np.full_like(demo.wrist, np.nan))):
            with self.assertRaises(ValueError):
                VideoDemonstration(time, wrist, demo.closure, demo.pose_valid,
                                   demo.hand_valid, demo.arm_length, {})

    def test_visual_translation_relocates_whole_human_path(self):
        demo = demonstration()
        a = demo.descent_targets(np.array([.43,0,.43]), .3)
        shift = np.array([.025,-.04,0])
        b = demo.descent_targets(np.array([.43,0,.43])+shift, .3)
        np.testing.assert_allclose(b-a, np.tile(shift,(len(a),1)), atol=1e-12)
        np.testing.assert_allclose(a[-1],[.43,0,.43],atol=1e-12)

    def test_human_shape_changes_motion_but_not_grasp_anchor(self):
        a = demonstration(.03).descent_targets(np.array([.43,0,.43]),0)
        b = demonstration(0).descent_targets(np.array([.43,0,.43]),0)
        self.assertGreater(np.max(np.abs(a-b)),.0001)
        np.testing.assert_allclose(a[[0,-1]],b[[0,-1]],atol=1e-12)

    def test_object_yaw_rotates_bounded_human_offsets(self):
        demo = demonstration()
        center = np.array([.43,0,.43])
        a = demo.descent_targets(center,0)
        b = demo.descent_targets(center,np.pi/2)
        np.testing.assert_allclose(b[:,0]-center[0],-(a[:,1]-center[1]),atol=1e-12)
        np.testing.assert_allclose(b[:,1]-center[1],a[:,0]-center[0],atol=1e-12)
        self.assertLessEqual(np.max(np.linalg.norm(a[:,:2]-center[:2],axis=1)),.012001)

    def test_long_pose_gap_is_not_bridged_as_a_demonstration(self):
        demo = demonstration()
        valid = demo.pose_valid.copy();valid[10:30]=False
        with self.assertRaises(ValueError):
            VideoDemonstration(demo.time,demo.wrist,demo.closure,valid,
                               demo.hand_valid,demo.arm_length,{})

    def test_smooth_path_has_zero_endpoint_velocity_and_respects_rate_cap(self):
        anchors=np.array([[0]*7,[.1,-.1,0,0,0,0,0],[.2,0,0,0,0,0,0]])
        time,q=sample_joint_path(anchors,1.5,.001)
        np.testing.assert_allclose(q[[0,-1]],anchors[[0,-1]],atol=1e-12)
        self.assertLess(np.max(np.abs(q[1]-q[0]))/.001,.01)
        self.assertLess(np.max(np.abs(q[-1]-q[-2]))/.001,.01)
        self.assertLessEqual(np.max(np.abs(np.diff(q,axis=0))/.001),2.5+1e-6)

    def test_joint_interpolation_never_overshoots_near_a_limit(self):
        anchors=np.repeat(np.array([0.,.9,1.,1.])[:,None],7,axis=1)
        _,q=sample_joint_path(anchors,1.5,.001)
        self.assertGreaterEqual(float(np.min(q)),0.)
        self.assertLessEqual(float(np.max(q)),1.)


class PlanningIsolationTests(unittest.TestCase):
    def test_only_visual_estimate_and_robot_encoders_enter_planning_state(self):
        model,data=build_scene()
        center=np.array([.43,0,.43])
        a=visual_planning_state(model,data,center,.2)
        before=data.qpos.copy()
        adr=int(model.joint('droid_object_freejoint').qposadr[0])
        data.qpos[adr:adr+3]=[.9,.9,.9]
        b=visual_planning_state(model,data,center,.2)
        np.testing.assert_allclose(a.qpos,b.qpos)
        c=visual_planning_state(model,data,center+[.02,0,0],.2)
        self.assertAlmostEqual(c.qpos[adr]-b.qpos[adr],.02)
        np.testing.assert_allclose(data.qpos[adr:adr+3],[.9,.9,.9])


if __name__=='__main__': unittest.main()
