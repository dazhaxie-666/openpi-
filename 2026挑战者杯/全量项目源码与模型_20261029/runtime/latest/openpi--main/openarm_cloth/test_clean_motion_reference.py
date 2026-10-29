"""Lock down pure imitation, missing-input holds and wrist branch continuity."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from a_reference_adapter import DEFAULT_REFERENCE
from live_motion_scene import build_motion_scene
from clean_motion_reference import rebuild_motion,bone_directions,WristContinuity,image_plane_targets,bounded_wrist_interval
from ik_solver import OpenArmIK
from member_b_actuators import ActuatorController
from run_a_reference_on_b import MotionFeedforwardController,simulate_reference
from a_reference_adapter import AReference
from pathlib import Path
import mujoco


def fixture(n=12):
    obs=np.zeros((n,18))
    for side,j in (('left',0),('right',1)):
        ik=OpenArmIK(side)
        for i in range(n):
            q=np.array([.2,-.4 if side=='left' else .4,.1,.8+.1*np.sin(i/8),0,0,0])
            u,f=bone_directions(ik,q)
            obs[i,9*j:9*j+9]=np.r_[u,f,np.cross(u,f)/np.linalg.norm(np.cross(u,f))]
    return dict(target_obs=obs,joint_angles=np.ones((n,14)),nominal_joint_angles=np.zeros((n,14)),
                frame_indices=np.arange(n),fps=np.array(30.),gripper_position=np.zeros((n,2)),
                source_arm_trusted=np.ones((n,2),bool))


class CleanReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model,_=build_motion_scene()

    def test_legacy_avoidance_joint_values_do_not_drive_new_motion(self):
        blob=fixture();other={k:v.copy() for k,v in blob.items()};other['joint_angles'][:]=-1.
        first,_=rebuild_motion(blob,self.model);second,_=rebuild_motion(other,self.model)
        np.testing.assert_allclose(first['joint_angles'],second['joint_angles'],atol=1e-10)

    def test_untrusted_tail_holds_entire_arm_instead_of_wrist_spin(self):
        blob=fixture();blob['source_arm_trusted'][-4:,0]=False
        blob['target_obs'][-4:,:9]*=-1
        clean,_=rebuild_motion(blob,self.model)
        np.testing.assert_allclose(clean['joint_angles'][-4:,:7],np.tile(clean['joint_angles'][-5,:7],(4,1)))
        self.assertTrue(np.any(np.ptp(clean['joint_angles'][-4:,7:11],axis=0)>0))

    def test_no_hand_pose_means_no_invented_wrist_rotation(self):
        clean,report=rebuild_motion(fixture(),self.model)
        np.testing.assert_array_equal(clean['joint_angles'][:,[4,5,6,11,12,13]],0.)
        self.assertLess(np.max(report['new_trusted_direction_mean_deg']),2.)

    def test_isolated_hand_rotation_flip_is_not_accepted(self):
        f=WristContinuity();identity=np.eye(3)
        f.update(identity,.033)
        result=f.update(Rotation.from_euler('z',170,degrees=True).as_matrix(),.033)
        self.assertLess(Rotation.from_matrix(result).magnitude(),1e-8)
        self.assertEqual(f.rejected,1)

    def test_wrist_target_has_explicit_angular_speed_bound(self):
        f=WristContinuity();f.update(np.eye(3),.033)
        result=f.update(Rotation.from_euler('z',80,degrees=True).as_matrix(),.033)
        self.assertLessEqual(Rotation.from_matrix(result).magnitude(),4.*.033+1e-8)

    def test_2d_fallback_uses_image_motion_not_nonmetric_model_xyz(self):
        blob=fixture(2);blob['source_arm_trusted'][:]=False
        xy=np.zeros((2,133,2));xy[:,[5,6,11,12]]=[[[800,400],[300,400],[750,900],[350,900]]]
        xy[:,7]=[850,650];xy[:,9]=[850,850];xy[:,8]=[250,650];xy[:,10]=[250,850]
        rt=dict(image_xy=xy,score=np.ones((2,133)),frame_width=np.array(1080),frame_height=np.array(1920),model_xyz=np.ones((2,133,3))*1e10)
        a,valid,modes=image_plane_targets(rt,np.arange(2),blob['target_obs'],blob['source_arm_trusted'])
        rt['model_xyz']*=-1
        b,_,_=image_plane_targets(rt,np.arange(2),blob['target_obs'],blob['source_arm_trusted'])
        np.testing.assert_array_equal(a,b);self.assertTrue(valid.all())
        self.assertTrue(np.all(modes=='rtmw_2d_depth_held'))
        np.testing.assert_allclose(a[:,[0,3,9,12]],0.)

    def test_bad_2d_fallback_does_not_invent_observations(self):
        blob=fixture(2);blob['source_arm_trusted'][:]=False
        rt=dict(image_xy=np.zeros((2,133,2)),score=np.zeros((2,133)),frame_width=1080,frame_height=1920)
        _,valid,modes=image_plane_targets(rt,np.arange(2),blob['target_obs'],blob['source_arm_trusted'])
        self.assertFalse(valid.any());self.assertTrue(np.all(modes=='held'))

    def test_wrist_acceleration_envelope_cannot_make_joint_limit_bounds_infeasible(self):
        limits=np.array([[-1.,1.]]*3);seed=np.array([.99999,-.99999,0.]);velocity=np.array([4.,-4.,0.])
        lower,higher=bounded_wrist_interval(seed,velocity,limits,.033)
        self.assertTrue(np.all(lower<higher));self.assertTrue(np.all(lower>=-1.));self.assertTrue(np.all(higher<=1.))

    @unittest.skipUnless(DEFAULT_REFERENCE.is_file(),'A fixture unavailable')
    def test_original_video_tail_does_not_keep_generating_joint_five_flips(self):
        with np.load(DEFAULT_REFERENCE,allow_pickle=False) as z:blob={k:z[k] for k in z.files}
        clean,report=rebuild_motion(blob,self.model)
        trusted=clean['source_arm_trusted'];q=clean['joint_angles']
        for j in range(2):
            for i in range(1,len(q)):
                if not trusted[i,j]:np.testing.assert_allclose(q[i,7*j:7*(j+1)],q[i-1,7*j:7*(j+1)],atol=1e-10)
        self.assertLess(np.mean(report['new_trusted_direction_mean_deg']),np.mean(report['old_trusted_direction_mean_deg']))


class MotionControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model,cls.initial=build_motion_scene()

    def test_zero_velocity_feedforward_is_exactly_B_original_controller(self):
        a=mujoco.MjData(self.model);a.qpos[:]=self.initial.qpos;a.qvel[:]=.1;mujoco.mj_forward(self.model,a)
        b=mujoco.MjData(self.model);b.qpos[:]=a.qpos;b.qvel[:]=a.qvel;mujoco.mj_forward(self.model,b)
        original=ActuatorController(self.model,a);motion=MotionFeedforwardController(self.model,b)
        original.apply();motion.apply();np.testing.assert_array_equal(a.ctrl,b.ctrl)

    def test_velocity_feedforward_never_bypasses_B_torque_limits(self):
        data=mujoco.MjData(self.model);data.qpos[:]=self.initial.qpos;mujoco.mj_forward(self.model,data)
        motion=MotionFeedforwardController(self.model,data);motion.target_velocity[:]=1e6;motion.apply()
        self.assertTrue(np.all(data.ctrl[motion.actuators]<=motion.limits[:,1]))
        self.assertTrue(np.all(data.ctrl[motion.actuators]>=motion.limits[:,0]))

    def test_continuous_motion_reduces_lag_without_changing_the_source_clock(self):
        base=ActuatorController(self.model,self.initial);q=np.tile(self.initial.qpos[base.qpos],(20,1))
        q[:,0]+=.4*np.arange(20)/20
        reference=AReference(q,np.zeros((20,2)),np.arange(20)/20,np.arange(20),np.ones((20,2),bool),20.,Path('x'),'','')
        results={}
        for tracking in ('step','continuous'):
            data=mujoco.MjData(self.model);data.qpos[:]=self.initial.qpos
            arrays,report=simulate_reference(self.model,data,reference,'dynamic',tracking)
            results[tracking]=report['mean_joint_error_deg']
            np.testing.assert_array_equal(arrays['source_time_s'],reference.source_time)
        self.assertLess(results['continuous'],.6*results['step'])


if __name__=='__main__':unittest.main()
