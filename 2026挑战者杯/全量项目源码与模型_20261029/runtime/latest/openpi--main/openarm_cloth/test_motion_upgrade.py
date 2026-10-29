"""Regressions for the user's v2 motion plan, exercising real mapping seams."""
import unittest
import numpy as np
import time
import json
import tempfile
from pathlib import Path
from scipy.spatial.transform import Rotation
from pose2arm_task_mapping import Pose2ArmTaskMapping,synthetic_pose_observation
from live_observation_quality import ConfirmedHands
from motion_observation import SourceClock,TorsoReference
from human_pinch_control import HumanPinchControl,PinchCalibration,pinch_features
from human_wrist_control import RelativePalmTracker,BWristRetargeter
from live_reference_conditioning import JointReferenceTrajectory
from live_motion_runtime import LivePhysicsWorker
from live_motion_scene import build_motion_scene
from run_pose2arm_live import AReferenceMotionController


def hand_fixture(angle=0.):
    hand=np.zeros((21,3))
    for base,x in ((1,-.04),(5,-.03),(9,0.),(13,.02),(17,.04)):
        for k in range(4):hand[base+k]=[x,.03+.02*k,0.]
    return hand @ Rotation.from_rotvec([0.,angle,0.]).as_matrix().T


class ExistingMotionGaps(unittest.TestCase):
    def test_palm_rotation_actually_reaches_wrist_joints(self):
        mapping=Pose2ArmTaskMapping()
        before=after=None
        for i in range(45):
            obs=synthetic_pose_observation(i,i*.05)
            obs['pose_raw']=synthetic_pose_observation(0,0.)['pose_raw']
            angle=0. if i<22 else min((i-22)*.04,.65)
            for arm in obs['arms'].values():arm['hand_raw']=hand_fixture(angle)
            result=mapping.process(obs)
            if i==21:before=result['joints']['right'].copy()
            if i==44:after=result['joints']['right'].copy()
        self.assertGreater(np.linalg.norm(after[4:]-before[4:]),.05)

    def test_duplicate_hand_source_never_resets_or_advances_confirmation(self):
        gate=ConfirmedHands();candidate=dict(wrist=np.array([.3,.4]))
        gate.update(dict(left=candidate),1.)
        gate.update(dict(left=candidate),1.05)
        count=gate.count['left']
        gate.update(dict(left=candidate),1.05)
        self.assertEqual(gate.count['left'],count)


class ObservationContractTests(unittest.TestCase):
    def test_source_clock_rejects_duplicate_reverse_and_large_gap_is_explicit(self):
        clock=SourceClock()
        self.assertTrue(clock.accept(1,1.)[0])
        self.assertFalse(clock.accept(1,1.1)[0]);self.assertFalse(clock.accept(2,.9)[0])
        self.assertEqual(clock.accept(3,1.6)[2],'SOURCE_GAP')

    def test_invisible_hips_cannot_rotate_calibrated_upper_body(self):
        torso=TorsoReference('upper_body')
        w=synthetic_pose_observation(0,0.)['pose_raw'];w[[23,24],3]=0.
        for i in range(10):accepted,info=torso.process(w,i*.05)
        first=info['rotation'].copy()
        w[[23,24],:3]=[[9.,-9.,6.],[-8.,4.,-3.]]
        accepted,info=torso.process(w,.55)
        self.assertTrue(info['valid']);self.assertTrue(info['virtual_hips'])
        np.testing.assert_allclose(first,info['rotation'],atol=1e-12)
        np.testing.assert_array_equal(accepted[[13,14,15,16]],w[[13,14,15,16]])

    def test_full_body_rejects_missing_hips_after_short_hold(self):
        torso=TorsoReference('full_body');w=synthetic_pose_observation(0,0.)['pose_raw']
        for i in range(11):torso.process(w,i*.05)
        w[[23,24],3]=0.
        self.assertTrue(torso.process(w,.6)[1]['valid'])
        self.assertFalse(torso.process(w,1.)[1]['valid'])

    def test_upper_body_large_turn_does_not_guess_hidden_spine(self):
        torso=TorsoReference('upper_body');w=synthetic_pose_observation(0,0.)['pose_raw']
        for i in range(10):torso.process(w,i*.05)
        w[:,:3]=w[:,:3]@Rotation.from_rotvec([0,np.pi/3,0]).as_matrix().T
        self.assertFalse(torso.process(w,.55)[1]['valid'])


class PinchCalibrationTests(unittest.TestCase):
    def profile(self):
        calibration=PinchCalibration()
        for _ in range(10):
            calibration.add_trial('closed',[dict(r3=.12,r2=.13,palm_scale_3d=.08) for _ in range(8)])
            calibration.add_trial('open',[dict(r3=1.15,r2=1.1,palm_scale_3d=.08) for _ in range(8)])
        return calibration

    def test_calibration_requires_trials_not_many_correlated_frames(self):
        c=PinchCalibration();c.add_trial('closed',[dict(r3=.1,r2=.1) for _ in range(100)])
        c.add_trial('open',[dict(r3=1.,r2=1.) for _ in range(100)])
        self.assertIsNone(c.profile)
        self.assertTrue(self.profile().profile)

    def test_overlapping_calibration_is_rejected(self):
        c=PinchCalibration()
        for _ in range(10):
            c.add_trial('closed',[dict(r3=.8,r2=.8) for _ in range(6)])
            c.add_trial('open',[dict(r3=.7,r2=.7) for _ in range(6)])
        self.assertIsNone(c.profile)

    def test_2d_3d_conflict_and_missing_hold_do_not_release(self):
        control=HumanPinchControl('binary',self.profile())
        for i in range(5):control.update(dict(r3=.1,r2=.1),i*.05)
        self.assertEqual(control.aperture,0.)
        control.update(None,.3);self.assertEqual(control.aperture,0.)
        control.update(dict(r3=.1,r2=1.2),.35)
        self.assertEqual(control.aperture,0.);self.assertIn('CONFLICT',control.reason)

    def test_scale_invariance_and_continuous_middle_aperture(self):
        hand=hand_fixture();a=pinch_features(hand);b=pinch_features(hand*2.2)
        self.assertAlmostEqual(a['r3'],b['r3'])
        control=HumanPinchControl(calibration=self.profile())
        for i in range(20):control.update(dict(r3=.6,r2=.6),i*.05)
        self.assertGreater(control.aperture,.01);self.assertLess(control.aperture,.03)

    def test_missing_frame_breaks_confirmation_evidence(self):
        control=HumanPinchControl('binary',self.profile())
        control.update(dict(r3=.1,r2=.1),0.)
        control.update(None,.05);control.update(dict(r3=.1,r2=.1),.1)
        self.assertEqual(control.aperture,.044)


class FullWristTests(unittest.TestCase):
    def test_missing_palm_retains_forearm_relative_rotation(self):
        tracker=RelativePalmTracker();upper=np.array([0,0,1.]);fore=np.array([1.,0,0])
        for i in range(4):frame,relative=tracker.update(upper,fore,hand_fixture(),np.eye(3),i*.05)
        before=relative.copy()
        frame,relative=tracker.update(upper,np.array([.9,.2,.1]),None,np.eye(3),.25)
        np.testing.assert_allclose(relative,before)
        self.assertEqual(tracker.state,'HOLD')

    def test_true_180_rotation_is_not_repaired_by_quaternion_sign(self):
        tracker=RelativePalmTracker();u=np.array([0,0,1.]);f=np.array([1.,0,0])
        for i in range(4):tracker.update(u,f,hand_fixture(),np.eye(3),i*.05)
        before=tracker.relative.copy()
        tracker.update(u,f,hand_fixture(np.pi),np.eye(3),.2)
        np.testing.assert_allclose(tracker.relative,before)
        self.assertIn('OUTLIER',tracker.reason)
        r=Rotation.from_rotvec([.2,.3,.4]);np.testing.assert_allclose(r.as_matrix(),Rotation.from_quat(-r.as_quat()).as_matrix())

    def test_straight_elbow_frames_are_proper_rotations(self):
        tracker=RelativePalmTracker()
        for f in ([1,0,0],[0,1,0],[0,0,1]):
            frame=tracker.forearm_frame(f,f)
            np.testing.assert_allclose(frame.T@frame,np.eye(3),atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(frame),1.)

    def test_b_wrist_solver_moves_all_three_axes_without_replacing_a_shoulder_elbow(self):
        model,data=build_motion_scene();solver=BWristRetargeter(model)
        side='right';nominal=data.qpos[solver.specs[side].qpos].copy()
        solver.solve(side,nominal,np.eye(3),np.eye(3))
        for axis in np.eye(3):
            q=solver.solve(side,nominal,Rotation.from_rotvec(axis*.2).as_matrix(),np.eye(3))
            np.testing.assert_array_equal(q[:4],nominal[:4])
            self.assertGreater(np.linalg.norm(q[4:]-nominal[4:]),.02)
            self.assertLess(solver.metrics[side]['rotation_residual_rad'],.15)


class ExecutionBudgetTests(unittest.TestCase):
    def test_responsive_profile_avoids_duplicate_reference_filter_lag(self):
        legacy=JointReferenceTrajectory(np.zeros(7));fast=JointReferenceTrajectory(np.zeros(7),frequency=30.,filter_enabled=False)
        old=[];new=[];truth=[]
        for i in range(160):
            wanted=np.full(7,.15*np.sin(i*.05*3.))
            legacy.observe(wanted,i*.05);fast.observe(wanted,i*.05)
            for _ in range(20):legacy.advance(.0025);fast.advance(.0025)
            old.append(legacy.position[0]);new.append(fast.position[0]);truth.append(wanted[0])
        self.assertLess(np.mean((np.array(new[30:])-truth[30:])**2),.6*np.mean((np.array(old[30:])-truth[30:])**2))

    def test_slow_ui_does_not_stop_physics_worker_or_replay_old_queue(self):
        model,data=build_motion_scene();control=AReferenceMotionController(model,data,responsive=True)
        worker=LivePhysicsWorker(control);self.addCleanup(worker.close)
        result=dict(seq=1,capture_ns=time.perf_counter_ns(),joints={s:q+.08 for s,q in control.joints.items()},fingers={})
        worker.submit(result,dict(left=True,right=True))
        time.sleep(.32)
        self.assertIsNone(worker.error);self.assertGreater(worker.snapshot().tick,3)
        first=worker.snapshot().tick;time.sleep(.1)
        self.assertGreater(worker.snapshot().tick,first)
        self.assertTrue(np.isfinite(worker.snapshot().qpos).all())


class CalibrationAndEvidenceTests(unittest.TestCase):
    def test_arm_recovers_after_source_gap_with_two_new_consistent_frames(self):
        mapping=Pose2ArmTaskMapping()
        fixed=synthetic_pose_observation(0,0.)['pose_raw']
        for i in range(18):
            obs=synthetic_pose_observation(i,i*.05);obs['pose_raw']=fixed
            mapping.process(obs)
        obs=synthetic_pose_observation(18,1.5);obs['pose_raw']=fixed
        self.assertFalse(mapping.process(obs)['joints'])
        obs=synthetic_pose_observation(19,1.55);obs['pose_raw']=fixed
        self.assertEqual(set(mapping.process(obs)['joints']),{'left','right'})

    def test_recalibrating_body_keeps_gripper_aperture(self):
        mapping=Pose2ArmTaskMapping();mapping.fingers={'left':.004,'right':.014}
        mapping.reset()
        self.assertEqual(mapping.grippers['left'].aperture,.004)
        self.assertEqual(mapping.grippers['right'].aperture,.014)
        result=mapping.process(synthetic_pose_observation(0,0.))
        self.assertFalse(result['finger_valid']['left'])
        self.assertEqual(result['fingers']['left'],.004)

    def test_pinch_remains_independent_when_body_is_missing(self):
        mapping=Pose2ArmTaskMapping()
        for i in range(8):
            obs=synthetic_pose_observation(i,i*.05);obs['pose_raw']=None
            for arm in obs['arms'].values():arm['hand_raw']=hand_fixture()
            result=mapping.process(obs)
        self.assertFalse(result['joints'])
        self.assertTrue(result['finger_valid']['right'])
        self.assertEqual(result['module_state']['right']['palm'],'LOST')

    def test_failed_new_personal_fit_does_not_keep_old_profile(self):
        c=PinchCalibrationTests().profile();self.assertIsNotNone(c.profile)
        for _ in range(12):c.add_trial('closed',[dict(r3=1.15,r2=1.1) for _ in range(8)])
        self.assertIsNone(c.profile);self.assertEqual(c.calibration_id,'uncalibrated')

    def test_independent_accuracy_is_not_inferred_from_missing_reference(self):
        from evaluate_motion_v2 import evaluate_reference,lower_success_bound
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(evaluate_reference(folder,{})['verdict'],'NOT_EVALUABLE')
        self.assertGreater(lower_success_bound(400,400),.99)
        self.assertLess(lower_success_bound(399,400),.99)
        self.assertLess(lower_success_bound(990,1000),.99)

    def test_nonfinite_independent_uncertainty_cannot_pass(self):
        from evaluate_motion_v2 import evaluate_reference
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'dataset').mkdir();(root/'physics_trace.npz').touch()
            (root/'dataset'/'manifest.json').write_text(json.dumps(dict(session_id='regression')))
            reference=dict(session_id='regression',independent_measurement=True,reviewers=['a','b'],
                coordinate_frame='B_model_world',time_basis='host_monotonic_ns',
                reference_uncertainty_deg=float('nan'),time_uncertainty_ms=1.)
            self.assertEqual(evaluate_reference(root,reference)['verdict'],'NOT_EVALUABLE')


if __name__=='__main__':unittest.main()
