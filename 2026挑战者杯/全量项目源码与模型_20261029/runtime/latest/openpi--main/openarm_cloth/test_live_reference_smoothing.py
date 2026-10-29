"""Regressions at the real A mapping -> B dynamic control boundary."""
import unittest
from unittest.mock import patch
import numpy as np

from live_motion_scene import build_motion_scene
from pose2arm_task_mapping import Pose2ArmTaskMapping, synthetic_pose_observation
from run_pose2arm_live import AReferenceMotionController
from live_reference_conditioning import AdaptiveReferenceFilter, ContinuousPinch, JointReferenceTrajectory
from run_human_task import pose2arm_reference_command


class LiveSmoothingTests(unittest.TestCase):
    def test_task_dispatch_keeps_intermediate_apertures_and_holds_them_on_source_loss(self):
        fingers=dict(left=.013,right=.027)
        result=dict(joints=dict(right=np.zeros(7)),fingers=fingers,capture_ns=1000000000)
        command=pose2arm_reference_command(result,dict(left=.044,right=0.))
        self.assertEqual(command.finger,.027)
        self.assertEqual(command.finger_targets,fingers)
        self.assertEqual(pose2arm_reference_command(None,fingers).finger,.027)

    def test_duplicate_source_does_not_advance_reference_filter(self):
        smoother=AdaptiveReferenceFilter()
        smoother.update(np.zeros(7),1.)
        once=smoother.update(np.ones(7),1.05)
        for _ in range(10):np.testing.assert_array_equal(smoother.update(np.ones(7),1.05),once)

    def test_missing_hand_holds_continuous_aperture_and_other_hand_is_independent(self):
        left,right=ContinuousPinch(),ContinuousPinch()
        a=left.update(.6,1.)
        right.update(1.2,1.)
        self.assertGreater(a,.01)
        self.assertLess(a,.03)
        self.assertEqual(left.update(None,1.1),a)
        self.assertEqual(left.update(float('nan'),1.2),a)
        self.assertEqual(right.aperture,.044)

    def test_small_stationary_noise_is_attenuated_without_hiding_an_intentional_step(self):
        trajectory=JointReferenceTrajectory(np.zeros(7))
        outputs=[]
        for i in range(160):
            goal=np.full(7,np.deg2rad(2)*(-1)**i)
            trajectory.observe(goal,i*.05)
            for _ in range(100):trajectory.advance(.0005)
            outputs.append(trajectory.position[0])
        self.assertLess(np.std(outputs[60:]),np.deg2rad(.5))
        trajectory=JointReferenceTrajectory(np.zeros(7))
        trajectory.observe(np.full(7,np.deg2rad(20)),0.)
        for _ in range(600):trajectory.advance(.0005)
        self.assertGreater(trajectory.position[0],.9*np.deg2rad(20))

    def test_hand_distance_produces_continuous_steady_apertures(self):
        mapping = Pose2ArmTaskMapping()
        values = []
        seq = 0
        for ratio in (.15, .35, .55, .75, .95, 1.2):
            for _ in range(30):
                obs = synthetic_pose_observation(seq, seq * .05)
                for arm in obs['arms'].values():
                    arm['hand_raw'][8, 0] = .08 * ratio
                result = mapping.process(obs)
                seq += 1
            values.append(result['fingers']['right'])
        self.assertTrue(np.all(np.diff(values) > .005), values)
        self.assertLess(values[0], .001)
        self.assertGreater(values[-1], .043)

    def test_feedforward_acceleration_is_bounded_inside_physics_steps(self):
        model, data = build_motion_scene()
        control = AReferenceMotionController(model, data)
        initial = {s:q.copy() for s,q in control.joints.items()}
        velocities = [np.zeros(14)]
        apply = control.control.apply
        def sample():
            velocities.append(control.control.target_velocity.copy())
            apply()
        with patch.object(control.control, 'apply', side_effect=sample):
            for i in range(18):
                offset = .35 if i < 8 else -.12
                result = dict(seq=i, capture_ns=int((1+i*.05)*1e9),
                              joints={s:q+offset for s,q in initial.items()}, fingers={})
                control.step(result, dict(left=True, right=True))
        acceleration = np.abs(np.diff(velocities, axis=0)) / model.opt.timestep
        self.assertLessEqual(acceleration.max(), 24.01)
        self.assertGreater(np.linalg.norm(control.joints['left']-initial['left']), .1)

    def test_tracking_loss_does_not_jump_target_to_lagging_encoders(self):
        model, data = build_motion_scene()
        control = AReferenceMotionController(model, data)
        initial = {s:q.copy() for s,q in control.joints.items()}
        for i in range(8):
            control.step(dict(seq=i, capture_ns=int((1+i*.05)*1e9),
                              joints={s:q+.4 for s,q in initial.items()}, fingers={}),
                         dict(left=True, right=True))
        before = control.joints['right'].copy()
        data.qpos[control.specs['right'].qpos] -= .3
        control.step(None, dict(left=True, right=False))
        self.assertLess(np.max(np.abs(control.joints['right']-before)), .06)


if __name__ == '__main__': unittest.main()
