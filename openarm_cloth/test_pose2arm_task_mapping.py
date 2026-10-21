"""A's real online calibration/retargeting is the task's reference generator."""
import time
import threading
import unittest
from unittest.mock import patch
import numpy as np

from pose2arm_task_mapping import Pose2ArmTaskMapping, LatestPose2ArmWorker, synthetic_pose_observation
from openarm_live import OpenArmLiveController


class MappingTests(unittest.TestCase):
    def test_uses_original_a_controller_and_matches_its_calibration_and_joint_output(self):
        mapping = Pose2ArmTaskMapping()
        original = {s: OpenArmLiveController(20., n_calib_frames=10, side=s) for s in ('left','right')}
        for i in range(16):
            obs = synthetic_pose_observation(i, i*.05)
            result = mapping.process(obs)
            for s in original:
                q, status = original[s].process(obs['pose_raw'])
                if q is not None and status == 'tracking':
                    np.testing.assert_allclose(result['joints'][s], q, atol=1e-12)
        for s in original:
            self.assertIsInstance(mapping.controllers[s], OpenArmLiveController)
            self.assertAlmostEqual(result['scales'][s], original[s].retargeter.scale)
            self.assertEqual(result['calibration_frames'][s], 10)

    def test_duplicate_and_out_of_order_frames_never_advance_calibration(self):
        mapping = Pose2ArmTaskMapping()
        obs = synthetic_pose_observation(10, 1.)
        mapping.process(obs)
        for _ in range(15): mapping.process(obs)
        mapping.process(synthetic_pose_observation(9, .9))
        self.assertEqual(len(mapping.controllers['right'].calib_frames), 1)

    def test_missing_right_does_not_disable_observed_left(self):
        mapping = Pose2ArmTaskMapping()
        for i in range(14):
            obs = synthetic_pose_observation(i, i*.05)
            obs['pose_raw'][16,3] = .05
            result = mapping.process(obs)
        self.assertIn('left', result['joints'])
        self.assertNotIn('right', result['joints'])
        self.assertIsNone(mapping.controllers['right'].retargeter.scale)

    def test_recalibration_discards_previous_scale(self):
        mapping = Pose2ArmTaskMapping()
        for i in range(12): mapping.process(synthetic_pose_observation(i, i*.05))
        mapping.reset()
        self.assertIsNone(mapping.controllers['right'].retargeter.scale)

    def test_worker_publishes_latest_without_blocking_consumer(self):
        worker = LatestPose2ArmWorker()
        self.addCleanup(worker.close)
        obs = synthetic_pose_observation(1,.05)
        with patch.object(worker.mapping, 'process', side_effect=lambda o: (time.sleep(.15) or dict(seq=o['seq']))):
            start=time.perf_counter()
            worker.submit(obs)
            self.assertIsNone(worker.snapshot())
            self.assertLess(time.perf_counter()-start,.05)
            deadline=time.perf_counter()+2.
            while worker.snapshot() is None and time.perf_counter()<deadline: time.sleep(.01)
            self.assertEqual(worker.snapshot()['seq'],1)

    def test_busy_worker_replaces_pending_frames_instead_of_replaying_a_queue(self):
        worker=LatestPose2ArmWorker()
        self.addCleanup(worker.close)
        entered=threading.Event();release=threading.Event();seen=[]
        def process(obs):
            seen.append(obs['seq'])
            if len(seen)==1:
                entered.set();release.wait(2.)
            return dict(seq=obs['seq'])
        with patch.object(worker.mapping,'process',side_effect=process):
            worker.submit(synthetic_pose_observation(1,.05))
            self.assertTrue(entered.wait(1.))
            for i in range(2,30):worker.submit(synthetic_pose_observation(i,i*.05))
            release.set()
            deadline=time.perf_counter()+2.
            while (worker.snapshot() or {}).get('seq')!=29 and time.perf_counter()<deadline:time.sleep(.01)
            self.assertEqual(seen,[1,29])


if __name__=='__main__': unittest.main()
