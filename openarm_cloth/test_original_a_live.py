"""Original A references execute on B's robot-only dynamic controller."""
import unittest
import numpy as np
from live_motion_scene import build_motion_scene
from pose2arm_task_mapping import Pose2ArmTaskMapping, synthetic_pose_observation
from run_pose2arm_live import AReferenceMotionController


class OriginalALiveTests(unittest.TestCase):
    def test_calibrated_a_references_move_both_b_arms_dynamically(self):
        model,data=build_motion_scene()
        control=AReferenceMotionController(model,data)
        mapping=Pose2ArmTaskMapping()
        initial={s:data.qpos[control.specs[s].qpos].copy() for s in ('left','right')}
        for i in range(50):
            result=mapping.process(synthetic_pose_observation(i,i*.05))
            control.step(result,dict(left=True,right=True))
        for side in initial:
            self.assertGreater(np.linalg.norm(data.qpos[control.specs[side].qpos]-initial[side]),.1)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertAlmostEqual(data.time,2.5)

    def test_unobservable_right_holds_while_left_reference_continues(self):
        model,data=build_motion_scene()
        control=AReferenceMotionController(model,data)
        old={s:data.qpos[control.specs[s].qpos].copy() for s in ('left','right')}
        result=dict(joints={s:q+.05 for s,q in old.items()},fingers=dict(left=.044,right=.044))
        for _ in range(5):control.step(result,dict(left=True,right=False))
        np.testing.assert_allclose(control.joints['right'],old['right'])
        self.assertGreater(np.linalg.norm(control.joints['left']-old['left']),.05)


if __name__=='__main__':unittest.main()
