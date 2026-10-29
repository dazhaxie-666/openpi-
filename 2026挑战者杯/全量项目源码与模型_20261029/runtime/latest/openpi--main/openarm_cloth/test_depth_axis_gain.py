import unittest
import numpy as np
from human_grasp_task import TaskConfig,DirectHumanTask,HumanGraspTask,HumanInput,Feedback


class DepthAxisGainTests(unittest.TestCase):
    def task(self,depth_gain):
        task=DirectHumanTask(TaskConfig(depth_gain=depth_gain));task.record_motion_trace=True
        f=Feedback(np.array([.43,0.,.54]),visual=dict(center=np.array([.43,0,.429]),time_s=0.,yaw=0.))
        task.update(HumanInput(0,np.zeros(3),None,True,0.),f,0.,start=True)
        return task,f

    def test_depth_gain_does_not_amplify_other_axes(self):
        t,f=self.task(1.0)
        t.update(HumanInput(1,np.full(3,.01),None,True,.05),f,.05)
        np.testing.assert_allclose(t.motion_trace['workspace_goal']-f.tcp,[.01,.007,.007])

    def test_depth_boundary_discards_excess_with_correct_axis_gain(self):
        t,f=self.task(1.0)
        t.update(HumanInput(1,np.array([1.,0,0]),None,True,.05),f,.05)
        self.assertTrue(t.follow_workspace_clipped[0])
        t.update(HumanInput(2,np.array([.99,0,0]),None,True,.1),f,.1)
        self.assertFalse(t.follow_workspace_clipped[0])
        self.assertAlmostEqual(t.motion_trace['workspace_goal'][0],.53)

    def test_assisted_task_preserves_original_uniform_gain(self):
        t=HumanGraspTask(TaskConfig(depth_gain=1.0))
        np.testing.assert_allclose(t.motion_gains,[.7,.7,.7])

    def test_invalid_depth_gain_is_rejected(self):
        for bad in (0.,-1.,np.nan,np.inf):
            with self.assertRaises(ValueError):TaskConfig(depth_gain=bad)
