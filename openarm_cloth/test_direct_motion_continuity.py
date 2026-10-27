"""Exercise the real direct task with slow measured-style wrist updates."""
import unittest
import numpy as np

from human_grasp_task import DirectHumanTask,Feedback,HumanInput,TaskConfig


class DirectMotionContinuityTests(unittest.TestCase):
    def sequence(self,increments):
        task=DirectHumanTask(TaskConfig())
        tcp=np.array([.43,0.,.54])
        feedback=Feedback(tcp.copy(),visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.))
        task.update(HumanInput(0,np.zeros(3),None,True,0.),feedback,0.,start=True)
        targets=[]
        for seq,delta in enumerate(increments,1):
            point=np.array([0.,float(delta)/task.config.gain,0.])
            command=task.update(HumanInput(seq,point,None,True,seq*.05),feedback,seq*.05)
            self.assertEqual(command.stage,'direct')
            targets.append(command.target.copy())
            feedback.tcp=command.target.copy()
        return task,np.asarray(targets)

    def test_slow_half_millimetre_motion_is_not_released_in_two_mm_jumps(self):
        task,targets=self.sequence(np.arange(1,81)*.0005)
        increments=np.diff(targets[:,1])
        self.assertLessEqual(np.max(increments),.0005+1e-12)
        np.testing.assert_allclose(increments[10:],.0005,atol=1e-12)
        self.assertAlmostEqual(targets[-1,1],.040-task.config.deadband_m)

    def test_stationary_submillimetre_noise_still_holds(self):
        task,targets=self.sequence(.0006*np.sin(np.arange(100)))
        np.testing.assert_allclose(targets, np.tile([.43,0.,.54],(100,1)),atol=1e-12)

    def test_fast_motion_preserves_original_speed_and_target_lead_limits(self):
        task,targets=self.sequence(np.full(40,.2))
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(targets,axis=0),axis=1)),task.config.speed*.05+1e-12)

    def test_depth_workspace_limit_is_exposed_separately_from_missing_observation(self):
        task=DirectHumanTask(TaskConfig());feedback=Feedback(np.array([.43,0.,.54]),visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.))
        task.update(HumanInput(0,np.zeros(3),None,True,0.),feedback,0.,start=True)
        command=task.update(HumanInput(1,np.array([1.,0,0]),None,True,.05),feedback,.05)
        self.assertEqual(command.stage,'direct')
        np.testing.assert_array_equal(task.follow_workspace_clipped,[True,False,False])
        task.update(HumanInput(2,np.array([.99,0,0]),None,True,.1),feedback,.1)
        self.assertFalse(task.follow_workspace_clipped.any())


if __name__=='__main__':unittest.main()
