"""Boundary motion must not accumulate unreachable human displacement."""
import unittest
import numpy as np
from human_grasp_task import DirectHumanTask, HumanGraspTask, HumanInput


class WorkspaceFollowingTests(unittest.TestCase):
    def task_at_boundary(self, axis, upper, cls=DirectHumanTask):
        task=cls();task.record_motion_trace=True
        tcp=np.array([.43,0.,.5])
        tcp[axis]=(task.config.upper if upper else task.config.lower)[axis]
        task.robot_anchor=tcp.copy();task.human_anchor=np.zeros(3);task.target=tcp.copy()
        return task,tcp

    def test_reversing_after_outward_motion_immediately_leaves_each_boundary(self):
        for axis in range(3):
            for upper in (False,True):
                with self.subTest(axis=axis,upper=upper):
                    task,tcp=self.task_at_boundary(axis,upper)
                    sign=1 if upper else -1
                    point=np.zeros(3);point[axis]=sign*.3
                    human=HumanInput(1,point.copy(),None,True,0.)
                    np.testing.assert_allclose(task._follow(human,.05,tcp),tcp)
                    # Another outward frame must not build a return dead zone.
                    human.point[axis]+=sign*.1
                    task._follow(human,.05,tcp)
                    human.point[axis]-=sign*.02
                    target=task._follow(human,.05,tcp)
                    self.assertLess(sign*(target[axis]-tcp[axis]),-.002)
                    self.assertLessEqual(np.linalg.norm(target-tcp),task.config.speed*.05+1e-9)
                    np.testing.assert_array_equal(task.human_anchor[np.arange(3)!=axis],0.)

    def test_trace_preserves_anchor_that_produced_unclipped_goal(self):
        task,tcp=self.task_at_boundary(0,False)
        human=HumanInput(1,np.array([-.3,.01,.02]),None,True,0.)
        before=task.human_anchor.copy()
        task._follow(human,.05,tcp)
        trace=task.motion_trace
        np.testing.assert_array_equal(trace['human_anchor'],before)
        np.testing.assert_allclose(trace['unclipped_goal'],tcp+task.config.gain*(human.point-before))
        np.testing.assert_allclose(trace['workspace_anchor_shift'],task.human_anchor-before)
        np.testing.assert_allclose(trace['workspace_anchor_shift'][1:],0.)

    def test_assisted_mapping_keeps_its_existing_fixed_anchor(self):
        task,tcp=self.task_at_boundary(0,False,HumanGraspTask)
        human=HumanInput(1,np.array([-.3,0.,0.]),None,True,0.)
        task._follow(human,.05,tcp)
        np.testing.assert_array_equal(task.human_anchor,np.zeros(3))

    def test_actuator_lag_inside_workspace_does_not_change_anchor(self):
        task=DirectHumanTask();task.record_motion_trace=True
        tcp=np.array([.43,0.,.5])
        task.robot_anchor=tcp.copy();task.human_anchor=np.zeros(3);task.target=tcp.copy()
        human=HumanInput(1,np.array([.1,0.,0.]),None,True,0.)
        for _ in range(10):task._follow(human,.05,tcp)
        np.testing.assert_array_equal(task.human_anchor,np.zeros(3))
        self.assertAlmostEqual(task.target[0],tcp[0]+task.config.max_target_lead_m)


if __name__=='__main__':unittest.main()
