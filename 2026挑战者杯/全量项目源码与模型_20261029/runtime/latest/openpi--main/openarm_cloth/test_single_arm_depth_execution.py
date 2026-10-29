"""Forward/retract demands must stay coordinated through B's physical joints."""
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from human_grasp_task import TaskCommand
from human_task_robot import HumanTaskRobot,coordinated_joint_step
from member_b_scene import build_scene


class SingleArmDepthExecutionTests(unittest.TestCase):
    def test_position_and_rotation_jacobian_agrees_with_measured_fk_differences(self):
        model,data=build_scene()
        with patch('human_task_robot.BVisualObserver'):
            robot=HumanTaskRobot(model,data)
        self.addCleanup(robot.close)
        robot.initialize()
        robot.visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.)
        robot.rotation=Rotation.from_euler('x',30,degrees=True).as_matrix()@robot.rotation
        def verify(residual,seed,**kwargs):
            for offset in (np.zeros(7),np.array([.1,.15,.2,-.15,.25,.1,.1])):
                q=seed+offset
                analytic=kwargs['jac'](q)
                numerical=np.column_stack([(residual(q+np.eye(7)[i]*1e-6)-residual(q-np.eye(7)[i]*1e-6))/2e-6 for i in range(7)])
                np.testing.assert_allclose(analytic,numerical,atol=1e-5,rtol=1e-4)
            raise ValueError('checked analytic Jacobian')
        with patch('human_task_robot.least_squares',side_effect=verify):
            with self.assertRaisesRegex(ValueError,'checked analytic Jacobian'):
                robot._direct_position_goal(robot.feedback(0.).tcp.copy(),robot._query_state(),bounded=True)

    def test_rate_limit_preserves_one_joint_path_instead_of_distorting_each_axis(self):
        start=np.arange(7)*.1
        delta=np.array([.2,-.1,.01,.005,.2,-.05,.001])
        cap=np.array([.04]*4+[.075]*3)
        result=coordinated_joint_step(start,start+delta,cap)
        np.testing.assert_allclose((result-start)/delta,np.full(7,.2))
        self.assertTrue(np.all(np.abs(result-start)<=cap+1e-12))
        np.testing.assert_allclose(coordinated_joint_step(start,start+delta*.01,cap),start+delta*.01)
        np.testing.assert_allclose(coordinated_joint_step(start,start,cap),start)

    def test_local_wrist_relaxation_cannot_accumulate_past_initial_attitude_limit(self):
        model,data=build_scene()
        with patch('human_task_robot.BVisualObserver'):
            robot=HumanTaskRobot(model,data)
        self.addCleanup(robot.close)
        robot.initialize()
        q=data.qpos[robot.spec.qpos].copy()
        robot.rotation=Rotation.from_euler('x',60,degrees=True).as_matrix()@data.geom_xmat[robot.geoms[0]].reshape(3,3)
        robot.visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.)
        with patch('human_task_robot.least_squares') as solve:
            solve.return_value.x=q
            with self.assertRaisesRegex(RuntimeError,'initial_wrist_delta_exceeds'):
                robot._direct_position_goal(robot.feedback(0.).tcp.copy(),robot._query_state(),bounded=True)
        self.assertTrue(robot.ik_diagnostic['attempted'])
        self.assertFalse(robot.ik_diagnostic['accepted'])
        self.assertGreater(robot.ik_diagnostic['initial_wrist_delta_rad'],np.deg2rad(20))

    def test_hold_clears_previous_ik_evidence_instead_of_reporting_a_stale_success(self):
        model,data=build_scene()
        with patch('human_task_robot.BVisualObserver'):
            robot=HumanTaskRobot(model,data)
        self.addCleanup(robot.close)
        robot.initialize();robot.direct_wrist_policy='position'
        robot.visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.)
        robot.execute(TaskCommand('direct',robot.feedback(0.).tcp.copy(),.044),0.)
        self.assertTrue(robot.ik_diagnostic['accepted'])
        robot.execute(TaskCommand('hold',finger=.044),.05)
        self.assertEqual(robot.ik_diagnostic,{})

    def test_forward_retract_moves_shoulder_elbow_and_wrist_without_sideways_lurch(self):
        model,data=build_scene()
        with patch('human_task_robot.BVisualObserver'):
            robot=HumanTaskRobot(model,data)
        self.addCleanup(robot.close)
        robot.initialize();robot.direct_wrist_policy='position'
        start=robot.feedback(0.).tcp.copy()
        positions=[];targets=[];joints=[];attitudes=[]
        for i in range(120):
            clock=i*.05
            target=start+np.array([.075*np.sin(2*np.pi*clock/6),0.,0.])
            robot.visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=clock)
            old=robot.joints['right'].copy()
            robot.execute(TaskCommand('direct',target,.044,'depth_regression'),clock,.05)
            self.assertIsNone(robot.error);self.assertIsNone(robot.warning)
            self.assertTrue(np.all(np.abs(robot.joints['right']-old)<=np.array([.8]*4+[1.5]*3)*.05+1e-10))
            for _ in range(round(.05/model.opt.timestep)):
                robot.control.apply();mujoco.mj_step(model,data)
            positions.append(robot.feedback(clock).tcp.copy());targets.append(target)
            joints.append(data.qpos[robot.spec.qpos].copy())
            attitudes.append(Rotation.from_matrix(robot.rotation@data.geom_xmat[robot.geoms[0]].reshape(3,3).T).magnitude())
        errors=np.linalg.norm(np.array(positions)-targets,axis=1)
        self.assertLess(float(np.percentile(errors,95)),.012)
        self.assertTrue(np.all(np.ptp(positions,axis=0)[1:]<.015))
        self.assertGreater(float(np.ptp(positions,axis=0)[0]),.14)
        self.assertTrue(np.all(np.ptp(joints,axis=0)[:4]>np.deg2rad(8)))
        self.assertLess(max(attitudes),np.deg2rad(20))


if __name__=='__main__':unittest.main()
