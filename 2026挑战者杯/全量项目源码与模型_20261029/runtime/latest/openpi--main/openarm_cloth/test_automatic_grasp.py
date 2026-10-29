"""Automatic assistance must preserve manual intent and reject false grasps."""
import unittest
import numpy as np
from human_grasp_task import Feedback,HumanInput,TaskCommand,TaskConfig
from automatic_grasp import AutomaticPrecisionTask,AutomaticGripper,allowed_pad_contact,predicted_contact_stop,bilateral_touch_center


def observation(center=(.43,0,.429),tcp=(.43,0,.44),q=.031,force=(3.,3.)):
    tcp=np.array(tcp,float)
    return dict(tcp=tcp,pads=np.array([tcp+[-.036,0,0],tcp+[.036,0,0]]),
        center=None if center is None else np.array(center,float),yaw=0.,visual_age_s=0.,
        finger_positions=np.full(2,q),finger_velocity=np.zeros(2),finger_force=np.array(force),finger_target=.020,
        table_top=.4,half_size=.0288)


class AutomaticPrecisionTests(unittest.TestCase):
    def task(self):
        task=AutomaticPrecisionTask(TaskConfig());task.context=observation(tcp=(.43,0,.54))
        fb=Feedback(np.array([.43,0,.54]),visual=dict(center=np.array([.43,0,.429]),yaw=0.,time_s=0.))
        task.update(HumanInput(0,np.zeros(3),1.2,True,0.),fb,0.,start=True)
        return task,fb

    def test_stationary_hand_never_moves_target_when_precision_ramps(self):
        task,fb=self.task();task.context=observation()
        start=task.target.copy()
        for i in range(1,25):
            task.update(HumanInput(i,np.zeros(3),.1,True,i*.05),fb,i*.05)
        self.assertLess(task.precision_scale,.4)
        np.testing.assert_allclose(task.target,start,atol=1e-12)

    def test_far_motion_matches_original_without_new_filter_or_gain_change(self):
        from human_grasp_task import DirectHumanTask
        task,fb=self.task();task.context=observation(center=(1.,1.,1.))
        baseline=DirectHumanTask(TaskConfig());baseline.update(HumanInput(0,np.zeros(3),1.2,True,0.),fb,0.,start=True)
        for i in range(1,15):
            h=HumanInput(i,np.array([i*.001,0,0]),1.2,True,i*.05)
            a=task.update(h,fb,i*.05);b=baseline.update(h,fb,i*.05)
            np.testing.assert_allclose(a.target,b.target,atol=1e-12)

    def test_small_near_target_movement_remains_observable_and_carry_recovers(self):
        task,fb=self.task();task.context=observation()
        for i in range(1,20):task.update(HumanInput(i,np.zeros(3),.1,True,i*.05),fb,i*.05)
        before=task.target.copy()
        task.update(HumanInput(20,np.array([.02,0,0]),.1,True,1.),fb,1.)
        self.assertGreater(task.target[0],before[0]);self.assertLess(task.target[0]-before[0],.008)
        task.carrying=True
        for i in range(21,45):task.update(HumanInput(i,np.array([.02,0,0]),.1,True,i*.05),fb,i*.05)
        self.assertGreater(task.precision_scale,.95)

    def test_collision_rebase_discards_only_blocked_axes(self):
        task,fb=self.task();task.context=observation(center=(1.,1.,1.))
        h=HumanInput(1,np.array([.02,.02,-.02]),1.2,True,.05)
        task.update(h,fb,.05);before=task.target.copy()
        task.reject_motion('task_motion_collision_rejected:fixture',h,fb,.05,blocked_axes=[False,False,True])
        np.testing.assert_allclose(task.target[:2],before[:2]);self.assertEqual(task.target[2],fb.tcp[2])

    def test_reducing_lead_does_not_reverse_an_unchanged_human_goal(self):
        task,fb=self.task();task.context=observation(center=(1.,1.,1.))
        for i in range(1,18):task.update(HumanInput(i,np.array([.2,0,0]),1.2,True,i*.05),fb,i*.05)
        before=task.target.copy();task.context=observation(center=tuple(fb.tcp+[.005,0,0]),tcp=tuple(fb.tcp))
        for i in range(18,40):task.update(HumanInput(i,np.array([.2,0,0]),.1,True,i*.05),fb,i*.05)
        self.assertGreaterEqual(task.target[0],before[0]-1e-12)


class AutomaticGripTests(unittest.TestCase):
    def tick(self,g,i,obs=None,wanted=0.,valid=True,ratio=.15):
        return g.apply(TaskCommand('direct',np.array([.43,0,.44]),wanted),
            dict(seq=i,capture_s=i*.05,valid=valid,aperture=wanted,intent_ratio=ratio),
            observation() if obs is None else obs,i*.05,.05,active=True)

    def confirmed(self):
        g=AutomaticGripper()
        for i in range(12):self.tick(g,i)
        self.assertEqual(g.state,'HELD')
        return g

    def test_confirmation_needs_bilateral_stall_and_camera_enclosure(self):
        for obs in (observation(center=None),observation(force=(3.,0.)),observation(force=(1.,1.)),observation(q=.001)):
            g=AutomaticGripper()
            for i in range(15):self.tick(g,i,obs)
            self.assertNotEqual(g.state,'HELD')
        self.confirmed()

    def test_diagonal_pinch_does_not_silently_become_a_confirmed_face_grasp(self):
        obs=observation();obs['pads']=obs['tcp']+np.array([[-.0395,0,0],[.0395,0,0]])
        g=AutomaticGripper()
        for i in range(15):self.tick(g,i,obs)
        self.assertNotEqual(g.state,'HELD')

    def test_grip_survives_missing_hand_and_jitter_but_fresh_open_releases(self):
        g=self.confirmed();held=g.held
        for i in range(12,18):
            command=self.tick(g,i,wanted=.025,ratio=.6)
            self.assertEqual(command.finger,held)
        self.assertEqual(self.tick(g,18,wanted=.044,valid=False).finger,held)
        # Repeated delivery of one image cannot complete release confirmation.
        for _ in range(10):self.tick(g,19,wanted=.044,ratio=1.2)
        self.assertEqual(g.state,'HELD')
        for i in range(20,28):command=self.tick(g,i,wanted=.044,ratio=1.2)
        self.assertEqual(g.state,'OPEN');self.assertGreater(command.finger,held)

    def test_missing_hand_does_not_continue_closing_and_arm_target_is_untouched(self):
        g=AutomaticGripper();obs=observation();obs['finger_target']=.04
        command=self.tick(g,0,obs)
        self.assertGreater(command.finger,.038)
        command=self.tick(g,1,obs,valid=False)
        self.assertEqual(command.finger,.04)
        np.testing.assert_array_equal(command.target,[.43,0,.44])

    def test_sensor_source_clock_is_separate_from_paused_task_clock(self):
        g=AutomaticGripper();hand=dict(seq=1,capture_s=1000.,valid=True,aperture=0.,intent_ratio=.15)
        obs=observation();obs['finger_target']=.04
        command=g.apply(TaskCommand('direct',np.zeros(3),0.),hand,obs,0.,.05,now_s=1000.05)
        self.assertLess(command.finger,.04);self.assertIn(g.state,('CLOSING','VERIFYING'))


class PadContactTests(unittest.TestCase):
    def test_touch_correction_is_bounded_and_cannot_be_inferred_from_one_finger(self):
        obs=observation(center=(.006,0,.43),tcp=(0,0,.43))
        obs['pads']=np.array([[-.036,0,.43],[.036,0,.43]])
        refined=bilateral_touch_center(obs)
        self.assertIsNotNone(refined);np.testing.assert_allclose(refined,[0,0,.43])
        obs['finger_force']=np.array([3.,0.]);self.assertIsNone(bilateral_touch_center(obs))
        obs['finger_force']=np.array([3.,3.]);obs['center']=np.array([.05,0,.43])
        self.assertIsNone(bilateral_touch_center(obs))

    def test_only_shallow_inward_pad_contact_is_allowed(self):
        kw=dict(point=np.array([-.0228,0,0]),normal=np.array([1.,0,0]),pad=np.array([-.03,0,0]),
            other=np.array([.03,0,0]),center=np.zeros(3),depth=.0007,enabled=True)
        self.assertTrue(allowed_pad_contact(**kw))
        for patch in (dict(normal=np.array([-1.,0,0])),dict(point=np.array([-.03,0,.04])),
                      dict(depth=.004),dict(enabled=False),dict(center=np.array([0.,.06,0]))):
            self.assertFalse(allowed_pad_contact(**dict(kw,**patch)))

    def test_servo_target_is_not_mistaken_for_actual_finger_penetration(self):
        pads=np.array([[-.04,0,.43],[.04,0,.43]])
        stop=predicted_contact_stop(pads,np.array([.04,.04]),0.,np.array([0,0,.43]),0.)
        self.assertTrue(np.all(stop>.03));self.assertTrue(np.all(stop<.04))
        # Opening and a cube outside the jaw region remain unprojected.
        np.testing.assert_array_equal(predicted_contact_stop(pads,np.array([.02,.02]),.044,np.array([0,0,.43]),0.),[.044,.044])
        np.testing.assert_array_equal(predicted_contact_stop(pads,np.array([.04,.04]),0.,np.array([0,.08,.43]),0.),[0,0])


class RobotObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from member_b_scene import build_scene
        cls.model,cls.initial=build_scene()

    def robot(self):
        import mujoco
        from unittest.mock import patch
        from human_task_robot import HumanTaskRobot
        data=mujoco.MjData(self.model);data.qpos[:]=self.initial.qpos;mujoco.mj_forward(self.model,data)
        with patch('human_task_robot.BVisualObserver'):robot=HumanTaskRobot(self.model,data)
        self.addCleanup(robot.close)
        robot.visual=dict(center=np.array([.43,0,.429]),yaw=0.,time_s=0.)
        return robot,data

    def test_object_truth_does_not_enter_assistance_geometry(self):
        import mujoco
        robot,data=self.robot();before=robot.grasp_observation(0.)
        data.qpos[robot.object_address:robot.object_address+3]=[4.,3.,2.]
        mujoco.mj_forward(self.model,data);after=robot.grasp_observation(0.)
        np.testing.assert_array_equal(after['center'],before['center'])
        np.testing.assert_allclose(after['pads'],before['pads'])

    def test_clear_view_prior_needs_fresh_camera_and_records_its_real_age(self):
        robot,_=self.robot();robot.direct_grasp_assist=True
        center=np.array([.43,0,.429]);robot.clear_view_visual=dict(center=center,yaw=.5,time_s=0.)
        robot.visual=dict(center=center+[.005,0,0],yaw=0.,time_s=1.)
        obs=robot.grasp_observation(1.)
        np.testing.assert_allclose(obs['center'],center)
        np.testing.assert_allclose(obs['raw_center'],robot.visual['center'])
        self.assertTrue(obs['clear_view_prior_active']);self.assertEqual(obs['clear_view_prior_age_s'],1.)
        self.assertEqual(obs['visual_age_s'],0.)
        obs=robot.grasp_observation(1.6)
        self.assertIsNone(obs['center']);self.assertFalse(obs['clear_view_prior_active'])

    def test_large_observed_displacement_invalidates_clear_view_prior(self):
        robot,_=self.robot();robot.direct_grasp_assist=True
        robot.clear_view_visual=dict(center=np.array([.43,0,.429]),yaw=.5,time_s=0.)
        robot.visual=dict(center=np.array([.46,0,.429]),yaw=0.,time_s=1.)
        obs=robot.grasp_observation(1.)
        self.assertIsNone(robot.clear_view_visual);self.assertFalse(obs['clear_view_prior_active'])
        np.testing.assert_allclose(obs['center'],robot.visual['center'])

    def test_confirmed_large_grasp_is_not_misread_as_opening(self):
        from unittest.mock import Mock
        robot,_=self.robot();robot.direct_grasp_assist=True;robot.auto_grip_confirmed=True
        robot.feedback=Mock(return_value=type('Feedback',(),dict(gripping=True))())
        robot._direct_payload(0.,opening=True)
        self.assertTrue(robot.attached)
        robot.auto_grip_confirmed=False;robot._direct_payload(0.,opening=False)
        self.assertFalse(robot.attached)

    def test_distant_object_needs_no_extra_contact_kinematics(self):
        from unittest.mock import patch
        robot,data=self.robot();robot.fast_query_mode=True
        robot.visual=dict(center=np.array([1.,1.,1.]),yaw=0.,time_s=0.)
        q=data.qpos[robot.spec.qpos].copy()
        expected=robot._contacts(q,.01,False,False,False)
        robot.direct_grasp_assist=True;robot.grasp_contact_enabled=True
        with patch('human_task_robot.mujoco.mj_kinematics',side_effect=AssertionError('distant contact projection ran')):
            actual=robot._contacts(q,.01,False,False,False)
        self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
