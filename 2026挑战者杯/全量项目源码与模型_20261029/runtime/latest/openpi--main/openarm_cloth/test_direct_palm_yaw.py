"""Image palm rotation and bounded, intent-gated near-grasp corrections."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from human_grasp_task import Feedback, HumanInput
from automatic_grasp import AutomaticPrecisionTask, AutomaticGripper
from direct_palm_yaw import DirectPalmYaw


def palm(angle=0.,scale=1.):
    p=np.zeros((21,3));p[:,:2]=[.5,.5]
    base=np.array([[0,.05],[-.035,0],[0,-.015],[.02,-.005],[.04,.015]])*scale
    c,s=np.cos(angle),np.sin(angle)
    p[[0,5,9,13,17],:2]=.5+base@np.array([[c,s],[-s,c]])/[1.,720/1280]
    return p


class PalmYawTests(unittest.TestCase):
    def calibrated(self):
        tracker=DirectPalmYaw()
        for i in range(5):tracker.update(palm(),i*.05)
        return tracker

    def test_image_rotation_drives_yaw_without_new_body_model(self):
        tracker=self.calibrated()
        for i in range(5,25):tracker.update(palm(np.deg2rad(30)),i*.05)
        self.assertAlmostEqual(np.degrees(tracker.yaw),-30.,delta=1.5)
        self.assertEqual(tracker.state,'TRACKING')

    def test_translation_scale_and_fingertip_closure_do_not_rotate(self):
        tracker=self.calibrated();p=palm(scale=.7);p[:,:2]+=[.1,.05];p[4]=p[8]
        for i in range(5,15):tracker.update(p,i*.05)
        self.assertAlmostEqual(tracker.yaw,0.,delta=1e-8)

    def test_missing_and_duplicate_frames_never_recenter_or_rotate(self):
        tracker=self.calibrated()
        for i in range(5,20):tracker.update(palm(.3),i*.05)
        before=tracker.yaw
        for _ in range(5):tracker.update(palm(-.6),.95)
        self.assertEqual(tracker.yaw,before)
        tracker.update(None,1.1);self.assertEqual(tracker.yaw,before)
        tracker.update(palm(.31),1.2);tracker.update(palm(.31),1.25)
        self.assertAlmostEqual(tracker.yaw,-.31,delta=.06)

    def test_edge_on_or_flipped_palm_cannot_create_a_large_rotation(self):
        tracker=self.calibrated();p=palm();p[[0,5,9,13,17],0]=.5
        tracker.update(p,.3);self.assertEqual(tracker.yaw,0.)
        tracker.update(palm(np.pi),.4);tracker.update(palm(np.pi),.45)
        self.assertEqual(tracker.yaw,0.)


class NearAlignmentTests(unittest.TestCase):
    def test_alignment_requires_fresh_closing_and_has_finite_budget(self):
        task=AutomaticPrecisionTask();task.visual_alignment_enabled=True
        tcp=np.array([.45,.0,.44]);center=np.array([.43,0,.429])
        fb=Feedback(tcp.copy(),visual=dict(center=center,yaw=0.,time_s=0.))
        task.context=dict(tcp=tcp.copy(),center=center,visual_age_s=0.)
        task.update(HumanInput(0,np.zeros(3),1.2,True,0.),fb,0.,start=True)
        initial=task.target.copy()
        for i in range(1,10):task.update(HumanInput(i,np.zeros(3),1.2,True,i*.05),fb,i*.05)
        np.testing.assert_allclose(task.target,initial)
        for i in range(10,60):
            cmd=task.update(HumanInput(i,np.zeros(3),.1,True,i*.05),fb,i*.05)
            fb.tcp=cmd.target.copy();task.context['tcp']=fb.tcp.copy()
        self.assertLess(np.linalg.norm(task.target-(center+[0,0,.003])),.005)
        self.assertLessEqual(np.linalg.norm(task.alignment_total),.025+1e-9)
        before=task.target.copy()
        task.update(HumanInput(59,np.zeros(3),.1,True,2.95),fb,3.1)
        np.testing.assert_allclose(task.target,before)

    def test_opening_and_stale_camera_do_not_generate_corrections(self):
        task=AutomaticPrecisionTask();task.visual_alignment_enabled=True
        tcp=np.array([.45,0,.44]);fb=Feedback(tcp.copy(),visual=dict(center=tcp,yaw=0.,time_s=0.))
        task.update(HumanInput(0,np.zeros(3),1.,True,0.),fb,0.,start=True)
        task.context=dict(center=np.array([.43,0,.429]),tcp=tcp,visual_age_s=.6)
        for i in range(1,20):task.update(HumanInput(i,np.zeros(3),.1,True,i*.05),fb,i*.05)
        np.testing.assert_allclose(task.target,tcp)

    def test_pending_alignment_keeps_clearance_instead_of_closing_on_edge(self):
        from test_automatic_grasp import observation
        from human_grasp_task import TaskCommand
        obs=observation();obs['alignment_pending']=True;obs['finger_target']=.044
        grip=AutomaticGripper()
        for i in range(10):
            cmd=grip.apply(TaskCommand('direct',np.array([.43,0,.44]),0.),
                dict(seq=i,capture_s=i*.05,valid=True,aperture=0.,intent_ratio=.1),obs,i*.05,.05)
            obs['finger_target']=cmd.finger
        self.assertAlmostEqual(cmd.finger,.044)
        self.assertEqual(grip.state,'ALIGNING');self.assertIsNone(grip.held)

    def test_small_camera_jitter_after_centering_cannot_reopen_jaws(self):
        task=AutomaticPrecisionTask();task.visual_alignment_enabled=True
        tcp=np.array([.43,0,.432]);center=np.array([.43,0,.429])
        fb=Feedback(tcp.copy(),visual=dict(center=center,yaw=0.,time_s=0.))
        task.context=dict(tcp=tcp,center=center,visual_age_s=0.)
        task.update(HumanInput(0,np.zeros(3),1.2,True,0.),fb,0.,start=True)
        task.update(HumanInput(1,np.zeros(3),.1,True,.05),fb,.05)
        self.assertFalse(task.alignment_pending)
        task.context['center']=center+np.array([.006,0,0])
        task.update(HumanInput(2,np.zeros(3),.1,True,.10),fb,.10)
        self.assertFalse(task.alignment_pending)
        task.context['center']=center+np.array([.020,0,0])
        task.update(HumanInput(3,np.zeros(3),.1,True,.15),fb,.15)
        self.assertTrue(task.alignment_pending)


class HumanYawIKTests(unittest.TestCase):
    def test_precision_reference_keeps_advancing_under_actuator_lag(self):
        from member_b_scene import build_scene
        from human_task_robot import HumanTaskRobot
        from human_grasp_task import TaskCommand
        model,data=build_scene();robot=HumanTaskRobot(model,data)
        try:
            robot.initialize();robot.observe(0.)
            robot.direct_wrist_policy='position';robot.direct_grasp_assist=True
            robot.direct_precision_scale=.25;robot.direct_cartesian_speed=.02
            target=robot.feedback(0.).tcp+np.array([.020,0.,0.])
            for i in range(60):
                if i%4==0:robot.observe(i*.05)
                robot.grasp_observation(i*.05)
                robot.execute(TaskCommand('direct',target,.044),i*.05)
                robot.step(.05)
            self.assertIsNone(robot.error)
            self.assertLess(np.linalg.norm(robot.feedback(3.).tcp-target),.004)
        finally:robot.close()

    def test_b_ik_can_turn_at_fixed_tcp_with_original_actuator_limits(self):
        from member_b_scene import build_scene
        from human_task_robot import HumanTaskRobot,pad_points
        model,data=build_scene();robot=HumanTaskRobot(model,data)
        try:
            robot.direct_wrist_policy='palm';robot.initialize();robot.observe(0.)
            tcp=robot.feedback(0.).tcp.copy()
            import mujoco
            self.assertGreater(min(data.qpos[robot.spec.qpos[5]]-robot.spec.q_range[5,0],robot.spec.q_range[5,1]-data.qpos[robot.spec.qpos[5]]),.3)
            for angle in (-30,30):
                wanted=Rotation.from_euler('z',angle,degrees=True).as_matrix()@robot.rotation
                robot.direct_rotation_target=wanted
                q=robot._direct_position_goal(tcp,robot._query_state(),bounded=True)
                state=robot.ik_query;state.qpos[robot.spec.qpos]=q
                mujoco.mj_kinematics(model,state)
                actual=state.geom_xmat[robot.geoms[0]].reshape(3,3)
                self.assertLess(np.linalg.norm(pad_points(model,state,'right').mean(0)-tcp),.003)
                self.assertLess(Rotation.from_matrix(wanted@actual.T).magnitude(),np.deg2rad(5))
        finally:robot.close()


if __name__=='__main__':unittest.main()
