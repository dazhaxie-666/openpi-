import unittest
import numpy as np
from human_grasp_task import HumanInput,Feedback
from shared_grasp import SharedGraspTask
from test_automatic_grasp import observation


class SharedTaskTests(unittest.TestCase):
    def setUp(self):
        self.task=SharedGraspTask();self.task.hand_intent_valid=True
        self.fb=Feedback(np.array([.43,0,.54]),visual=dict(center=np.array([.43,0,.429]),yaw=0.,time_s=0.))
        self.task.context=observation();self.i=0
        self.tick(1.2,start=True)
        for _ in range(5):self.tick(1.2)

    def tick(self,ratio=.1,valid=True,**kwargs):
        self.i+=1
        self.task.request_ratio=ratio  # explicit software hand fixture
        return self.task.update(HumanInput(self.i,np.zeros(3),ratio,valid,self.i*.05),self.fb,self.i*.05,**kwargs)

    def request(self):
        self.fb.tcp[2]=.49  # human has approached; home is outside assistance radius
        for _ in range(6):self.tick(1.2)
        for _ in range(6):cmd=self.tick()
        self.assertEqual(self.task.phase,'plan');return cmd

    def test_near_pinch_requests_planning_without_millimetre_alignment(self):
        self.request();self.assertEqual(self.task.trial_id,1)
        self.assertTrue(self.task.intent_latched)

    def test_collection_reset_gate_blocks_next_trial_until_countdown_finishes(self):
        self.task.trial_limit=2;self.task.inter_trial_reset_s=.5;self.task.final_trial_notice_s=.2
        self.task.automatic_scene_reset=True
        self.request();ended_at=self.i*.05
        self.task.end_trial(ended_at,'failed','fixture attempt ended')
        self.assertTrue(self.task.scene_reset_pending)
        self.task.scene_reset_completed(ended_at,1)
        self.assertFalse(self.task.scene_reset_pending);self.assertEqual(self.task.scene_reset_count,1)
        for _ in range(5):
            command=self.tick(.1)
            self.assertEqual(command.owner,'shared_inter_trial_reset')
            self.assertEqual(self.task.trial_id,1)
        while self.i*.05<=ended_at+.5:self.tick(.1)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(self.task.trial_id,1)

    def test_final_trial_holds_a_visible_completion_notice(self):
        self.task.trial_limit=1;self.task.inter_trial_reset_s=.5;self.task.final_trial_notice_s=.2
        self.request();ended_at=self.i*.05
        self.task.end_trial(ended_at,'completed','fixture placement observed')
        command=self.tick(1.2)
        self.assertEqual(command.owner,'shared_collection_complete')
        self.assertEqual(self.task.phase,'session_complete')
        self.assertAlmostEqual(self.task.final_notice_until,ended_at+.2)

    def test_missing_hand_after_request_does_not_deadlock_approach_or_close(self):
        self.request();self.task.hand_intent_valid=False;self.fb.plan_ready=True
        self.tick(None,valid=False)
        self.assertEqual(self.task.phase,'transit')
        self.task.enter('close',self.i*.05,'fixture')
        for _ in range(8):cmd=self.tick(None,valid=False)
        self.assertEqual(cmd.stage,'lift');self.assertTrue(self.task.grasp_confirmed)

    def test_stale_repeated_close_never_requests_grasp(self):
        self.task.hand_intent_valid=False
        for _ in range(10):self.tick()
        self.assertEqual(self.task.phase,'manual')

    def test_fresh_associated_hand_can_request_grasp_without_body_position(self):
        self.fb.tcp[2]=.49
        for _ in range(6):self.tick(1.2,valid=False)
        for _ in range(6):self.tick(.1,valid=False)
        self.assertEqual(self.task.phase,'plan')

    def test_human_open_cancels_before_lift_and_records_cancelled_trial(self):
        self.request()
        for _ in range(16):self.tick(1.2)
        self.assertEqual(self.task.trials[-1]['status'],'cancelled')
        self.assertIn(self.task.phase,('recover','manual'))

    def test_missing_wrist_while_carrying_holds_grip_and_open_requests_lowering(self):
        self.request();self.task.grasp_confirmed=True;self.fb.gripping=True
        self.task.enter('transport',self.i*.05,'fixture')
        self.task.hand_intent_valid=False
        cmd=self.tick(None,valid=False)
        self.assertEqual((cmd.stage,cmd.finger),('hold',0.))
        self.task.hand_intent_valid=True
        for _ in range(6):self.tick(.1)
        for _ in range(10):cmd=self.tick(1.2)
        self.assertEqual(cmd.stage,'lower');self.assertEqual(cmd.finger,0.)

    def test_collection_hold_task_enforces_three_seconds_then_returns_at_source(self):
        self.request();self.task.grasp_confirmed=True;self.fb.gripping=True
        self.task.collection_task_profile='hold';self.task.collection_hold_seconds=3.
        began=self.i*.05;self.task.enter('transport',began,'fixture lift complete')
        start_tcp=self.fb.tcp.copy()
        for _ in range(59):
            cmd=self.tick(1.2)
            self.assertEqual((cmd.stage,cmd.owner),('hold','shared_collection_payload_hold'))
        self.assertFalse(self.task.collection_hold_complete)
        for _ in range(3):cmd=self.tick(.1)
        self.assertTrue(self.task.collection_hold_complete)
        for _ in range(10):cmd=self.tick(1.2)
        self.assertEqual(cmd.stage,'lower')
        np.testing.assert_allclose(cmd.target[:2],start_tcp[:2])

    def test_jaw_closure_is_not_claimed_as_grasp_without_bilateral_evidence(self):
        self.request();self.task.enter('close',self.i*.05,'fixture')
        self.task.context=observation(force=(3.,0.));self.task.hand_intent_valid=False
        for _ in range(9):cmd=self.tick(None)
        self.assertEqual(cmd.stage,'close');self.assertFalse(self.task.grasp_confirmed)

    def test_pause_preserves_current_aperture_and_discards_release_confirmation(self):
        self.request();self.task.enter('close',self.i*.05,'fixture')
        self.task.context['finger_target']=.020
        cmd=self.tick(1.2,paused=True)
        self.assertEqual(cmd.stage,'hold');self.assertEqual(cmd.finger,.020)
        self.assertEqual(self.task.phase,'close');self.assertIsNone(self.task.gesture)

    def test_close_at_occluded_target_does_not_start_a_grasp_or_automatic_rise(self):
        self.fb.tcp=np.array([.435,.005,.438])
        self.tick(1.2)  # real camera target was seen immediately before occlusion
        self.fb.visual=None
        for _ in range(6):cmd=self.tick(.1)
        self.assertEqual(self.task.phase,'manual')
        self.assertEqual(cmd.stage,'hold');self.assertEqual(cmd.finger,.044)
        self.assertFalse(self.task.intent_latched);self.assertEqual(self.task.trial_id,0)

    def test_human_view_recovery_requires_new_full_pose_and_returns_without_grasping(self):
        self.test_missing_object_view_accepts_only_observed_upward_escape_without_grasp()
        self.task.hand_intent_valid=False
        self.fb.visual=dict(center=np.array([.44,.01,.429]),time_s=self.i*.05)
        self.tick(None,valid=False)
        self.assertEqual(self.task.phase,'restore_view')
        self.fb.command_stage='reacquire';self.fb.reached=self.fb.opened=True
        self.fb.visual['pose_is_prior']=True
        for _ in range(6):self.tick(None,valid=False)
        self.assertEqual(self.task.phase,'restore_view')
        self.fb.visual['pose_is_prior']=False;self.fb.visual['time_s']=self.i*.05
        for _ in range(4):cmd=self.tick(None,valid=False)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(self.task.trial_id,0)

    def test_no_target_or_expired_camera_geometry_cannot_authorise_blind_retreat(self):
        self.fb.visual=None;self.task.last_target=None
        for _ in range(6):cmd=self.tick(.1)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(self.task.trial_id,0)
        self.task.last_target=dict(center=np.array([.43,0,.429]),time_s=-10.)
        for _ in range(6):self.tick(.1)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(self.task.trial_id,0)

    def test_open_hand_during_human_view_recovery_does_not_create_or_cancel_a_trial(self):
        self.test_missing_object_view_accepts_only_observed_upward_escape_without_grasp()
        for _ in range(16):self.tick(1.2)
        self.assertEqual(self.task.phase,'restore_view');self.assertEqual(self.task.trials,[])
        self.assertFalse(self.task.intent_latched)

    def test_repeated_source_frame_cannot_confirm_a_request(self):
        human=HumanInput(99,np.zeros(3),.1,True,1.)
        for i in range(20):self.task.update(human,self.fb,1.+i*.05)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(self.task.trial_id,0)

    def test_brief_open_during_automatic_grasp_does_not_cancel(self):
        self.request()
        for _ in range(6):self.tick(1.2)
        self.assertEqual(self.task.phase,'plan')
        self.assertTrue(self.task.intent_latched)

    def test_brief_position_dropout_does_not_discard_follow_displacement(self):
        self.tick(1.2,valid=False)
        self.i+=1
        cmd=self.task.update(HumanInput(self.i,np.array([0.,.02,0.]),1.2,True,self.i*.05),
            self.fb,self.i*.05)
        self.assertEqual(cmd.stage,'direct')
        self.assertGreater(cmd.target[1],.001)

    def test_start_posture_and_pinched_hand_do_not_generate_autonomous_approach(self):
        for _ in range(8):cmd=self.tick(.1)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(cmd.stage,'direct')
        self.assertEqual(self.task.trial_id,0)

    def test_pinch_near_cube_surface_is_not_blocked_by_center_distance(self):
        self.fb.tcp=np.array([.43,0,.519])  # 90mm centre distance, 61.2mm surface clearance
        for _ in range(6):self.tick(1.2)
        for _ in range(6):cmd=self.tick(.1)
        self.assertEqual(self.task.phase,'plan');self.assertTrue(self.task.intent_latched)

    def test_missing_object_view_accepts_only_observed_upward_escape_without_grasp(self):
        self.fb.tcp=np.array([.435,.005,.438]);self.tick(1.2)
        self.fb.visual=None
        self.i+=1
        cmd=self.task.update(HumanInput(self.i,np.array([0.,0.,.04]),1.2,True,self.i*.05),self.fb,self.i*.05)
        self.assertEqual(self.task.phase,'restore_view');self.assertEqual(cmd.stage,'reacquire')
        self.assertEqual(cmd.owner,'human_restore_view');self.assertEqual(cmd.finger,.044)
        self.assertFalse(self.task.intent_latched);self.assertEqual(self.task.trial_id,0)
        np.testing.assert_array_equal(cmd.target[:2],self.fb.tcp[:2])

    def test_missing_view_does_not_authorise_lateral_approach(self):
        self.fb.tcp=np.array([.435,.005,.438]);self.tick(1.2)
        self.fb.visual=None;self.i+=1
        cmd=self.task.update(HumanInput(self.i,np.array([0.,.04,0.]),1.2,True,self.i*.05),self.fb,self.i*.05)
        self.assertEqual(cmd.stage,'hold');self.assertEqual(self.task.phase,'manual')

    def test_view_recovery_returns_to_manual_without_creating_a_grasp_trial(self):
        self.test_missing_object_view_accepts_only_observed_upward_escape_without_grasp()
        for _ in range(10):self.tick(1.2)
        self.assertEqual(self.task.phase,'restore_view')
        self.fb.visual=dict(center=np.array([.43,0,.429]),yaw=0.,time_s=self.i*.05)
        self.fb.command_stage='reacquire';self.fb.reached=self.fb.opened=True
        for _ in range(5):cmd=self.tick(1.2)
        self.assertEqual(self.task.phase,'manual');self.assertEqual(self.task.trial_id,0)
        self.assertFalse(self.task.intent_latched)

    def test_upward_recovery_requires_recent_geometry_and_respects_pause(self):
        self.fb.tcp=np.array([.435,.005,.438]);self.tick(1.2)
        self.fb.visual=None;self.i+=1
        human=HumanInput(self.i,np.array([0.,0.,.04]),1.2,True,self.i*.05)
        cmd=self.task.update(human,self.fb,self.i*.05,paused=True)
        self.assertEqual(cmd.stage,'hold');self.assertEqual(self.task.phase,'manual')
        self.task.last_target['time_s']=-40.;self.task.was_missing=False
        self.task.view_loss_reference=np.zeros(3);self.i+=1
        human=HumanInput(self.i,np.array([0.,0.,.08]),1.2,True,self.i*.05)
        cmd=self.task.update(human,self.fb,self.i*.05)
        self.assertEqual(cmd.stage,'hold');self.assertEqual(self.task.phase,'manual')

    def test_carry_handoff_rejects_old_frames_but_accepts_new_sustained_open_without_repinch(self):
        self.request();self.task.grasp_confirmed=True;self.fb.gripping=True
        self.task.enter('transport',self.i*.05,'fixture')
        human=HumanInput(self.i,np.zeros(3),1.2,True,self.i*.05)
        for i in range(10):cmd=self.task.update(human,self.fb,self.i*.05+i*.01)
        self.assertEqual(self.task.phase,'transport');self.assertEqual(cmd.finger,0.)
        self.i+=2
        for _ in range(4):cmd=self.tick(1.2)
        self.assertEqual(self.task.phase,'transport');self.assertEqual(cmd.finger,0.)
        for _ in range(6):cmd=self.tick(1.2)
        self.assertEqual(self.task.phase,'lower');self.assertEqual(cmd.finger,0.)

    def test_unreachable_carry_request_rebases_and_allows_next_human_move(self):
        self.request();self.task.grasp_confirmed=True;self.fb.gripping=True
        self.task.enter('transport',self.i*.05,'fixture')
        self.tick(.1)
        human=HumanInput(self.i,np.zeros(3),.1,True,self.i*.05)
        self.task.reject_motion('cartesian_target_unreachable: fixture',human,self.fb,self.i*.05)
        self.assertEqual(self.task.phase,'transport')
        self.i+=1
        cmd=self.task.update(HumanInput(self.i,np.array([0.,-.02,0.]),.1,True,self.i*.05),self.fb,self.i*.05)
        self.assertEqual(cmd.stage,'transport');self.assertEqual(cmd.finger,0.)
        self.assertLess(cmd.target[1],self.fb.tcp[1]-.001)

    def test_rejected_placement_restores_human_carry_and_never_retries_at_same_point(self):
        self.request();self.task.grasp_confirmed=True;self.fb.gripping=True
        self.task.enter('lower',self.i*.05,'fixture accepted open')
        human=HumanInput(self.i,np.zeros(3),1.2,True,self.i*.05)
        self.task.reject_motion('cartesian_target_unreachable: fixture',human,self.fb,self.i*.05)
        self.assertEqual(self.task.phase,'transport')
        self.assertIsNotNone(self.task.placement_blocked_origin)
        for _ in range(24):
            cmd=self.tick(1.2)
            self.assertNotEqual(cmd.stage,'lower');self.assertEqual(cmd.finger,0.)
        # Changing the measured place position permits a NEW open confirmation.
        self.fb.tcp[1]+=.021
        for _ in range(10):cmd=self.tick(1.2)
        self.assertEqual(cmd.stage,'lower');self.assertEqual(cmd.finger,0.)

    def test_colliding_placement_returns_control_but_uncertain_grip_stays_held(self):
        self.request();self.task.grasp_confirmed=True;self.fb.gripping=True
        self.task.enter('lower',self.i*.05,'fixture')
        human=HumanInput(self.i,np.zeros(3),1.2,True,self.i*.05)
        self.task.reject_motion('task_motion_collision_rejected:lower:fixture',human,self.fb,self.i*.05)
        self.assertEqual(self.task.phase,'transport')
        self.assertTrue(self.task.grasp_confirmed)
        self.task.reject_motion('uncertain_fixture',human,self.fb,self.i*.05)
        self.assertEqual(self.task.phase,'holding_payload')


class SharedBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from member_b_scene import build_scene
        cls.model,cls.initial=build_scene()

    def robot(self,shared=False):
        import mujoco
        from unittest.mock import patch
        from human_task_robot import HumanTaskRobot
        from shared_grasp_robot import SharedGraspRobot
        data=mujoco.MjData(self.model);data.qpos[:]=self.initial.qpos;mujoco.mj_forward(self.model,data)
        with patch('human_task_robot.BVisualObserver'):
            robot=(SharedGraspRobot if shared else HumanTaskRobot)(self.model,data)
        self.addCleanup(robot.close)
        robot.visual=dict(center=np.array([.43,0,.429]),yaw=.2,time_s=0.)
        return robot,data

    def test_grasp_tool_rotation_is_never_used_as_object_rotation(self):
        import mujoco
        from scipy.spatial.transform import Rotation
        robot,data=self.robot();robot.direct_grasp_assist=True;robot.fast_query_mode=True
        robot.plan=dict(robot.visual,rotation=Rotation.from_euler('x',90,degrees=True).as_matrix())
        robot._contacts(data.qpos[robot.spec.qpos],.044,False,False,False)
        actual=np.empty(9);mujoco.mju_quat2Mat(actual,robot.query.qpos[robot.object_address+3:robot.object_address+7])
        np.testing.assert_allclose(actual.reshape(3,3),Rotation.from_euler('z',.2).as_matrix(),atol=1e-9)

    def test_repeated_payload_hold_preserves_latched_actuator_target(self):
        from human_grasp_task import TaskCommand
        robot,data=self.robot(shared=True);robot.attached=True
        expected=robot.joints['right'].copy();robot.finger_targets['right']=0.
        data.qpos[robot.spec.qpos[1]]+=.005
        for i in range(5):robot.execute(TaskCommand('hold',finger=0.),i*.05)
        np.testing.assert_array_equal(robot.joints['right'],expected)
        self.assertEqual(robot.finger_targets['right'],0.)

    def test_collection_scene_reset_restores_robot_and_cube_without_rewinding_time(self):
        robot,data=self.robot(shared=True)
        snapshot=robot.capture_collection_reset_state();expected=snapshot['qpos'].copy()
        before_time=float(data.time);data.qpos[:]+=0.01;data.qvel[:]=1.
        robot.attached=True;robot.plan={'fixture':True};robot.visual={'fixture':True}
        robot.restore_collection_reset_state(snapshot)
        np.testing.assert_allclose(data.qpos,expected,atol=0,rtol=0)
        np.testing.assert_array_equal(data.qvel,np.zeros_like(data.qvel))
        self.assertEqual(data.time,before_time);self.assertFalse(robot.attached)
        self.assertIsNone(robot.plan);self.assertIsNone(robot.visual)
        self.assertTrue(all(value==.044 for value in robot.finger_targets.values()))

    def test_failed_carry_preserves_previous_checked_target_and_clears_fallback_mode(self):
        from unittest.mock import patch
        from human_grasp_task import TaskCommand
        from human_task_robot import HumanTaskRobot
        robot,data=self.robot(shared=True);robot.attached=True
        expected=robot.joints['right'].copy();robot.finger_targets['right']=0.
        data.qpos[robot.spec.qpos[1]]+=.005
        with patch.object(HumanTaskRobot,'_execute',side_effect=RuntimeError('cartesian_target_unreachable:fixture')):
            robot.execute(TaskCommand('transport',np.zeros(3),0.),.1)
        np.testing.assert_array_equal(robot.joints['right'],expected)
        self.assertFalse(robot.shared_carry_query);self.assertIsNone(robot.error)
        self.assertIn('cartesian_target_unreachable:',robot.warning)

    def test_carry_fallback_is_bounded_by_position_and_fixed_grasp_attitude(self):
        from unittest.mock import patch
        from human_task_robot import HumanTaskRobot
        from scipy.spatial.transform import Rotation
        robot,data=self.robot(shared=True);robot.shared_carry_query=True
        state=robot._query_state();robot.rotation=state.geom_xmat[robot.geoms[0]].reshape(3,3).copy()
        previous=(robot.direct_wrist_policy,robot.direct_rotation_target)
        with patch.object(HumanTaskRobot,'_cartesian',side_effect=RuntimeError('cartesian_target_unreachable:fixture')), \
             patch.object(robot,'_query_state',return_value=state), \
             patch.object(robot,'_direct_position_goal',return_value=data.qpos[robot.spec.qpos].copy()):
            robot.ik_diagnostic={'position_error_m':.003}
            robot._cartesian(np.zeros(3),True)
            robot.ik_diagnostic={'position_error_m':.0041}
            with self.assertRaisesRegex(RuntimeError,'carry position='):robot._cartesian(np.zeros(3),True)
            robot.ik_diagnostic={'position_error_m':.003}
            state.geom_xmat[robot.geoms[0]]=(Rotation.from_euler('z',9,degrees=True).as_matrix()@robot.rotation).ravel()
            with self.assertRaisesRegex(RuntimeError,'grasp_angle='):robot._cartesian(np.zeros(3),True)
        self.assertEqual(robot.direct_wrist_policy,previous[0]);self.assertIs(robot.direct_rotation_target,previous[1])

    def test_placement_can_relax_attitude_but_cannot_bypass_original_collision_check(self):
        from unittest.mock import patch
        from human_task_robot import HumanTaskRobot
        from human_grasp_task import TaskCommand
        robot,data=self.robot(shared=True)
        goal=data.qpos[robot.spec.qpos].copy();old=robot.joints['right'].copy()
        state=robot._query_state();robot.rotation=state.geom_xmat[robot.geoms[0]].reshape(3,3).copy()
        robot.attached=True
        def solve(target):
            robot.ik_diagnostic={'position_error_m':.003}
            robot.placement_log.append({})
            return robot._cartesian(target,True),np.asarray(target)
        with patch.object(HumanTaskRobot,'_lower_goal',side_effect=solve), \
             patch.object(HumanTaskRobot,'_cartesian',side_effect=RuntimeError('cartesian_target_unreachable:fixture')), \
             patch.object(robot,'_query_state',return_value=state), \
             patch.object(robot,'_direct_position_goal',return_value=goal), \
             patch.object(robot,'_check_segment',side_effect=RuntimeError('task_motion_collision_rejected:lower:fixture')) as check:
            robot.execute(TaskCommand('lower',np.array([.43,0.,.43]),0.),1.)
        self.assertTrue(robot.ik_diagnostic['place_bounded_attitude_fallback'])
        check.assert_called_once();self.assertIn('task_motion_collision_rejected:',robot.warning)
        np.testing.assert_array_equal(robot.joints['right'],old)
        self.assertFalse(robot.shared_place_query)

    def test_release_exit_prior_is_fixed_short_lived_and_never_becomes_fresh_visual(self):
        from unittest.mock import patch
        from human_task_robot import pad_points
        from human_grasp_task import TaskCommand
        robot,data=self.robot(shared=True);robot.visual=None
        data.qpos[robot.fingers[0]]=.044
        robot.release_visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=1.)
        target=pad_points(robot.model,data,'right').mean(0)+[0,0,.07]
        prior=robot.release_visual['center'].copy()
        with patch.object(robot,'_execute_checked') as execute:
            robot.execute(TaskCommand('retract',target,.044),1.5)
            execute.assert_called_once()
            temporary=execute.call_args.args[0]
        self.assertEqual(temporary.stage,'retract');self.assertIsNone(robot.fresh_visual(1.5))
        self.assertEqual(robot.release_visual['time_s'],1.)
        np.testing.assert_array_equal(robot.release_visual['center'],prior)
        with patch.object(robot,'_execute_checked') as execute:
            robot.execute(TaskCommand('retract',target+[.005,0,0],.044),1.6)
            execute.assert_not_called()
        self.assertIn('target_changed',robot.warning)
        with patch.object(robot,'_execute_checked') as execute:
            robot.execute(TaskCommand('retract',target,.044),7.1)
            execute.assert_not_called()
        self.assertIn('expired',robot.warning)

    def test_release_exit_prior_cannot_open_closed_jaws_or_authorize_sideways_motion(self):
        from unittest.mock import patch
        from human_task_robot import pad_points
        from human_grasp_task import TaskCommand
        robot,data=self.robot(shared=True);robot.visual=None
        robot.release_visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=1.)
        target=pad_points(robot.model,data,'right').mean(0)+[0,0,.07]
        data.qpos[robot.fingers[0]]=.020
        with patch.object(robot,'_execute_checked') as execute:
            robot.execute(TaskCommand('retract',target,.044),1.5)
            execute.assert_not_called()
        data.qpos[robot.fingers[0]]=.044
        with patch.object(robot,'_execute_checked') as execute:
            robot.execute(TaskCommand('retract',target+[.005,0,0],.044),1.5)
            execute.assert_not_called()

    def test_duplicate_sensor_read_preserves_encoder_velocity(self):
        import mujoco
        robot,data=self.robot();robot.grasp_observation(0.)
        data.qpos[robot.fingers[0]]-=.001;mujoco.mj_forward(self.model,data)
        first=robot.grasp_observation(.1)
        data.qvel[robot.fingers[1]]=10.
        second=robot.grasp_observation(.1)
        np.testing.assert_allclose(first['finger_velocity'],[-.01,-.01],atol=1e-9)
        np.testing.assert_array_equal(second['finger_velocity'],first['finger_velocity'])

    def test_pending_planner_returns_without_consuming_future_or_object_truth(self):
        from concurrent.futures import Future
        from unittest.mock import patch
        from human_grasp_task import TaskCommand
        robot,data=self.robot(shared=True);pending=Future()
        with patch.object(robot.shared_executor,'submit',return_value=pending) as submit:
            robot.execute(TaskCommand('plan',finger=.044),0.)
        self.assertFalse(pending.done());self.assertIsNone(robot.plan)
        self.assertEqual(robot.command_stage,'plan')
        self.assertNotIn(robot.object_address,submit.call_args.args[3])
        robot.reset_shared_plan();self.assertIsNone(robot.shared_future);self.assertTrue(pending.cancelled())

    def test_restore_view_cannot_request_horizontal_motion_or_skip_collision_check(self):
        from unittest.mock import patch
        from human_grasp_task import TaskCommand
        from human_task_robot import pad_points
        robot,data=self.robot(shared=True);robot.shared_last_full_visual=robot.visual
        tcp=pad_points(robot.model,data,robot.side).mean(0)
        robot.rotation=data.geom_xmat[robot.geoms[0]].reshape(3,3).copy()
        robot.execute(TaskCommand('reacquire',tcp+[.01,0,.07]),.1)
        self.assertIn('outside_vertical_retreat',robot.warning)
        before=robot.joints[robot.side].copy()
        with patch.object(robot,'_direct_position_goal',return_value=before),patch.object(robot,'_check_segment',
                side_effect=RuntimeError('task_motion_collision_rejected:fixture')) as check:
            robot.execute(TaskCommand('reacquire',tcp+[0,0,.07]),.1)
        check.assert_called_once()
        self.assertIn('collision_rejected',robot.warning)
        np.testing.assert_array_equal(robot.joints[robot.side],before)


class SharedDatasetTests(unittest.TestCase):
    def test_waiting_before_start_cannot_corrupt_saved_frame_indices(self):
        from single_arm_runtime import EpisodeRows
        from shared_dataset import saved_control_index
        rows=EpisodeRows(ready_limit=40)
        for i in range(120):
            self.assertIsNone(saved_control_index(rows,'ready'))
            rows.append(dict(stage='ready',time_s=i*.05))
        index=saved_control_index(rows,'direct');rows.append(dict(stage='direct',time_s=6.))
        self.assertEqual(index,40);self.assertEqual(rows[index]['time_s'],6.)

    def test_rgbd_files_keep_the_control_row_index_and_metric_depth(self):
        import tempfile,json
        from pathlib import Path
        from shared_dataset import SharedFrameWriter
        with tempfile.TemporaryDirectory() as directory:
            writer=SharedFrameWriter(directory)
            writer.submit(17,{'test_camera':dict(rgb=np.full((12,16,3),128,np.uint8),depth=np.full((12,16),.4321,np.float32),time_s=.5)})
            report=writer.close();self.assertTrue(report['valid'])
            sample=json.loads((Path(directory)/report['index']).read_text(encoding='utf-8'))
            self.assertEqual(sample['control_frame'],17);self.assertEqual(sample['size_px'],[16,12])
            with np.load(Path(directory)/sample['depth']) as blob:
                np.testing.assert_allclose(blob['depth_m'],.4321,atol=1e-7)

    def test_partial_depth_presence_requires_observed_red_depth(self):
        from shared_grasp_robot import partial_depth_presence
        rgb=np.zeros((24,24,3),np.uint8);rgb[:,:,0]=255
        depth=np.full((24,24),.4,np.float32);matrix=np.array([[100.,0,12],[0,100.,12],[0,0,1.]])
        calibration=dict(xy_coeff=[[1.,0],[0,1.],[0,0]],z_bias_m=0.)
        hint=partial_depth_presence(rgb,depth,np.eye(4),matrix,calibration)
        self.assertGreater(hint['mask_pixels'],80)
        self.assertAlmostEqual(hint['center'][2],.4-.0288,places=6)
        self.assertIsNone(partial_depth_presence(rgb,np.full_like(depth,np.nan),np.eye(4),matrix,calibration))


if __name__=='__main__':unittest.main()
