"""One B executor with asynchronous, camera-only B grasp planning."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time
import numpy as np
import mujoco
import cv2
from scipy.spatial.transform import Rotation
from human_task_robot import HumanTaskRobot,pad_points,coordinated_joint_step
from human_grasp_task import TaskCommand
from automatic_grasp import bilateral_touch_estimate
from task_workspace import TABLE_TOP
from demo_grasp_viewer import CUBE_HALF
from handeye_grasp_red_cube import red_cube_mask
from calibration_simulation import body_transform
from demo_grasp_viewer import apply_workspace_rgbd_calibration


def partial_depth_presence(rgb,depth,camera_to_world,matrix,calibration):
    """A rough visible red depth component, NEVER an identified full cube pose."""
    mask=red_cube_mask(rgb)>0
    ys,xs=np.nonzero(mask & np.isfinite(depth) & (depth>.05) & (depth<2.))
    if len(xs)<80:return None
    pick=np.arange(0,len(xs),max(1,len(xs)//2000));x=xs[pick];y=ys[pick];z=depth[y,x]
    camera=np.column_stack(((x-matrix[0,2])*z/matrix[0,0],(y-matrix[1,2])*z/matrix[1,1],z,np.ones(len(z))))
    cloud=(camera_to_world@camera.T).T[:,:3]
    hint=np.median(cloud,axis=0);hint[2]=np.percentile(cloud[:,2],90)-CUBE_HALF
    return dict(center=apply_workspace_rgbd_calibration(hint,calibration),mask_pixels=len(xs))


def snapshot_plan(model,positions,addresses,spec,geoms,fingers,visual):
    state=mujoco.MjData(model);state.qpos[:]=model.qpos0
    state.qpos[addresses]=positions;mujoco.mj_forward(model,state)
    holder=SimpleNamespace(model=model,data=state,spec=spec,geoms=geoms,fingers=fingers,plan_log=[])
    # Reuse B's calibrated camera geometry and validated face-grasp solver.
    # The worker receives robot encoders and a camera estimate, no object truth.
    HumanTaskRobot._make_plan(holder,visual)
    return holder.plan,holder.rotation,holder.plan_log[-1]


class SharedGraspRobot(HumanTaskRobot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.shared_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='B-camera-grasp')
        self.shared_future=None;self.shared_plan_locked=False
        self.shared_planning_s=0.;self.shared_plan_started=None
        self.observer.known_cube_tracking=True
        self.observer.retain_frames=True
        self.shared_touch_conditioned=False
        self.shared_last_clear_time=None
        self.shared_last_full_visual=None;self.reacquire_origin=None
        self.shared_carry_query=False;self.shared_place_query=False
        self.shared_retract_target=None
        self.reacquire_owner=None

    def observe(self,clock):
        super().observe(clock)
        if self.visual is not None and not self.visual.get('pose_is_prior',False) and 0<=clock-self.visual['time_s']<=.5:
            self.shared_last_full_visual=dict(self.visual,center=np.asarray(self.visual['center']).copy())
        if self.plan is None or not self.shared_plan_locked or self.attached or self.command_stage not in ('transit','approach','descend','close'):return
        full=self.observer.measurements
        if full:
            matching=[v for v in full if np.linalg.norm(v['center']-self.plan['center'])<=.015]
            if matching:
                self.visual=min(matching,key=lambda v:np.linalg.norm(v['center']-self.plan['center']))
                self.shared_last_clear_time=clock
            return  # An independently complete moved target must still reject.
        # Closing jaws can hide the full top. A fresh, nearby red RGB-D
        # component corroborates the latched plan's presence, not a new centre
        # measurement. Encoders/effort still independently confirm the grasp.
        partial=list(self.observer.partial_measurements)
        for camera,frame in self.observer.frames.items():
            if any(v['source']==camera for v in partial):continue
            if camera=='chest_d435i_preview':
                transform=self.observer.chest_transform;matrix=self.observer.chest_matrix;calibration=self.observer.calibration['chest']
            else:
                handeye,matrix=self.observer.handeye[self.side]
                transform=body_transform(self.data,self.spec.body_id)@handeye
                calibration=self.observer.calibration['wrist'][self.side]
            hint=partial_depth_presence(frame['rgb'],frame['depth'],transform,matrix,calibration)
            if hint is not None:partial.append(dict(hint,source=camera,time_s=clock))
        compatible=[v for v in partial if
            np.linalg.norm(v['center']-self.plan['center'])<=.035 and
            abs(v['center'][2]-self.plan['center'][2])<=.010]
        clear_time=self.shared_last_clear_time if self.shared_last_clear_time is not None else self.plan.get('camera_time_s',clock)
        if compatible and clock-clear_time<=6.:
            latest=max(compatible,key=lambda v:v['mask_pixels'])
            self.visual=dict(center=self.plan['center'].copy(),yaw=self.plan['yaw'],time_s=latest['time_s'],
                source='fresh_partial_rgbd_presence_with_latched_plan',pose_is_prior=True,
                prior_time_s=self.plan.get('camera_time_s',self.plan.get('time_s',clock)))

    def reset_shared_plan(self):
        if self.attached:return
        if self.shared_future is not None:self.shared_future.cancel()
        self.shared_future=None;self.shared_plan_locked=False;self.plan=None
        self.shared_touch_conditioned=False
        self.contact_visual=None;self.pregrasp_visual=None;self.release_visual=None
        self.reacquire_origin=None
        self.reacquire_owner=None
        self.shared_retract_target=None
        self.shared_last_clear_time=None;self.shared_last_full_visual=None
        self.shared_carry_query=False;self.shared_place_query=False

    def feedback(self,clock):
        result=super().feedback(clock)
        result.plan_entry_stage=self.plan.get('entry_stage','transit') if self.plan else 'transit'
        return result

    def _make_plan(self,visual):
        # Descend uses the accepted, clear-view plan. Fresh camera consistency
        # is checked by B before each approach stage; no blocking replan occurs.
        if not self.shared_plan_locked:raise RuntimeError('camera_grasp_plan_not_ready')

    def _cartesian(self,target,closed):
        try:return super()._cartesian(target,closed)
        except RuntimeError as error:
            if not (self.shared_carry_query or self.shared_place_query) or not closed or not str(error).startswith('cartesian_target_unreachable:'):
                raise
            # The exact grasp attitude can be infeasible at a carry boundary,
            # even when retreating to the measured TCP. Permit at most 8deg
            # from the fixed grasp attitude, with <=4mm position error. The
            # same B joint limits and complete payload collision checks follow.
            # Placement uses the same bounded region: exact human-selected XY,
            # support height from the HELD geometry, at most 8deg wrist relief.
            previous=(self.direct_wrist_policy,self.direct_rotation_target)
            self.direct_wrist_policy='palm';self.direct_rotation_target=self.rotation
            state=self._query_state()
            try:
                goal=self._direct_position_goal(np.asarray(target),state,bounded=True)
                angle=float(Rotation.from_matrix(self.rotation@state.geom_xmat[self.geoms[0]].reshape(3,3).T).magnitude())
                position=float(self.ik_diagnostic['position_error_m'])
                if angle>np.deg2rad(8.) or position>.004:
                    raise RuntimeError(f'cartesian_target_unreachable: carry position={position}, grasp_angle={angle}')
                self.ik_diagnostic['place_bounded_attitude_fallback' if self.shared_place_query else
                    'carry_bounded_attitude_fallback']=True
                return goal
            finally:self.direct_wrist_policy,self.direct_rotation_target=previous

    def _lower_goal(self,requested):
        self.shared_place_query=True
        try:
            goal,xyz=super()._lower_goal(requested)
            if self.ik_diagnostic.get('place_bounded_attitude_fallback'):
                self.placement_log[-1].update(attitude_region_deg=8.,xy_shift_m=0.,
                    method='bounded_place_attitude_camera_attachment')
            return goal,xyz
        finally:self.shared_place_query=False

    def execute(self,command,clock,dt=.05):
        if command.stage=='retract' and not self.attached and self.fresh_visual(clock) is None:
            # The newly placed object can remain hidden by the open jaws. Use
            # its FIXED release estimate solely for the checked 7cm upward
            # exit. Never move the released collision object with the hand,
            # refresh its timestamp, or advertise this prior as a new view.
            prior=self.release_visual
            target=np.asarray(command.target)
            if self.shared_retract_target is None:
                origin=pad_points(self.model,self.data,self.side).mean(0)
                delta=target-origin
                if (prior is None or not 0<=clock-prior['time_s']<=6.
                    or np.min(self.data.qpos[self.fingers[0]])<=.040
                    or np.linalg.norm(delta[:2])>.001 or not 0<=delta[2]<=.071):
                    self.warning='camera_target_retract_requires_open_jaws_recent_release_and_vertical_exit'
                    return
                self.shared_retract_target=target.copy()
            if (prior is None or clock-prior['time_s']>6.
                or np.linalg.norm(target-self.shared_retract_target)>.001):
                self.warning='camera_target_retract_prior_expired_or_target_changed';return
            previous=self.visual
            self.visual=dict(prior,pose_is_prior=True,source='stationary_release_estimate_for_vertical_exit')
            try:self._execute_checked(command,clock,dt)
            finally:self.visual=previous
            return
        self._execute_checked(command,clock,dt)

    def _execute_checked(self,command,clock,dt):
        if command.stage=='hold' and self.attached:
            # HOLD is a fixed servo reference. Re-capturing lagging encoders
            # every tick erases restoring error and ratchets the loaded arm
            # downward under gravity, despite closed fingers.
            self.warning=self.error=None;self.ik_diagnostic={}
            self.execution_timing=dict(ik_ms=0.,collision_check_ms=0.,planning_query_ms=0.,payload_ms=0.)
            self.rejected_motion_axes=np.zeros(3,dtype=bool);self.contact_diagnostics={}
            self.contact_query_center=np.full(3,np.nan);self.contact_query_rotation=np.full((3,3),np.nan)
            self.reference_trajectories.clear()
            self.finger_targets[self.side]=float(command.finger)
            self.control.set_targets(self.joints,self.finger_targets)
            self.command_stage='hold';self.goal_q=self.goal_xyz=None
            return
        if command.stage=='reacquire':
            self._execute_reacquire(command,clock,dt)
            return
        if command.stage=='release' and self.plan is not None:
            if self.attached:
                frame=self.data.xmat[self.spec.body_id].reshape(3,3)
                center=pad_points(self.model,self.data,self.side).mean(0)+frame@self.attachment
                rotation=frame@self.attachment_rotation
                self.plan['center']=center.copy();self.plan['object_rotation']=rotation.copy()
                self.release_visual=dict(center=center.copy(),yaw=self.plan['yaw'],rotation=rotation.copy(),time_s=clock)
                self.release_query_active=True;self.release_query_until=clock+.6
            if self.release_query_active and (clock>self.release_query_until or np.min(self.data.qpos[self.fingers[0]])>.041):
                self.release_query_active=False
            self.pressure_releasing=True
        if command.stage=='plan':
            self.warning=self.error=None;self.ik_diagnostic={}
            self.execution_timing=dict(ik_ms=0.,collision_check_ms=0.,planning_query_ms=0.,payload_ms=0.)
            if self.plan is not None and self.shared_plan_locked:return
            visual=self.fresh_visual(clock)
            if visual is None:
                self.warning='camera_target_missing_at_shared_plan';return
            if self.shared_future is None:
                self.joints[self.side]=self.data.qpos[self.spec.qpos].copy()
                self.finger_targets[self.side]=.044;self.control.set_targets(self.joints,self.finger_targets)
                self.shared_plan_started=time.perf_counter()
                self.shared_future=self.shared_executor.submit(snapshot_plan,self.model,
                    self.data.qpos[self.robot_addresses].copy(),self.robot_addresses,self.spec,self.geoms,self.fingers,
                    dict(visual,center=np.asarray(visual['center']).copy()))
            if self.shared_future.done():
                try:
                    plan,rotation,log=self.shared_future.result()
                    if np.linalg.norm(plan['center']-visual['center'])>.015:
                        raise RuntimeError('camera_target_moved_during_shared_plan')
                    self.plan=plan;self.rotation=rotation;self.shared_plan_locked=True
                    self.plan['camera_time_s']=visual['time_s'];self.shared_touch_conditioned=False
                    self.shared_last_clear_time=visual['time_s']
                    # Skip the high transit only when B accepts the complete
                    # current->approach path. Grasp/descend checks are unchanged.
                    try:
                        self._check_segment(plan['q_approach'],.044,'approach')
                        self.plan['entry_stage']='approach'
                    except (RuntimeError,ValueError):self.plan['entry_stage']='transit'
                    self.shared_planning_s=time.perf_counter()-self.shared_plan_started
                    log.update(planning_s=self.shared_planning_s,worker='B camera/encoder snapshot')
                    log['entry_stage']=self.plan['entry_stage']
                    self.plan_log.append(log)
                except (RuntimeError,ValueError) as error:
                    self.warning='camera_grasp_plan_rejected:'+str(error)
                finally:self.shared_future=None
            self.command_stage='plan';self.goal_q=self.goal_xyz=None
            return
        if command.stage=='close':
            # Final pressure can build after physical fingers stop. Close at
            # 20mm/s rather than the legacy per-tick aperture increment.
            command=TaskCommand(command.stage,command.target,max(command.finger,self.finger_targets[self.side]-.020*dt),command.owner)
        if command.stage in ('close','lift') and not self.attached and self.plan is not None and not self.shared_touch_conditioned:
            obs=self.grasp_observation(clock)
            if obs['center'] is not None and np.linalg.norm(obs['center']-self.plan['center'])<=.015:
                obs=dict(obs,center=self.plan['center'].copy(),yaw=self.plan['yaw'])
                touch=bilateral_touch_estimate(obs,max_velocity=.008)
                if touch is not None:
                    center=touch['center'].copy();rotation=touch['rotation']
                    center[2]=max(center[2],TABLE_TOP+CUBE_HALF*np.abs(rotation[2]).sum()+.0005)
                    self.plan['center']=center;self.plan['object_rotation']=rotation
                    self.shared_touch_conditioned=True
                    self.plan_log[-1]['encoder_conditioned_object_estimate']=dict(center=center.tolist(),rotation=rotation.tolist())
        if (command.stage=='gripper_reference' and not self.attached and self.plan is not None
            and command.finger>=self.finger_targets[self.side] and self.fresh_visual(clock) is None):
            # An explicitly requested opening at a fixed arm pose need not wait
            # for the very object that the closed jaws currently occlude.
            self.warning=self.error=None
            self.execution_timing=dict(ik_ms=0.,collision_check_ms=0.,planning_query_ms=0.,payload_ms=0.)
            held=self.data.qpos[self.spec.qpos].copy();finger=min(command.finger,self.finger_targets[self.side]+.132*dt)
            try:self._check_segment(held,finger,'direct_grip')
            except (RuntimeError,ValueError) as error:self.warning=str(error);return
            self.joints[self.side]=held;self.finger_targets[self.side]=finger
            self.control.set_targets(self.joints,self.finger_targets);self.command_stage='gripper_reference'
            return
        last_safe_joints={side:q.copy() for side,q in self.joints.items()} if self.attached else None
        last_safe_fingers=dict(self.finger_targets) if self.attached else None
        self.shared_carry_query=command.stage=='transport' and self.attached
        try:super().execute(command,clock,dt)
        finally:self.shared_carry_query=False
        if self.error is not None and self.error.startswith(('task_motion_collision_rejected:',
                'camera_target_', 'camera_grasp_', 'cartesian_target_unreachable:')):
            self.warning=self.error;self.error=None
        if self.attached and self.warning and last_safe_joints is not None:
            # B rejected the NEW command; retain the previously checked
            # reference rather than moving the hold target to encoder lag.
            self.joints=last_safe_joints;self.finger_targets=last_safe_fingers
            self.control.set_targets(self.joints,self.finger_targets)

    def _execute_reacquire(self,command,clock,dt):
        """Open-jaw vertical escape using explicitly old camera geometry.

        Never feeds a prior to fresh_visual or to grasp planning. Every segment
        retains B's table/link checks and non-deepening existing-contact check.
        """
        self.warning=self.error=None;self.ik_diagnostic={}
        self.execution_timing=dict(ik_ms=0.,collision_check_ms=0.,planning_query_ms=0.,payload_ms=0.)
        self.rejected_motion_axes=np.zeros(3,dtype=bool);self.contact_diagnostics={}
        self.contact_query_center=np.full(3,np.nan);self.contact_query_rotation=np.full((3,3),np.nan)
        previous=(self.visual,self.direct_rotation_target,self.fast_query_mode)
        try:
            if self.attached:raise RuntimeError('camera_target_reacquire_cannot_move_payload')
            human_controlled=command.owner=='human_restore_view'
            if self.reacquire_origin is None or (not human_controlled and self.command_stage!='reacquire'):
                prior=self.shared_last_full_visual
                maximum_age=30. if command.owner=='human_restore_view' else 8.
                if prior is None or not 0<=clock-prior['time_s']<=maximum_age:
                    raise RuntimeError('camera_target_no_recent_geometry_for_retreat')
                self.reacquire_visual=dict(prior,center=prior['center'].copy())
                self.reacquire_origin=pad_points(self.model,self.data,self.side).mean(0)
                self.reacquire_rotation=self.data.geom_xmat[self.geoms[0]].reshape(3,3).copy()
                self.reacquire_started=clock
                self.reacquire_owner=command.owner
            target=np.asarray(command.target)
            delta=target-self.reacquire_origin
            if (target.shape!=(3,) or not np.isfinite(target).all() or np.linalg.norm(delta[:2])>.0001
                or not 0<=delta[2]<=.071 or clock-self.reacquire_started>(30. if human_controlled else 6.)
                or human_controlled and clock-self.reacquire_visual['time_s']>30.):
                raise RuntimeError('camera_target_reacquire_outside_vertical_retreat')
            self.visual=self.reacquire_visual;self.fast_query_mode=True
            self.direct_rotation_target=self.reacquire_rotation
            phase=time.perf_counter();state=self._query_state()
            # Advance a small vertical reference, at most 8mm ahead of the
            # encoders; avoid solving a large jump out of an existing contact.
            state.qpos[self.spec.qpos]=self.joints[self.side];mujoco.mj_kinematics(self.model,state)
            reference=pad_points(self.model,state,self.side).mean(0)
            actual=pad_points(self.model,self.data,self.side).mean(0)
            waypoint=self.reacquire_origin.copy()
            waypoint[2]=min(target[2],reference[2]+.040*dt,actual[2]+.008)
            goal=self._direct_position_goal(waypoint,state,bounded=True)
            self.execution_timing['ik_ms']=(time.perf_counter()-phase)*1000
            cap=np.array([.8]*4+[1.5]*3)*dt
            commanded=coordinated_joint_step(self.joints[self.side],goal,cap)
            old_finger=float(np.mean(self.data.qpos[self.fingers[0]]))
            finger=min(.044,self.finger_targets[self.side]+.132*dt)
            phase=time.perf_counter()
            self._check_segment(commanded,old_finger,'direct_retreat')
            self._check_segment(commanded,finger,'direct_retreat')
            self.execution_timing['collision_check_ms']=(time.perf_counter()-phase)*1000
            self.joints[self.side]=commanded;self.finger_targets[self.side]=finger
            self.control.set_targets(self.joints,self.finger_targets)
            self.command_stage='reacquire';self.goal_q=goal.copy();self.goal_xyz=target.copy()
        except (RuntimeError,ValueError) as error:
            self.warning=str(error);self.goal_q=self.goal_xyz=None
            self.joints[self.side]=self.data.qpos[self.spec.qpos].copy()
            self.control.set_targets(self.joints,self.finger_targets)
        finally:self.visual,self.direct_rotation_target,self.fast_query_mode=previous

    def _contacts(self,q,finger,attached,allow_grasp,allow_support,**kwargs):
        # Intended inner-pad contacts only. All other B object/table/robot
        # collisions still reject the path; no whole finger/cube exemption.
        previous=(self.direct_grasp_assist,self.fast_query_mode,self.grasp_contact_enabled,self.contact_scene_near)
        self.direct_grasp_assist=True;self.fast_query_mode=True
        self.grasp_contact_enabled=bool(allow_grasp)
        visual=self.plan or self.visual
        self.contact_scene_near=bool(attached or self.release_query_active or visual is not None and
            np.linalg.norm(pad_points(self.model,self.data,self.side).mean(0)-visual['center'])<.060)
        try:return super()._contacts(q,finger,attached,allow_grasp,allow_support,**kwargs)
        finally:
            self.direct_grasp_assist,self.fast_query_mode,self.grasp_contact_enabled,self.contact_scene_near=previous

    def close(self):
        if self.shared_future is not None:self.shared_future.cancel()
        self.shared_executor.shutdown(wait=True,cancel_futures=True)
        super().close()
