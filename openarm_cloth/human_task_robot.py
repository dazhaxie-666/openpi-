"""B robot execution adapter for human_grasp_task (red cube, simulation only).

Planning copies robot encoders plus CAMERA object estimates into a private query
state. Object ground truth is confined to TaskEvaluator, never fed to control.
"""
import mujoco
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from b_scene_fusion import BVisualObserver
from demonstration_transfer import visual_planning_state
from demo_grasp_viewer import CUBE_HALF, PAD_CONTACT_PROXY_LOCAL
from handeye_grasp_red_cube import finger_joint_indices, solve_gripper_contact_proxy_q
from human_grasp_task import Feedback
from human_teleop_core import PadCartesianIK
from member_b_actuators import ActuatorController
from task_workspace import TABLE_TOP
from visual_grasp_planner import plan_face_grasp
from live_reference_conditioning import JointReferenceTrajectory
from live_observation_quality import b_joint_reference


def pad_points(model, data, side):
    return np.array([data.geom_xpos[model.geom(f'openarm_{side}_{finger}_finger_collision').id]
                     + data.geom_xmat[model.geom(f'openarm_{side}_{finger}_finger_collision').id].reshape(3, 3) @ proxy
                     for finger, proxy in zip(('right', 'left'), PAD_CONTACT_PROXY_LOCAL)])


def supported_tcp_height(object_rotation, attachment_world, clearance=.0005):
    """Height at which all corners of the estimated held cube clear the table."""
    half_height = CUBE_HALF * np.sum(np.abs(np.asarray(object_rotation)[2]))
    return float(TABLE_TOP + half_height + clearance - np.asarray(attachment_world)[2])


def coordinated_joint_step(start,goal,cap):
    """One time scale for all seven joints keeps the checked IK path intact."""
    delta=np.asarray(goal)-start
    fraction=min(1.,float(np.min(np.asarray(cap)/np.maximum(np.abs(delta),1e-12))))
    return start+fraction*delta


def rotation_log_right_inverse(vector):
    """Derivative of Log(Exp(vector) Exp(delta)) at delta=0."""
    x,y,z=vector
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    angle=float(np.linalg.norm(vector))
    factor=(1./12.+angle*angle/720. if angle<1e-4 else
            (1.-.5*angle/np.tan(.5*angle))/(angle*angle))
    return np.eye(3)+.5*skew+factor*(skew@skew)


class HumanTaskRobot:
    def __init__(self, model, data, side='right', lift_m=.07):
        if side != 'right':
            raise ValueError('First task version uses the calibrated right arm')
        self.model, self.data, self.side = model, data, side
        self.control = ActuatorController(model, data)
        self.spec = self.control.specs[side]
        self.fingers = finger_joint_indices(model, side)
        self.geoms = tuple(model.geom(f'openarm_{side}_{f}_finger_collision').id for f in ('right', 'left'))
        self.object_geom = model.geom('droid_object_geom').id
        self.table_geom = model.geom('folding_table').id
        self.object_address = int(model.joint('droid_object_freejoint').qposadr[0])
        self.joints = {s: data.qpos[spec.qpos].copy() for s, spec in self.control.specs.items()}
        self.finger_targets = {s: .044 for s in self.joints}
        self.control.set_targets(self.joints, self.finger_targets)
        self.observer = BVisualObserver(model, self.control.specs, camera_sides=(None, side))
        self.visual = None
        self.plan = None
        self.command_stage = ''
        self.goal_q = self.goal_xyz = None
        self.error = None
        self.warning = None
        self.direct_ik_mode = 'fixed_wrist'
        self.direct_wrist_policy = 'fixed'
        self.direct_rotation_target=None
        self.ik_diagnostic = {}
        self.direct_grasp_assist=False;self.grasp_contact_enabled=False;self.auto_grip_confirmed=False
        self.rejected_motion_axes=np.zeros(3,dtype=bool);self.contact_normals={};self.contact_diagnostics={}
        self.contact_visual=None;self.release_visual=None;self.release_query_active=False;self.release_query_until=0.;self.pressure_releasing=False
        self.clear_view_visual=None;self.pregrasp_visual=None
        self.finger_encoder_sample=None
        self.finger_encoder_velocity=np.zeros(2)
        self.contact_scene_near=False
        self.contact_query_center=np.full(3,np.nan);self.contact_query_rotation=np.full((3,3),np.nan)
        self.reference_sides = ()
        self.reference_blocked_sides = []
        self.reference_step_fraction = dict(left=0.,right=0.)
        self.reference_trajectories = {}
        self.lift_m = lift_m
        self.attached = False
        self.attachment = None
        self.attachment_rotation = None
        self.query = mujoco.MjData(model)
        self.ik_query = mujoco.MjData(model)
        self.fast_query_mode = False
        self.execution_timing = {}
        self.robot_addresses = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
                                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').startswith('openarm_')]
        self.robot_geoms = {g for g in range(model.ngeom)
                            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or '').startswith('openarm_')}
        self.rotation = None
        self.grip_span = None
        self.grip_finger = None
        self.plan_log = []
        self.placement_log = []
        self.motion_origin = None
        self.motion_progress = 0.
        self.motion_duration = 0.
        self.ready_tcp=np.array([.43,0.,.54])

    def initialize(self):
        """Episode reset posture only, before input/recording; no object is moved.

        Uses B's pad IK at a fixed workspace coordinate, independent of the cube.
        Subsequent episode motion uses only dynamic actuators.
        """
        # Start above the cube rather than above the goal zone. The old
        # y=-.10 start forced the operator to spend most of the relative wrist
        # travel just reaching the cube, and made the workspace clip early.
        ready=np.asarray(self.ready_tcp,float)
        if ready.shape!=(3,) or not np.isfinite(ready).all():raise ValueError('ready TCP must be a finite XYZ vector')
        ik = PadCartesianIK(self.model, self.data, ready=ready)
        self.data.qpos[self.spec.qpos] = ik.ready_q
        self.data.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.joints[self.side] = ik.ready_q.copy()
        self.rotation = self.data.geom_xmat[self.geoms[0]].reshape(3, 3).copy()
        if self.direct_wrist_policy=='palm':
            # The legacy ready solve puts wrist joint 6 on its negative stop.
            # A fixed +30deg tabletop yaw puts that joint near its centre,
            # allowing useful motion in BOTH directions. This reset is before
            # input/recording and independent of the target object's pose.
            self.direct_rotation_target=Rotation.from_euler('z',30,degrees=True).as_matrix()@self.rotation
            state=self.ik_query;state.qpos[:]=self.model.qpos0
            state.qpos[self.robot_addresses]=self.data.qpos[self.robot_addresses]
            q=self._direct_position_goal(ready,state,bounded=True)
            self.data.qpos[self.spec.qpos]=q;self.data.qvel[:]=0.
            mujoco.mj_forward(self.model,self.data);self.joints[self.side]=q.copy()
            self.rotation=self.data.geom_xmat[self.geoms[0]].reshape(3,3).copy()
            self.direct_rotation_target=self.rotation.copy()
        self.control.set_targets(self.joints, self.finger_targets)
        for _ in range(round(.6 / self.model.opt.timestep)):
            self.control.apply()
            mujoco.mj_step(self.model, self.data)

    def capture_collection_reset_state(self):
        """Capture the post-initialisation simulator state used between trials."""
        return dict(qpos=self.data.qpos.copy(),act=self.data.act.copy(),ctrl=self.data.ctrl.copy(),
                    ready_tcp=np.asarray(self.ready_tcp).copy())

    def restore_collection_reset_state(self,state):
        """Reset the simulator between labelled trials; never called mid-trial."""
        if np.shape(state.get('qpos'))!=(self.model.nq,):raise ValueError('invalid collection reset qpos')
        simulation_time=float(self.data.time)
        self.data.qpos[:]=state['qpos'];self.data.qvel[:]=0.
        if self.model.na:self.data.act[:]=state['act']
        self.data.ctrl[:]=state['ctrl'];self.data.qacc_warmstart[:]=0.
        self.data.time=simulation_time
        mujoco.mj_forward(self.model,self.data)
        self.joints={s:self.data.qpos[spec.qpos].copy() for s,spec in self.control.specs.items()}
        self.finger_targets={s:.044 for s in self.joints}
        self.control.set_targets(self.joints,self.finger_targets)
        self.attached=False;self.attachment=self.attachment_rotation=None
        if hasattr(self,'reset_shared_plan'):self.reset_shared_plan()
        self.visual=self.plan=None;self.command_stage='';self.goal_q=self.goal_xyz=None
        self.error=self.warning=None;self.ik_diagnostic={};self.direct_ik_mode='fixed_wrist'
        self.rotation=self.data.geom_xmat[self.geoms[0]].reshape(3,3).copy()
        if self.direct_wrist_policy=='palm':self.direct_rotation_target=self.rotation.copy()
        self.clear_view_visual=self.pregrasp_visual=self.contact_visual=self.release_visual=None
        self.release_query_active=False;self.release_query_until=0.;self.pressure_releasing=False
        self.finger_encoder_sample=None;self.finger_encoder_velocity=np.zeros(2)
        self.contact_scene_near=False;self.contact_query_center=np.full(3,np.nan)
        self.contact_query_rotation=np.full((3,3),np.nan);self.contact_diagnostics={}
        self.reference_trajectories.clear();self.motion_origin=None;self.motion_progress=0.;self.motion_duration=0.
        self.observer.last=None;self.observer.attempts=[];self.observer.measurements=[]
        self.observer.partial_measurements=[];self.observer.frames={}

    def observe(self, clock):
        self.observer.observe(self.data, clock)
        self.visual = self.observer.last
        # Keep the last unobstructed approach estimate. Closing jaws can
        # truncate a cube's visible top and bias its centre / square yaw.
        if self.visual is not None and 0<=clock-self.visual['time_s']<=.5 and not self.attached:
            tcp=pad_points(self.model,self.data,self.side).mean(0)
            if np.linalg.norm(self.visual['center']-tcp)>.075 and np.min(self.data.qpos[self.fingers[0]])>.035:
                self.clear_view_visual=dict(self.visual,center=np.asarray(self.visual['center']).copy())

    def fresh_visual(self, clock):
        if self.visual is not None and 0 <= clock - self.visual['time_s'] <= .5:
            return self.visual
        return None

    def feedback(self, clock):
        tcp = pad_points(self.model, self.data, self.side).mean(axis=0)
        reached = self.goal_q is not None and np.max(np.abs(self.data.qpos[self.spec.qpos] - self.goal_q)) < .025
        if self.goal_xyz is not None:
            reached = reached and np.linalg.norm(tcp - self.goal_xyz) < .008
        grip = (np.all(self.data.qpos[self.fingers[0]] > .008)
                and np.all(np.abs(self.data.actuator_force[self.control.fingers[self.side]]) > .8))
        return Feedback(tcp=tcp, visual=self.fresh_visual(clock), command_stage=self.command_stage,
                        reached=bool(reached), gripping=bool(grip),
                        opened=bool(np.min(self.data.qpos[self.fingers[0]]) > .040),
                        plan_ready=self.plan is not None, error=self.error)

    def grasp_observation(self,clock):
        """Robot sensors and CAMERA object estimate; never object qpos/contact truth."""
        visual=self.fresh_visual(clock)
        self.pregrasp_visual=None
        if self.direct_grasp_assist and visual is not None and self.clear_view_visual is not None and not self.attached:
            # This is a stationary-target PRIOR corroborated by a fresh image,
            # not a fresh measurement at the prior coordinate. Large observed
            # displacement invalidates it rather than moving a hidden object.
            if np.linalg.norm(np.asarray(visual['center'])-self.clear_view_visual['center'])<=.015:
                self.pregrasp_visual=self.clear_view_visual
            else:self.clear_view_visual=None
        if self.contact_visual is not None and not 0<=clock-self.contact_visual['time_s']<=.5:
            self.contact_visual=None
        points=pad_points(self.model,self.data,self.side)
        positions=self.data.qpos[self.fingers[0]].copy();velocity=self.data.qvel[self.fingers[1]].copy()
        sample=self.finger_encoder_sample
        if sample is not None and clock>sample[0]+1e-6:velocity=(positions-sample[1])/(clock-sample[0])
        elif sample is not None and clock==sample[0]:velocity=self.finger_encoder_velocity.copy()
        if sample is None or clock>sample[0]:
            self.finger_encoder_sample=(clock,positions.copy());self.finger_encoder_velocity=velocity.copy()
        obs=dict(tcp=points.mean(0),pads=points,
            center=None if visual is None else np.asarray(visual['center']).copy(),
            yaw=0. if visual is None else float(visual['yaw']),
            visual_age_s=np.inf if visual is None else clock-visual['time_s'],
            finger_positions=positions,finger_velocity=velocity,
            finger_force=self.data.actuator_force[self.control.fingers[self.side]].copy(),
            finger_target=float(self.finger_targets[self.side]),table_top=TABLE_TOP,half_size=CUBE_HALF)
        if self.pregrasp_visual is not None:
            obs['raw_center']=obs['center'].copy();obs['center']=self.pregrasp_visual['center'].copy()
            obs['yaw']=float(self.pregrasp_visual['yaw'])
        obs['clear_view_prior_active']=self.pregrasp_visual is not None
        obs['clear_view_prior_age_s']=np.nan if self.pregrasp_visual is None else clock-self.pregrasp_visual['time_s']
        self.contact_scene_near=bool(visual is not None and np.linalg.norm(visual['center']-points.mean(0))<.055)
        if self.direct_grasp_assist and self.grasp_contact_enabled and not self.attached:
            from automatic_grasp import bilateral_touch_estimate
            refined=bilateral_touch_estimate(obs)
            if refined is not None:self.contact_visual=dict(visual,**refined)
        if self.contact_visual is not None and visual is not None and not self.release_query_active:
            # Keep source age honest and reuse only during the same closing
            # episode. Opening clears this contact-conditioned estimate.
            if self.grasp_contact_enabled:
                obs.setdefault('raw_center',obs['center'].copy());obs['center']=self.contact_visual['center'].copy()
        return obs

    def _make_plan(self, visual):
        # Known calibrated support plane + known cube size, not simulator pose.
        # A noisy RGB-D centre must not place the private cube below its table.
        visual = dict(visual, center=np.asarray(visual['center']).copy())
        raw_center = visual['center'].copy()
        visual['center'][2] = max(visual['center'][2], TABLE_TOP + CUBE_HALF + .0005)
        state = visual_planning_state(self.model, self.data, visual['center'], visual['yaw'])
        seed = self.data.qpos[self.spec.qpos].copy()
        plan = plan_face_grasp(self.model, state, self.spec, self.geoms, self.fingers,
                               PAD_CONTACT_PROXY_LOCAL, visual['center'], visual['yaw'],
                               seed, seed, .044, .07, .12)
        metrics = plan['metrics']
        if (metrics['midpoint_error_m'] > .006 or metrics['perpendicular_error_m'] > .008
                or metrics['max_new_penetration_mm'] > .5):
            raise RuntimeError('camera_grasp_plan_rejected')
        self.plan = plan
        self.plan['center'] = np.asarray(visual['center']).copy()
        self.plan['yaw'] = float(visual['yaw'])
        self.rotation = plan['rotation'].copy()
        self.plan_log.append(dict(center=self.plan['center'].tolist(), raw_camera_center=raw_center.tolist(),
                                  support_plane_constraint=True, yaw=self.plan['yaw'], metrics=metrics))

    def _query_state(self):
        visual = self.plan if self.attached else (self.contact_visual or self.pregrasp_visual or self.visual) if self.direct_grasp_assist else self.visual
        if visual is None:
            raise RuntimeError('camera_target_missing')
        if getattr(self,'fast_query_mode',False):
            state=self.ik_query
            state.qpos[:]=self.model.qpos0;state.qvel[:]=0.;state.ctrl[:]=0.
            state.qpos[self.robot_addresses]=self.data.qpos[self.robot_addresses]
            adr=self.object_address;yaw=visual['yaw']
            state.qpos[adr:adr+3]=visual['center']
            state.qpos[adr+3:adr+7]=[np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
            mujoco.mj_kinematics(self.model,state)
            return state
        return visual_planning_state(self.model, self.data, visual['center'], visual['yaw'])

    def _cartesian(self, target, closed):
        state = self._query_state()
        points = pad_points(self.model, self.data, self.side)
        finger = self.grip_finger if closed else .044
        span = self.grip_span if closed else float(np.linalg.norm(points[0] - points[1]))
        direction = self.plan['direction'] if self.plan else (points[0] - points[1]) / span
        q, _, _, metrics = solve_gripper_contact_proxy_q(
            self.model, state, self.spec, self.geoms, self.fingers, PAD_CONTACT_PROXY_LOCAL,
            target, 1, 1., span, finger, self.joints[self.side], reference_rotation=self.rotation,
            multi_start=False, closing_direction=direction)
        if metrics['midpoint_error_m'] > .008 or metrics['perpendicular_error_m'] > .008:
            raise RuntimeError(f'cartesian_target_unreachable: {metrics}')
        return q

    def _contacts(self, q, finger, attached, allow_grasp, allow_support, joint_overrides=None, finger_overrides=None):
        query = self.query
        controlled=bool(getattr(self,'direct_grasp_assist',False) and getattr(self,'fast_query_mode',False))
        project_fingers=bool(controlled and (self.grasp_contact_enabled or self.pressure_releasing)
            and (self.contact_scene_near or attached or self.release_query_active))
        query.qpos[:] = self.model.qpos0
        query.qpos[self.robot_addresses] = self.data.qpos[self.robot_addresses]
        query.qpos[self.spec.qpos] = q
        query.qpos[self.fingers[0]] = self.data.qpos[self.fingers[0]] if project_fingers else finger
        for side, target in (joint_overrides or {}).items():
            query.qpos[self.control.specs[side].qpos] = target
        for side, aperture in (finger_overrides or {}).items():
            query.qpos[finger_joint_indices(self.model, side)[0]] = aperture
        fast=getattr(self,'fast_query_mode',False)
        if not fast:
            mujoco.mj_forward(self.model, query)
        elif attached or controlled and self.release_query_active:
            mujoco.mj_kinematics(self.model,query)
        # Keep the same measured object geometry as the accepted docking plan.
        # Fresh frames still gate movement detection; occlusion must not silently
        # move collision geometry while the joint goal stays on an old target.
        visual = self.plan if self.plan is not None else (self.release_visual if self.release_query_active else self.contact_visual or self.pregrasp_visual or self.visual) if controlled else self.visual
        center = np.asarray(visual['center']).copy()
        yaw = visual['yaw']
        predicted_payload=attached or bool(controlled and self.release_query_active)
        if predicted_payload:
            tcp = pad_points(self.model, query, self.side).mean(axis=0)
            frame = query.xmat[self.spec.body_id].reshape(3, 3)
            center = tcp + frame @ self.attachment
            object_rotation = frame @ self.attachment_rotation
        query.qpos[self.object_address:self.object_address + 3] = center
        if predicted_payload:
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, object_rotation.ravel())
            query.qpos[self.object_address + 3:self.object_address + 7] = quat
        else:
            # A B grasp PLAN's rotation is the tool frame. Camera/contact
            # estimates may carry an object rotation; never confuse the two.
            object_rotation=visual.get('object_rotation') if visual is self.plan else visual.get('rotation')
            if controlled and object_rotation is not None:
                quat=np.empty(4);mujoco.mju_mat2Quat(quat,np.asarray(object_rotation).ravel())
                query.qpos[self.object_address+3:self.object_address+7]=quat
            else:query.qpos[self.object_address + 3:self.object_address + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        if project_fingers:
            from automatic_grasp import predicted_contact_stop
            actual_fingers=self.data.qpos[self.fingers[0]].copy()
            query.qpos[self.fingers[0]]=actual_fingers
            mujoco.mj_kinematics(self.model,query)
            stop=predicted_contact_stop(pad_points(self.model,query,self.side),actual_fingers,
                                       finger,center,yaw,CUBE_HALF,object_rotation)
            query.qpos[self.fingers[0]]=stop
        if controlled:
            self.contact_query_center=center.copy()
            self.contact_query_rotation=(np.asarray(object_rotation).copy() if object_rotation is not None else
                np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]]))
        if fast:
            # Position pipeline includes collision detection and constraints;
            # query code consumes contact distances, not forces/accelerations.
            mujoco.mj_fwdPosition(self.model,query)
        else:
            mujoco.mj_forward(self.model, query)
        hits = {}
        pads=None;contact_enabled=False
        self.contact_normals={}
        for c in query.contact:
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            # Same coupled finger-pair exemption as B's TaskGeometry. The
            # physical servo/contact solver, not query teleportation, closes it.
            if set(pair) == set(self.geoms):
                continue
            if finger_overrides is not None and set(pair)=={
                self.model.geom('openarm_left_right_finger_collision').id,
                self.model.geom('openarm_left_left_finger_collision').id}:
                continue
            if not set(pair) & self.robot_geoms and not (attached and self.object_geom in pair):
                continue
            if self.object_geom in pair and set(pair) & set(self.geoms):
                if controlled:
                    if self.grasp_contact_enabled or self.pressure_releasing:
                        if pads is None:
                            from automatic_grasp import allowed_pad_contact
                            pads=pad_points(self.model,query,self.side)
                            travel=float(np.max(np.linalg.norm(pads-pad_points(self.model,self.data,self.side),axis=1)))
                            contact_enabled=bool(predicted_payload or travel<=.0025)
                        finger=next(g for g in self.geoms if g in pair);j=self.geoms.index(finger)
                        normal=np.asarray(c.frame[:3])*(1 if int(c.geom1)==finger else -1)
                        if allowed_pad_contact(np.asarray(c.pos),normal,pads[j],pads[1-j],center,
                                max(0.,-float(c.dist)),contact_enabled):
                            self.contact_diagnostics['expected_pad_contacts']=self.contact_diagnostics.get('expected_pad_contacts',0)+1
                            continue
                elif allow_grasp:continue
            if attached and allow_support and set(pair) == {self.object_geom, self.table_geom}:
                if c.dist >= -.001:
                    continue
            if c.dist < 0:
                hits[pair] = -float(c.dist)
                self.contact_normals[pair]=np.asarray(c.frame[:3]).copy()
        return hits

    def _lower_goal(self, requested):
        # A tilted payload needs its lowest corner, rather than its nominal
        # centre height, aligned with the calibrated support plane. XY remains
        # exactly the human-selected release position.
        xyz = np.asarray(requested, float).copy()
        for _ in range(3):
            goal = self._cartesian(xyz, True)
            query = self._query_state()
            query.qpos[self.spec.qpos] = goal
            mujoco.mj_forward(self.model, query)
            frame = query.xmat[self.spec.body_id].reshape(3, 3)
            minimum = supported_tcp_height(frame @ self.attachment_rotation, frame @ self.attachment)
            correction = max(0., minimum - xyz[2])
            if correction < .00005:
                break
            xyz[2] += correction
        if xyz[2] - requested[2] > .02:
            raise RuntimeError('held_geometry_requires_excessive_support_correction')
        self.placement_log.append(dict(requested_tcp=np.asarray(requested).tolist(),
                                       supported_tcp=xyz.tolist(), source='camera_grip_attachment_and_known_cube_geometry'))
        return self._cartesian(xyz, True), xyz

    def _check_segment(self, goal, finger, stage):
        start = self.data.qpos[self.spec.qpos].copy()
        old_finger = float(np.mean(self.data.qpos[self.fingers[0]]))
        grasp = stage in ('close', 'lift', 'transport', 'lower', 'release', 'direct_grip', 'direct_carry')
        support = stage in ('lift', 'lower', 'release', 'direct_carry', 'direct_grip')
        baseline = self._contacts(start, old_finger, self.attached, grasp, support)
        # Assisted paths never exempt object penetrations as a baseline. Direct
        # retries allow escape from existing contact, never deepening it (1 um
        # numerical tolerance). New contact retains the 0.5 mm rejection limit.
        if not stage.startswith('direct_'):
            baseline = {pair: depth for pair, depth in baseline.items() if self.object_geom not in pair}
        count = max(1, int(np.ceil(np.max(np.abs(goal - start)) / .02)),
                    int(np.ceil(abs(finger - old_finger) / .004)))
        for a in np.linspace(0, 1, count + 1):
            hits = self._contacts(start + a * (goal - start), old_finger + a * (finger - old_finger),
                                  self.attached, grasp, support)
            rejected = [(pair, depth, baseline.get(pair, 0.)) for pair, depth in hits.items()
                        if depth - baseline.get(pair, 0.) > (
                            .000001 if stage.startswith('direct_') and pair in baseline else .0005)]
            if rejected:
                if getattr(self,'direct_grasp_assist',False) and stage.startswith('direct_'):
                    for pair,_,_ in rejected:
                        normal=self.contact_normals.get(pair)
                        if normal is not None:self.rejected_motion_axes|=np.abs(normal)>=.5*np.max(np.abs(normal))
                detail = [dict(geoms=[self.model.geom(g).name for g in pair], depth_m=depth,
                               baseline_m=old) for pair, depth, old in rejected]
                raise RuntimeError(f'task_motion_collision_rejected:{stage}:{detail}')

    def execute(self, command, clock, dt=.05):
        self.warning = None
        self.ik_diagnostic = {}
        self.rejected_motion_axes=np.zeros(3,dtype=bool);self.contact_diagnostics={}
        self.contact_query_center=np.full(3,np.nan);self.contact_query_rotation=np.full((3,3),np.nan)
        if getattr(self,'direct_grasp_assist',False):
            self.pressure_releasing=command.finger>self.finger_targets[self.side]+1e-9
        self.fast_query_mode=command.stage in ('direct','gripper_reference')
        self.execution_timing=dict(ik_ms=0.,collision_check_ms=0.,planning_query_ms=0.,payload_ms=0.)
        try:
            self._execute(command, clock, dt)
        except (RuntimeError, ValueError) as error:
            reason = str(error)
            recoverable = command.stage in ('direct', 'pose2arm_reference','gripper_reference') and reason.startswith((
                'cartesian_target_unreachable:', 'task_motion_collision_rejected:', 'camera_target_missing'))
            self.warning = reason if recoverable else None
            self.error = None if recoverable else reason
            self.joints[self.side] = self.data.qpos[self.spec.qpos].copy()
            if command.stage == 'pose2arm_reference':
                for side in self.reference_sides:
                    self.joints[side] = self.data.qpos[self.control.specs[side].qpos].copy()
            self.goal_q = self.goal_xyz = None
            # Hold the existing finger target on a failed command; do not drop payload.
            self.control.set_targets(self.joints, self.finger_targets)
            if recoverable and command.stage=='direct':
                # An unreachable arm target does not invalidate an independently
                # observed pinch. Check fingers at the actual held arm pose.
                try:self._execute_gripper_reference(command,clock,dt)
                except (RuntimeError,ValueError) as grip_error:
                    grip_reason=str(grip_error)
                    if grip_reason.startswith(('task_motion_collision_rejected:','camera_target_missing')):
                        self.warning=reason+'; gripper held: '+grip_reason
                    else:self.error=grip_reason;self.warning=None

    def _execute_gripper_reference(self,command,clock,dt):
        if not np.isfinite(command.finger) or not 0<=command.finger<=.044:raise ValueError('invalid gripper aperture')
        phase=time.perf_counter()
        self._direct_payload(clock,opening=command.finger>.022)
        self.execution_timing['payload_ms']+=(time.perf_counter()-phase)*1000
        if not self.attached and self.fresh_visual(clock) is None and not (self.direct_grasp_assist and self.release_query_active):
            raise RuntimeError('camera_target_missing_for_gripper_collision_check')
        held=self.data.qpos[self.spec.qpos].copy()
        finger=float(self.finger_targets[self.side]+np.clip(command.finger-self.finger_targets[self.side],-.132*dt,.132*dt))
        phase=time.perf_counter()
        try:self._check_segment(held,finger,'direct_grip')
        finally:self.execution_timing['collision_check_ms']+=(time.perf_counter()-phase)*1000
        self.joints[self.side]=held;self.finger_targets[self.side]=finger
        self.control.set_targets(self.joints,self.finger_targets)
        self.command_stage='gripper_reference';self.goal_q=self.goal_xyz=None

    def _check_reference_path(self, targets, fingers, allow_grasp=False):
        start={s:self.data.qpos[self.control.specs[s].qpos].copy() for s in self.reference_sides}
        aperture={s:float(np.mean(self.data.qpos[finger_joint_indices(self.model,s)[0]])) for s in self.reference_sides}
        count=max([1]+[int(np.ceil(np.max(np.abs(targets[s]-start[s]))/.02)) for s in start]
                  +[int(np.ceil(abs(fingers[s]-aperture[s])/.004)) for s in start])
        def contacts(qs, fs):
            return self._contacts(qs.get('right',self.data.qpos[self.spec.qpos]),
                                  fs.get('right',float(np.mean(self.data.qpos[self.fingers[0]]))),
                                  self.attached, allow_grasp or self.attached, self.attached,
                                  joint_overrides=qs, finger_overrides=fs)
        baseline=contacts(start,aperture)
        for a in np.linspace(0.,1.,count+1):
            qs={s:start[s]+a*(targets[s]-start[s]) for s in start}
            fs={s:aperture[s]+a*(fingers[s]-aperture[s]) for s in start}
            for pair,depth in contacts(qs,fs).items():
                if depth-baseline.get(pair,0.) > (.000001 if pair in baseline else .0005):
                    names=[self.model.geom(g).name for g in pair]
                    raise RuntimeError(f'task_motion_collision_rejected:pose2arm_reference:{names}: depth={depth}')

    def _execute_reference(self, command, clock, dt):
        self._direct_payload(clock, opening=command.finger > .022)
        if not self.attached and self.fresh_visual(clock) is None:
            raise RuntimeError('camera_target_missing_for_reference_collision_check')
        targets={s:self.data.qpos[self.control.specs[s].qpos].copy() for s in self.reference_sides}
        fingers={s:float(self.finger_targets[s]) for s in self.reference_sides}
        # Same causal conditioner as free following, with the existing slower
        # tabletop speed caps. B's task collision checks still have final say.
        stamp=command.reference_capture_s if command.reference_capture_s is not None else clock
        rejected={}
        for side in self.reference_sides:
            if side not in self.reference_trajectories:
                self.reference_trajectories[side]=JointReferenceTrajectory(self.joints[side],
                    speed=[.8]*4+[1.5]*3,acceleration=[4.]*4+[7.5]*3,limits=self.control.specs[side].q_range)
        for side,q in (command.joint_targets or {}).items():
            if side not in targets: raise ValueError('unconfigured reference side')
            limits=self.control.specs[side].q_range
            q,reason=b_joint_reference(q,limits)
            if reason:
                rejected[side]=reason
                continue
            trajectory=self.reference_trajectories[side]
            trajectory.observe(q,stamp)
            for _ in range(20):targets[side],_=trajectory.advance(dt/20.)
        for side,value in (command.finger_targets or {}).items():
            if side in fingers:
                if not np.isfinite(value) or not 0<=value<=.044:
                    rejected[side]='invalid A gripper reference: hold'
                    continue
                fingers[side]=float(self.finger_targets[side]+np.clip(value-self.finger_targets[side],-.0066,.0066))
        # Resolve each side locally: a rejected right-arm step must not freeze
        # the observed left arm or prevent an explicit pinch/open command.
        # Nominal A references are untouched; B only bounds the applied step.
        old_fingers={s:float(np.mean(self.data.qpos[finger_joint_indices(self.model,s)[0]])) for s in targets}
        origin={s:self.data.qpos[self.control.specs[s].qpos].copy() for s in targets}
        accepted={s:q.copy() for s,q in origin.items()}
        applied_fingers=old_fingers.copy()
        blocked=[s+': '+reason for s,reason in rejected.items()]
        self.reference_blocked_sides=[]
        self.reference_step_fraction=dict(left=0.,right=0.)
        for side in targets:
            reason=None
            for fraction in (1.,.5,.25,.125):
                trial=dict(accepted)
                trial[side]=origin[side]+fraction*(targets[side]-origin[side])
                try:
                    self._check_reference_path(trial,old_fingers)
                    # Simultaneous interpolation is checked with all previously
                    # accepted sides, not only the isolated moving arm.
                    accepted=trial
                    self.reference_step_fraction[side]=fraction
                    reason=None
                    break
                except RuntimeError as error:
                    if not str(error).startswith('task_motion_collision_rejected:'):raise
                    reason=str(error)
            if reason:
                blocked.append(reason)
                self.reference_blocked_sides.append(side)
        for side in fingers:
            trial=dict(applied_fingers,**{side:fingers[side]})
            try:
                self._check_reference_path(accepted,trial,allow_grasp=True)
                applied_fingers=trial
            except RuntimeError as error:
                if not str(error).startswith('task_motion_collision_rejected:'):raise
                blocked.append(str(error))
        for side in accepted:
            self.joints[side]=accepted[side]
            self.finger_targets[side]=applied_fingers[side]
            trajectory=self.reference_trajectories[side]
            if side in rejected or side not in (command.joint_targets or {}) or self.reference_step_fraction[side]<1.:
                # A collision must not leave hidden accumulated velocity that
                # surges forward as soon as the side becomes unblocked.
                trajectory.position=accepted[side].copy()
                trajectory.velocity[:]=0.
                trajectory.filter.stamp=None
        self.control.set_targets(self.joints,self.finger_targets)
        self.warning='; '.join(blocked) if blocked else None
        self.command_stage='pose2arm_reference'
        self.direct_ik_mode='pose2arm_original_joint_reference'
        self.goal_q=self.goal_xyz=None

    def _direct_payload(self, clock, opening):
        """Estimate attachment from encoders/effort plus camera, never truth."""
        gripping = self.feedback(clock).gripping
        if getattr(self,'direct_grasp_assist',False):
            gripping=gripping and self.auto_grip_confirmed
            opening=not self.auto_grip_confirmed
        if self.attached and (opening or not gripping):
            if getattr(self,'direct_grasp_assist',False) and opening:
                self.release_query_active=True;self.release_query_until=clock+.6
                self.release_visual=dict(self.plan or self.contact_visual or self.visual)
            self.attached = False
            self.plan = None
        if getattr(self,'direct_grasp_assist',False) and self.release_query_active:
            if np.min(self.data.qpos[self.fingers[0]])>.041 or clock>self.release_query_until:
                self.release_query_active=False;self.contact_visual=None
                self.release_visual=None
        if not opening and gripping and not self.attached:
            visual = (self.contact_visual or self.pregrasp_visual) if self.direct_grasp_assist else None
            if visual is None:visual=self.fresh_visual(clock)
            if visual is None:
                raise RuntimeError('camera_target_missing_for_grip_geometry')
            points = pad_points(self.model, self.data, self.side)
            frame = self.data.xmat[self.spec.body_id].reshape(3, 3)
            self.attachment = frame.T @ (np.asarray(visual['center']) - points.mean(axis=0))
            c, s = np.cos(visual['yaw']), np.sin(visual['yaw'])
            self.attachment_rotation = frame.T @ visual.get('rotation',np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]))
            self.plan = dict(center=np.asarray(visual['center']).copy(), yaw=float(visual['yaw']))
            self.attached = True

    def _direct_position_goal(self, target, state, bounded=False):
        """Position-first IK, optionally tethered to the initial wrist frame.

        No object pose selects this goal. Keep the measured attitude softly,
        rather than requiring an unreachable global grasp-ready attitude.
        """
        seed = self.data.qpos[self.spec.qpos].copy()
        reference = self.data.geom_xmat[self.geoms[0]].reshape(3, 3).copy()
        human_rotation=self.direct_wrist_policy=='palm' and self.direct_rotation_target is not None
        if human_rotation:reference=np.asarray(self.direct_rotation_target).copy()
        angular_weight=1.5 if human_rotation else .15
        tether_reference=reference if human_rotation else self.rotation
        lower, upper = self.spec.q_range[:, 0] + 1e-7, self.spec.q_range[:, 1] - 1e-7
        cached={}
        def residual(q):
            if bounded and 'q' in cached and np.array_equal(q,cached['q']):
                return cached['terms']
            state.qpos[self.spec.qpos] = q
            mujoco.mj_kinematics(self.model, state)
            center = pad_points(self.model, state, self.side).mean(0)
            frame = state.geom_xmat[self.geoms[0]].reshape(3, 3)
            angular = Rotation.from_matrix(reference @ frame.T).as_rotvec()
            terms=np.r_[50 * (center - target), angular_weight * angular, .03 * (q - seed)]
            if bounded:
                # Local attitude continuity avoids switching IK branches merely
                # to retain the initial wrist. Tether to that initial frame too,
                # so small local relaxations cannot accumulate unlimited drift.
                tether=Rotation.from_matrix(tether_reference @ frame.T).as_rotvec()
                angle=np.linalg.norm(tether)
                terms=np.r_[terms,2.*tether*max(0.,angle-np.deg2rad(15))/max(angle,1e-12)]
                cached.update(q=q.copy(),terms=terms,center=center,angular=angular,tether=tether)
            return terms
        def jacobian(q):
            residual(q)
            # mj_jac needs the current subtree COM frames as well as FK.
            mujoco.mj_comPos(self.model,state)
            jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
            mujoco.mj_jac(self.model,state,jp,jr,cached['center'],int(self.model.geom_bodyid[self.geoms[0]]))
            angular_jac=-rotation_log_right_inverse(cached['angular'])@jr[:,self.spec.dof]
            tether=cached['tether'];angle=float(np.linalg.norm(tether));radius=np.deg2rad(15)
            cone_jac=np.zeros((3,7))
            if angle>radius:
                radial=2.*((1.-radius/angle)*np.eye(3)+radius*np.outer(tether,tether)/angle**3)
                cone_jac=radial@(-rotation_log_right_inverse(tether)@jr[:,self.spec.dof])
            return np.vstack((50.*jp[:,self.spec.dof],angular_weight*angular_jac,.03*np.eye(7),cone_jac))
        result = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                               jac=jacobian if bounded else '2-point',
                               max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        residual(result.x)
        position_error = float(np.linalg.norm(pad_points(self.model, state, self.side).mean(0) - target))
        frame = state.geom_xmat[self.geoms[0]].reshape(3, 3)
        angle = float(Rotation.from_matrix(reference @ frame.T).magnitude())
        initial_angle=float(Rotation.from_matrix(tether_reference @ frame.T).magnitude()) if bounded else np.nan
        accepted=position_error<=.008 and angle<=np.deg2rad(20) and (not bounded or initial_angle<=np.deg2rad(20))
        self.ik_diagnostic=dict(attempted=True,accepted=bool(accepted),solver_success=bool(result.success),
            solver_status=int(result.status),nfev=int(result.nfev),position_error_m=position_error,
            wrist_delta_rad=angle,initial_wrist_delta_rad=initial_angle,
            orientation_limited=bool(human_rotation and angle>np.deg2rad(5)),
            goal_q=result.x.copy(),target_xyz=np.asarray(target).copy())
        if position_error > .008 or angle > np.deg2rad(20):
            raise RuntimeError(f'cartesian_target_unreachable: position_priority_error_m={position_error}, wrist_delta_rad={angle}')
        if bounded and initial_angle>np.deg2rad(20):
            raise RuntimeError('cartesian_target_unreachable: initial_wrist_delta_exceeds_20_degrees')
        return result.x.copy()

    def _execute_direct(self, command, clock, dt):
        phase=time.perf_counter()
        self._direct_payload(clock, opening=command.finger > .022)
        self.execution_timing['payload_ms']=(time.perf_counter()-phase)*1000
        if not self.attached and self.fresh_visual(clock) is None:
            raise RuntimeError('camera_target_missing_for_collision_geometry')
        phase=time.perf_counter()
        state = self._query_state()
        self.execution_timing['planning_query_ms']=(time.perf_counter()-phase)*1000
        phase=time.perf_counter()
        points = pad_points(self.model, self.data, self.side)
        span = float(np.linalg.norm(points[0] - points[1]))
        # The pad direction and orientation residual must describe the SAME
        # requested wrist frame. Mixing a direction from lagging encoders with
        # the fixed desired attitude makes even a 2 mm translation infeasible.
        current_rotation = self.data.geom_xmat[self.geoms[0]].reshape(3, 3)
        direction = self.rotation @ (current_rotation.T @ ((points[0] - points[1]) / max(span, 1e-9)))
        old_finger = float(np.mean(self.data.qpos[self.fingers[0]]))
        try:
            if self.direct_wrist_policy in ('position','palm'):
                goal=self._direct_position_goal(np.asarray(command.target),state,bounded=True)
                self.direct_ik_mode = 'human_palm_yaw' if self.direct_wrist_policy=='palm' else 'position_priority'
            else:
                goal, _, _, metrics = solve_gripper_contact_proxy_q(
                    self.model, state, self.spec, self.geoms, self.fingers, PAD_CONTACT_PROXY_LOCAL,
                    np.asarray(command.target), 1, 1., span, old_finger, self.joints[self.side],
                    reference_rotation=self.rotation, multi_start=False,
                    closing_direction=direction,kinematics_only=True)
                self.direct_ik_mode = 'fixed_wrist'
                if metrics['midpoint_error_m'] > .008 or metrics['perpendicular_error_m'] > .008:
                    goal = self._direct_position_goal(np.asarray(command.target), state)
                    self.direct_ik_mode = 'position_priority'
        finally:
            self.execution_timing['ik_ms']=(time.perf_counter()-phase)*1000
        finger = float(self.finger_targets[self.side] + np.clip(
            command.finger - self.finger_targets[self.side], -.132*dt, .132*dt))
        cap = np.array([.8] * 4 + [1.5] * 3) * dt
        commanded = (coordinated_joint_step(self.joints[self.side],goal,cap) if self.direct_wrist_policy in ('position','palm')
                     else self.joints[self.side] + np.clip(goal - self.joints[self.side], -cap, cap))
        if self.direct_wrist_policy=='palm' or self.direct_grasp_assist and getattr(self,'direct_precision_scale',1.)<.99:
            # Advance the REFERENCE from its preceding target. Re-starting
            # every tick from lagging encoders limits servo error to one tiny
            # increment and can make a 20mm command take many seconds.
            origin=self.joints[self.side].copy()
            commanded=coordinated_joint_step(origin,goal,cap)
            limit=float(getattr(self,'direct_cartesian_speed',.12))*dt
            angular_limit=np.deg2rad(30 if getattr(self,'direct_precision_scale',1.)<.99 else 60)*dt
            state.qpos[self.spec.qpos]=origin;state.qpos[self.fingers[0]]=old_finger
            mujoco.mj_kinematics(self.model,state)
            reference_tcp=pad_points(self.model,state,self.side).mean(0)
            reference_rotation=state.geom_xmat[self.geoms[0]].reshape(3,3).copy()
            for _ in range(3):
                state.qpos[self.spec.qpos]=commanded;state.qpos[self.fingers[0]]=old_finger
                mujoco.mj_kinematics(self.model,state)
                travel=float(np.linalg.norm(pad_points(self.model,state,self.side).mean(0)-reference_tcp))
                turn=float(Rotation.from_matrix(reference_rotation@state.geom_xmat[self.geoms[0]].reshape(3,3).T).magnitude())
                fraction=min(1.,limit/max(travel,1e-9),angular_limit/max(turn,1e-9))
                if fraction>=1.-1e-7:break
                commanded=origin+(commanded-origin)*fraction
            # Separately bound tracking debt, rather than shrinking each
            # reference step to the encoder error. Query the whole actual->goal
            # path below; payload/table/link checks remain authoritative.
            for _ in range(3):
                state.qpos[self.spec.qpos]=commanded;mujoco.mj_kinematics(self.model,state)
                lead=float(np.linalg.norm(pad_points(self.model,state,self.side).mean(0)-points.mean(0)))
                turn=float(Rotation.from_matrix(current_rotation@state.geom_xmat[self.geoms[0]].reshape(3,3).T).magnitude())
                max_lead=.008 if getattr(self,'direct_precision_scale',1.)<.99 else .04
                fraction=min(1.,max_lead/max(lead,1e-9),np.deg2rad(15)/max(turn,1e-9))
                if fraction>=1.-1e-7:break
                actual=self.data.qpos[self.spec.qpos]
                commanded=actual+(commanded-actual)*fraction
        # Check arm displacement at the measured aperture before allowing pad
        # contact from finger closure. Closing never exempts an unchecked arm path.
        phase=time.perf_counter()
        try:
            self._check_segment(commanded, old_finger, 'direct_carry' if self.attached else 'direct_move')
            self._check_segment(commanded, finger, 'direct_grip')
        finally:
            self.execution_timing['collision_check_ms']=(time.perf_counter()-phase)*1000
        self.joints[self.side] = commanded
        self.finger_targets[self.side] = finger
        self.control.set_targets(self.joints, self.finger_targets)
        self.command_stage, self.goal_q = 'direct', goal.copy()
        self.goal_xyz = np.asarray(command.target).copy()

    def _execute(self, command, clock, dt):
        stage = command.stage
        if stage == 'pose2arm_reference':
            self._execute_reference(command, clock, dt)
            return
        if stage == 'direct':
            self._execute_direct(command, clock, dt)
            return
        if stage == 'gripper_reference':
            self._execute_gripper_reference(command,clock,dt)
            return
        if stage == 'plan':
            visual = self.fresh_visual(clock)
            if visual is None:
                raise RuntimeError('camera_target_missing_at_plan')
            if self.plan is None:
                self._make_plan(visual)
            self.command_stage = stage
            return
        if stage == 'hold':
            self.reference_trajectories.clear()
            for side in self.reference_sides or (self.side,):
                self.joints[side] = self.data.qpos[self.control.specs[side].qpos].copy()
            self.finger_targets[self.side] = command.finger
            self.control.set_targets(self.joints, self.finger_targets)
            self.command_stage = stage
            self.goal_q = self.goal_xyz = None
            return
        new_stage = stage != self.command_stage
        if stage == 'lift' and not self.attached:
            points = pad_points(self.model, self.data, self.side)
            self.grip_span = float(np.linalg.norm(points[0] - points[1]))
            self.grip_finger = float(np.mean(self.data.qpos[self.fingers[0]]))
            frame = self.data.xmat[self.spec.body_id].reshape(3, 3)
            self.attachment = frame.T @ (self.plan['center'] - points.mean(axis=0))
            c, s = np.cos(self.plan['yaw']), np.sin(self.plan['yaw'])
            self.attachment_rotation = frame.T @ self.plan.get('object_rotation',np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]))
            self.attached = True
        if stage in ('transit', 'approach', 'descend', 'close'):
            if stage != 'close' or new_stage:
                if self.fresh_visual(clock) is None:
                    raise RuntimeError('camera_target_lost_before_grasp')
                if np.linalg.norm(self.visual['center'] - self.plan['center']) > .015:
                    raise RuntimeError('camera_target_moved_before_grasp')
            if stage == 'descend' and new_stage:
                self._make_plan(self.fresh_visual(clock))
            key = dict(transit='q_transit', approach='q_approach', descend='q_grasp', close='q_grasp')[stage]
            goal = self.plan[key]
            xyz = self.plan['center'] if stage in ('descend', 'close') else None
        elif stage == 'lift':
            xyz = self.plan['center'] + [0, 0, self.lift_m]
            goal = self._cartesian(xyz, True) if new_stage else self.goal_q
        elif stage in ('follow', 'transport', 'lower', 'retract'):
            if stage == 'lower':
                goal, xyz = self._lower_goal(command.target) if new_stage else (self.goal_q, self.goal_xyz)
            else:
                xyz = np.asarray(command.target)
                goal = self._cartesian(xyz, stage == 'transport')
        elif stage in ('release', 'verify'):
            xyz = self.goal_xyz
            goal = self.joints[self.side].copy()
            if stage == 'release':
                self.attached = False
        else:
            raise ValueError('Unknown robot task command ' + stage)
        finger = self.finger_targets[self.side] + np.clip(command.finger - self.finger_targets[self.side], -.0066, .0066)
        if stage in ('transit', 'approach', 'descend', 'lift', 'lower', 'retract'):
            if new_stage:
                self.motion_origin = self.data.qpos[self.spec.qpos].copy()
                self.motion_progress = 0.
                # Smooth start/stop instead of saturating each joint immediately.
                self.motion_duration = max(1.2, float(np.max(np.abs(goal - self.motion_origin))) / .4)
                self._check_segment(goal, command.finger, stage)
            self.motion_progress = min(self.motion_duration, self.motion_progress + dt)
            u = self.motion_progress / self.motion_duration
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            commanded = self.motion_origin + blend * (goal - self.motion_origin)
        else:
            commanded = goal
        self._check_segment(commanded, finger, stage)
        cap = np.array([.8] * 4 + [1.5] * 3) * dt
        self.joints[self.side] += np.clip(commanded - self.joints[self.side], -cap, cap)
        self.finger_targets[self.side] = float(finger)
        self.control.set_targets(self.joints, self.finger_targets)
        self.command_stage, self.goal_q = stage, np.asarray(goal).copy()
        self.goal_xyz = None if xyz is None else np.asarray(xyz).copy()

    def step(self, dt=.05, evaluator=None, stage=None):
        for _ in range(round(dt / self.model.opt.timestep)):
            self.control.apply()
            mujoco.mj_step(self.model, self.data)
            if evaluator is not None:
                evaluator.sample(self.data, stage, float(self.model.opt.timestep))
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('nonfinite_robot_state')

    def close(self):
        self.observer.close()


class TaskEvaluator:
    """Read-only simulation ground truth. Its outputs never enter the controller."""
    def __init__(self, model, data, config, side='right'):
        self.model, self.config = model, config
        self.body = model.body('droid_object').id
        self.object_geom = model.geom('droid_object_geom').id
        self.fingers = {model.geom(f'openarm_{side}_{f}_finger_collision').id for f in ('right', 'left')}
        self.finger_qpos = finger_joint_indices(model, side)[0]
        self.initial_z = float(data.xpos[self.body, 2])
        self.lift_run = self.best_lift = self.placed_run = self.best_placed = 0.
        self.max_lift = 0.

    def sample(self, data, stage, dt):
        p = data.xpos[self.body]
        contacts = set()
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.object_geom in pair:
                contacts.update(pair & self.fingers)
        dz = float(p[2] - self.initial_z)
        self.max_lift = max(self.max_lift, dz)
        lifted = dz > .03 and len(contacts) == 2
        self.lift_run = self.lift_run + dt if lifted else 0.
        self.best_lift = max(self.best_lift, self.lift_run)
        released_stage = stage in ('retract', 'verify', 'success') or (
            stage == 'direct' and np.min(data.qpos[self.finger_qpos]) > .040)
        placed = (released_stage and (stage != 'direct' or self.best_lift >= .5 - 1e-8) and not contacts
                  and np.linalg.norm(p[:2] - self.config.goal_xy) <= self.config.goal_radius_m
                  and abs(p[2] - (TABLE_TOP + CUBE_HALF)) < .01
                  and np.linalg.norm(data.cvel[self.body, 3:]) < .02
                  and np.linalg.norm(data.cvel[self.body, :3]) < .2)
        self.placed_run = self.placed_run + dt if placed else 0.
        self.best_placed = max(self.best_placed, self.placed_run)
        return np.array([*p, dz, float(len(contacts)), float(placed)])

    def report(self, task):
        return dict(scope='Independent simulated object/contact evaluation; never control input',
                    max_lift_m=self.max_lift, sustained_lift_s=self.best_lift,
                    stable_released_in_goal_s=self.best_placed,
                    physical_task_success=bool((task.stage == 'success' or (
                                               getattr(task, 'control_mode', '') == 'direct' and task.stage != 'failed'))
                                               and self.best_lift >= .5 - 1e-8
                                               and self.best_placed >= .5 - 1e-8))
