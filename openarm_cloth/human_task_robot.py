"""B robot execution adapter for human_grasp_task (red cube, simulation only).

Planning copies robot encoders plus CAMERA object estimates into a private query
state. Object ground truth is confined to TaskEvaluator, never fed to control.
"""
import mujoco
import numpy as np

from b_scene_fusion import BVisualObserver
from demonstration_transfer import visual_planning_state
from demo_grasp_viewer import CUBE_HALF, PAD_CONTACT_PROXY_LOCAL
from handeye_grasp_red_cube import finger_joint_indices, solve_gripper_contact_proxy_q
from human_grasp_task import Feedback
from human_teleop_core import PadCartesianIK
from member_b_actuators import ActuatorController
from task_workspace import TABLE_TOP
from visual_grasp_planner import plan_face_grasp


def pad_points(model, data, side):
    return np.array([data.geom_xpos[model.geom(f'openarm_{side}_{finger}_finger_collision').id]
                     + data.geom_xmat[model.geom(f'openarm_{side}_{finger}_finger_collision').id].reshape(3, 3) @ proxy
                     for finger, proxy in zip(('right', 'left'), PAD_CONTACT_PROXY_LOCAL)])


def supported_tcp_height(object_rotation, attachment_world, clearance=.0005):
    """Height at which all corners of the estimated held cube clear the table."""
    half_height = CUBE_HALF * np.sum(np.abs(np.asarray(object_rotation)[2]))
    return float(TABLE_TOP + half_height + clearance - np.asarray(attachment_world)[2])


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
        self.lift_m = lift_m
        self.attached = False
        self.attachment = None
        self.attachment_rotation = None
        self.query = mujoco.MjData(model)
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

    def initialize(self):
        """Episode reset posture only, before input/recording; no object is moved.

        Uses B's pad IK at a fixed workspace coordinate, independent of the cube.
        Subsequent episode motion uses only dynamic actuators.
        """
        ik = PadCartesianIK(self.model, self.data, ready=np.array([.43, -.10, .54]))
        self.data.qpos[self.spec.qpos] = ik.ready_q
        self.data.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.joints[self.side] = ik.ready_q.copy()
        self.rotation = self.data.geom_xmat[self.geoms[0]].reshape(3, 3).copy()
        self.control.set_targets(self.joints, self.finger_targets)
        for _ in range(round(.6 / self.model.opt.timestep)):
            self.control.apply()
            mujoco.mj_step(self.model, self.data)

    def observe(self, clock):
        self.observer.observe(self.data, clock)
        self.visual = self.observer.last

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
        visual = self.plan if self.attached else self.visual
        if visual is None:
            raise RuntimeError('camera_target_missing')
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

    def _contacts(self, q, finger, attached, allow_grasp, allow_support):
        query = self.query
        query.qpos[:] = self.model.qpos0
        query.qpos[self.robot_addresses] = self.data.qpos[self.robot_addresses]
        query.qpos[self.spec.qpos] = q
        query.qpos[self.fingers[0]] = finger
        mujoco.mj_forward(self.model, query)
        # Keep the same measured object geometry as the accepted docking plan.
        # Fresh frames still gate movement detection; occlusion must not silently
        # move collision geometry while the joint goal stays on an old target.
        visual = self.plan if self.plan is not None else self.visual
        center = np.asarray(visual['center']).copy()
        yaw = visual['yaw']
        if attached:
            tcp = pad_points(self.model, query, self.side).mean(axis=0)
            frame = query.xmat[self.spec.body_id].reshape(3, 3)
            center = tcp + frame @ self.attachment
            object_rotation = frame @ self.attachment_rotation
        query.qpos[self.object_address:self.object_address + 3] = center
        if attached:
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, object_rotation.ravel())
            query.qpos[self.object_address + 3:self.object_address + 7] = quat
        else:
            query.qpos[self.object_address + 3:self.object_address + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, query)
        hits = {}
        for c in query.contact:
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            # Same coupled finger-pair exemption as B's TaskGeometry. The
            # physical servo/contact solver, not query teleportation, closes it.
            if set(pair) == set(self.geoms):
                continue
            if not set(pair) & self.robot_geoms and not (attached and self.object_geom in pair):
                continue
            if allow_grasp and self.object_geom in pair and set(pair) & set(self.geoms):
                continue
            if attached and allow_support and set(pair) == {self.object_geom, self.table_geom}:
                if c.dist >= -.001:
                    continue
            if c.dist < 0:
                hits[pair] = -float(c.dist)
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
        grasp = stage in ('close', 'lift', 'transport', 'lower', 'release')
        support = stage in ('lift', 'lower', 'release')
        baseline = self._contacts(start, old_finger, self.attached, grasp, support)
        # Never exempt object penetrations against obstacles as a baseline.
        baseline = {pair: depth for pair, depth in baseline.items() if self.object_geom not in pair}
        count = max(1, int(np.ceil(np.max(np.abs(goal - start)) / .02)),
                    int(np.ceil(abs(finger - old_finger) / .004)))
        for a in np.linspace(0, 1, count + 1):
            hits = self._contacts(start + a * (goal - start), old_finger + a * (finger - old_finger),
                                  self.attached, grasp, support)
            rejected = [(pair, depth, baseline.get(pair, 0.)) for pair, depth in hits.items()
                        if depth - baseline.get(pair, 0.) > .0005]
            if rejected:
                detail = [dict(geoms=[self.model.geom(g).name for g in pair], depth_m=depth,
                               baseline_m=old) for pair, depth, old in rejected]
                raise RuntimeError(f'task_motion_collision_rejected:{stage}:{detail}')

    def execute(self, command, clock, dt=.05):
        try:
            self._execute(command, clock, dt)
        except (RuntimeError, ValueError) as error:
            self.error = str(error)
            self.joints[self.side] = self.data.qpos[self.spec.qpos].copy()
            # Hold the existing finger target on a failed command; do not drop payload.
            self.control.set_targets(self.joints, self.finger_targets)

    def _execute(self, command, clock, dt):
        stage = command.stage
        if stage == 'plan':
            visual = self.fresh_visual(clock)
            if visual is None:
                raise RuntimeError('camera_target_missing_at_plan')
            if self.plan is None:
                self._make_plan(visual)
            self.command_stage = stage
            return
        if stage == 'hold':
            self.joints[self.side] = self.data.qpos[self.spec.qpos].copy()
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
            self.attachment_rotation = frame.T @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
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
        placed = (stage in ('retract', 'verify', 'success') and not contacts
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
                    physical_task_success=bool(task.stage == 'success' and self.best_lift >= .5 - 1e-8
                                               and self.best_placed >= .5 - 1e-8))
