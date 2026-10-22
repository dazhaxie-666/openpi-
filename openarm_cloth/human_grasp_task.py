"""Single-arm shared task state machine, independent of MuJoCo and camera APIs.

Humans choose approach, transport and release. Vision only docks the gripper.
No object ground truth, joint commands or autonomous demonstration live here.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HumanInput:
    seq: int
    point: np.ndarray | None
    pinch_ratio: float | None
    valid: bool
    capture_s: float = 0.

    @property
    def gesture_valid(self):
        return self.pinch_ratio is not None and np.isfinite(self.pinch_ratio) and self.pinch_ratio >= 0


@dataclass
class Feedback:
    tcp: np.ndarray
    visual: dict | None = None
    command_stage: str = ''
    reached: bool = False
    gripping: bool = False
    opened: bool = False
    plan_ready: bool = False
    error: str | None = None


@dataclass
class TaskCommand:
    stage: str
    target: np.ndarray | None = None
    finger: float = .044
    owner: str = 'hold'
    joint_targets: dict | None = None
    finger_targets: dict | None = None
    reference_capture_s: float | None = None


@dataclass
class TaskConfig:
    gain: float = .7
    speed: float = .12
    deadband_m: float = .002
    max_target_lead_m: float = .04
    timeout_s: float = 15.
    input_loss_s: float = 2.
    close_ratio: float = .35
    open_ratio: float = .8
    dwell_s: float = .15
    near_m: float = .10
    lift_m: float = .07
    goal_xy: np.ndarray = field(default_factory=lambda: np.array([.43, -.10]))
    goal_radius_m: float = .03
    lower: np.ndarray = field(default_factory=lambda: np.array([.32, -.18, .429]))
    upper: np.ndarray = field(default_factory=lambda: np.array([.54, .18, .69]))

    def __post_init__(self):
        for name in ('gain', 'speed', 'input_loss_s', 'dwell_s', 'near_m',
                     'lift_m', 'goal_radius_m', 'max_target_lead_m'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not np.isfinite(self.timeout_s) or self.timeout_s < 0:
            raise ValueError('timeout_s must be nonnegative and finite (0 disables timeout)')
        if not np.isfinite(self.deadband_m) or self.deadband_m < 0:
            raise ValueError('deadband_m must be finite and nonnegative')
        if not 0 < self.close_ratio < self.open_ratio:
            raise ValueError('Pinch thresholds must have hysteresis')
        self.goal_xy = np.asarray(self.goal_xy, float)
        self.lower, self.upper = np.asarray(self.lower, float), np.asarray(self.upper, float)
        if (self.goal_xy.shape != (2,) or self.lower.shape != (3,) or self.upper.shape != (3,)
                or not np.isfinite(np.r_[self.goal_xy, self.lower, self.upper]).all()
                or np.any(self.lower >= self.upper)):
            raise ValueError('Invalid task workspace')


def normalized_pinch(hand):
    """Thumb/index tip distance divided by palm width; no metric hand claim."""
    if hand is None:
        return None
    points = np.asarray(hand, float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return None
    width = np.linalg.norm(points[5] - points[17])
    return float(np.linalg.norm(points[4] - points[8]) / width) if width > 1e-5 else None


class HumanGraspTask:
    terminal = ('success', 'failed', 'cancelled')
    held_stages = ('lift', 'transport', 'lower')

    def __init__(self, config=None):
        self.config = config or TaskConfig()
        self.stage = 'ready'
        self.events = []
        self.reason = None
        self.started = self.entered = None
        self.last_clock = None
        self.last_seq = None
        self.last_capture = None
        self.gesture = None
        self.candidate = None
        self.candidate_since = None
        self.missing_since = None
        self.hand_missing_since = None
        self.start_requested = False
        self.was_missing = False
        self.human_anchor = self.robot_anchor = self.target = None
        self.object_anchor = None
        self.stable_since = None
        self.place_target = None
        self.release_requested = False
        self.holding = False
        self.grip_missing_since = None
        self.message = 'Open hand, then SPACE to start'

    def _enter(self, stage, clock, reason):
        self.events.append(dict(time_s=float(clock), previous=self.stage, stage=stage, reason=reason))
        self.stage, self.entered, self.stable_since = stage, clock, None
        self.message = reason
        if stage == 'close':
            self.holding = True
        elif stage == 'release':
            self.holding = False

    def fail(self, reason, clock):
        if self.stage not in self.terminal:
            self.reason = reason
            self._enter('failed', clock, reason)

    def cancel(self, clock):
        if self.stage not in self.terminal:
            self.reason = 'user_cancelled'
            self._enter('cancelled', clock, self.reason)

    def _stable(self, good, clock, duration):
        if not good:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = clock
        return clock - self.stable_since >= duration - 1e-9

    def _gesture(self, human, clock):
        if self.last_seq is not None and human.seq <= self.last_seq:
            return
        if self.last_capture is not None and human.capture_s <= self.last_capture:
            return
        self.last_seq = human.seq
        if self.last_capture is not None and human.capture_s - self.last_capture > .35:
            self.candidate = self.candidate_since = None
            self.gesture = None
        self.last_capture = human.capture_s
        ratio = human.pinch_ratio
        label = 'closed' if ratio <= self.config.close_ratio else 'open' if ratio >= self.config.open_ratio else None
        if label is None:
            self.candidate = self.candidate_since = None
        elif label != self.candidate:
            self.candidate, self.candidate_since = label, human.capture_s
        elif human.capture_s - self.candidate_since >= self.config.dwell_s - 1e-9:
            self.gesture = label

    def rebase(self, human, feedback):
        self.human_anchor = np.asarray(human.point, float).copy()
        self.robot_anchor = feedback.tcp.copy()
        self.target = feedback.tcp.copy()

    def _follow(self, human, dt, actual_tcp):
        goal = self.robot_anchor + self.config.gain * (human.point - self.human_anchor)
        tracing=getattr(self,'record_motion_trace',False)
        unclipped=goal.copy() if tracing else None
        goal = np.clip(goal, self.config.lower, self.config.upper)
        workspace_goal=goal.copy() if tracing else None
        lead = goal - actual_tcp
        goal = actual_tcp + lead * min(1., self.config.max_target_lead_m / max(np.linalg.norm(lead), 1e-12))
        if tracing:
            self.motion_trace=dict(human_anchor=self.human_anchor.copy(),robot_anchor=self.robot_anchor.copy(),
                unclipped_goal=unclipped,workspace_goal=workspace_goal,
                workspace_clipped=np.abs(unclipped-workspace_goal)>1e-9,lead_goal=goal.copy())
        delta = goal - self.target
        if np.linalg.norm(delta) > self.config.deadband_m:
            self.target += delta * min(1., self.config.speed * dt / max(np.linalg.norm(delta), 1e-12))
        return self.target.copy()

    def update(self, human, feedback, clock, start=False, paused=False, recenter=False):
        if not np.isfinite(clock) or self.last_clock is not None and clock < self.last_clock:
            raise ValueError('Task clock must be finite and monotonic')
        dt = .05 if self.last_clock is None else min(.1, clock - self.last_clock)
        self.last_clock = clock
        if self.stage == 'ready' and start:
            self.start_requested = True
        finger = 0. if self.holding else .044
        hold = TaskCommand('hold', finger=finger)
        if self.stage in self.terminal:
            return hold
        if feedback.error:
            self.fail(feedback.error, clock)
            return hold
        if paused:
            self.candidate = self.candidate_since = None
            self.stable_since = None
            self.was_missing = True
            return hold
        valid = (human is not None and human.valid and human.point is not None
                 and np.asarray(human.point).shape == (3,) and np.isfinite(human.point).all()
                 and np.isfinite(human.capture_s))
        if self.config.timeout_s > 0 and self.started is not None and clock - self.started > self.config.timeout_s:
            self.fail('task_timeout', clock)
            return hold
        if not valid:
            self.candidate = self.candidate_since = None
            self.gesture = None
            self.stable_since = None
            self.was_missing = True
            if self.missing_since is None:
                self.missing_since = clock
            self.message = ('START REQUESTED: show shoulders, torso and right wrist' if self.stage == 'ready' and self.start_requested
                            else 'Right wrist missing: hold (keep grip if carrying)')
            if self.started is not None and clock - self.missing_since >= self.config.input_loss_s:
                self.fail('human_input_lost', clock)
            return hold
        self.missing_since = None
        if human.gesture_valid:
            self.hand_missing_since = None
            self._gesture(human, clock)
        else:
            self.gesture = self.candidate = self.candidate_since = None
            if self.hand_missing_since is None:
                self.hand_missing_since = clock
            if self.stage not in ('ready', 'follow', 'transport'):
                self.stable_since = None
                self.message = 'Right fingers missing: assisted motion held; show your hand'
                if clock - self.hand_missing_since >= self.config.input_loss_s:
                    self.fail('human_hand_lost_during_assist', clock)
                return hold
        if self.was_missing or recenter:
            if self.stage in ('follow', 'transport'):
                self.rebase(human, feedback)
            self.was_missing = False
        visual = feedback.visual
        if self.stage == 'ready':
            if not self.start_requested:
                self.message = 'READY: open thumb/index, press SPACE or click START'
            elif not human.gesture_valid:
                self.message = 'START REQUESTED: show right thumb and index finger'
            elif self.gesture != 'open':
                self.message = 'START REQUESTED: spread thumb/index and hold 0.2 s'
            elif visual is None:
                self.message = 'START REQUESTED: waiting for robot camera target'
            else:
                self.start_requested = False
                self.started = clock
                self.rebase(human, feedback)
                self._enter('follow', clock, 'human_start_with_open_hand')
            return hold
        if self.stage in ('follow', 'transport'):
            self.message = ('FOLLOW: move right wrist; pinch near cube' if self.stage == 'follow'
                            else 'TRANSPORT: move over GREEN goal, then open') if human.gesture_valid else 'Wrist follow active; fingers missing, gripper held'
            if self.stage == 'follow' and visual is None:
                self.was_missing = True
                self.message = 'Camera target missing: hold until fresh localization'
                return hold
            target = self._follow(human, dt, feedback.tcp)
            if self.stage == 'follow':
                if self.gesture == 'closed' and visual is not None and np.linalg.norm(feedback.tcp - visual['center']) <= self.config.near_m:
                    self.object_anchor = np.asarray(visual['center']).copy()
                    self._enter('plan', clock, 'human_close_near_camera_target')
                    return TaskCommand('plan', owner='vision_docking')
            else:
                if not feedback.gripping:
                    if self.grip_missing_since is None:
                        self.grip_missing_since = clock
                    if clock - self.grip_missing_since > .4:
                        self.fail('grip_lost_during_transport', clock)
                        return hold
                else:
                    self.grip_missing_since = None
                if self.gesture == 'open':
                    if np.linalg.norm(feedback.tcp[:2] - self.config.goal_xy) <= self.config.goal_radius_m:
                        self.place_target = feedback.tcp.copy()
                        self.place_target[2] = self.object_anchor[2] + .002
                        self.release_requested = True
                        self._enter('lower', clock, 'human_open_in_goal_zone: assisted_lower_before_release')
                        return TaskCommand('lower', self.place_target.copy(), 0., 'human_release_with_vision_height')
                    self.message = 'Release blocked: move over green goal, then open hand'
            return TaskCommand(self.stage, target, 0. if self.stage == 'transport' else .044, 'human_delta')
        if self.stage == 'plan':
            if feedback.plan_ready:
                self._enter('transit', clock, 'camera_grasp_plan_ready')
            return TaskCommand(self.stage, owner='vision_docking')
        if self.stage in ('transit', 'approach', 'descend'):
            if self.gesture == 'open':
                self.fail('human_cancelled_grasp', clock)
                return hold
            if self._stable(feedback.command_stage == self.stage and feedback.reached, clock, .10):
                next_stage = dict(transit='approach', approach='descend', descend='close')[self.stage]
                self._enter(next_stage, clock, 'robot_reached_' + self.stage)
            return TaskCommand(self.stage, finger=0. if self.stage == 'close' else .044, owner='vision_docking')
        if self.stage == 'close':
            if self.gesture == 'open':
                self.holding = False
                self.fail('human_cancelled_grasp', clock)
                return TaskCommand('hold', finger=.044)
            if self._stable(feedback.gripping, clock, .35):
                self._enter('lift', clock, 'encoders_and_effort_confirm_grip')
            elif clock - self.entered > 2.5:
                self.fail('closure_not_confirmed', clock)
            return TaskCommand(self.stage if self.stage not in self.terminal else 'hold', finger=0., owner='vision_docking')
        if self.stage == 'lift':
            if self._stable(feedback.command_stage == 'lift' and feedback.reached and feedback.gripping, clock, .5):
                self.rebase(human, feedback)
                self._enter('transport', clock, 'lift_motion_and_grip_stable: resume_human_delta')
                return TaskCommand('hold', finger=0.)
            return TaskCommand('lift', finger=0., owner='assisted_lift')
        if self.stage == 'lower':
            if not feedback.gripping:
                self.fail('grip_lost_before_release', clock)
                return TaskCommand('hold', finger=0.)
            if self._stable(feedback.command_stage == 'lower' and feedback.reached, clock, .15):
                self._enter('release', clock, 'support_height_reached_after_human_release_request')
            return TaskCommand(self.stage, self.place_target.copy(), 0. if self.stage == 'lower' else .044, 'assisted_place')
        if self.stage == 'release':
            if self._stable(feedback.opened, clock, .25):
                self.target = feedback.tcp + np.array([0., 0., .07])
                self._enter('retract', clock, 'gripper_opened')
            return TaskCommand(self.stage, self.target, .044, 'assisted_place')
        if self.stage == 'retract':
            if self._stable(feedback.command_stage == 'retract' and feedback.reached, clock, .15):
                self._enter('verify', clock, 'gripper_retracted')
            return TaskCommand(self.stage, self.target, .044, 'assisted_place')
        if self.stage == 'verify':
            good = (visual is not None
                    and np.linalg.norm(np.asarray(visual['center'])[:2] - self.config.goal_xy) <= self.config.goal_radius_m
                    and abs(visual['center'][2] - self.object_anchor[2]) <= .015)
            if self._stable(good, clock, .5):
                self._enter('success', clock, 'camera_confirms_released_object_in_goal')
            elif clock - self.entered > 2.:
                self.fail('placement_not_confirmed_by_camera', clock)
            return TaskCommand('hold', finger=.044)
        raise RuntimeError('Unknown task stage ' + self.stage)


class DirectHumanTask(HumanGraspTask):
    """Continuous human position and gripper control. Scoring never drives it.

    No plan, docking, lift, lower or goal-zone release gate is generated here.
    Tracking loss and rejected motion hold/rebase rather than end a session.
    """
    control_mode = 'direct'
    terminal = ('success', 'failed', 'cancelled', 'finished')

    def finish(self, clock, reason='operator_finished'):
        if self.stage not in self.terminal:
            self.reason = reason
            self._enter('finished', clock, reason)

    def reject_motion(self, reason, human, feedback, clock):
        if human is not None and human.valid and human.point is not None:
            self.rebase(human, feedback)
        else:
            self.was_missing = True
        # Log transitions, not identical warnings at camera/control frequency.
        if not self.events or self.events[-1]['reason'] != reason:
            self.events.append(dict(time_s=float(clock), previous=self.stage,
                                    stage=self.stage, reason=reason, recoverable=True))
        cause = ('Collision check blocked motion' if reason.startswith('task_motion_collision_rejected:')
                 else 'IK target unreachable' if reason.startswith('cartesian_target_unreachable:')
                 else 'Camera geometry missing')
        self.message = cause + ': retreat / adjust / C recenter; retry stays active'

    def update(self, human, feedback, clock, start=False, paused=False, recenter=False):
        if not np.isfinite(clock) or self.last_clock is not None and clock < self.last_clock:
            raise ValueError('Task clock must be finite and monotonic')
        dt = .05 if self.last_clock is None else min(.1, clock - self.last_clock)
        self.last_clock = clock
        if self.stage == 'ready' and start:
            self.start_requested = True
        hold = TaskCommand('hold', finger=0. if self.holding else .044)
        if self.stage in self.terminal:
            return hold
        if feedback.error:
            self.fail(feedback.error, clock)
            return hold
        if paused:
            self.candidate = self.candidate_since = self.gesture = None
            self.was_missing = True
            self.message = 'PAUSED: reposition your hand, then resume'
            return hold
        valid = (human is not None and human.valid and human.point is not None
                 and np.asarray(human.point).shape == (3,) and np.isfinite(human.point).all()
                 and np.isfinite(human.capture_s))
        if not valid:
            self.gesture = self.candidate = self.candidate_since = None
            if self.missing_since is None:
                self.missing_since = clock
            if clock - self.missing_since >= .35:
                self.was_missing = True
            self.message = 'Wrist missing: holding; show shoulders and right wrist to resume'
            return hold
        if self.missing_since is not None and clock - self.missing_since >= .35:
            self.was_missing = True
        self.missing_since = None
        if human.gesture_valid:
            self._gesture(human, clock)
        else:
            self.gesture = self.candidate = self.candidate_since = None
        if self.stage == 'ready':
            if self.start_requested:
                self.started = clock
                self.start_requested = False
                self.rebase(human, feedback)
                self.was_missing = False
                self._enter('direct', clock, 'human_direct_start')
            else:
                self.message = 'Press SPACE or click START; wrist required, fingers optional'
            return hold
        if self.was_missing or recenter:
            self.rebase(human, feedback)
            self.was_missing = False
        if self.gesture in ('open', 'closed'):
            closed = self.gesture == 'closed'
            if closed != self.holding:
                self.holding = closed
                self.events.append(dict(time_s=float(clock), previous=self.stage, stage=self.stage,
                                        reason='human_direct_close' if closed else 'human_direct_open'))
        # Object localization serves collision geometry only, never an arm goal.
        if feedback.visual is None and not feedback.gripping:
            self.was_missing = True
            self.message = 'Robot camera target missing: holding; tracking recovery resumes control'
            return TaskCommand('hold', finger=0. if self.holding else .044)
        target = self._follow(human, dt, feedback.tcp)
        self.message = ('DIRECT: move wrist; PINCH close / OPEN release; missed grasp? open and retry'
                        if human.gesture_valid else 'DIRECT wrist follow; FINGERS MISSING - keep thumb/index visible, gripper unchanged')
        if self.holding and not feedback.gripping:
            self.message = 'DIRECT closing / grip unconfirmed: OPEN, adjust wrist and PINCH again'
        return TaskCommand('direct', target, 0. if self.holding else .044, 'human_direct')
