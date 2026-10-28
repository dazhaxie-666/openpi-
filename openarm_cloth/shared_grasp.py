"""Continuous shared autonomy: measured human intent + B camera/encoder feedback.

No simulator object state or contact truth enters this controller. Independent
evaluation labels completed trials in the runner, outside this state machine.
"""
import numpy as np
from human_grasp_task import DirectHumanTask,TaskCommand
from automatic_grasp import camera_enclosure,PAD_INNER_OFFSET


def grasp_evidence(obs):
    if not camera_enclosure(obs):return False
    q=np.asarray(obs['finger_positions']);effort=np.abs(obs['finger_force'])
    width=np.linalg.norm(np.asarray(obs['pads'])[1]-obs['pads'][0])-2*PAD_INNER_OFFSET
    return bool(abs(width-2*obs.get('half_size',.0288))<=.004 and q.min()>.008 and q.max()<.041
        and np.ptp(q)<.018 and effort.min()>=1.8 and effort.sum()>=5.6
        and np.max(np.abs(obs['finger_velocity']))<.004)


class SharedGraspTask(DirectHumanTask):
    """Outer stage stays direct; phase explicitly identifies who owns motion."""
    def __init__(self,config=None):
        super().__init__(config)
        self.phase='manual';self.context=None;self.hand_intent_valid=False
        self.intent_latched=False;self.trial_id=0;self.trials=[];self.phase_since=0.
        self.grasp_confirmed=False;self.release_target=None;self.carry_missing=False
        self.rearm=False;self.phase_events=[];self.verify_since=None
        self.request_radius_m=.075
        self.last_target=None;self.reacquire_target=None
        self.open_since=self.open_stamp=None;self.open_seq=-1
        self.placement_blocked_origin=None
        self.release_armed=False;self.carry_lost_since=None
        self.last_visible_human_point=None;self.view_loss_reference=None
        self.input_seq=-1;self.input_capture=None
        self.release_handoff_seq=-1;self.release_handoff_capture=None
        self.request_ratio=None;self.request_seq=-1;self.request_stamp=None
        self.request_open_since=self.request_close_since=None;self.request_close_count=0
        self.view_robot_origin=self.view_hand_origin=None;self.view_limit_z=None
        # Collection extensions may enable explicit, protected attempt boundaries.
        # Defaults preserve the normal continuous shared-grasp workflow.
        self.trial_limit=0;self.inter_trial_reset_s=0.;self.final_trial_notice_s=0.
        self.reset_until=None;self.final_notice_until=None;self.last_trial_result=None
        self.automatic_scene_reset=False;self.scene_reset_pending=False;self.scene_reset_count=0
        self.collection_task_profile='pickplace';self.collection_hold_seconds=0.
        self.collection_source_tolerance_m=.03;self.collection_hold_complete=False

    def surface_clearance(self,tcp,visual):
        """TCP reference to the observed known cube's surface, not its centre."""
        yaw=float(visual.get('yaw',0.));c,s=np.cos(yaw),np.sin(yaw)
        rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        half=float(self.context.get('half_size',.0288)) if self.context else .0288
        local=rotation.T@(np.asarray(tcp)-visual['center'])
        return float(np.linalg.norm(np.maximum(np.abs(local)-half,0.)))

    def grasp_readiness(self,feedback,clock):
        target=feedback.visual
        if target is None and self.last_target is not None and 0<=clock-self.last_target['time_s']<=8.:
            target=self.last_target
        distance=np.inf if target is None else self.surface_clearance(feedback.tcp,target)
        fresh=feedback.visual is not None and not feedback.visual.get('pose_is_prior',False)
        clear=target is not None and feedback.tcp[2]>=target['center'][2]+.040
        return dict(target=target,distance_m=distance,in_range=distance<=self.request_radius_m,
            fresh=fresh,height_clear=clear,ready=bool(self.stage=='direct' and self.phase=='manual'
                and self.hand_intent_valid and self.request_ratio is not None and fresh and clear
                and distance<=self.request_radius_m))

    def reset_grasp_entry(self):
        self.rearm=False;self.request_open_since=self.request_close_since=None
        self.request_close_count=0

    def fresh_grasp_request(self,human,status):
        """Remember natural open approach; accept a NEW pinch only in range."""
        eligible=status['fresh'] and status['in_range'] and status['height_clear']
        if not eligible:
            self.request_close_since=None;self.request_close_count=0
        if (human is None or not self.hand_intent_valid or self.request_ratio is None
            or not np.isfinite(self.request_ratio) or not np.isfinite(human.capture_s)):
            self.request_open_since=self.request_close_since=None;self.request_close_count=0
            if human is None or self.request_stamp is None or human.capture_s-self.request_stamp>.35:
                self.rearm=False
            return False
        if human.seq<=self.request_seq or self.request_stamp is not None and human.capture_s<=self.request_stamp:
            return False
        if self.request_stamp is not None and human.capture_s-self.request_stamp>.35:self.reset_grasp_entry()
        self.request_seq=human.seq;self.request_stamp=human.capture_s
        if self.request_ratio>=self.config.open_ratio:
            self.request_close_since=None;self.request_close_count=0
            # Opening naturally during approach is enough; no in-range
            # preparation step or separate OPEN dwell is required.
            self.request_open_since=human.capture_s;self.rearm=True
        elif self.request_ratio<=self.config.close_ratio:
            if not eligible:
                # A pinch already made outside range must not fire merely
                # because the robot later crosses the range boundary.
                self.reset_grasp_entry();return False
            if self.rearm:
                self.request_open_since=None
                if self.request_close_since is None:self.request_close_since=human.capture_s
                self.request_close_count+=1
                return bool(self.request_close_count>=3 and human.capture_s-self.request_close_since>=.20-1e-9)
        else:
            self.request_open_since=self.request_close_since=None;self.request_close_count=0
        return False

    def human_view_command(self,human,feedback,clock,recenter=False):
        """Only measured upward hand displacement can raise the open jaws."""
        self.fresh_grasp_request(human,dict(fresh=False,in_range=False,height_clear=False))
        visual=feedback.visual
        fresh=visual is not None and not visual.get('pose_is_prior',False) and visual['time_s']>=self.phase_since
        if self._stable(fresh,clock,.10):
            self.enter('manual',clock,'View restored: human wrist control resumes; pinch in range')
            self.was_missing=True;self.view_loss_reference=None
            self.view_robot_origin=self.view_hand_origin=None
            return TaskCommand('hold',finger=.044,owner='human_view_restored')
        prior=self.last_target
        if prior is None or not 0<=clock-prior['time_s']<=30.:
            self.enter('manual',clock,'view geometry expired; hold until a new camera view')
            self.was_missing=True;return TaskCommand('hold',finger=.044)
        valid=(human is not None and human.valid and human.point is not None
            and np.asarray(human.point).shape==(3,) and np.isfinite(human.point).all())
        if not valid:
            if self.missing_since is None:self.missing_since=clock
            if clock-self.missing_since>=.35:self.was_missing=True
            self.message='HUMAN UP: wrist missing, holding open jaws'
            return TaskCommand('hold',finger=.044,owner='human_view_input_hold')
        if self.missing_since is not None and clock-self.missing_since>=.35:self.was_missing=True
        if recenter or self.was_missing:
            self.view_hand_origin=np.asarray(human.point).copy()
            self.view_robot_origin[2]=feedback.tcp[2];self.was_missing=False
        self.missing_since=None
        upward=max(0.,(human.point[2]-self.view_hand_origin[2])*self.motion_gains[2])
        target=self.view_robot_origin.copy()
        target[2]=min(self.view_limit_z,max(feedback.tcp[2],self.view_robot_origin[2]+upward))
        self.reacquire_target=target
        self.message='HUMAN UP: open jaws follow measured upward hand displacement; no automatic rise'
        return TaskCommand('reacquire',target.copy(),.044,'human_restore_view')

    def enter(self,phase,clock,reason):
        self.phase_events.append(dict(time_s=float(clock),trial_id=self.trial_id,
            previous=self.phase,phase=phase,reason=reason))
        self.events.append(dict(time_s=float(clock),previous=self.stage,stage=self.stage,
            reason='shared_phase_'+phase,trial_id=self.trial_id,detail=reason))
        self.phase=phase;self.phase_since=clock;self.stable_since=None;self.message=reason
        if phase in ('transport','holding_payload'):
            # The accepted grasp pinch already supplies the closed baseline.
            # Require NEW open observations after handoff, not another pinch.
            self.release_armed=self.grasp_confirmed;self.open_since=None
            self.release_handoff_seq=self.input_seq;self.release_handoff_capture=self.input_capture
            self.gesture=self.candidate=self.candidate_since=None

    def confirmed_open(self,human,duration):
        """Advance only on fresh measured OPEN frames, never control polls."""
        if not self.hand_intent_valid or human is None or not human.gesture_valid or human.pinch_ratio<self.config.open_ratio:
            self.open_since=None;return False
        if self.phase in ('transport','holding_payload') and (human.seq<=self.release_handoff_seq or
                self.release_handoff_capture is not None and human.capture_s<=self.release_handoff_capture):
            self.open_since=None;return False
        if human.seq>self.open_seq and (self.open_stamp is None or human.capture_s>self.open_stamp):
            if self.open_stamp is None or human.capture_s-self.open_stamp>.35:self.open_since=None
            self.open_seq=human.seq;self.open_stamp=human.capture_s
            if self.open_since is None:self.open_since=human.capture_s
        return bool(self.open_since is not None and self.open_stamp-self.open_since>=duration-1e-9)

    def end_trial(self,clock,status,reason):
        ended=False
        if self.trials and 'end_s' not in self.trials[-1]:
            self.trials[-1].update(end_s=float(clock),status=status,reason=reason)
            ended=True
        if ended:
            self.last_trial_result=dict(status=status,reason=reason)
            completed=sum('end_s' in trial for trial in self.trials)
            if self.trial_limit and completed>=self.trial_limit and self.final_trial_notice_s>0:
                self.final_notice_until=float(clock)+self.final_trial_notice_s
                self.reset_until=None;self.scene_reset_pending=False
            elif self.inter_trial_reset_s>0 or self.automatic_scene_reset:
                self.reset_until=float(clock)+self.inter_trial_reset_s
                self.scene_reset_pending=bool(self.automatic_scene_reset)
        self.intent_latched=False;self.grasp_confirmed=False;self.rearm=False
        self.reset_grasp_entry()
        self.placement_blocked_origin=None

    def scene_reset_completed(self,clock,count):
        """Forget pre-reset perception/intent while retaining trial history."""
        self.scene_reset_pending=False;self.scene_reset_count=int(count)
        self.last_target=self.reacquire_target=self.object_anchor=None
        self.release_target=self.placement_blocked_origin=None
        self.last_visible_human_point=self.view_loss_reference=None
        self.view_robot_origin=self.view_hand_origin=self.view_limit_z=None
        self.was_missing=True;self.missing_since=None
        self.open_since=self.open_stamp=None;self.open_seq=-1
        self.request_ratio=None;self.request_seq=-1;self.request_stamp=None
        self.reset_grasp_entry()
        self.events.append(dict(time_s=float(clock),previous=self.stage,stage=self.stage,
            reason='collection_scene_reset',trial_id=self.trial_id,reset_count=self.scene_reset_count))

    def reject_motion(self,reason,human,feedback,clock,blocked_axes=None):
        if self.phase=='manual':return super().reject_motion(reason,human,feedback,clock,blocked_axes)
        if self.phase=='restore_view':
            self.enter('manual',clock,'upward view recovery blocked: '+reason)
            self.was_missing=True;self.view_loss_reference=None
            return
        if self.phase=='lower' and self.grasp_confirmed and reason.startswith((
                'cartesian_target_unreachable:','task_motion_collision_rejected:',
                'held_geometry_requires_excessive_support_correction')):
            # OPEN was recognized; it is the PLACE goal that failed. Return
            # wrist control instead of freezing it and retrying the same goal.
            self.enter('transport',clock,'placement blocked: grip held; move wrist to another place')
            self.placement_blocked_origin=feedback.tcp.copy();self.open_since=None
            self.carry_missing=True
            return
        if self.phase=='transport' and self.grasp_confirmed and reason.startswith((
                'cartesian_target_unreachable:','task_motion_collision_rejected:')):
            # Reject just this carry increment. Rebase its human reference so
            # moving away from a limit can resume; keep B's collision checks.
            super().reject_motion(reason,human,feedback,clock,blocked_axes)
            self.message='HUMAN CARRY limited: grip held; move wrist back from the limit to continue'
            return
        if self.phase not in ('recover','holding_payload'):
            self.message='B motion held: '+reason
            if self.grasp_confirmed and self.phase not in ('release','retract','verify'):
                self.enter('holding_payload',clock,'motion blocked; keep payload, open to request safe lowering')
            else:
                self.end_trial(clock,'failed',reason)
                self.enter('recover',clock,'grasp not completed; open hand, adjust and retry')

    def finish(self,clock,reason='operator_finished'):
        self.end_trial(clock,'interrupted',reason)
        super().finish(clock,reason)

    def update(self,human,feedback,clock,start=False,paused=False,recenter=False):
        if human is not None and np.isfinite(human.capture_s):
            self.input_seq=human.seq;self.input_capture=human.capture_s
        if self.stage in self.terminal:return TaskCommand('hold',finger=0. if self.grasp_confirmed else .044)
        if self.final_notice_until is not None:
            if self.phase!='session_complete':
                self.enter('session_complete',clock,'Final trial recorded; saving collection')
            self.message='Final trial recorded; saving collection'
            return TaskCommand('hold',finger=.044,owner='shared_collection_complete')
        if self.reset_until is not None:
            if clock<self.reset_until:
                if self.phase!='reset':self.enter('reset',clock,'Trial recorded; protected operator reset time')
                self.message=f'RESET TIME: {max(0.,self.reset_until-clock):.1f}s before the next trial can start'
                self.reset_grasp_entry();self.open_since=None
                return TaskCommand('hold',finger=.044,owner='shared_inter_trial_reset')
            self.reset_until=None;self.was_missing=True
            self.enter('manual',clock,'Reset complete; approach the cube with an open hand, then pinch to start')
            return TaskCommand('hold',finger=.044,owner='shared_inter_trial_ready')
        if self.phase=='manual':
            if self.stage=='direct' and feedback.visual is None:
                if not np.isfinite(clock) or self.last_clock is not None and clock<self.last_clock:raise ValueError('Task clock must be finite and monotonic')
                self.last_clock=clock;command=TaskCommand('hold',finger=.044)
                valid=human is not None and human.valid and human.point is not None and np.isfinite(human.point).all()
                if valid:
                    if self.view_loss_reference is None:self.view_loss_reference=self.last_visible_human_point
                    if recenter or self.was_missing or self.missing_since is not None and clock-self.missing_since>=.35:
                        self.view_loss_reference=np.asarray(human.point).copy();self.was_missing=False
                    self.missing_since=None
                    prior=self.last_target
                    recent=prior is not None and 0<=clock-prior['time_s']<=30.
                    retreat=self.view_loss_reference is not None and (human.point[2]-self.view_loss_reference[2])*self.motion_gains[2]>.002
                    if (not paused and retreat and recent and np.linalg.norm(feedback.tcp-prior['center'])<.15):
                        self.view_robot_origin=feedback.tcp.copy();self.view_hand_origin=np.asarray(self.view_loss_reference).copy()
                        self.view_limit_z=min(self.config.upper[2],feedback.tcp[2]+.070)
                        self.enter('restore_view',clock,'Human-controlled upward motion; no fixed automatic retreat')
                        return self.human_view_command(human,feedback,clock)
                else:
                    if self.missing_since is None:self.missing_since=clock
                    if clock-self.missing_since>=.35:self.was_missing=True
            elif self.stage=='direct' and human is not None and not human.valid and self.hand_intent_valid:
                if self.last_clock is not None and clock<self.last_clock:raise ValueError('Task clock must be monotonic')
                self.last_clock=clock
                if self.missing_since is None:self.missing_since=clock
                if clock-self.missing_since>=.35:self.was_missing=True
                command=TaskCommand('hold',finger=.044)
            else:command=super().update(human,feedback,clock,start=start,paused=paused,recenter=recenter)
            if feedback.visual is not None and human is not None and human.valid and human.point is not None and not paused:
                self.last_visible_human_point=np.asarray(human.point).copy();self.view_loss_reference=None
            if self.hand_intent_valid and human is not None and human.gesture_valid:self._gesture(human,clock)
            # A pinch in shared mode requests a grasp, rather than closing on
            # the cube's edge while awaiting a sub-millimetre hand alignment.
            command.finger=.044
        else:
            if not np.isfinite(clock) or self.last_clock is not None and clock<self.last_clock:
                raise ValueError('Task clock must be finite and monotonic')
            self.last_clock=clock
            command=TaskCommand('hold',finger=0. if self.grasp_confirmed or self.phase=='close' else .044)
            if self.hand_intent_valid and human is not None and human.gesture_valid:self._gesture(human,clock)
            else:self.gesture=None;self.candidate=None;self.candidate_since=None
        if paused:
            self.stable_since=None;self.was_missing=True
            self.gesture=self.candidate=self.candidate_since=None
            self.open_since=None
            self.reset_grasp_entry()
            held=float(self.context.get('finger_target',0. if self.grasp_confirmed else .044)) if self.context else .044
            return TaskCommand('hold',finger=held,owner='shared_pause_hold')
        if self.stage!='direct':return command
        if self.phase=='manual' and feedback.visual is not None and not feedback.visual.get('pose_is_prior',False):
            self.last_target=dict(feedback.visual,center=np.asarray(feedback.visual['center']).copy())
        open_intent=self.hand_intent_valid and self.gesture=='open'
        closed_intent=self.hand_intent_valid and self.gesture=='closed'
        cancel_open=self.confirmed_open(human,.7)
        if self.grasp_confirmed and self.phase in ('lift','transport'):
            if not feedback.gripping:
                if self.grip_missing_since is None:self.grip_missing_since=clock
                if clock-self.grip_missing_since>.4:
                    self.enter('holding_payload',clock,'grip feedback uncertain; hold motion, OPEN requests safe lowering')
                    return TaskCommand('hold',finger=0.,owner='shared_uncertain_grip_hold')
            else:self.grip_missing_since=None
        if self.phase=='manual':
            visual=feedback.visual
            readiness=self.grasp_readiness(feedback,clock);target=readiness['target']
            if self.fresh_grasp_request(human,readiness):
                self.trial_id+=1;self.trials.append(dict(trial_id=self.trial_id,start_s=float(clock),
                    status='active',source='fresh in-range open-to-pinch + B RGB-D grasp planning'))
                self.collection_hold_complete=False
                self.reset_grasp_entry()
                self.intent_latched=True;self.object_anchor=np.asarray(target['center']).copy()
                self.enter('plan',clock,'New in-range pinch confirmed by 2D/3D hand input: planning')
                return TaskCommand('plan',finger=.044,owner='shared_rgbd_plan')
            if visual is None:self.message='Target occluded: move hand UP; open jaws follow your measured upward movement'
            else:self.message='HUMAN APPROACH: wrist follow; near cube new PINCH requests precision grasp'
            return command
        if self.phase=='restore_view':return self.human_view_command(human,feedback,clock,recenter)
        if self.phase in ('reacquire','plan','transit','approach','descend','close') and cancel_open:
            self.end_trial(clock,'cancelled','fresh human open held 0.7s before lift')
            self.enter('recover',clock,'grasp cancelled by sustained OPEN; adjust and pinch to retry')
            return TaskCommand('gripper_reference',finger=.044,owner='human_cancel_grasp')
        if self.phase in ('reacquire','restore_view'):
            if self._stable(feedback.command_stage=='reacquire' and feedback.reached and feedback.opened,clock,.15):
                visual=feedback.visual
                if visual is not None and not visual.get('pose_is_prior',False) and visual['time_s']>=self.phase_since:
                    if self.phase=='restore_view':
                        self.enter('manual',clock,'View restored: human wrist control resumes; PINCH requests grasp')
                        self.was_missing=True;self.view_loss_reference=None
                        return TaskCommand('hold',finger=.044,owner='human_view_restored')
                    self.object_anchor=np.asarray(visual['center']).copy()
                    self.enter('plan',clock,'View restored: planning from NEW robot RGB-D pose')
                    return TaskCommand('plan',finger=.044,owner='shared_rgbd_plan')
            if clock-self.phase_since>6.:
                if self.phase=='restore_view':
                    self.enter('manual',clock,'view recovery timed out; no blind grasp')
                    self.was_missing=True;self.view_loss_reference=None
                    return TaskCommand('hold',finger=.044)
                self.end_trial(clock,'failed','fresh depth view not restored after open-jaw retreat')
                self.enter('recover',clock,'view still blocked; open hand and adjust upward to retry')
                return TaskCommand('hold',finger=.044)
            self.message='AUTO VIEW: open jaws, retreat upward, then obtain a NEW depth pose'
            return TaskCommand('reacquire',self.reacquire_target.copy(),.044,
                'human_restore_view' if self.phase=='restore_view' else 'shared_restore_view')
        if self.phase=='plan':
            if feedback.plan_ready:
                entry=getattr(feedback,'plan_entry_stage','transit')
                self.enter(entry,clock,'Plan ready: '+('direct safe approach' if entry=='approach' else 'clearance transit required'))
            elif clock-self.phase_since>12:
                self.end_trial(clock,'failed','planning timeout');self.enter('recover',clock,'no feasible plan; open and retry')
                return TaskCommand('hold',finger=.044)
            return TaskCommand(self.phase if self.phase!='recover' else 'hold',finger=.044,owner='shared_rgbd_plan')
        if self.phase in ('transit','approach','descend'):
            phase=self.phase
            if self._stable(feedback.command_stage==phase and feedback.reached,clock,.15):
                self.enter(dict(transit='approach',approach='descend',descend='close')[phase],clock,'B reached '+phase)
            elif clock-self.phase_since>15:
                self.end_trial(clock,'failed',phase+' timeout');self.enter('recover',clock,'approach held; open and retry')
                return TaskCommand('hold',finger=.044)
            return TaskCommand(self.phase,finger=0. if self.phase=='close' else .044,owner='shared_depth_alignment')
        if self.phase=='close':
            # The HUMAN request is already latched. Confirmation now uses
            # new ROBOT observations; a briefly occluded hand cannot deadlock it.
            if self._stable(grasp_evidence(self.context),clock,.25):
                self.grasp_confirmed=True;self.enter('lift',clock,'Grip confirmed: lifting')
                return TaskCommand('lift',finger=0.,owner='shared_confirmed_lift')
            if clock-self.phase_since>4.:
                self.end_trial(clock,'failed','grip not confirmed');self.enter('recover',clock,'grip not confirmed; open and retry')
                return TaskCommand('gripper_reference',finger=.044,owner='shared_failed_grasp_open')
            aperture=float(np.min(self.context['finger_positions']))*1000 if self.context else float('nan')
            gap=(np.linalg.norm(np.asarray(self.context['pads'])[1]-self.context['pads'][0])-2*PAD_INNER_OFFSET)*1000 if self.context else float('nan')
            self.message=f'AUTO CLOSE: pad gap {gap:.1f} mm / cube 57.6 mm; encoder {aperture:.1f} mm; confirming grip'
            return TaskCommand('close',finger=0.,owner='shared_grasp_close')
        if self.phase=='lift':
            if self._stable(feedback.command_stage=='lift' and feedback.reached and feedback.gripping,clock,.4):
                self.enter('transport',clock,'Lift complete: move your hand to carry')
                self.carry_missing=True
                return TaskCommand('hold',finger=0.,owner='shared_to_human_handoff')
            if clock-self.phase_since>15:
                self.enter('holding_payload',clock,'lift held; preserve grip, open to request safe lowering')
            return TaskCommand('lift' if self.phase=='lift' else 'hold',finger=0.,owner='shared_confirmed_lift')
        if self.phase in ('transport','holding_payload'):
            if self.collection_task_profile=='hold':
                remaining=max(0.,self.collection_hold_seconds-(clock-self.phase_since)) if not self.collection_hold_complete else 0.
                if remaining>0.:
                    self.open_since=None
                    self.message=f'PAYLOAD HOLD: keep still for {remaining:.1f}s before returning to source'
                    return TaskCommand('hold',finger=0.,owner='shared_collection_payload_hold')
                self.collection_hold_complete=True;self.release_armed=True
                if self.confirmed_open(human,.35):
                    self.release_target=feedback.tcp.copy();self.release_target[2]=self.object_anchor[2]+.002
                    self.enter('lower',clock,'hold complete: return to source support before release')
                    return TaskCommand('lower',self.release_target.copy(),0.,'shared_source_return')
                self.message='HOLD COMPLETE: keep wrist still and OPEN 0.35s to return the cube to its source'
                return TaskCommand('hold',finger=0.,owner='shared_collection_wait_open')
            if closed_intent:self.release_armed=True
            if self.placement_blocked_origin is not None:
                if np.linalg.norm(feedback.tcp-self.placement_blocked_origin)>=.020:
                    self.placement_blocked_origin=None
                # A NEW position needs a NEW confirmation. Continued OPEN at
                # the rejected point cannot start a retry loop.
                self.open_since=None
            if (self.placement_blocked_origin is None and (self.release_armed or self.phase=='holding_payload')
                and self.confirmed_open(human,.35)):
                self.release_target=feedback.tcp.copy();self.release_target[2]=self.object_anchor[2]+.002
                self.enter('lower',clock,'human open: lower to support before release')
                return TaskCommand('lower',self.release_target.copy(),0.,'shared_supported_release')
            valid=human is not None and human.valid and human.point is not None
            if not valid or self.phase=='holding_payload':
                if self.carry_lost_since is None:self.carry_lost_since=clock
                if clock-self.carry_lost_since>=.35:self.carry_missing=True
                self.message=('PAYLOAD HOLD: grip feedback uncertain; grip preserved, OPEN requests lowering'
                    if self.phase=='holding_payload' else 'HUMAN CARRY held: wrist missing; grip preserved')
                return TaskCommand('hold',finger=0.,owner='shared_payload_hold')
            if self.carry_lost_since is not None and clock-self.carry_lost_since>=.35:self.carry_missing=True
            self.carry_lost_since=None
            if self.carry_missing or recenter or self.was_missing:
                self.rebase(human,feedback);self.carry_missing=False;self.was_missing=False
            dt=.033 if not hasattr(self,'carry_clock') else min(.1,max(0.,clock-self.carry_clock))
            self.carry_clock=clock
            target=self._follow(human,dt,feedback.tcp)
            self.message=('HUMAN CARRY: move wrist XYZ; OPEN 0.35s to place' if self.release_armed else
                'HUMAN CARRY: wrist control active; pinch once to arm OPEN-to-place')
            if self.placement_blocked_origin is not None:
                self.message='PLACE UNREACHABLE: grip held; wrist control restored, move at least 2cm to retry'
            return TaskCommand('transport',target,0.,'human_carry_xyz')
        if self.phase=='lower':
            if self._stable(feedback.command_stage=='lower' and feedback.reached,clock,.2):
                self.enter('release',clock,'support height reached; unload and open jaws')
            elif clock-self.phase_since>15:
                self.enter('holding_payload',clock,'placement held; keep grip and adjust')
                return TaskCommand('hold',finger=0.)
            return TaskCommand(self.phase,self.release_target.copy(),.044 if self.phase=='release' else 0.,'shared_supported_release')
        if self.phase=='release':
            if self._stable(feedback.opened,clock,.25):
                self.grasp_confirmed=False;self.target=feedback.tcp+[0,0,.07]
                self.enter('retract',clock,'jaws open; retract before next trial')
                return TaskCommand('retract',self.target.copy(),.044,'shared_retract')
            if clock-self.phase_since>3.:
                self.end_trial(clock,'failed','jaws did not open in time');self.enter('recover',clock,'release held; open and retry')
                return TaskCommand('gripper_reference',finger=.044,owner='shared_release_retry')
            return TaskCommand('release',finger=.044,owner='shared_release')
        if self.phase=='retract':
            if self._stable(feedback.command_stage=='retract' and feedback.reached,clock,.15):
                self.enter('verify',clock,'camera checks supported placement')
            return TaskCommand(self.phase,self.target.copy(),.044,'shared_retract')
        if self.phase=='verify':
            visual=feedback.visual
            good=visual is not None and abs(visual['center'][2]-self.object_anchor[2])<.015
            if self._stable(good,clock,.5) or clock-self.phase_since>3.:
                self.end_trial(clock,'completed' if good else 'failed','placement observed' if good else 'placement not observed')
                self.enter('manual',clock,'Trial recorded; open then pinch for next try. Q saves session')
                self.was_missing=True
            return TaskCommand('hold',finger=.044,owner='shared_trial_verification')
        if self.phase=='recover':
            if open_intent:
                self.enter('manual',clock,'retry after explicit open; confirm OPEN again in range')
                self.reset_grasp_entry();self.was_missing=True
            return TaskCommand('gripper_reference',finger=.044,owner='shared_retry')
        raise RuntimeError('unknown shared phase '+self.phase)
