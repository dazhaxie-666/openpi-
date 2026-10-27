"""Opt-in direct assistance, including bounded intent-gated camera centering.

Only camera object estimates and robot encoder/servo feedback enter this file.
Simulation object state/contact truth are reserved for independent evaluation.
"""
from dataclasses import replace
import numpy as np
from human_grasp_task import DirectHumanTask


def camera_enclosure(obs):
    if obs is None or obs.get('center') is None or not 0<=obs.get('visual_age_s',np.inf)<=.5:return False
    pads=np.asarray(obs['pads']);center=np.asarray(obs['center']);tcp=pads.mean(0)
    span=np.linalg.norm(pads[1]-pads[0])
    if span<.005 or not np.isfinite(np.r_[pads.ravel(),center]).all():return False
    axis=(pads[1]-pads[0])/span;delta=center-tcp
    yaw=obs.get('yaw',0.);c,s=np.cos(yaw),np.sin(yaw)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    width=2*obs.get('half_size',.0288)*np.sum(np.abs(rotation.T@axis))
    return bool(abs(delta@axis)<=.012 and np.linalg.norm(delta-axis*(delta@axis))<=.023
                and .6*width<=span<=width+.025 and pads[:,2].min()>obs.get('table_top',.4)+.005)


def bilateral_touch_estimate(obs,max_velocity=.0015):
    """Refine only the closing-axis position from corroborated encoder stalls.

    This uses servo effort, aperture and the known target width, not simulator
    contacts. Camera position/orientation remain the prior in other axes.
    """
    if not camera_enclosure(obs):return None
    q=np.asarray(obs['finger_positions']);pads=np.asarray(obs['pads'])
    if (np.min(np.abs(obs['finger_force']))<.08 or np.max(np.abs(obs['finger_velocity']))>max_velocity
            or np.ptp(q)>.018 or np.min(q)<.008 or np.max(q)>.041):return None
    axis=pads[1]-pads[0];span=np.linalg.norm(axis);axis/=span
    c,s=np.cos(obs.get('yaw',0.)),np.sin(obs.get('yaw',0.));rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    width=2*obs.get('half_size',.0288)
    if abs(span-2*PAD_INNER_OFFSET-width)>.004:return None
    center=np.asarray(obs['center']);offset=float((center-pads.mean(0))@axis)
    # Matching the known edge width corroborates opposing faces, rather than
    # a diagonal/corner pinch. Refine the face normal from jaw encoders; retain
    # the closest camera symmetry class and the upright prior about that axis.
    agreement=np.abs(rotation[:,:2].T@axis)
    if np.max(agreement)<.6:return None
    j=int(np.argmax(agreement))
    normal=axis*np.sign(rotation[:,j]@axis)
    up=np.array([0.,0.,1.])-normal*normal[2]
    if np.linalg.norm(up)<.7:return None
    up/=np.linalg.norm(up);refined=np.empty((3,3));refined[:,2]=up
    if j==0:refined[:,0]=normal;refined[:,1]=np.cross(up,normal)
    else:refined[:,1]=normal;refined[:,0]=np.cross(normal,up)
    return dict(center=center-axis*offset,rotation=refined)


def bilateral_touch_center(obs):
    result=bilateral_touch_estimate(obs)
    return None if result is None else result['center']


PAD_INNER_OFFSET=.0072  # Measured B mesh inner surface relative to its TCP proxy.


def predicted_contact_stop(pads,positions,requested,center,yaw,half_size=.0288,object_rotation=None):
    """Private collision-query finger position, not an actuator command.

    A position-servo target can lie inside an object while the actual finger
    stalls on its surface. Query that expected stop, retaining all non-pad
    collision checks; never teleport the real fingers or object.
    """
    pads=np.asarray(pads);q=np.asarray(positions);result=np.full(2,float(requested))
    obs=dict(pads=pads,center=center,yaw=yaw,visual_age_s=0.,half_size=half_size,table_top=-np.inf)
    if not camera_enclosure(obs):return result
    c,s=np.cos(yaw),np.sin(yaw)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]]) if object_rotation is None else np.asarray(object_rotation)
    for j in (0,1):
        axis=pads[1-j]-pads[j];axis/=np.linalg.norm(axis)
        extent=half_size*np.sum(np.abs(rotation.T@axis))
        gap=float((np.asarray(center)-pads[j])@axis)-extent-PAD_INNER_OFFSET
        stop=q[j]-gap-.0004
        if requested<q[j]:result[j]=max(float(requested),min(float(q[j]),float(stop)))
    return result


def allowed_pad_contact(point,normal,pad,other,center,depth,enabled,max_depth=.0015):
    """An intended inner-pad patch only; never a whole finger/object exemption."""
    if not enabled or center is None or not 0<=depth<=max_depth:return False
    values=np.r_[point,normal,pad,other,center]
    if not np.isfinite(values).all():return False
    axis=np.asarray(other)-pad;span=np.linalg.norm(axis)
    if span<.005:return False
    axis/=span;offset=np.asarray(point)-pad
    middle=(np.asarray(pad)+other)/2;delta=np.asarray(center)-middle
    return bool(np.dot(normal,axis)>=.7 and abs(offset@axis-PAD_INNER_OFFSET)<=.003
        and np.linalg.norm(offset-axis*(offset@axis))<=.018
        and abs(delta@axis)<=.012 and np.linalg.norm(delta-axis*(delta@axis))<=.023)


class AutomaticPrecisionTask(DirectHumanTask):
    def __init__(self,config=None):
        super().__init__(config);self.context=None;self.carrying=False
        self.precision_scale=1.;self.near=False;self.approach_since=None
        self.previous_point=None;self.applied_gains=None
        self.manual_human_anchor=self.manual_robot_anchor=None;self.manual_goal=None
        self.object_error=np.full(3,np.nan);self.object_distance=np.nan
        self.visual_alignment_enabled=False;self.alignment_pending=False;self.alignment_state='disabled'
        self.alignment_total=np.zeros(3);self.alignment_step=np.zeros(3)
        self.alignment_since=None;self.alignment_stamp=None;self.alignment_ready=False

    def rebase(self,human,feedback):
        super().rebase(human,feedback)
        self.previous_point=np.asarray(human.point).copy()
        self.applied_gains=self.motion_gains*self.precision_scale
        self.manual_human_anchor=self.previous_point.copy();self.manual_robot_anchor=feedback.tcp.copy()
        self.alignment_total=np.zeros(3);self.alignment_since=None;self.alignment_ready=False

    def motion_parameters(self):
        weight=(1-self.precision_scale)/.75
        return (self.motion_gains*self.precision_scale,
                self.config.speed*(1-weight)+.02*weight,
                self.config.max_target_lead_m*(1-weight)+.008*weight)

    @property
    def motion_deadband_m(self):return self.config.deadband_m*self.precision_scale

    def rebase_axes(self,human,feedback,axes):
        super().rebase_axes(human,feedback,axes)
        if self.manual_human_anchor is not None:
            self.manual_human_anchor[axes]=human.point[axes]
            self.manual_robot_anchor[axes]=feedback.tcp[axes]

    def update(self,human,feedback,clock,**kwargs):
        dt=.05 if self.last_clock is None else min(.1,max(0.,clock-self.last_clock))
        desired=1.;obs=self.context;retreat=False
        valid=human is not None and human.valid and human.point is not None and np.isfinite(human.point).all()
        if obs is not None and obs.get('center') is not None and 0<=obs.get('visual_age_s',np.inf)<=.5:
            self.object_error=np.asarray(obs['center'])-np.asarray(obs['tcp'])
            self.object_distance=float(np.linalg.norm(self.object_error))
            delta=np.zeros(3) if not valid or self.previous_point is None else self.motion_gains*(human.point-self.previous_point)
            direction=float(delta@self.object_error/max(self.object_distance,1e-9))
            retreat=np.linalg.norm(delta)>.0005 and direction<-.3*np.linalg.norm(delta)
            approach=direction>.0002
            closing=valid and human.gesture_valid and human.pinch_ratio<self.config.close_ratio
            corridor=abs(self.object_error[0])<.085 and abs(self.object_error[1])<.085 and -.025<-self.object_error[2]<.13
            if self.carrying or self.object_distance>.13 or retreat or not corridor:
                self.near=False;self.approach_since=None
            elif not self.near and self.object_distance<.095 and valid and (approach or closing):
                if self.approach_since is None:self.approach_since=clock
                if clock-self.approach_since>=.1:self.near=True
            if self.near and not self.carrying and valid and not kwargs.get('paused',False):
                x=float(np.clip((self.object_distance-.025)/.07,0,1))
                desired=.25+.75*x*x*(3-2*x)
        else:self.near=False;self.approach_since=None;self.object_distance=np.nan;self.object_error[:]=np.nan
        self.precision_scale+=float(np.clip(desired-self.precision_scale,-1.875*dt,1.875*dt))
        command=super().update(human,feedback,clock,**kwargs)
        self.alignment_step[:]=0.;self.alignment_pending=False
        self.alignment_state='idle' if self.visual_alignment_enabled else 'disabled'
        if not self.visual_alignment_enabled:return command
        eligible=(command.stage=='direct' and valid and human.gesture_valid and getattr(self,'alignment_hand_valid',True)
            and human.pinch_ratio<self.config.close_ratio and not self.carrying
            and not kwargs.get('paused',False) and obs is not None and obs.get('center') is not None
            and 0<=obs.get('visual_age_s',np.inf)<=.5 and self.object_distance<.060
            and not retreat)
        if not eligible:
            self.alignment_since=None
            if (valid and human.gesture_valid and human.pinch_ratio>=self.config.open_ratio) or retreat:
                self.alignment_ready=False;self.alignment_total[:]=0.
            return command
        error=np.asarray(obs['center'])+np.array([0.,0.,.003])-np.asarray(obs['tcp'])
        distance=float(np.linalg.norm(error))
        if distance<=.005:self.alignment_ready=True
        elif distance>.012:self.alignment_ready=False
        self.alignment_pending=not self.alignment_ready
        self.alignment_state='aligning' if self.alignment_pending else 'ready to close'
        new=self.alignment_stamp is None or human.capture_s>self.alignment_stamp
        if not new:return command
        self.alignment_stamp=float(human.capture_s)
        if self.alignment_since is None:self.alignment_since=float(human.capture_s)
        if not self.alignment_pending or human.capture_s-self.alignment_since<.15:return command
        # A bounded visual correction joins the SAME B Cartesian reference.
        # It never creates an independent joint controller or lifts a payload.
        remaining=max(0.,.025-float(np.linalg.norm(self.alignment_total)))
        if remaining<1e-6:
            self.alignment_state='correction limit: move closer / open';return command
        delta=np.asarray(obs['center'])+np.array([0.,0.,.003])-command.target
        amount=min(.015*dt,remaining,float(np.linalg.norm(delta)))
        correction=delta*(amount/max(np.linalg.norm(delta),1e-9))
        target=np.clip(command.target+correction,self.config.lower,self.config.upper)
        correction=target-command.target
        self.robot_anchor+=correction;self.target=target.copy()
        self.alignment_total+=correction;self.alignment_step=correction.copy()
        return replace(command,target=target,owner='human_with_visual_centering')

    def _follow(self,human,dt,actual_tcp):
        old_target=self.target.copy()
        gains=self.motion_gains*self.precision_scale
        if self.previous_point is not None and self.applied_gains is not None:
            # Preserve the prior desired position when gain changes. Apply the
            # new gain only to subsequent human displacement, never to old debt.
            before=self.robot_anchor+self.applied_gains*(self.previous_point-self.human_anchor)
            self.robot_anchor=before;self.human_anchor=self.previous_point.copy()
        self.applied_gains=gains.copy();self.previous_point=np.asarray(human.point).copy()
        self.manual_goal=self.manual_robot_anchor+self.motion_gains*(human.point-self.manual_human_anchor)
        result=super()._follow(human,dt,actual_tcp)
        workspace_goal=np.clip(self.robot_anchor+gains*(human.point-self.human_anchor),self.config.lower,self.config.upper)
        # A shrinking lead bound may delay progress, but must not pull an
        # already queued target away from the unchanged human goal.
        away=(result-old_target)*(workspace_goal-old_target)<-1e-12
        self.target[away]=old_target[away]
        return self.target.copy()


class AutomaticGripper:
    def __init__(self):
        self.state='OPEN';self.held=None;self.confirm_since=None;self.confirm_count=0
        self.open_since=None;self.open_count=0;self.last_seq=None;self.last_stamp=None
        self.lost_since=None;self.contact_enabled=False;self.reason='human aperture'

    @property
    def carrying(self):return self.state=='HELD'

    def apply(self,command,hand,obs,clock,dt,active=True,paused=False,now_s=None):
        held_target=float(obs['finger_target'])
        now_s=clock if now_s is None else float(now_s)
        fresh=bool(hand and hand.get('valid') and hand.get('capture_s') is not None
                   and 0<=now_s-hand['capture_s']<=.35)
        new=bool(fresh and (self.last_seq is None or hand['seq']>self.last_seq)
                 and (self.last_stamp is None or hand['capture_s']>self.last_stamp))
        if new:self.last_seq=int(hand['seq']);self.last_stamp=float(hand['capture_s'])
        if not active or paused:
            self.confirm_since=self.open_since=None;self.confirm_count=self.open_count=0
            return command
        if command.stage not in ('direct','gripper_reference','hold'):return command
        wanted=float(hand['aperture']) if fresh else held_target
        q=np.asarray(obs['finger_positions']);forces=np.abs(obs['finger_force']);velocity=np.abs(obs['finger_velocity'])
        if self.held is not None:
            opened=fresh and hand.get('intent_ratio') is not None and hand['intent_ratio']>=.8 and wanted>=.033
            if opened and new:
                if self.open_since is None:self.open_since=hand['capture_s'];self.open_count=0
                self.open_count+=1
                if hand['capture_s']-self.open_since>=.2-1e-9 and self.open_count>=3:
                    self.held=None;self.state='OPEN';self.contact_enabled=False
                    self.confirm_since=None;self.confirm_count=0;self.reason='explicit human release'
            elif not opened:self.open_since=None;self.open_count=0
            if self.held is not None:
                suspect=bool(np.min(q)<.006 or np.min(forces)<.4)
                if suspect:
                    if self.lost_since is None:self.lost_since=clock
                    if clock-self.lost_since>.3:self.state='SLIP_SUSPECT'
                else:self.lost_since=None;self.state='HELD'
                self.contact_enabled=self.state=='HELD'
                self.reason='grip held; waiting for explicit open' if self.state=='HELD' else 'grip uncertain; held aperture'
                return replace(command,finger=self.held)
        if not fresh or command.stage=='hold':
            self.confirm_since=self.open_since=None;self.confirm_count=self.open_count=0
            self.contact_enabled=False;self.reason='hand missing: hold target'
            return replace(command,finger=held_target)
        if obs.get('alignment_pending',False) and wanted<float(np.mean(q))-.001:
            self.confirm_since=None;self.confirm_count=0;self.contact_enabled=False
            self.state='ALIGNING';self.reason='camera centering; keep jaw clearance'
            finger=held_target+float(np.clip(.044-held_target,-.132*dt,.132*dt))
            return replace(command,finger=finger)
        closing=wanted<float(np.mean(q))-.001
        enclosed=camera_enclosure(obs)
        observed_width=float(np.linalg.norm(np.asarray(obs['pads'])[1]-obs['pads'][0])-2*PAD_INNER_OFFSET)
        opposing_faces=abs(observed_width-2*obs.get('half_size',.0288))<=.004
        near=obs.get('center') is not None and np.linalg.norm(np.asarray(obs['center'])-obs['tcp'])<.1
        rate=.020 if closing and near else .132
        finger=float(held_target+np.clip(wanted-held_target,-rate*dt,rate*dt))
        stall=(closing and enclosed and opposing_faces and np.min(q)>.008 and np.max(q)<.041
               and np.ptp(q)<.018 and np.min(forces)>=1.8 and np.sum(forces)>=5.6 and np.max(velocity)<.004)
        self.contact_enabled=bool(closing and near)
        if stall:
            if self.confirm_since is None:self.confirm_since=clock;self.confirm_count=0
            if new:self.confirm_count+=1
            self.state='VERIFYING';self.reason='bilateral stall and camera enclosure'
            if clock-self.confirm_since>=.25-1e-9 and self.confirm_count>=3:
                self.held=held_target;self.state='HELD';self.reason='grip confirmed';finger=self.held
        else:
            self.confirm_since=None;self.confirm_count=0
            self.state='CLOSING' if closing else 'OPEN'
            self.reason=('contact not settled; align or open' if closing and near and not opposing_faces and np.min(forces)>.8 else
                         'align cube between pads; slow close' if closing and near and not enclosed else
                         'slow close near object' if closing and near else 'human aperture')
        return replace(command,finger=finger)
