"""Forearm-relative SO(3) palm tracking and one B-model wrist target solve."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from motion_observation import unit
from teleop_towel_dataset import build_arm_specs


class RelativePalmTracker:
    def __init__(self):
        self.frame=None;self.neutral=None;self.relative=None;self.last_stamp=None
        self.last_valid=None;self.candidate=None;self.candidate_stamp=None;self.state='LOST';self.reason='PALM_MISSING'
        self.raw_relative=None

    def forearm_frame(self,upper,fore):
        z=unit(fore);normal=np.cross(unit(upper),z)
        if np.linalg.norm(normal)>.08:x=unit(np.cross(unit(normal),z))
        elif self.frame is not None:
            projected=self.frame[:,0]-z*np.dot(z,self.frame[:,0])
            x=unit(projected) if np.linalg.norm(projected)>1e-5 else unit(np.cross(np.eye(3)[np.argmin(np.abs(z))],z))
        else:x=unit(np.cross(np.eye(3)[np.argmin(np.abs(z))],z))
        if self.frame is not None and np.dot(x,self.frame[:,0])<0:x=-x
        self.frame=np.column_stack((x,np.cross(z,x),z))
        return self.frame

    def update(self,upper,fore,hand,transform,stamp,side='right'):
        frame=self.forearm_frame(upper,fore)
        h=None if hand is None else np.asarray(hand,dtype=float)
        observed=None
        if h is not None and h.shape==(21,3) and np.isfinite(h[[0,5,9,17]]).all():
            longitudinal=transform@(h[9]-h[0]);lateral=transform@(h[5]-h[17])
            if min(np.linalg.norm(longitudinal),np.linalg.norm(lateral))>.015:
                z=unit(longitudinal);x=lateral-z*np.dot(lateral,z)
                if np.linalg.norm(x)>.008:
                    x=unit(x)*(1. if side=='right' else -1.)
                    observed=frame.T@np.column_stack((x,np.cross(z,x),z))
        if observed is None:
            self.state='HOLD' if self.relative is not None else 'LOST';self.reason='PALM_GEOMETRY_MISSING'
            self.candidate=None;self.last_stamp=stamp
            return frame,self.relative
        dt=.05 if self.last_stamp is None else stamp-self.last_stamp
        if dt<=0:return frame,self.relative
        gap=self.last_valid is None or stamp-self.last_valid>.2 or self.state in ('HOLD','LOST','REACQUIRING')
        if self.relative is not None and not gap:
            innovation=Rotation.from_matrix((self.raw_relative if self.raw_relative is not None else self.relative).T@observed).magnitude()
            if innovation>np.deg2rad(12)+np.deg2rad(600)*min(dt,.1):
                self.reason='PALM_ROTATION_OUTLIER';self.state='HOLD';self.candidate=None;self.last_stamp=stamp
                return frame,self.relative
        if gap:
            consistent=(self.candidate is not None and 0<stamp-self.candidate_stamp<=.2 and
                        Rotation.from_matrix(self.candidate.T@observed).magnitude()<np.deg2rad(18))
            if not consistent:
                self.candidate=observed.copy();self.candidate_stamp=stamp;self.state='REACQUIRING';self.reason='PALM_CONFIRMING';self.last_stamp=stamp
                return frame,self.relative
            # A large reacquisition is not accepted just because it repeats.
            # Recalibration is explicit; impossible orientation remains unknown.
            if self.relative is not None and Rotation.from_matrix(self.relative.T@observed).magnitude()>np.deg2rad(100):
                self.reason='PALM_REACQUISITION_AMBIGUOUS';self.state='REACQUIRING';self.last_stamp=stamp
                return frame,self.relative
        if self.relative is None:self.relative=observed.copy()
        else:
            delta=Rotation.from_matrix(observed@self.relative.T).as_rotvec()
            speed=Rotation.from_matrix((self.raw_relative if self.raw_relative is not None else self.relative).T@observed).magnitude()/max(.01,min(.1,dt))
            alpha=1-np.exp(-2*np.pi*(2.+.5*speed)*min(.1,dt))
            self.relative=Rotation.from_rotvec(delta*alpha).as_matrix()@self.relative
        self.raw_relative=observed.copy();self.last_valid=self.last_stamp=stamp
        if self.neutral is None:self.neutral=observed.copy()
        self.candidate=None;self.state='TRACKING';self.reason=''
        return frame,self.relative

    @property
    def delta(self):
        return np.eye(3) if self.neutral is None or self.relative is None else self.relative@self.neutral.T


class BWristRetargeter:
    """A remains the arm reference; B solves its wrist orientation, not a sum
    of independently generated arm commands. First four A joints are fixed.
    """
    def __init__(self,model):
        self.model=model;self.data=mujoco.MjData(model);self.specs=build_arm_specs(model)
        self.jp=np.zeros((3,model.nv));self.jr=np.zeros_like(self.jp)
        self.alignment={};self.previous={};self.metrics={}

    def solve(self,side,nominal,delta,human_forearm):
        spec=self.specs[side];q=np.asarray(nominal,dtype=float).copy()
        q=np.clip(q,spec.q_range[:,0],spec.q_range[:,1])
        self.data.qpos[spec.qpos]=q;mujoco.mj_forward(self.model,self.data)
        nominal_rotation=self.data.xmat[spec.body_id].reshape(3,3).copy()
        if side not in self.alignment:self.alignment[side]=nominal_rotation.T@human_forearm
        c=self.alignment[side];wanted=nominal_rotation@c@delta@c.T
        if side in self.previous:q[4:]=self.previous[side]
        for _ in range(8):
            self.data.qpos[spec.qpos]=q;mujoco.mj_forward(self.model,self.data)
            current=self.data.xmat[spec.body_id].reshape(3,3)
            error=Rotation.from_matrix(wanted@current.T).as_rotvec()
            if np.linalg.norm(error)<.008:break
            mujoco.mj_jacBody(self.model,self.data,self.jp,self.jr,spec.body_id)
            jac=self.jr[:,spec.dof[4:]]
            step=np.linalg.solve(jac.T@jac+.003*np.eye(3),jac.T@error)
            q[4:]+=np.clip(step,-.18,.18)
            q=np.clip(q,spec.q_range[:,0],spec.q_range[:,1])
        self.data.qpos[spec.qpos]=q;mujoco.mj_forward(self.model,self.data)
        residual=float(Rotation.from_matrix(wanted@self.data.xmat[spec.body_id].reshape(3,3).T).magnitude())
        self.previous[side]=q[4:].copy();self.metrics[side]=dict(rotation_residual_rad=residual,limited=residual>np.deg2rad(15),target_rotation=wanted.copy())
        return q
