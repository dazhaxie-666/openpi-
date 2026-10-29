"""Rebuild A observations into pure imitation, without historical avoidance.

The arm fit uses the actual upper/forearm centreline, not the robot base or
the gripper tip. Wrist orientation has its own measured-hand target; a missing
hand cannot drive a free IK/null-space spin to improve a wrist position.
"""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from a_reference_adapter import SIDES,resample_reference
from ik_solver import OpenArmIK
from human_teleop_core import human_arm_pose
from teleop_towel_dataset import build_arm_specs


def unit(v):
    norm=np.linalg.norm(v)
    if not np.isfinite(norm) or norm<1e-8:raise ValueError('Degenerate arm direction')
    return v/norm


def bone_directions(ik,q):
    p=ik.forward_points(q)['joint_positions']
    # Joint3->joint4 is the upper arm; joint4->joint7 is the forearm.
    # Exclude the lateral shoulder mount and the wrist-dependent tool offset.
    return unit(p[3]-p[2]),unit(p[6]-p[3])


class WristContinuity:
    def __init__(self):
        self.rotation=None;self.candidate=None;self.confirmations=0;self.rejected=0
        self.velocity=np.zeros(3)

    def update(self,wanted,dt):
        if self.rotation is None:
            self.rotation=np.asarray(wanted,float).copy()
            return self.rotation.copy()
        delta=Rotation.from_matrix(wanted@self.rotation.T).as_rotvec()
        size=np.linalg.norm(delta)
        if size>np.deg2rad(110):
            consistent=self.candidate is not None and Rotation.from_matrix(wanted@self.candidate.T).magnitude()<np.deg2rad(20)
            self.confirmations=self.confirmations+1 if consistent else 1
            self.candidate=wanted.copy()
            if self.confirmations<3:
                self.rejected+=1
                self.velocity[:]=0.
                return self.rotation.copy()
        else:self.candidate=None;self.confirmations=0
        alpha=1-np.exp(-dt/.06)
        desired=delta*min(alpha,4.*dt/max(size,1e-12))/dt
        acceleration=desired-self.velocity
        self.velocity+=acceleration*min(1.,20.*dt/max(np.linalg.norm(acceleration),1e-12))
        step=self.velocity*dt
        if np.dot(step,delta)<=0 or np.linalg.norm(step)>size:
            step=delta*min(alpha,4.*dt/max(size,1e-12));self.velocity=step/dt
        self.rotation=Rotation.from_rotvec(step).as_matrix()@self.rotation
        return self.rotation.copy()


def bounded_wrist_interval(seed,velocity,limits,dt):
    lower=np.maximum(limits[:,0],seed+dt*np.maximum(-4.,velocity-30.*dt))
    higher=np.minimum(limits[:,1],seed+dt*np.minimum(4.,velocity+30.*dt))
    # A physical stop has priority over a soft acceleration envelope. Near a
    # limit the latter can become infeasible; stop rather than flip IK branch.
    blocked=lower>=higher
    if np.any(blocked):
        lower[blocked]=np.maximum(limits[blocked,0],seed[blocked]-4.*dt)
        higher[blocked]=np.minimum(limits[blocked,1],seed[blocked]+4.*dt)
    return lower,higher


def image_plane_targets(rtmw,frames,obs,trusted):
    """Only image-plane motion is observed; missing depth is held, not inferred."""
    targets=obs.copy();usable=trusted.copy();modes=np.where(trusted,'observed_3d','held').astype('U24')
    if rtmw is None:return targets,usable,modes
    if int(np.max(frames))>=len(rtmw['image_xy']):raise ValueError('RTMW does not cover A source frames')
    width,height=float(rtmw['frame_width']),float(rtmw['frame_height'])
    for j in range(2):
        depth=np.zeros(2)
        for i,fid in enumerate(frames):
            if trusted[i,j]:
                depth=obs[i,9*j+np.array([0,3])].copy()
                continue
            ids=[5+j,7+j,9+j]
            points=np.asarray(rtmw['image_xy'][fid,ids],float)
            if (not np.isfinite(points).all() or np.min(rtmw['score'][fid,ids])<.45
                or (points<0).any() or (points[:,0]>=width).any() or (points[:,1]>=height).any()):continue
            anchors=np.asarray(rtmw['image_xy'][fid,[5,6,11,12]],float)
            if not np.isfinite(anchors).all() or np.min(rtmw['score'][fid,[5,6,11,12]])<.3:continue
            horizontal=anchors[1]-anchors[0];vertical=(anchors[0]+anchors[1]-anchors[2]-anchors[3])/2
            if min(np.linalg.norm(horizontal),np.linalg.norm(vertical))<20:continue
            horizontal=unit(horizontal);vertical=vertical-horizontal*np.dot(vertical,horizontal)
            if np.linalg.norm(vertical)<.01:continue
            vertical=unit(vertical);directions=[]
            for k,v in enumerate(np.diff(points,axis=0)):
                if np.linalg.norm(v)<12:break
                lateral=np.array([-np.dot(v,horizontal),np.dot(v,vertical)])
                if np.linalg.norm(lateral)<1e-8:break
                x=float(np.clip(depth[k],-.9,.9))
                yz=unit(lateral)*np.sqrt(1-x*x)
                directions.append(np.r_[x,yz])
            if len(directions)!=2:continue
            u,f=directions;normal=np.cross(u,f)
            normal=unit(normal) if np.linalg.norm(normal)>1e-5 else targets[i,9*j+6:9*j+9]
            targets[i,9*j:9*j+9]=np.r_[u,f,normal]
            usable[i,j]=True;modes[i,j]='rtmw_2d_depth_held'
    return targets,usable,modes


def rebuild_motion(blob,model,hand_cache=None,rtmw=None):
    sampled,legacy,closure,trusted=resample_reference(blob,hz=None)
    obs=sampled['target_obs'];frames=sampled['source_frame_indices'];fps=float(blob['fps']);dt=1/fps
    if not np.isfinite(obs).all():raise ValueError('Nonfinite A body observations')
    targets,usable,modes=image_plane_targets(rtmw,frames,obs,trusted)
    specs=build_arm_specs(model);data=mujoco.MjData(model)
    output=np.zeros_like(legacy,dtype=float);hand_used=np.zeros((len(output),2),bool)
    old_errors=np.zeros((len(output),2,2));new_errors=np.zeros_like(old_errors)
    wrist_filters={s:WristContinuity() for s in SIDES}
    for j,side in enumerate(SIDES):
        ik=OpenArmIK(side);valid=np.flatnonzero(usable[:,j])
        if not len(valid):raise ValueError(f'No trusted {side} arm observation')
        limits=specs[side].q_range
        q=np.r_[ik.home_seed[:4],np.zeros(3)]
        wrist_velocity=np.zeros(3)

        def fit_arm(row,previous,initial=False):
            upper=unit(row[:3]);fore=unit(row[3:6])
            lower=limits[:4,0].copy();higher=limits[:4,1].copy()
            if not initial:
                step=np.array([5.,5.,6.,6.])*dt
                lower=np.maximum(lower,previous[:4]-step)
                higher=np.minimum(higher,previous[:4]+step)
            def residual(arm):
                u,f=bone_directions(ik,np.r_[arm,np.zeros(3)])
                return np.r_[4*(u-upper),4*(f-fore),(.002 if initial else .025)*(arm-previous[:4])]
            result=least_squares(residual,np.clip(previous[:4],lower+1e-9,higher-1e-9),
                                 bounds=(lower,higher),max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            return result.x

        # Offline prefix has no trustworthy measurement: hold the first
        # trustworthy pose explicitly, never invent a prefix trajectory.
        q[:4]=fit_arm(targets[valid[0],9*j:9*j+9],q,initial=True)
        for i,row in enumerate(targets):
            if usable[i,j]:
                q[:4]=fit_arm(row[9*j:9*j+9],q,initial=i==0)
                if hand_cache is not None and trusted[i,j]:
                    fid=int(frames[i]);present=bool(hand_cache['hand_valid'][fid,j])
                    hand=hand_cache['hand_world'][fid,j] if present else None
                    pose=human_arm_pose(hand_cache['pose_world'][fid],hand,side=side) if present else None
                    if pose is not None and pose['palm']:
                        x=unit(pose['palm_normal']);z=pose['rotation'][:,2];z=unit(z-x*np.dot(x,z))
                        wanted=np.column_stack((x,np.cross(z,x),z))
                        data.qpos[specs[side].qpos]=q;mujoco.mj_forward(model,data)
                        filt=wrist_filters[side]
                        if filt.rotation is None:filt.rotation=data.xmat[specs[side].body_id].reshape(3,3).copy()
                        wanted=filt.update(wanted,dt)
                        seed=q[4:].copy()
                        lower,higher=bounded_wrist_interval(seed,wrist_velocity,limits[4:],dt)
                        def wrist_residual(wrist):
                            data.qpos[specs[side].qpos]=np.r_[q[:4],wrist];mujoco.mj_forward(model,data)
                            actual=data.xmat[specs[side].body_id].reshape(3,3)
                            return np.r_[Rotation.from_matrix(wanted@actual.T).as_rotvec(),.05*(wrist-seed)]
                        fitted=least_squares(wrist_residual,np.clip(seed,lower+1e-9,higher-1e-9),bounds=(lower,higher),
                                             max_nfev=40,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                        q[4:]=fitted.x;wrist_velocity=(q[4:]-seed)/dt;hand_used[i,j]=True
            if not hand_used[i,j]:
                wrist_velocity[:]=0.;wrist_filters[side].velocity[:]=0.
            # No trusted body -> hold this arm, including the wrist; the other
            # arm remains independent. No wrist inference from body-only IK.
            output[i,7*j:7*(j+1)]=q
            target=(unit(obs[i,9*j:9*j+3]),unit(obs[i,9*j+3:9*j+6]))
            angle=lambda a,b:float(np.degrees(np.arccos(np.clip(np.dot(a,b),-1,1))))
            old_errors[i,j]=[angle(a,b) for a,b in zip(bone_directions(ik,legacy[i,7*j:7*(j+1)]),target)]
            new_errors[i,j]=[angle(a,b) for a,b in zip(bone_directions(ik,q),target)]
        if len(output)>100:print(f'Clean imitation: {side} complete ({len(output)} frames)',flush=True)
    # Keep the existing RTMW/MediaPipe continuous grip, not hinge radians.
    clean=dict(target_obs=obs.copy(),joint_angles=output,nominal_joint_angles=output.copy(),
               frame_indices=frames.copy(),fps=np.array(fps),joint_order=np.array('left-first'),
               gripper_position=closure.copy(),gripper_cmd=(closure>=.5).astype(float),gripper_order=np.array('left-first'),
               source_arm_trusted=trusted.copy(),source_hand_orientation_used=hand_used,
               motion_target_obs=targets,source_arm_input_mode=modes,
               legacy_joint_angles=legacy.copy(),trajectory_optimizer=np.array('pure_anatomical_motion_no_avoidance'),
               old_direction_error_deg=old_errors,new_direction_error_deg=new_errors)
    means=lambda values:np.array([values[trusted[:,j],j].mean(axis=0) for j in range(2)])
    report=dict(mode='Pure imitation from A body targets, measured hands and actual B wrist geometry',
                historical_avoidance_used=False,frames=len(output),
                old_trusted_direction_mean_deg=means(old_errors).tolist(),
                new_trusted_direction_mean_deg=means(new_errors).tolist(),
                trusted_body_frames=trusted.sum(0).tolist(),measured_hand_orientation_frames=hand_used.sum(0).tolist(),
                image_plane_fallback_frames=np.sum(modes=='rtmw_2d_depth_held',axis=0).tolist(),
                held_body_frames=np.sum(modes=='held',axis=0).tolist(),
                wrist_flip_rejections={s:wrist_filters[s].rejected for s in SIDES},
                max_adjacent_wrist_step_deg=float(np.degrees(np.max(np.abs(np.diff(output[:,[4,5,6,11,12,13]],axis=0))))),
                metric_note='Arm centreline vs A identified directions; not human ground truth or gripper-tip position',
                prefix_policy='Use verified image-plane observations when available; otherwise hold first usable pose',
                missing_policy='2D fallback holds depth; no usable body holds arm; no trusted body/hand holds wrist')
    return clean,report
