"""Human path shape -> B visual grasp anchor -> checked B joint trajectory."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from handeye_grasp_red_cube import solve_gripper_contact_proxy_q


def visual_planning_state(model,encoders,center,yaw):
    """Copy robot state only; the virtual object's pose comes from perception."""
    state=mujoco.MjData(model)
    for j in range(model.njnt):
        if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or '').startswith('openarm_'):
            adr=int(model.jnt_qposadr[j]);state.qpos[adr]=encoders.qpos[adr]
    adr=int(model.joint('droid_object_freejoint').qposadr[0])
    state.qpos[adr:adr+3]=center
    state.qpos[adr+3:adr+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    mujoco.mj_forward(model,state)
    return state


def sample_joint_path(anchors,seconds,timestep):
    anchors=np.asarray(anchors,float)
    if (anchors.ndim!=2 or anchors.shape[1]!=7 or len(anchors)<2
            or not np.isfinite(anchors).all() or not np.isfinite([seconds,timestep]).all()
            or seconds<=0 or timestep<=0): raise ValueError('Invalid B joint path')
    # Shape-preserving interpolation cannot overshoot an anchor's joint range.
    # Quintic time warping gives zero velocity/acceleration at both endpoints.
    spline=PchipInterpolator(np.linspace(0,1,len(anchors)),anchors,axis=0)
    velocity_caps=np.array([2.5]*4+[4.4]*3)
    probe=np.linspace(0,1,601)
    phase=10*probe**3-15*probe**4+6*probe**5
    phase_rate=30*probe**2*(1-probe)**2
    duration=max(float(seconds),float(np.max(np.abs(spline(phase,1))*phase_rate[:,None]/velocity_caps))*1.02)
    count=int(np.ceil(duration/timestep));time=np.arange(count+1)*timestep
    u=time/time[-1];q=spline(10*u**3-15*u**4+6*u**5)
    q[0]=anchors[0];q[-1]=anchors[-1]
    return time,q


def plan_demo_descent(demo,model,data,spec,finger_geoms,finger_indices,proxies,
                      center,yaw,start,goal,rotation,span,closing_direction,gain=1.):
    planning=visual_planning_state(model,data,center,yaw)
    prefix=(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,finger_geoms[0]) or '').split('_')[1]
    active={g for g in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or '').startswith(f'openarm_{prefix}_')}
    planning.qpos[finger_indices[0]]=.044

    def penetrations(q):
        planning.qpos[spec.qpos]=q;mujoco.mj_forward(model,planning)
        hits={}
        for c in planning.contact:
            pair=tuple(sorted((int(c.geom1),int(c.geom2))))
            if set(pair)&active:
                hits[pair]=max(hits.get(pair,0.),max(0.,-float(c.dist)))
        return hits

    baseline=penetrations(start)
    initial_limit_error=float(max(np.max(spec.q_range[:,0]-start),np.max(start-spec.q_range[:,1]),0.))
    if initial_limit_error>.025: raise RuntimeError('B encoders too far outside joint range to start a transfer')
    lower=np.minimum(spec.q_range[:,0],start);upper=np.maximum(spec.q_range[:,1],start)
    attempts=[]
    for weight in (gain,gain*.5,gain*.25,0.):
        targets=demo.descent_targets(center,yaw,weight)
        anchors=[np.asarray(start).copy()];max_error=0.
        for point in targets[1:-1]:
            q,_,_,metrics=solve_gripper_contact_proxy_q(
                model,planning,spec,finger_geoms,finger_indices,proxies,point,1,1.,span,.044,
                anchors[-1],reference_rotation=rotation,multi_start=False,closing_direction=closing_direction)
            max_error=max(max_error,float(metrics['midpoint_error_m']));anchors.append(q)
        anchors.append(np.asarray(goal).copy())
        time,q=sample_joint_path(anchors,1.5,float(model.opt.timestep))
        within_limits=bool(np.all(q>=lower-1e-10) and np.all(q<=upper+1e-10)
                           and np.all(q[-1]>=spec.q_range[:,0]) and np.all(q[-1]<=spec.q_range[:,1]))
        # A sweep is sampled with at most 0.01 rad between checked states.
        max_penetration=0.
        last=q[0]
        for current in q[::max(1,int(.01/model.opt.timestep))].tolist()+[q[-1].tolist()]:
            current=np.asarray(current)
            count=max(1,int(np.ceil(np.max(np.abs(current-last))/.01)))
            for alpha in np.linspace(0,1,count+1):
                hits=penetrations(last+alpha*(current-last))
                max_penetration=max(max_penetration,max((depth-baseline.get(pair,0.) for pair,depth in hits.items()),default=0.))
            last=current
        accepted=bool(within_limits and max_error<=.006 and max_penetration<=.0005 and time[-1]<=2.5)
        attempts.append(dict(gain=float(weight),position_error_m=max_error,
                             max_new_penetration_m=max_penetration,within_limits=within_limits,accepted=accepted))
        if accepted:
            demo.plan_report=dict(accepted_gain=float(weight),attempts=attempts,
                                  visual_anchor_m=np.asarray(center).tolist(),visual_yaw_rad=float(yaw),
                                  cartesian_targets_m=targets.tolist(),duration_s=float(time[-1]),
                                  safety='B current geometry, sampled sweep; not continuous collision guarantee',
                                  active_arm=prefix)
            demo.plan_report['initial_soft_limit_error_rad']=initial_limit_error
            return time,q
    raise RuntimeError(f'Human-conditioned descent rejected by B IK/collision checks: {attempts}')
