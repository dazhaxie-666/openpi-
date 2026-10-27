"""Full A motion in B's task scene, with camera-based collision context."""
from __future__ import annotations

from types import SimpleNamespace
import time

import mujoco
import numpy as np

from bimanual_teleop import BimanualController,SIDES
from realtime_bimanual_planner import RealtimeBimanualPlanner,InterarmGeometry,TimedRRT
from calibration_simulation import body_transform,render_rgb_depth
from handeye_grasp_red_cube import estimate_cube_center,load_handeye,latest_handeye_report
from demo_grasp_viewer import (load_chest_calibration,load_workspace_rgbd_calibration,
                               apply_workspace_rgbd_calibration,CUBE_HALF)


class BVisualObserver:
    def __init__(self,model,specs,camera_sides=(None,'left','right')):
        self.model=model;self.specs=specs
        self.camera_sides=tuple(camera_sides)
        self.chest_transform,self.chest_matrix,_=load_chest_calibration()
        self.calibration=load_workspace_rgbd_calibration()
        self.chest=mujoco.Renderer(model,480,640);self.wrist=mujoco.Renderer(model,240,320)
        self.handeye={s:load_handeye(latest_handeye_report(s),s) for s in SIDES}
        self.last=None;self.attempts=[]
        self.retain_frames=False;self.frames={}
        self.known_cube_tracking=False;self.measurements=[];self.partial_measurements=[]

    def observe(self,data,clock):
        measured=[];self.attempts=[];self.partial_measurements=[]
        if self.retain_frames:self.frames={}
        for side in self.camera_sides:
            try:
                if side is None:
                    camera='chest_d435i_preview';renderer=self.chest
                    transform=self.chest_transform;matrix=self.chest_matrix;correction=self.calibration['chest']
                else:
                    camera=f'v22_{side}_d405_rgb';renderer=self.wrist
                    handeye,matrix=self.handeye[side]
                    transform=body_transform(data,self.specs[side].body_id)@handeye
                    correction=self.calibration['wrist'][side]
                rgb,depth=render_rgb_depth(renderer,data,camera)
                if self.retain_frames:self.frames[camera]=dict(rgb=rgb.copy(),depth=depth.copy(),time_s=float(clock))
                observation=estimate_cube_center(rgb,depth,np.eye(4),transform,matrix,CUBE_HALF)
                partial_center=apply_workspace_rgbd_calibration(observation.center_world,correction)
                self.partial_measurements.append(dict(center=partial_center.copy(),yaw=float(observation.yaw_world_rad),
                    source=camera,time_s=float(clock),mask_pixels=int(observation.mask_pixels)))
                observed_center=observation.center_world
                if self.known_cube_tracking and observation.top_center_world is not None:
                    # A complete square top localises the cube centre without
                    # the visibility-dependent bias of its side-face extrema.
                    sizes=np.asarray(observation.top_size_xy)
                    if np.max(np.abs(sizes-2*CUBE_HALF))>.010:
                        raise RuntimeError('cube top incomplete; do not interpret occlusion as target motion')
                    observed_center=observation.top_center_world
                center=apply_workspace_rgbd_calibration(observed_center,correction)
                # All observations originate in camera pixels; no object state
                # is used to decide which estimate agrees with the others.
                measured.append(dict(center=np.asarray(center).copy(),yaw=float(observation.yaw_world_rad),
                                     source=camera,mask_pixels=int(observation.mask_pixels),time_s=float(clock)))
                self.attempts.append(dict(camera=camera,detected=True))
            except RuntimeError as error:
                self.attempts.append(dict(camera=camera,detected=False,reason=str(error)))
        self.measurements=measured
        if measured:
            # Chest is calibrated for this whole workspace; a wrist refines it
            # only when both views agree to 15 mm. Never average contradictions.
            selected=measured[0]
            compatible=[m for m in measured[1:] if np.linalg.norm(m['center']-selected['center'])<.015]
            if compatible:
                selected=dict(compatible[0]);selected['source']='chest/wrist_agreement' if measured[0]['source'].startswith('chest') else selected['source']
            self.last=selected
        return self.last

    def close(self): self.chest.close();self.wrist.close()


class TaskGeometry(InterarmGeometry):
    def __init__(self,model,specs,grasp_side=None):
        super().__init__(model,specs)
        self.allowed=set()
        if grasp_side is not None:
            object_geom=int(model.geom('droid_object_geom').id)
            self.allowed={tuple(sorted((object_geom,int(model.geom(f'openarm_{grasp_side}_{f}_finger_collision').id)))) for f in ('left','right')}

    def distance(self,q):
        self.data.qpos[self.qpos]=q
        mujoco.mj_kinematics(self.model,self.data);mujoco.mj_collision(self.model,self.data)
        minimum=.04
        for contact in self.data.contact:
            a,b=int(contact.geom1),int(contact.geom2)
            if a not in self.sides and b not in self.sides: continue
            if tuple(sorted((a,b))) in self.allowed: continue
            if self.sides.get(a)==self.sides.get(b) and 'finger' in self.names[a] and 'finger' in self.names[b]: continue
            minimum=min(minimum,float(contact.dist))
        return minimum


class VisionTaskPlanner(RealtimeBimanualPlanner):
    def __init__(self,model,data,specs,grasp_side=None):
        super().__init__(model,data,specs)
        self.geometry=TaskGeometry(model,specs,grasp_side)
        self.worker_geometry=TaskGeometry(model,specs,grasp_side);self.rrt=TimedRRT(self.worker_geometry)
        self.object_address=int(model.joint('droid_object_freejoint').qposadr[0])
        self.robot_addresses=np.array([int(model.jnt_qposadr[j]) for j in range(model.njnt)
            if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or '').startswith('openarm_')])
        self.visual=None

    def query_state(self,data):
        if self.visual is None: raise ValueError('B task collision checking requires camera localization')
        qpos=self.model.qpos0.copy()
        qpos[self.robot_addresses]=data.qpos[self.robot_addresses]
        center=self.visual['center'];yaw=self.visual['yaw'];a=self.object_address
        qpos[a:a+3]=center;qpos[a+3:a+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        return SimpleNamespace(qpos=qpos)

    def coordinate(self,previous,goals,data,steps,fingers=None,enabled=True):
        if self.visual is None:
            self.reset();self.status.update(mode='vision_missing',clearance_m=float('nan'),planning_ms=0.)
            return {s:previous[s].copy() for s in SIDES}
        return super().coordinate(previous,goals,self.query_state(data),steps,fingers,enabled)


class BSceneFusionController(BimanualController):
    def __init__(self,model,data,wrist_mode='full'):
        super().__init__(model,data,wrist_mode=wrist_mode)
        self.coordinator.close();self.coordinator=VisionTaskPlanner(model,data,self.specs)
        self.grasp_side=None

    def set_visual(self,visual): self.coordinator.visual=visual

    def allow_grasp(self,side):
        if side==self.grasp_side: return
        visual=self.coordinator.visual;self.coordinator.close()
        self.coordinator=VisionTaskPlanner(self.model,self.data,self.specs,side)
        self.coordinator.visual=visual;self.grasp_side=side


class NearObjectAssist:
    """Visual precision is gated by human close intent AND robot proximity.

    Free gestures continue to drive the full arm. A grasp locks only its
    selected side, while the other arm remains under human control.
    """
    def __init__(self,control):
        self.control=control;self.side=None;self.stage='human_follow'
        self.last_closed={s:False for s in SIDES};self.plan=None;self.started=0.
        self.human_anchor=None;self.target_anchor=None;self.error=None
        self.starts=0;self.holds=0

    def update(self,obs,visual,closure,clock):
        from handeye_grasp_red_cube import finger_joint_indices,solve_gripper_contact_proxy_q
        from demo_grasp_viewer import PAD_CONTACT_PROXY_LOCAL
        from visual_grasp_planner import plan_face_grasp
        from demonstration_transfer import visual_planning_state
        c=self.control;model=c.model;data=c.data
        closed={s:bool(closure[s]>=.65) for s in SIDES}
        rising={s:closed[s] and not self.last_closed[s] for s in SIDES}
        self.last_closed=closed
        if self.side is None and visual is not None:
            for s in SIDES:
                arm=obs['arms'][s]
                if not rising[s] or not arm['hand_valid'] or arm['pose'] is None: continue
                pad=c.pads[s];pad.kin.qpos[pad.spec.qpos]=data.qpos[pad.spec.qpos]
                pad.kin.qpos[pad.finger_indices[0]]=data.qpos[pad.finger_indices[0]]
                mujoco.mj_forward(model,pad.kin)
                if np.linalg.norm(pad.midpoint()-visual['center'])>.10: continue
                try:
                    fingers=finger_joint_indices(model,s)
                    geoms=tuple(int(model.geom(f'openarm_{s}_{f}_finger_collision').id) for f in ('right','left'))
                    state=visual_planning_state(model,data,visual['center'],visual['yaw'])
                    seed=data.qpos[c.specs[s].qpos].copy()
                    plan=plan_face_grasp(model,state,c.specs[s],geoms,fingers,PAD_CONTACT_PROXY_LOCAL,
                                         visual['center'],visual['yaw'],seed,seed,.044,.07,.16)
                    if plan['metrics']['midpoint_error_m']>.006 or plan['metrics']['max_new_penetration_mm']>.5:
                        raise RuntimeError('B visual grasp candidate rejected')
                    self.side=s;self.stage='visual_approach';self.plan=plan
                    self.started=clock;self.human_anchor=None;self.target_anchor=visual['center'].copy()
                    self.geoms=geoms;self.fingers=fingers;self.starts+=1;c.allow_grasp(s)
                except (RuntimeError,ValueError) as error: self.error=str(error)
                break
        if self.side is None: return {},{}
        s=self.side;arm=obs['arms'][s]
        if arm['hand_valid'] and closure[s]<=.25:
            self.side=None;self.stage='human_follow';self.plan=None;c.allow_grasp(None)
            return {},{}
        if visual is None or arm['pose'] is None or not arm['hand_valid']:
            return {s:c.targets[s].copy()},{s:0. if self.stage in ('visual_close','human_transport') else .044}
        q=self.plan['q_approach'] if self.stage=='visual_approach' else self.plan['q_grasp']
        error=float(np.max(np.abs(data.qpos[c.specs[s].qpos]-q)))
        if self.stage=='visual_approach' and error<.025:
            self.stage='visual_descend';self.started=clock;q=self.plan['q_grasp']
        elif self.stage=='visual_descend' and error<.025:
            self.stage='visual_close';self.started=clock
        elif self.stage=='visual_close':
            finger_ids=c.actuators.fingers[s]
            gripping=np.all(data.qpos[self.fingers[0]]>.008) and np.all(np.abs(data.actuator_force[finger_ids])>.8)
            if gripping and clock-self.started>=.25:
                self.stage='human_transport';self.human_anchor=arm['point'].copy();self.holds+=1
            elif clock-self.started>2.:
                self.error='closure_not_confirmed';return {s:q},{s:.044}
        if self.stage=='human_transport':
            delta=np.clip(.7*(arm['point']-self.human_anchor),-.12,.12)
            center=self.target_anchor+delta;center[2]=max(center[2],self.target_anchor[2])
            state=visual_planning_state(model,data,visual['center'],visual['yaw'])
            q,*_=solve_gripper_contact_proxy_q(
                model,state,c.specs[s],self.geoms,self.fingers,PAD_CONTACT_PROXY_LOCAL,
                center,1,1.,2*CUBE_HALF,0.,c.targets[s],reference_rotation=self.plan['rotation'],
                multi_start=False,closing_direction=self.plan['direction'])
        return {s:q},{s:0. if self.stage in ('visual_close','human_transport') else .044}
