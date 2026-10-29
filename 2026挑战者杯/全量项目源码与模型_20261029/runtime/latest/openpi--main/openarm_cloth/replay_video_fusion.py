"""Replay recorded B physics beside the phase-aligned original human video."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

import cv2
import mujoco
import numpy as np

from task_workspace import load_task_model,CONFIG
from member_b_scene import ROOT,SCENE
from source_video_reader import SourceVideoReader

sys.path.insert(0,str(ROOT.parent/'pose2arm-main'))
from replay_controls import ReplayControls,configure_replay_camera


def add_reference_skeletons(scene,model,state,recording,index):
    """Display-only A reference (magenta) and B adapted target (cyan)."""
    for field,color in (('a_reference_joint',[1.,.15,.8,.65]),('joint_target',[.1,.9,1.,.55])):
        if field not in recording: continue
        ghost=mujoco.MjData(model);ghost.qpos[:]=state
        for j,side in enumerate(('left','right')):
            ids=[int(model.joint(f'openarm_{side}_joint{k}').qposadr[0]) for k in range(1,8)]
            ghost.qpos[ids]=recording[field][index,7*j:7*(j+1)]
        mujoco.mj_forward(model,ghost)
        for side in ('left','right'):
            points=[ghost.xanchor[model.joint(f'openarm_{side}_joint{k}').id].copy() for k in (1,4,7)]
            for start,stop in zip(points[:-1],points[1:]):
                if scene.ngeom>=scene.maxgeom: return
                geom=scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.005,start,stop)
                scene.ngeom+=1


def replay_native(model,recording,cap,landmarks,speed=1.,smoke_seconds=0.):
    # Same native passive viewer/camera controls as A's scripts/replay_mujoco.
    import mujoco.viewer as mjviewer
    controls=ReplayControls(len(recording['qpos']),speed=speed)
    fps=float(recording.get('fps',20.))
    data=mujoco.MjData(model)
    title='A source | SPACE pause A/D step +/- speed R restart G ghosts Q quit'
    cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,480,750)
    started=time.perf_counter()
    print('Native 3D replay: mouse rotate/pan, wheel zoom. SPACE pause, A/D or arrows step, +/- speed, R restart, G ghosts, Q quit.',flush=True)
    print('Magenta skeleton = existing A reference; cyan = B target; solid robot = recorded B model state. See manifest execution_mode for kinematic vs dynamic.',flush=True)
    try:
        with mjviewer.launch_passive(model,data,key_callback=controls.key) as view:
            configure_replay_camera(view,lookat=(.30,0,.45))
            view.cam.distance=1.4
            while view.is_running() and not controls.quit:
                tick=time.perf_counter();index=controls.index
                data.qpos[:]=recording['qpos'][index]
                mujoco.mj_forward(model,data)
                # Never reset view.cam in the frame loop: mouse navigation persists.
                with view.lock():
                    view.user_scn.ngeom=0
                    if controls.ghosts:
                        add_reference_skeletons(view.user_scn,model,data.qpos,recording,index)
                view.sync()
                source_frame=int(recording['source_frame'][index])
                arm_modes=recording['source_arm_input_mode'][index] if 'source_arm_input_mode' in recording else None
                frame=draw_landmarks(cap.frame(source_frame),source_frame,landmarks,arm_modes).copy()
                height,width=frame.shape[:2]
                frame=cv2.resize(frame,(max(1,int(width*min(1.,720/height))),min(height,720)))
                source_time=float(recording['source_time_s'][index])
                execution_time=float(recording.get('execution_time_s',recording['source_time_s'])[index])
                lines=[f'source {source_time:.2f}s frame {source_frame} | execution {execution_time:.2f}s',
                       f'{controls.speed:g}x | {recording["phase"][index]}',
                       'magenta: A reference | cyan: B target',
                       str(recording.get('avoidance_mode',recording['phase'])[index])]
                if 'source_arm_input_mode' in recording:
                    labels={'observed_3d':'3D','rtmw_2d_depth_held':'2D(depth held)','held':'HOLD'}
                    lines.append('arm input L/R: '+' / '.join(labels.get(str(v),str(v)) for v in recording['source_arm_input_mode'][index]))
                for row,line in enumerate(lines):
                    cv2.putText(frame,line,(8,22+22*row),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,240,255),1)
                cv2.imshow(title,frame)
                key=cv2.waitKey(1)&255
                if key!=255: controls.key(key)
                if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1: break
                # A user step/restart takes priority over automatic advancement.
                if controls.index==index: controls.advance()
                if smoke_seconds>0 and time.perf_counter()-started>=smoke_seconds: break
                time.sleep(max(0.,1/fps/controls.speed-(time.perf_counter()-tick)))
    finally:
        cv2.destroyAllWindows()


def draw_landmarks(frame,index,landmarks,arm_modes=None):
    if landmarks is None: return frame
    frame=frame.copy();height,width=frame.shape[:2]
    pose=landmarks['pose_norm'][index]
    for side,color in ((0,(255,180,0)),(1,(0,240,255))):
        use_image_plane=(arm_modes is not None and arm_modes[side]=='rtmw_2d_depth_held' and 'rtmw_xy' in landmarks)
        if use_image_plane:
            points=landmarks['rtmw_xy'][index,[5+side,7+side,9+side]]
            for a,b in ((0,1),(1,2)):
                cv2.line(frame,tuple(np.round(points[a]).astype(int)),tuple(np.round(points[b]).astype(int)),color,3)
        else:
            for a,b in ((11+side,13+side),(13+side,15+side)):
                if np.isfinite(pose[[a,b]]).all() and min(pose[a,3],pose[b,3])>=.45:
                    points=[tuple(np.round(pose[p,:2]*[width,height]).astype(int)) for p in (a,b)]
                    cv2.line(frame,*points,color,3)
        if not landmarks['hand_valid'][index,side]: continue
        hand=landmarks['hand_norm'][index,side]
        if not np.isfinite(hand).all(): continue
        pixels=np.round(hand[:,:2]*[width,height]).astype(int)
        for start in (1,5,9,13,17):
            for a,b in ((0,start),(start,start+1),(start+1,start+2),(start+2,start+3)):
                cv2.line(frame,tuple(pixels[a]),tuple(pixels[b]),color,2)
        for point in pixels: cv2.circle(frame,tuple(point),3,color,-1)
    return frame


def load_recording(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version')!=1: raise ValueError('Unsupported fusion replay schema')
    if manifest['workspace']!=CONFIG: raise ValueError('B workspace changed since recording; restore recorded configuration')
    for name,expected in manifest['scene_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'B scene changed since recording: {name}')
    with np.load(directory/'episode.npz',allow_pickle=False) as z:
        recording={k:z[k] for k in z.files}
    if manifest.get('scene_kind')=='motion_only':
        from live_motion_scene import build_motion_scene
        model,_=build_motion_scene()
    else:
        model=load_task_model(SCENE)
    qpos=recording['qpos'];times=recording['sim_time_s']
    fps=float(recording.get('fps',20.))
    if (qpos.ndim!=2 or not len(qpos) or qpos.shape[1]!=model.nq
            or not np.isfinite(qpos).all() or times.shape!=(len(qpos),)
            or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or recording['source_frame'].shape!=(len(qpos),)
            or recording['phase'].shape!=(len(qpos),) or not np.isfinite(fps) or fps<=0):
        raise ValueError('Invalid physics replay arrays')
    return directory,manifest,recording,model


def default_recording():
    pointer=ROOT/'outputs/a_motion/latest.json'
    if pointer.is_file():
        return Path(json.loads(pointer.read_text(encoding='utf-8'))['recording'])
    return ROOT/'outputs/a_motion/p2a_test1_kinematic'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path)
    parser.add_argument('--video',type=Path,help='Relocated original human video, if its path changed')
    parser.add_argument('--render',type=Path,help='Save a combined MP4 without interactive windows')
    parser.add_argument('--flat',action='store_true',help='Legacy fixed-camera OpenCV preview instead of native 3D')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--smoke-seconds',type=float,default=0.,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if not np.isfinite(args.speed) or not .125<=args.speed<=4 or not np.isfinite(args.smoke_seconds) or args.smoke_seconds<0:
        parser.error('Invalid playback speed or smoke-test duration')
    directory,manifest,recording,model=load_recording(args.recording or default_recording())
    print(f'Recording: {directory} | reference={manifest.get("reference_mode","legacy")} | execution={manifest.get("execution_mode","physics")}',flush=True)
    source=args.video or Path(manifest['source_video'])
    with source.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['source_sha256']:
            raise ValueError('Video hash differs from the measured demonstration')
    cap=SourceVideoReader(source)
    # Replay-only lighting; never touches physics, materials or the saved qpos.
    model.vis.headlight.active=1
    model.vis.headlight.ambient[:]=[.65,.65,.65]
    model.vis.headlight.diffuse[:]=[.9,.9,.9]
    landmark_path=Path(manifest.get('human_landmark_cache',ROOT/'outputs/ab_motion/cache'/f'{source.stem}_full.npz'))
    landmarks=None
    if landmark_path.is_file():
        with np.load(landmark_path,allow_pickle=False) as z:
            if str(z['source_sha256'].item())==manifest['source_sha256']:
                landmarks={k:z[k] for k in ('pose_norm','hand_norm','hand_valid')}
    # Show the actual image-plane fallback, not a failed MediaPipe wrist.
    if landmarks is not None and 'source_arm_input_mode' in recording:
        rtmw_path=ROOT.parent/'pose2arm-main/out/pose_ab/rtmw3d.npz'
        bindings=json.loads((ROOT/'a_reference_bindings.json').read_text(encoding='utf-8'))
        if rtmw_path.is_file():
            with rtmw_path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if any(entry.get('rtmw_sha256')==digest and entry['source_sha256']==manifest['source_sha256'] for entry in bindings['references']):
                with np.load(rtmw_path,allow_pickle=False) as z:landmarks['rtmw_xy']=z['image_xy']
    if not args.render and not args.flat:
        try: replay_native(model,recording,cap,landmarks,args.speed,args.smoke_seconds)
        finally: cap.close()
        return
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,480,640)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
    camera.lookat[:]=[.36,0,.48];camera.distance=1.25;camera.azimuth=130;camera.elevation=-25
    writer=None;paused=False;index=0;last_source=-1;human=None
    fps=float(recording.get('fps',20.))
    if args.render:
        args.render.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(args.render),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,480))
        if not writer.isOpened(): raise OSError(f'Cannot write replay: {args.render}')
    else:
        cv2.namedWindow('A video + B recorded physics | SPACE pause R restart Q quit',cv2.WINDOW_NORMAL)
    try:
        while index<len(recording['qpos']):
            tick=time.perf_counter()
            frame_id=int(recording['source_frame'][index])
            if frame_id!=last_source or human is None:
                frame=cap.frame(frame_id)
                arm_modes=recording['source_arm_input_mode'][index] if 'source_arm_input_mode' in recording else None
                frame=draw_landmarks(frame,frame_id,landmarks,arm_modes)
                h,w=frame.shape[:2];scale=min(640/w,420/h)
                thumb=cv2.resize(frame,(int(w*scale),int(h*scale)))
                human=np.zeros((480,640,3),np.uint8)
                y=45+(420-thumb.shape[0])//2;x=(640-thumb.shape[1])//2
                human[y:y+thumb.shape[0],x:x+thumb.shape[1]]=thumb
                last_source=frame_id
            data.qpos[:]=recording['qpos'][index];mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera)
            if 'a_reference_joint' in recording:
                add_reference_skeletons(renderer.scene,model,data.qpos,recording,index)
            if 'visual_center_m' in recording and np.isfinite(recording['visual_center_m'][index]).all():
                scene=renderer.scene
                if scene.ngeom<len(scene.geoms):
                    geom=scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.009,.009,.009]),
                                       recording['visual_center_m'][index]+[0,0,.04],np.eye(3).ravel(),np.array([0.,1.,1.,.8]))
                    scene.ngeom+=1
            robot=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
            human_view=human.copy()
            cv2.putText(human_view,f'A measured video | source frame {frame_id}',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,240,255),1)
            mode=manifest.get('execution_mode','physics')
            cv2.putText(robot,f'B {mode} | {recording["phase"][index]}',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,230,0),1)
            if 'visual_source' in recording:
                cv2.putText(robot,f'B vision: {recording["visual_source"][index]}',(12,46),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,240,240),1)
            combined=np.concatenate([human_view,robot],axis=1)
            if writer:
                writer.write(combined);index+=1
            else:
                cv2.imshow('A video + B recorded physics | SPACE pause R restart Q quit',combined)
                key=cv2.waitKey(max(1,int(1000*max(0.,1/fps-(time.perf_counter()-tick)))))&255
                if key in (ord('q'),27): break
                if key==32: paused=not paused
                if key==ord('r'): index=0;last_source=-1;continue
                if not paused: index=(index+1)%len(recording['qpos'])
                if cv2.getWindowProperty('A video + B recorded physics | SPACE pause R restart Q quit',cv2.WND_PROP_VISIBLE)<1: break
    finally:
        cap.close();renderer.close()
        if writer: writer.release()
        else: cv2.destroyAllWindows()
    print(f'Replay rendered: {args.render}' if args.render else 'Replay closed')


if __name__=='__main__': main()
