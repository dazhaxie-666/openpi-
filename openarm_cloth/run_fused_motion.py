"""Continuous A arms/wrists/grippers + B vision, task scene and actuators."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import cv2
import mujoco
import numpy as np

from member_b_scene import ROOT,SCENE,build_scene
from full_video_source import FullVideoSource,load_full_landmarks,SIDES
from b_scene_fusion import BSceneFusionController,BVisualObserver,NearObjectAssist
from demo_grasp_viewer import place_cube_for_blind_test
from task_workspace import CONFIG
from live_motion_scene import initialize_motion_posture


def run(args):
    out=args.outdir.resolve();out.mkdir(parents=True,exist_ok=True)
    cache_path=ROOT/'outputs/ab_motion/cache'/f'{args.video.stem}_full.npz'
    blob=load_full_landmarks(args.video,cache_path,args.refresh)
    source=FullVideoSource(blob)
    model,data=build_scene();place_cube_for_blind_test(model,data,args.cube_test_pose)
    # Grasp standby has only 7.16 mm between left link3/link5; use B's
    # established follow posture so the 8 mm safety margin starts feasible.
    initialize_motion_posture(model,data)
    control=BSceneFusionController(model,data,args.wrist_mode)
    observer=None;capture=None;rows=[];vision_log=[];failure=None
    flags=dict(paused=False,quit=False)
    completed=False
    try:
        for _ in range(round(.6/model.opt.timestep)): control.actuators.apply();mujoco.mj_step(model,data)
        observer=BVisualObserver(model,control.specs)
        assist=NearObjectAssist(control)
        steps=round(.05/model.opt.timestep)
        if abs(steps*model.opt.timestep-.05)>1e-10: raise ValueError('B physical interval must equal 50 ms')
        duration=min(source.duration,args.seconds) if args.seconds>0 else source.duration
        count=int(np.ceil(duration/.05-1e-9));next_vision=0.;last_fingers=dict(control.fingers)
        with ExitStack() as stack:
            viewer=None
            if not args.check:
                import mujoco.viewer as mjviewer
                def key(code):
                    if code==32: flags['paused']=not flags['paused']
                    elif code in (81,113): flags['quit']=True
                viewer=stack.enter_context(mjviewer.launch_passive(model,data,key_callback=key))
                viewer.cam.lookat[:]=[.36,0,.48];viewer.cam.distance=1.3;viewer.cam.azimuth=130;viewer.cam.elevation=-25
                from source_video_reader import SourceVideoReader
                capture=SourceVideoReader(args.video)
            for index in range(count):
                if flags['quit'] or (viewer is not None and not viewer.is_running()): break
                while flags['paused'] and viewer is not None and viewer.is_running() and not flags['quit']:
                    viewer.sync();time.sleep(.02)
                if flags['quit']: break
                begin=time.perf_counter();clock=index*.05;obs=source.at(clock)
                if obs is None: break
                if clock+1e-9>=next_vision:
                    observer.observe(data,clock);next_vision=clock+1/args.vision_hz
                    vision_log.append(dict(time_s=clock,attempts=observer.attempts))
                visual=observer.last if observer.last is not None and clock-observer.last['time_s']<=2. else None
                control.set_visual(visual)
                closure={s:float(obs['arms'][s]['closure']) for s in SIDES}
                joints={};fingers={s:.044*(1-closure[s]) for s in SIDES} if args.gripper_mode=='pinch' else {}
                if args.assist_grasp:
                    if args.gripper_mode=='fist':
                        closure={s:float(control.closed[s]) for s in SIDES}
                    joints,assisted_fingers=assist.update(obs,visual,closure,clock);fingers.update(assisted_fingers)
                if visual is None: fingers=dict(last_fingers)
                control.update(obs,source.now_ns(clock),joint_goals=joints,finger_goals=fingers)
                # Smooth the physical slide command; the coordinator checked
                # the larger open-finger envelope before this closing slew.
                for s in SIDES:
                    control.fingers[s]=float(last_fingers[s]+np.clip(control.fingers[s]-last_fingers[s],-.0066,.0066))
                last_fingers=dict(control.fingers);control.actuators.set_targets(control.targets,control.fingers)
                contact_count=0;min_distance=0.
                for _ in range(steps):
                    control.actuators.apply();mujoco.mj_step(model,data)
                    for contact in data.contact:
                        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
                               for g in (int(contact.geom1),int(contact.geom2))]
                        if any('openarm_left_' in n for n in names) and any('openarm_right_' in n for n in names):
                            contact_count+=1;min_distance=min(min_distance,float(contact.dist))
                errors=control.errors()
                rows.append(dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy(),
                                 target=np.concatenate([control.targets[s] for s in SIDES]),
                                 finger_target=np.array([control.fingers[s] for s in SIDES]),
                                 source_frame=obs['seq'],time=clock,sim_time=float(data.time),
                                 phase=assist.stage if args.assist_grasp else 'human_follow',
                                 avoidance=control.coordinator.status['mode'],errors=errors,
                                 pose_valid=np.array([control.valid[s] for s in SIDES]),
                                 hand_valid=np.array([control.hand_valid[s] for s in SIDES]),
                                 vision=np.full(3,np.nan) if visual is None else visual['center'].copy(),
                                 vision_source='missing' if visual is None else visual['source'],
                                 vision_time=np.nan if visual is None else visual['time_s'],
                                 interarm_contacts=contact_count,contact_distance=min_distance,
                                 compute_ms=(time.perf_counter()-begin)*1000))
                if index%20==0:
                    print(f'A+B full video {index+1}/{count} source={obs["seq"]}/{source.count} '
                          f'vision={rows[-1]["vision_source"]} mode={rows[-1]["avoidance"]}',flush=True)
                if viewer is not None:
                    viewer.sync();frame=capture.frame(obs['seq']).copy()
                    cv2.putText(frame,f'FULL A motion | source {obs["seq"]} | B vision {rows[-1]["vision_source"]}',
                                (10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                    cv2.imshow('A full video input (Q exits)',frame)
                    if cv2.waitKey(1)&255==ord('q'): flags['quit']=True
                    time.sleep(max(0.,.05-(time.perf_counter()-begin)))
            completed=len(rows)==count
        if not rows: raise RuntimeError('No full-motion control frames recorded')
        def arrays(key): return np.asarray([row[key] for row in rows])
        np.savez_compressed(out/'episode.npz',qpos=arrays('qpos'),qvel=arrays('qvel'),ctrl=arrays('ctrl'),
                            joint_target=arrays('target'),finger_target=arrays('finger_target'),
                            source_frame=arrays('source_frame'),source_time_s=arrays('time'),sim_time_s=arrays('sim_time'),
                            phase=arrays('phase'),avoidance_mode=arrays('avoidance'),errors_deg=arrays('errors'),
                            pose_valid=arrays('pose_valid'),hand_valid=arrays('hand_valid'),
                            visual_center_m=arrays('vision'),visual_source=arrays('vision_source'),
                            visual_time_s=arrays('vision_time'),fps=np.array(20.))
        summary=dict(scope='Full video human arms/wrists/grippers in B task scene with B visual perception',
                     verdict='COMPLETED' if completed else 'STOPPED',not_a_grasp_success_claim=True,
                     frames=len(rows),source_frames=source.count,source_duration_s=source.duration,
                     consumed_source_frame_range=[int(rows[0]['source_frame']),int(rows[-1]['source_frame'])],
                     wrist_mode=args.wrist_mode,gripper_mode=args.gripper_mode,
                     side_order=list(SIDES),joint_order='left7,right7',
                     pose_valid_frames=np.sum(arrays('pose_valid'),axis=0).tolist(),
                     hand_valid_frames=np.sum(arrays('hand_valid'),axis=0).tolist(),
                     visual_available_frames=int(np.isfinite(arrays('vision')[:,0]).sum()),
                     mean_pose_errors_deg=np.nanmean(arrays('errors'),axis=0).tolist(),
                     avoidance_modes=dict(Counter(arrays('avoidance').tolist())),
                     interarm_contact_count=int(arrays('interarm_contacts').sum()),
                     deepest_interarm_distance_m=float(arrays('contact_distance').min()),
                     actual_joint_range_rad=np.ptp(arrays('qpos')[:,control.coordinator.qpos],axis=0).tolist(),
                     assist_starts=assist.starts,assist_holds=assist.holds,assist_error=assist.error,
                     compute_p95_ms=float(np.percentile(arrays('compute_ms'),95)),
                     timing='Offline inference; each 20 Hz tick advances all 100 physical substeps',
                     vision_log=vision_log)
        (out/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        manifest=dict(schema_version=1,source_video=str(args.video.resolve()),
                      source_sha256=str(blob['source_sha256'].item()),workspace=CONFIG,
                      scene_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                          'simulation/v1/droid_cube_scene_perception.xml','simulation/v1/droid_cube_scene.xml')},
                      recording='episode.npz',report='result.json',frames=len(rows),
                      human_landmark_cache=str(cache_path.resolve()),
                      verdict=summary['verdict'],scope='Full video, original source clock; no selected-fragment replay')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in summary.items() if k!='vision_log'},indent=2))
        (out/'failure.json').unlink(missing_ok=True)
        return 0
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)),indent=2),encoding='utf-8')
        raise
    finally:
        control.close()
        if observer is not None: observer.close()
        if capture is not None: capture.close();cv2.destroyAllWindows()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,default=ROOT.parent/'pose2arm-main/videos/p2a_test1.mp4')
    parser.add_argument('--outdir',type=Path,default=ROOT/'outputs/ab_motion/p2a_test1')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--seconds',type=float,default=0.)
    parser.add_argument('--wrist-mode',choices=('full','twist'),default='full')
    parser.add_argument('--gripper-mode',choices=('pinch','fist'),default='pinch')
    parser.add_argument('--assist-grasp',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--vision-hz',type=float,default=2.)
    parser.add_argument('--cube-test-pose',nargs=3,type=float)
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<0 or not np.isfinite(args.vision_hz) or args.vision_hz<=0:
        parser.error('Invalid duration or vision frequency')
    return run(args)


if __name__=='__main__': raise SystemExit(main())
