"""Timed saved-camera -> real MediaPipe/A/B stress test, with optional renderer.

Loops are development load only, never independent trials or fresh human data.
"""
import argparse
import hashlib
from datetime import datetime
import json
from pathlib import Path
import time
import cv2,mujoco,numpy as np
from recorded_motion_camera import RecordedMotionCamera
from live_bimanual_stream import BimanualHumanStream
from live_motion_scene import build_motion_scene
from pose2arm_task_mapping import LatestPose2ArmWorker,observed_sides
from run_pose2arm_live import AReferenceMotionController
from live_motion_runtime import LivePhysicsWorker


def run(args):
    cv2.setNumThreads(1)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    def source_hashes():
        root=Path(__file__).resolve().parent
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/name for name in (
            'benchmark_motion_runtime.py','pose2arm_task_mapping.py','human_pinch_control.py','human_wrist_control.py',
            'motion_observation.py','live_motion_runtime.py','run_pose2arm_live.py','live_bimanual_stream.py',
            'live_observation_quality.py','live_reference_conditioning.py','recorded_motion_camera.py'))}
    initial_hashes=source_hashes()
    model,data=build_motion_scene();control=AReferenceMotionController(model,data,responsive=True)
    stream=BimanualHumanStream(camera_factory=lambda *a:RecordedMotionCamera(args.dataset,loop=True))
    reference=LatestPose2ArmWorker(model=model,torso_mode=args.torso_mode,wrist_mode='full',source_dt=True)
    def provider():
        obs=stream.snapshot()
        if stream.error:raise RuntimeError(stream.error)
        if reference.error:raise RuntimeError(reference.error)
        reference.submit(obs)
        return reference.snapshot(),observed_sides(obs),False,{s:bool(obs and obs['arms'][s]['hand_valid']) for s in ('left','right')}
    physics=LivePhysicsWorker(control,reference_provider=provider)
    renderer=None;view=mujoco.MjData(model);frames=0;failure=None;start=time.perf_counter();last_log=start
    try:
        if args.render:renderer=mujoco.Renderer(model,height=360,width=640)
        while time.perf_counter()-start<args.seconds:
            if physics.error:raise RuntimeError(physics.error)
            if renderer:
                state=physics.snapshot();view.qpos[:]=state.qpos;view.qvel[:]=state.qvel;mujoco.mj_forward(model,view)
                renderer.update_scene(view);renderer.render();frames+=1
            time.sleep(.1)  # deliberate slow UI; it must not own physics time
            if time.perf_counter()-last_log>=30:
                print('STRESS',json.dumps(dict(elapsed_s=round(time.perf_counter()-start,1),**physics.report())),flush=True);last_log=time.perf_counter()
    except Exception as error:failure=f'{type(error).__name__}: {error}';raise
    finally:
        physics.close();reference.close();stream.close()
        if renderer:renderer.close()
        physics.save_trace(out/'physics_trace.npz')
        trace=physics.trace;coverage={}
        if trace:
            dt=np.array([r['wall_dt_s'] for r in trace]);weights=dt/max(dt.sum(),1e-9)
            modules=np.stack([r['module_state'] for r in trace])
            for j,side in enumerate(('left','right')):
                coverage[side]={name:float(np.sum(weights*(modules[:,j,k]=='TRACKING')))
                    for k,name in enumerate(('torso','arm','palm','pinch','association'))}
        report=dict(source='saved-camera real-time loop',fresh_human_validation=False,independent_trials=0,
                    release_verdict='NOT_EVALUABLE',renderer=bool(args.render),rendered_frames=frames,
                    runtime=physics.report(),error=failure,camera_metadata=stream.metadata,
                    source_hashes=initial_hashes,source_changed_during_run=initial_hashes!=source_hashes(),
                    torso_mode=args.torso_mode,wrist_mode='full',gripper_mode='continuous',
                    module_tracking_time_fraction=coverage,coverage_is_accuracy=False,
                    calibration={s:c.summary() for s,c in reference.mapping.pinch_calibrations.items()})
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=60.);parser.add_argument('--render',action='store_true')
    parser.add_argument('--torso-mode',choices=('auto','upper_body','full_body','original'),default='auto')
    run(parser.parse_args())
