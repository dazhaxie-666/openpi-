"""Replay a saved A-live observation/reference trace through current B control.

This compares command smoothness on identical nominal A references, not human
tracking ground truth or a new live-camera acceptance run. Personal data stays
under the local outputs directory. Never changes the original recording.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from human_grasp_task import normalized_pinch
from live_motion_scene import build_motion_scene
from live_reference_conditioning import ContinuousPinch, JointReferenceTrajectory
from run_pose2arm_live import AReferenceMotionController


def run(dataset, output):
    dataset, output = Path(dataset), Path(output)
    files = sorted(dataset.rglob('*.npz'))
    chunks = [np.load(p, allow_pickle=False) for p in files]
    keys = chunks[0].files
    trace = {k:np.concatenate([c[k] for c in chunks]) for k in keys}
    for chunk in chunks: chunk.close()
    model, data = build_motion_scene()
    control = AReferenceMotionController(model, data)
    pinches = {s:ContinuousPinch() for s in ('left','right')}
    source_rows = {}
    for i, seq in enumerate(trace['source_seq']):source_rows.setdefault(int(seq), i)
    applied, actual, fingers, durations = [], [], [], []
    for i in range(len(trace['action'])):
        seq = int(trace['a_source_seq'][i])
        source = source_rows.get(seq, i)
        stamp = float(trace['a_capture_s'][i])
        apertures = {}
        for j,s in enumerate(('left','right')):
            hand = trace[s+'_hand_raw'][source]
            ratio = normalized_pinch(hand) if trace['side_hand_valid'][source,j] else None
            apertures[s] = pinches[s].update(ratio,stamp)
        q = trace['a_reference_joint'][i].reshape(2,7)
        result = dict(seq=seq, capture_ns=int(stamp*1e9),
                      joints={s:q[j] for j,s in enumerate(('left','right')) if np.isfinite(q[j]).all()},
                      fingers=apertures) if np.isfinite(stamp) else None
        visible = dict(zip(('left','right'), trace['side_pose_valid'][i]))
        start = time.perf_counter()
        control.step(result, visible, bool(trace['paused'][i]))
        durations.append((time.perf_counter()-start)*1000)
        applied.append(np.concatenate([control.joints[s] for s in ('left','right')]))
        actual.append(np.concatenate([data.qpos[control.specs[s].qpos] for s in ('left','right')]))
        fingers.append([control.fingers[s] for s in ('left','right')])
    applied, actual, fingers = map(np.asarray,(applied,actual,fingers))
    old = trace['action'][:,[ *range(7), *range(9,16) ]]
    report = dict(input=str(dataset),frames=len(old),input_kind='recorded human A references',
                  fresh_human_validation=False,task_success='not assessed',
                  nominal_a_reference_unchanged=True,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  control_work_ms_p95=float(np.percentile(durations,95)),sides={})
    for j,s in enumerate(('left','right')):
        valid = trace['side_pose_valid'][40:,j]
        metrics = {}
        for label,commands in (('before',old),('after',applied)):
            q = commands[40:,j*7:(j+1)*7]
            velocity = np.diff(q,axis=0)/.05
            acceleration = np.diff(velocity,axis=0)/.05
            selected = valid[2:] & valid[1:-1] & valid[:-2]
            metrics[label] = dict(
                command_acceleration_p95_rad_s2=float(np.percentile(np.abs(acceleration[selected]),95)),
                command_step_p95_deg=float(np.rad2deg(np.percentile(np.max(np.abs(np.diff(q,axis=0)),axis=1)[valid[1:] & valid[:-1]],95))),
                command_path_length_rad=float(np.sum(np.abs(np.diff(q,axis=0)))))
        report['sides'][s] = metrics
    # An intentional 20-degree step must still respond within a useful time.
    trajectory = JointReferenceTrajectory(np.zeros(7))
    response = []
    for i in range(1600):
        if i%100==0:trajectory.observe(np.full(7,np.deg2rad(20)),i*.0005)
        response.append(trajectory.advance(.0005)[0][0])
    report['synthetic_20deg_step_t90_s'] = float(np.flatnonzero(np.array(response)>=.9*np.deg2rad(20))[0]*.0005)
    report['synthetic_20deg_step_overshoot_deg'] = float(np.rad2deg(max(response)-np.deg2rad(20)))
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'comparison.npz',before=old,after=applied,after_actual=actual,
                        before_fingers=trace['action'][:,[7,16]],after_fingers=fingers,
                        nominal_a=trace['a_reference_joint'])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    run(args.dataset,args.output)
