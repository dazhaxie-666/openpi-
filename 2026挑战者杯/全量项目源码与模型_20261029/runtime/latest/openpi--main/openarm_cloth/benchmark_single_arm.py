"""Repeatable saved-image inference benchmark; no fresh human accuracy claim."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import importlib.util

import cv2
import numpy as np

from human_teleop_core import ROOT


def frames(dataset, limit):
    rows=list(csv.DictReader((dataset/'video_timestamps.csv').open(encoding='utf-8')))
    cap=None;chunk=None
    try:
        for row in rows[:limit]:
            current=int(row['video_chunk'])
            if chunk!=current:
                if cap is not None:cap.release()
                cap=cv2.VideoCapture(str(dataset/f'human_camera_{current:05d}.mp4'));chunk=current
            ok,bgr=cap.read()
            if not ok:raise RuntimeError('Cannot decode saved frame')
            yield int(row['capture_read_end_ns'])//1_000_000,bgr
    finally:
        if cap is not None:cap.release()


def run(args):
    if args.pipeline:return run_pipeline(args)
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    cv2.setNumThreads(1)
    source=list(frames(args.dataset,args.frames))
    reports={}
    for count in args.hands:
        options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(
            model_asset_path=str(ROOT.parent/'pose2arm-main/models/hand_landmarker.task')),
            running_mode=vision.RunningMode.VIDEO,num_hands=count)
        times=[];found=[];landmarks=[]
        with vision.HandLandmarker.create_from_options(options) as detector:
            for stamp,bgr in source:
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                start=time.perf_counter()
                result=detector.detect_for_video(image,stamp)
                times.append((time.perf_counter()-start)*1000)
                found.append(len(result.hand_landmarks))
                landmarks.append([[p.x,p.y,p.z] for p in result.hand_landmarks[0]] if result.hand_landmarks else None)
        reports[str(count)]=dict(frames=len(times),inference_ms_p50=float(np.percentile(times[5:],50)),
            inference_ms_p95=float(np.percentile(times[5:],95)),candidate_frames=int(np.count_nonzero(found)),
            multiple_hand_frames=int(np.count_nonzero(np.array(found)>1)),
            first_hand_landmarks=landmarks)
    report=dict(scope='identical saved frames; isolated HandLandmarker inference; no robot/GUI',
        fresh_human_validation=False,accuracy_measured=False,dataset=str(args.dataset),results=reports,
        stream_source_sha256=hashlib.sha256((ROOT/'live_bimanual_stream.py').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:{n:v for n,v in r.items() if n!='first_hand_landmarks'} for k,r in reports.items()},indent=2))


def run_pipeline(args):
    """Consume every identical saved input through the real stream orchestration."""
    cv2.setNumThreads(1)
    source=list(frames(args.dataset,args.frames))
    source_hash=hashlib.sha256(args.pipeline.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('bench_stream',args.pipeline)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Camera:
        metadata=dict(source='unpaced saved frames',fresh_human_validation=False)
        def __init__(self,*args):self.index=0
        def read_new(self,after):
            if self.index>=len(source):raise EOFError()
            stamp,bgr=source[self.index];self.index+=1
            return self.index,stamp*1_000_000,stamp*1_000_000,bgr
        def close(self):pass
    start=time.perf_counter();results=[];seen=set()
    stream=module.BimanualHumanStream(camera_factory=Camera,expected_sides=('right',))
    try:
        deadline=start+120
        while time.perf_counter()<deadline:
            if stream.error:raise RuntimeError(stream.error)
            obs=stream.snapshot()
            if obs is not None and obs['seq'] not in seen:
                seen.add(obs['seq'])
                results.append(dict(seq=obs['seq'],work_ms=(obs['done']-obs['infer_start'])/1e6,
                    hand_valid=obs['arms']['right']['hand_valid'],position_valid=obs['arms']['right']['position_point'] is not None,
                    roi=obs['arms']['right']['roi_used'],full_scan=obs.get('hand_full_scan',True)))
            if stream.finished:break
            time.sleep(.0005)
        else:raise TimeoutError('stream benchmark did not finish')
    finally:stream.close()
    values=[r['work_ms'] for r in results[5:]]
    report=dict(scope='unpaced identical saved frames; real pose/hand stream, no robot or GUI; throughput is not live end-to-end latency',
        fresh_human_validation=False,accuracy_measured=False,frames=len(results),input_frames=len(source),
        processed_sources=int(stream.snapshot()['processed_sources']),
        inference_ms_p50=float(np.percentile(values,50)),inference_ms_p95=float(np.percentile(values,95)),
        hand_valid_frames=sum(r['hand_valid'] for r in results),position_valid_frames=sum(r['position_valid'] for r in results),
        roi_frames=sum(r['roi'] for r in results),full_scan_frames=sum(r['full_scan'] for r in results),
        stream_source_sha256=source_hash,source_changed_during_run=source_hash!=hashlib.sha256(args.pipeline.read_bytes()).hexdigest(),metadata=stream.metadata)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(report,rows=results),indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=180)
    parser.add_argument('--hands',type=int,nargs='+',default=[2,1])
    parser.add_argument('--pipeline',type=Path,help='Benchmark this stream source file instead of isolated hand counts')
    run(parser.parse_args())
