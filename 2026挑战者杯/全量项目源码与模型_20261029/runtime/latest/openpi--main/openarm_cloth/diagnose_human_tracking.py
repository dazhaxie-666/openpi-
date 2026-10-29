"""Camera-only local diagnosis; no simulator, robot controller or calibration.

Record the production visual stream, then optionally compare VIDEO tracking
with independent IMAGE detection on the same saved images. Detection coverage
is not accuracy: both modes can confidently identify the wrong image region.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from live_bimanual_stream import BimanualHumanStream
from live_observation_quality import body_hand_consistency
from live_test_dataset import TestDatasetRecorder,array_or_nan
from member_b_scene import ROOT

SIDES=('left','right')


def observation_row(obs,index,posture):
    return dict(control_frame=np.asarray(index),source_seq=np.asarray(obs['seq']),
        capture_read_end_ns=np.asarray(obs['read_end'],dtype=np.int64),
        vision_generation=np.asarray(obs.get('vision_generation',0)),posture=np.asarray(posture),
        pose_norm=array_or_nan(obs.get('pose_norm'),(33,4)),
        pose_raw=array_or_nan(obs.get('pose_raw'),(33,4)),
        body_detected=np.asarray(obs.get('pose_norm') is not None),
        position_trusted=np.asarray([obs.get('position_trusted',{}).get(s,False) for s in SIDES]),
        position_reason=np.asarray([obs.get('position_quality',{}).get(s,'body missing') for s in SIDES]),
        arm_reason=np.asarray([obs.get('pose_quality',{}).get(s,'body missing') for s in SIDES]),
        inference_ms=np.asarray((obs['done']-obs['read_end'])/1e6),
        hand_valid=np.asarray([obs['arms'][s]['hand_valid'] for s in SIDES]))


def record_camera(args,stream_factory=BimanualHumanStream,synthetic_input=False):
    if not np.isfinite(args.seconds) or args.seconds<=0:raise ValueError('seconds must be positive')
    recorder=TestDatasetRecorder(args.output/'dataset',dict(
        source='explicit_software_fixture' if synthetic_input else 'live_usb',purpose='perception diagnosis only; no robot execution or ground truth',
        synthetic_input=bool(synthetic_input),human_validation_performed=False,pose_model=args.pose_model,
        side_array_order=list(SIDES),posture_labels='operator 1=sitting, 2=standing; unknown until labeled'),
        save_video=args.save_video)
    stream=None;error=None;last_seq=-1;index=0;posture='unknown';events=[];window_created=False
    window='CAMERA DIAG | 1 sitting | 2 standing | R restart vision | Q save/exit'
    try:
        stream=stream_factory(args.camera,args.pose_model,(1280,720))
        start=time.perf_counter()
        while time.perf_counter()-start<args.seconds:
            obs=stream.snapshot()
            if stream.error:raise RuntimeError(stream.error)
            if obs is not None and obs['seq']>last_seq:
                row=observation_row(obs,index,posture)
                recorder.submit(row,obs['raw_image'].copy() if args.save_video else None)
                index+=1;last_seq=obs['seq']
            if not args.no_preview:
                frame=cv2.resize(obs['image'],(960,540)) if obs is not None else np.full((540,960,3),40,np.uint8)
                frame=cv2.flip(frame,1)
                lines=['CAMERA ONLY - NO ROBOT | '+posture.upper(),
                    '1 sitting | 2 standing | R vision restart | Q save/exit']
                lines.extend(s.upper()+': '+(obs.get('position_quality',{}).get(s,'unknown') if obs else 'waiting') for s in SIDES)
                for i,line in enumerate(lines):
                    cv2.putText(frame,line,(12,28+30*i),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),3)
                    cv2.putText(frame,line,(12,28+30*i),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                cv2.imshow(window,frame);window_created=True;key=cv2.waitKey(10)&255
                if key in (81,113,27):break
                if key in (49,50,82,114):
                    if key in (49,50):posture='sitting' if key==49 else 'standing'
                    else:stream.request_reacquire()
                    events.append(dict(host_ns=time.perf_counter_ns(),last_source_seq=last_seq,
                        event=posture if key in (49,50) else 'vision_reacquire'))
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            else:time.sleep(.005)
            if stream.finished:break
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        if stream is not None:
            stream.close()
            error=error or stream.error
            thread=getattr(stream,'thread',None)
            if thread is not None and thread.is_alive():error=error or 'Vision thread did not stop'
        recorder.update_metadata(camera_metadata=stream.metadata if stream else {},events=events)
        recorder.close()
        if window_created:cv2.destroyWindow(window)
    report=dict(scope='camera-only recording; not standing accuracy validation',
        consumed_fresh_sources=index,saved_rows=recorder.saved_rows,video_frames=recorder.video_frames,
        recording_error=recorder.error,dropped_rows=recorder.dropped_rows,run_error=error,
        recording_valid=bool(index and not error and not recorder.error and not recorder.dropped_rows
                             and recorder.finished and recorder.saved_rows==index),
        output=str(args.output),robot_execution=False)
    report['synthetic_input']=bool(synthetic_input)
    (args.output/'recording_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def result_pose(result):
    if not result.pose_landmarks:return None
    return np.array([[p.x,p.y,p.z,p.visibility] for p in result.pose_landmarks[0]])


def compare_recording(path,pose_model='full',stride=3):
    """Replay every saved frame through VIDEO; sample IMAGE without tracking.

    Compare only paired frames. IMAGE calls run offline, after the live stream
    closes, so their CPU cost cannot change the recorded tracking behaviour.
    """
    if stride<1:raise ValueError('stride must be positive')
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    path=Path(path);dataset=path/'dataset'
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    if not manifest['finished'] or manifest['recording_error'] or manifest['dropped_recording_rows']:
        raise ValueError('Recording incomplete or contains dropped rows')
    if not manifest['save_video'] or not manifest['video_frames']:raise ValueError('No saved video; record with --save-video')
    if manifest.get('pose_model')!=pose_model:raise ValueError('Comparison must use the recorded pose model')
    rows=[];capture=None;chunk=None;timestamp=-1
    model=ROOT.parent/'pose2arm-main/models'/f'pose_landmarker_{pose_model}.task'
    def options(mode):return vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),running_mode=mode,num_poses=1)
    try:
        with vision.PoseLandmarker.create_from_options(options(vision.RunningMode.VIDEO)) as video, \
             vision.PoseLandmarker.create_from_options(options(vision.RunningMode.IMAGE)) as image:
            with (dataset/'video_timestamps.csv').open(newline='',encoding='utf-8') as f:
                for i,entry in enumerate(csv.DictReader(f)):
                    current=int(entry['video_chunk'])
                    if current!=chunk:
                        if capture is not None:capture.release()
                        capture=cv2.VideoCapture(str(dataset/f'human_camera_{current:05d}.mp4'));chunk=current
                    if int(capture.get(cv2.CAP_PROP_POS_FRAMES))!=int(entry['frame_in_chunk']):
                        raise ValueError('Video/frame timestamp alignment mismatch')
                    ok,bgr=capture.read()
                    if not ok:raise ValueError('Recorded frame unreadable')
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    frame=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    timestamp=max(timestamp+1,int(entry['capture_read_end_ns'])//1_000_000)
                    tracked=video.detect_for_video(frame,timestamp)
                    if i%stride:continue
                    fresh=image.detect(frame)
                    row=dict(source_seq=np.asarray(int(entry['source_seq'])),
                        capture_ns=np.asarray(int(entry['capture_read_end_ns']),dtype=np.int64))
                    for name,result in (('video',tracked),('image',fresh)):
                        pose=result_pose(result)
                        trusted,reasons=body_hand_consistency(pose,[],(bgr.shape[1],bgr.shape[0]),require_elbow=False)
                        row[name+'_pose']=array_or_nan(pose,(33,4))
                        row[name+'_body']=np.asarray(pose is not None)
                        row[name+'_right_wrist']=np.asarray(trusted['right'])
                        row[name+'_reason']=np.asarray(reasons['right'])
                    rows.append(row)
    finally:
        if capture is not None:capture.release()
    if not rows:raise ValueError('No saved video frames to compare; record with --save-video')
    arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(path/'pose_mode_comparison.npz',**arrays)
    report=dict(scope='same saved images, VIDEO vs IMAGE; coverage is not accuracy',
        paired_frames=len(rows),model=pose_model,stride=stride,hand_consistency_check=False,
        note='Offline VIDEO starts fresh and is not an exact replay of the live tracker or R events',
        independent_ground_truth=False,
        body_detected={m:int(arrays[m+'_body'].sum()) for m in ('video','image')},
        right_wrist_gate_passed={m:int(arrays[m+'_right_wrist'].sum()) for m in ('video','image')},
        image_only_body=int(np.sum(arrays['image_body']&~arrays['video_body'])),
        video_only_body=int(np.sum(arrays['video_body']&~arrays['image_body'])))
    (path/'pose_mode_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--pose-model',choices=('lite','full'),default='full')
    parser.add_argument('--seconds',type=float,default=35.)
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--save-video',action='store_true',help='Opt in to local raw camera video')
    parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--compare',action='store_true',help='After recording, compare IMAGE and VIDEO offline')
    parser.add_argument('--compare-existing',type=Path,help='Only compare an existing diagnostic output directory')
    parser.add_argument('--stride',type=int,default=3)
    args=parser.parse_args()
    if args.compare_existing:
        print(json.dumps(compare_recording(args.compare_existing,args.pose_model,args.stride),indent=2));return
    if args.compare and not args.save_video:parser.error('--compare requires --save-video')
    args.output=args.output or ROOT/'outputs/pose_diagnosis'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    print('CAMERA ONLY; no robot. 1=sitting, 2=standing, R=restart vision, Q=save/exit. Output: '+str(args.output),flush=True)
    report=record_camera(args);print(json.dumps(report,indent=2),flush=True)
    if not report['recording_valid']:raise SystemExit(1)
    if args.compare:print(json.dumps(compare_recording(args.output,args.pose_model,args.stride),indent=2),flush=True)


if __name__=='__main__':main()
