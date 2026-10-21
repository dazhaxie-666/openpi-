"""Existing human video -> A relative intent -> B visual grasp and physics."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import hashlib

import numpy as np

from video_demonstration import load_video
from demo_grasp_viewer import main as grasp, interactive_cube_placement

ROOT=Path(__file__).resolve().parent


class EpisodeRecording:
    def __init__(self,path,demo):
        self.path=Path(path);self.demo=demo;self.next_time=None
        self.states=[];self.times=[];self.phases=[];self.frames=[]

    def sample(self,data,phase,source_frame):
        if self.next_time is None: self.next_time=float(data.time)
        if data.time+1e-8<self.next_time: return
        self.states.append(data.qpos.copy());self.times.append(float(data.time))
        self.phases.append(phase);self.frames.append(source_frame);self.next_time+=.05

    def finish(self,result):
        self.path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(self.path/'episode.npz',qpos=np.asarray(self.states),
                            sim_time_s=np.asarray(self.times),phase=np.asarray(self.phases),
                            source_frame=np.asarray(self.frames,dtype=np.int64),fps=np.array(20.))
        manifest=dict(schema_version=1,source_video=self.demo.metadata['source_video'],
                      source_sha256=self.demo.metadata['source_sha256'],
                      scene='simulation/v1/droid_cube_scene_perception.xml',
                      recording='episode.npz',report='result.json',verdict=result['verdict'],
                      frames=len(self.states),scope='B physics recording; video source is phase-normalized',
                      workspace=result['task_workspace'],
                      scene_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                    for name in ('simulation/v1/droid_cube_scene_perception.xml',
                                                 'simulation/v1/droid_cube_scene.xml')},
                      human_demonstration=self.demo.report())
        (self.path/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,default=ROOT.parent/'pose2arm-main/videos/p2a_test1.mp4')
    parser.add_argument('--side',choices=('left','right'),default='right')
    parser.add_argument('--event',type=int,default=-1,help='-1 selects strongest valid open->close event')
    parser.add_argument('--refresh',action='store_true',help='Re-extract landmarks even if source hash matches')
    parser.add_argument('--check',action='store_true',help='Execute actual B physics without interactive windows')
    parser.add_argument('--outdir',type=Path,default=ROOT/'outputs/video_fusion/p2a_test1')
    placement=parser.add_mutually_exclusive_group()
    placement.add_argument('--cube-test-pose',nargs=3,type=float,metavar=('X','Y','YAW_DEG'))
    placement.add_argument('--interactive-place',action='store_true')
    args=parser.parse_args()
    args.outdir=args.outdir.resolve();args.outdir.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'outputs/video_fusion/cache'/f'{args.video.stem}_A_landmarks.npz'
    try:
        demo=load_video(args.video,cache,args.side,args.refresh,args.event)
        print('Selected measured A phase:',json.dumps(demo.report()),flush=True)
        fixture=args.cube_test_pose
        if args.interactive_place:
            if args.check: parser.error('--check cannot use --interactive-place')
            fixture=interactive_cube_placement(86400)
            if fixture is None: return 0
        recorder=EpisodeRecording(args.outdir,demo)
        result=grasp(check_only=args.check,test_cube_pose=fixture,
                     report_path=args.outdir/'result.json',demonstration=demo,
                     episode_recorder=recorder,preview_initial_s=3. if not args.check else 0.)
        (args.outdir/'failure.json').unlink(missing_ok=True)
        print(f'Fusion replay: {args.outdir / "episode.npz"}',flush=True)
        return result
    except Exception as exc:
        (args.outdir/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__,message=str(exc)),indent=2),encoding='utf-8')
        raise


if __name__=='__main__': raise SystemExit(main())
