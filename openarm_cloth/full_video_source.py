"""Full-video A landmarks -> continuous anatomical two-arm B observations."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from human_teleop_core import ROOT,human_arm_pose,human_wrist
from human_gripper_gestures import fist_features
from smoothing import VectorOneEuroFilter

SIDES=('left','right')


def load_full_landmarks(video,cache,refresh=False,expected_sides=SIDES):
    video=Path(video).resolve();cache=Path(cache)
    expected=(expected_sides,) if isinstance(expected_sides,str) else tuple(expected_sides)
    if not expected or any(s not in SIDES for s in expected):raise ValueError('unknown expected hand side')
    with video.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if cache.is_file() and not refresh:
        with np.load(cache,allow_pickle=False) as z:
            cached_sides=tuple(z['expected_sides'].tolist()) if 'expected_sides' in z.files else None
            if int(z['schema_version'])==1 and str(z['source_sha256'].item())==digest and cached_sides==expected:
                return {k:z[k] for k in z.files}
    from extract_pose import extract_from_video
    from extract_hand import extract_hands_from_video,normalized_pinch_distance
    from gripper_map import map_positions
    print('A full motion: extracting every body and hand frame',flush=True)
    pose=extract_from_video(str(video))
    hands=extract_hands_from_video(str(video),pose_norm=pose['pose'],expected_sides=expected)
    capture=cv2.VideoCapture(str(video))
    width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT));capture.release()
    n=len(pose['pose_world']);fps=float(pose['fps'])
    if len(hands['hand_world'])!=n: raise ValueError('A body/hand video clocks differ')
    closure=np.column_stack([map_positions(normalized_pinch_distance(hands['hand_world'],s)[0],fps) for s in SIDES])
    result=dict(schema_version=np.array(1),source_sha256=np.array(digest),source_video=np.array(str(video)),
                expected_sides=np.array(expected),fps=np.array(fps),image_size=np.array([width,height]),
                pose_world=pose['pose_world'],pose_norm=pose['pose'],hand_world=hands['hand_world'],
                hand_norm=hands['hand_img'],hand_valid=hands['hand_present'],closure=closure)
    cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**result)
    return result


class FullVideoSource:
    """Every source frame retains both arms and all hand/wrist information.

    Filters follow the source video clock, not offline processing wall time.
    The control consumer samples the last source frame at each 20 Hz tick.
    """
    epoch_ns=1_000_000_000_000

    def __init__(self,blob):
        self.blob=blob;self.fps=float(blob['fps']);self.count=len(blob['pose_world'])
        if not np.isfinite(self.fps) or self.fps<=0 or self.count<2: raise ValueError('Invalid full video clock')
        if np.asarray(blob['pose_world']).shape!=(self.count,33,4): raise ValueError('Invalid body landmark shape')
        self.duration=self.count/self.fps
        self.frames=[]
        body_filters={i:VectorOneEuroFilter(3,self.fps,min_cutoff=1.5,beta=4) for i in (11,12,13,14,15,16,23,24)}
        hand_filters={s:VectorOneEuroFilter(63,self.fps,min_cutoff=2,beta=5) for s in SIDES}
        normals={s:None for s in SIDES};last_seen={s:-1_000_000 for s in SIDES}
        for index,raw in enumerate(blob['pose_world']):
            world=np.array(raw,dtype=float,copy=True)
            for i,f in body_filters.items():
                if np.isfinite(world[i]).all() and world[i,3]>=.3: world[i,:3]=f.filter(world[i,:3])
            arms={}
            for j,s in enumerate(SIDES):
                present=bool(blob['hand_valid'][index,j])
                hand=np.asarray(blob['hand_world'][index,j],float) if present else None
                norm=np.asarray(blob['hand_norm'][index,j],float) if present else None
                present=present and np.isfinite(hand).all()
                features=fist_features(hand,norm,blob['image_size']) if present else None
                filtered=None
                if present:
                    if (index-last_seen[s])/self.fps>.4:
                        hand_filters[s]=VectorOneEuroFilter(63,self.fps,min_cutoff=2,beta=5)
                    filtered=hand_filters[s].filter(hand.reshape(-1)).reshape(21,3);last_seen[s]=index
                pose=human_arm_pose(world,filtered,normals[s],side=s)
                if pose is not None: normals[s]=pose['normal']
                arms[s]=dict(pose=pose,point=human_wrist(world,s),hand_valid=present,
                             position_point=human_wrist(world,s,require_elbow=False),
                             fist=features,closure=float(blob['closure'][index,j]),
                             hand_raw=hand,hand_filtered=filtered,hand_norm=norm)
            self.frames.append(dict(seq=index,read_end=self.epoch_ns+int(index/self.fps*1e9),arms=arms))

    def at(self,time_s):
        if not np.isfinite(time_s) or time_s<0: raise ValueError('Invalid source time')
        if time_s>=self.duration: return None
        return self.frames[min(self.count-1,int(np.floor(time_s*self.fps+1e-9)))]

    def now_ns(self,time_s): return self.epoch_ns+int(time_s*1e9)
