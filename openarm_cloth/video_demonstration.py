"""Offline Member A landmarks and pinch intent, independent of simulator state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'pose2arm-main'))


def runs(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


class VideoDemonstration:
    """Select a measured open->close phase and transfer its bounded path shape.

    The video's clock is normalized into B's approach phase, not played back
    as a continuous full-body imitation. Grasp position/orientation remain B's.
    """
    def __init__(self,time,wrist,closure,pose_valid,hand_valid,arm_length,metadata,event=-1):
        self.time=np.asarray(time,float);self.wrist=np.asarray(wrist,float)
        self.closure=np.asarray(closure,float);self.pose_valid=np.asarray(pose_valid,bool)
        self.hand_valid=np.asarray(hand_valid,bool);self.arm_length=np.asarray(arm_length,float)
        n=len(self.time)
        if (n<12 or self.time.shape!=(n,) or self.wrist.shape!=(n,3)
                or any(x.shape!=(n,) for x in (self.closure,self.pose_valid,self.hand_valid,self.arm_length))
                or not np.isfinite(self.time).all() or np.any(np.diff(self.time)<=0)
                or not np.isfinite(self.closure).all() or np.any((self.closure<0)|(self.closure>1))
                or not np.isfinite(self.wrist[self.pose_valid]).all()):
            raise ValueError('Invalid human demonstration values/shapes/timestamps')
        self.metadata=dict(metadata)
        self.fps=1/float(np.median(np.diff(self.time)))
        stable=max(3,int(np.ceil(.15*self.fps)))
        opens=[(a,b) for a,b in runs((self.closure<=.25)&self.hand_valid&self.pose_valid) if b-a>=stable]
        events=[]
        for a,b in runs((self.closure>=.65)&self.hand_valid&self.pose_valid):
            previous=[(c,d) for c,d in opens if d<=a and self.time[a]-self.time[d-1]<=3.]
            if b-a<stable or not previous: continue
            start=max(previous[-1][0],a-int(round(1.5*self.fps)))
            if a-start<max(6,int(.3*self.fps)): continue
            valid=self.pose_valid[start:a+1]
            gaps=runs(~valid)
            if np.mean(valid)<.85 or any(y-x>max(2,int(.12*self.fps)) for x,y in gaps): continue
            # A short close spike is not enough; take a stable measured window.
            level=max(float(np.median(self.closure[j:j+stable])) for j in range(a,b-stable+1))
            events.append((start,a,b,level))
        if not events: raise ValueError('No valid measured open -> close phase; no autonomous substitution')
        if event < -1 or event>=len(events): raise ValueError(f'event must be -1 or 0..{len(events)-1}')
        self.event_index=max(range(len(events)),key=lambda j:events[j][3]) if event==-1 else event
        self.start,self.close,self.end,self.close_level=events[self.event_index]
        ids=np.arange(self.start,self.close+1);valid=self.pose_valid[ids]
        points=np.column_stack([np.interp(ids,ids[valid],self.wrist[ids[valid],j]) for j in range(3)])
        points=gaussian_filter1d(points,1.5,axis=0,mode='nearest')
        length=self.arm_length[ids][valid]
        if not np.isfinite(length).all() or np.min(length)<.1:
            raise ValueError('Human arm length invalid; cannot normalize motion')
        self.scale=float(np.clip(.455/np.median(length),.35,1.25))
        self.phase=np.linspace(0,1,len(points))
        # Endpoint correction preserves B's measured approach/grasp anchors.
        chord=(1-self.phase[:,None])*points[0]+self.phase[:,None]*points[-1]
        self.curve=(points-chord)*self.scale
        self.current_frame=int(self.start)
        self.stage='B autonomous setup'
        self.plan_report=None

    def descent_targets(self,center,yaw,gain=1.,count=9):
        center=np.asarray(center,float)
        if center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(yaw):
            raise ValueError('Invalid visual grasp anchor')
        if not np.isfinite(gain) or not 0<=gain<=1: raise ValueError('motion gain must be in [0,1]')
        phase=np.linspace(0,1,count)
        offsets=np.column_stack([np.interp(phase,self.phase,self.curve[:,j]) for j in range(3)])*gain
        # The last quarter is B's precision docking corridor. Never move the
        # endpoint off the observed object to copy an unrelated video gesture.
        envelope=np.sin(np.pi*phase)**2*np.clip((1-phase)/.3,0,1)
        offsets*=envelope[:,None]
        norms=np.linalg.norm(offsets[:,:2],axis=1)
        offsets[:,:2]*=np.minimum(1,.012/np.maximum(norms,1e-12))[:,None]
        offsets[:,2]=np.clip(offsets[:,2],-.004,.004)
        c,s=np.cos(yaw),np.sin(yaw)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        targets=center+offsets@rotation.T
        targets[:,2]+=.07*(1-(10*phase**3-15*phase**4+6*phase**5))
        targets[:,2]=np.maximum(targets[:,2],center[2])
        targets[0]=center+[0,0,.07];targets[-1]=center
        return targets

    @property
    def finger_target(self): return .044*(1-self.close_level)

    def set_phase(self,stage,progress=0.):
        self.stage=stage
        self.current_frame=int(round(self.start+(self.close-self.start)*np.clip(progress,0,1))) if stage=='human_descent' else int(self.close if stage in ('grip','prelift','lift','hold') else self.start)

    def report(self):
        return dict(self.metadata,source_phase='offline_open_to_close',event_index=self.event_index,
                    start_frame=int(self.start),close_frame=int(self.close),end_frame=int(self.end),
                    source_interval_s=[float(self.time[self.start]),float(self.time[self.close])],
                    human_to_robot_scale=self.scale,human_closure=self.close_level,
                    finger_target_m=self.finger_target,orientation_control='B visual face orientation',
                    timing='phase normalized; B setup/lift hold source frame, not full-video imitation',
                    release_policy='hold after lift; no transfer/release implemented in this phase',
                    transfer=self.plan_report)


def load_video(video,cache,side='right',refresh=False,event=-1):
    video=Path(video).resolve();cache=Path(cache)
    if side not in ('left','right'): raise ValueError('Invalid demo side')
    with video.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    blob=None
    if cache.is_file() and not refresh:
        with np.load(cache,allow_pickle=False) as z:
            if str(z['source_sha256'].item())==digest: blob={k:z[k] for k in z.files}
    if blob is None:
        from extract_pose import extract_from_video
        from extract_hand import extract_hands_from_video,normalized_pinch_distance
        from gripper_map import map_positions
        from human_teleop_core import human_wrist
        print(f'A: extracting body landmarks from {video.name}',flush=True)
        pose=extract_from_video(str(video))
        print('A: extracting hands with anatomical one-to-one association',flush=True)
        hands=extract_hands_from_video(str(video),pose_norm=pose['pose'])
        n=len(pose['pose_world']);fps=float(pose['fps'])
        if len(hands['hand_world'])!=n: raise ValueError('Body/hand frame count mismatch')
        wrist=np.full((n,2,3),np.nan);valid=np.zeros((n,2),bool);length=np.full((n,2),np.nan)
        closure=[]
        for j,name in enumerate(('left','right')):
            ratio,_=normalized_pinch_distance(hands['hand_world'],name)
            closure.append(map_positions(ratio,fps))
            a,b,c=(11,13,15) if j==0 else (12,14,16)
            for i,w in enumerate(pose['pose_world']):
                point=human_wrist(w,name)
                if point is not None:
                    wrist[i,j]=point;valid[i,j]=True
                    length[i,j]=np.linalg.norm(w[b,:3]-w[a,:3])+np.linalg.norm(w[c,:3]-w[b,:3])
        blob=dict(time_s=np.arange(n)/fps,wrist=wrist,pose_valid=valid,
                  hand_valid=hands['hand_present'],arm_length=length,
                  closure=np.column_stack(closure),source_sha256=np.array(digest),
                  source_video=np.array(str(video)),fps=np.array(fps),schema_version=np.array(1))
        cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**blob)
        print(f'A: saved measured demonstration cache {cache}',flush=True)
    index=0 if side=='left' else 1
    return VideoDemonstration(blob['time_s'],blob['wrist'][:,index],blob['closure'][:,index],
                              blob['pose_valid'][:,index],blob['hand_valid'][:,index],blob['arm_length'][:,index],
                              dict(source_video=str(video),source_sha256=digest,source_side=side,
                                   cache=str(cache.resolve()),backend='A MediaPipe heavy + hand landmarker'),event)
