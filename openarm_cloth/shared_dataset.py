"""Bounded asynchronous robot RGB-D recording, indexed to control rows."""
from pathlib import Path
from queue import Queue,Full
from threading import Thread
import json
import cv2
import numpy as np


def saved_control_index(rows,stage):
    """Idle history may be trimmed; indices always address the SAVED array."""
    return None if stage=='ready' else len(rows)


class SharedEpisodeSegments:
    """Index complete approaches and trials against final saved control rows."""
    def __init__(self):
        self.approach=None;self.episode_id=0;self.requests=set();self.ends=set()

    def update(self,task,rows,clock):
        index=len(rows);began=False
        active=bool(task.trials and 'end_s' not in task.trials[-1])
        new_request=bool(task.trials and task.trial_id not in self.requests)
        new_end=bool(task.trials and 'end_s' in task.trials[-1] and task.trials[-1]['trial_id'] not in self.ends)
        collection_boundary=task.phase in ('reset','session_complete')
        if (task.stage=='direct' and not collection_boundary and self.approach is None
                and not new_end and (not active or new_request)):
            self.approach=dict(start_frame=index,start_s=float(clock));began=True
            self.episode_id=task.trial_id if new_request else task.trial_id+1
        if new_request:
            trial=task.trials[-1];self.requests.add(task.trial_id);self.episode_id=task.trial_id
            approach=self.approach or dict(start_frame=index,start_s=float(clock))
            trial.update(request_frame=index,request_s=trial['start_s'],**approach,segment_schema='full_task_v2')
            self.approach=None
        if task.trials:
            trial=task.trials[-1]
            if 'end_s' in trial and trial['trial_id'] not in self.ends:
                self.ends.add(trial['trial_id']);trial['end_frame_exclusive']=index+1
        return began

    def finalize(self,task,rows):
        if task.trials and 'end_frame_exclusive' not in task.trials[-1]:
            task.trials[-1]['end_frame_exclusive']=len(rows)
        if self.approach is not None and self.approach['start_frame']<len(rows):
            task.trials.append(dict(self.approach,trial_id=self.episode_id,request_frame=None,request_s=None,
                end_frame_exclusive=len(rows),end_s=float(rows[-1]['time_s']),status='interrupted',
                reason='session ended during human approach before grasp request',source='human approach only',
                segment_schema='full_task_v2',physics=dict(physical_task_success=False)))
        for trial in task.trials:
            segment=rows[trial['start_frame']:trial['end_frame_exclusive']]
            phases={str(r['shared_phase']) for r in segment}
            request=trial.get('request_frame')
            before_request=rows[trial['start_frame']:request] if request is not None else segment
            trial['includes_human_approach']=any(str(r['shared_phase'])=='manual' for r in before_request)
            trial['complete_task_segment']=bool(trial['includes_human_approach'] and
                {'close','lift','transport','release','verify'}<=phases)


class SharedFrameWriter:
    def __init__(self,output):
        self.root=Path(output);self.queue=Queue(maxsize=4)
        self.saved=self.dropped=0;self.error=None;self.finished=False
        self.thread=Thread(target=self._run,name='shared-rgbd-writer',daemon=True);self.thread.start()

    def submit(self,index,frames):
        if not frames:return
        try:self.queue.put_nowait((int(index),frames))
        except Full:self.dropped+=len(frames)

    def _run(self):
        try:
            with (self.root/'shared_frames.jsonl').open('w',encoding='utf-8') as stream:
                while True:
                    sample=self.queue.get()
                    try:
                        if sample is None:break
                        index,frames=sample
                        for camera,frame in frames.items():
                            rgb=frame['rgb'];depth=np.asarray(frame['depth'],dtype=np.float32)
                            directory=self.root/'shared_rgbd'/camera;directory.mkdir(parents=True,exist_ok=True)
                            rgb_path=directory/f'{index:06d}.jpg';depth_path=directory/f'{index:06d}.npz'
                            if not cv2.imwrite(str(rgb_path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90]):
                                raise OSError('robot RGB write failed')
                            np.savez_compressed(depth_path,depth_m=depth)
                            row=dict(control_frame=index,camera=camera,time_s=float(frame['time_s']),
                                rgb=str(rgb_path.relative_to(self.root)).replace('\\','/'),
                                depth=str(depth_path.relative_to(self.root)).replace('\\','/'),
                                size_px=[int(rgb.shape[1]),int(rgb.shape[0])])
                            stream.write(json.dumps(row)+'\n');self.saved+=1
                    finally:self.queue.task_done()
        except Exception as error:
            self.error=f'{type(error).__name__}: {error}'
            # Drain work so a disk error cannot deadlock simulator shutdown.
            while True:
                sample=self.queue.get();self.queue.task_done()
                if sample is None:break
        finally:self.finished=True

    def close(self):
        self.queue.put(None);self.thread.join()
        return dict(index='shared_frames.jsonl',saved=self.saved,dropped=self.dropped,error=self.error,
            finished=self.finished,target_observation_hz=5,encoding='robot RGB JPEG90 + metric float32 depth NPZ',
            timing='RGB-D observed before the indexed row action/physics; camera time separate from human source time',
            valid=bool(self.finished and self.error is None and self.dropped==0 and self.saved>0))
