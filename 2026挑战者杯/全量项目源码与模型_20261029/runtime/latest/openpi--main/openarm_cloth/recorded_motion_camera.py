"""Saved consumed human frames as a timed latest-frame input for regression.

Not a fresh camera or independent acceptance recording. Missing original sensor
frames remain missing; looping the same session is only a load/regression test.
"""
import csv
from pathlib import Path
import threading
import time
import cv2


class RecordedMotionCamera:
    def __init__(self,dataset,loop=False):
        self.root=Path(dataset);self.rows=list(csv.DictReader((self.root/'video_timestamps.csv').open(encoding='utf-8')))
        if not self.rows:raise ValueError('recording contains no human video frames')
        self.loop=loop;self.closed=False;self.finished=False;self.error=None;self.latest=None;self.condition=threading.Condition()
        self.metadata=dict(source='saved human recording',fresh_human_validation=False,original_consumed_frames_only=True,loop=loop)
        self.thread=threading.Thread(target=self._reader,daemon=True,name='recorded-camera');self.thread.start()

    def _reader(self):
        cap=None;chunk=-1;next_frame=0;base=time.perf_counter_ns();first=int(self.rows[0]['capture_read_end_ns']);offset=0;seq=0
        duration=max(.05,(int(self.rows[-1]['capture_read_end_ns'])-first)/1e9+.05)
        try:
            while not self.closed:
                for row in self.rows:
                    if self.closed:return
                    when=base+int((offset+((int(row['capture_read_end_ns'])-first)/1e9))*1e9)
                    delay=(when-time.perf_counter_ns())/1e9
                    if delay>0:
                        with self.condition:self.condition.wait_for(lambda:self.closed,timeout=delay)
                    if self.closed:return
                    next_chunk=int(row['video_chunk'])
                    if cap is None or chunk!=next_chunk:
                        if cap:cap.release()
                        cap=cv2.VideoCapture(str(self.root/f'human_camera_{next_chunk:05d}.mp4'));chunk=next_chunk;next_frame=0
                    indexed_frame=int(row['frame_in_chunk'])
                    if indexed_frame!=next_frame:cap.set(cv2.CAP_PROP_POS_FRAMES,indexed_frame)
                    start=time.perf_counter_ns();ok,frame=cap.read();end=time.perf_counter_ns()
                    if not ok:raise RuntimeError('saved human video decode failed')
                    next_frame=indexed_frame+1
                    seq+=1
                    with self.condition:self.latest=(seq,start,end,frame);self.condition.notify_all()
                if not self.loop:break
                offset+=duration
                if cap:cap.release();cap=None;chunk=-1
        except Exception as error:self.error=str(error)
        finally:
            if cap:cap.release()
            with self.condition:self.finished=True;self.condition.notify_all()

    def read_new(self,after,timeout=2.):
        with self.condition:
            self.condition.wait_for(lambda:self.closed or self.error or self.finished or (self.latest and self.latest[0]>after),timeout)
            if self.error:raise RuntimeError(self.error)
            if self.latest is not None and self.latest[0]>after:return self.latest
            raise EOFError('recorded camera finished' if self.finished or self.closed else 'recorded camera timeout')

    def close(self):
        self.closed=True
        with self.condition:self.condition.notify_all()
        self.thread.join(timeout=3)
