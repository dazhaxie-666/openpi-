"""Diagnostic recordings preserve fresh source/frame identity and missing data."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from diagnose_human_tracking import record_camera,compare_recording


class PoseDiagnosisTests(unittest.TestCase):
    def record(self,output):
        class Stream:
            error=None;metadata={'source':'explicit software fixture'}
            def __init__(self):self.count=0
            def snapshot(self):
                seq=(1,1,2,3)[min(self.count,3)];self.count+=1
                frame=np.full((120,160,3),seq*30,np.uint8)
                return dict(seq=seq,read_end=seq*100_000_000,done=seq*100_000_000+1,
                    pose_norm=None,pose_raw=None,image=frame,raw_image=frame,
                    arms={s:dict(hand_valid=False) for s in ('left','right')})
            @property
            def finished(self):return self.count>=4
            def close(self):pass
        args=SimpleNamespace(output=output,seconds=5,camera=0,pose_model='full',
            save_video=True,no_preview=True)
        return record_camera(args,lambda *args:Stream(),synthetic_input=True)

    def test_missing_body_and_duplicate_poll_do_not_invent_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'recording';report=self.record(path)
            self.assertTrue(report['recording_valid']);self.assertFalse(report['robot_execution'])
            self.assertTrue(report['synthetic_input'])
            self.assertEqual(report['saved_rows'],3);self.assertEqual(report['video_frames'],3)
            with np.load(next((path/'dataset').glob('chunk_*.npz')),allow_pickle=False) as blob:
                np.testing.assert_array_equal(blob['source_seq'],[1,2,3])
                self.assertTrue(np.isnan(blob['pose_norm']).all())
                self.assertFalse(blob['body_detected'].any());self.assertFalse(blob['position_trusted'].any())

    def test_offline_image_samples_pair_with_same_video_frames_and_source_ids(self):
        from mediapipe.tasks.python import vision
        calls={'video':[],'image':[]}
        class Detector:
            def __init__(self,mode):self.mode=mode
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def detect_for_video(self,image,timestamp):
                calls['video'].append(timestamp);return SimpleNamespace(pose_landmarks=[])
            def detect(self,image):
                calls['image'].append(1);return SimpleNamespace(pose_landmarks=[])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'recording';self.record(path)
            with patch.object(vision.PoseLandmarker,'create_from_options',side_effect=lambda opts:Detector(opts.running_mode)):
                report=compare_recording(path,stride=2)
            self.assertEqual(calls['video'],[100,200,300]);self.assertEqual(len(calls['image']),2)
            self.assertEqual(report['paired_frames'],2);self.assertFalse(report['independent_ground_truth'])
            self.assertEqual(report['body_detected'],{'video':0,'image':0})
            with np.load(path/'pose_mode_comparison.npz',allow_pickle=False) as blob:
                np.testing.assert_array_equal(blob['source_seq'],[1,3])
                self.assertTrue(np.isnan(blob['image_pose']).all())


if __name__=='__main__':unittest.main()
