"""Guided labels and motion trace retain source/target meaning without driving it."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from human_task_diagnostics import diagnostic_phase,diagnostic_fields
from human_grasp_task import DirectHumanTask,TaskCommand,HumanInput,Feedback
from live_test_dataset import TestDatasetRecorder


class TaskDiagnosticRecordingTests(unittest.TestCase):
    def test_protocol_waits_for_start_and_has_distinct_depth_and_hand_back_labels(self):
        self.assertEqual(diagnostic_phase(None)[0],'WAIT_START')
        self.assertEqual(diagnostic_phase(0)[0],'rest')
        self.assertEqual(diagnostic_phase(5)[0],'left_right')
        self.assertEqual(diagnostic_phase(15)[0],'up_down')
        self.assertEqual(diagnostic_phase(25)[0],'forward_back')
        self.assertEqual(diagnostic_phase(35)[0],'palm_pinch')
        self.assertEqual(diagnostic_phase(45)[0],'back_pinch')
        self.assertEqual(diagnostic_phase(55)[0],'rest_end')

    def test_recording_distinguishes_workspace_clip_from_actual_robot_motion(self):
        task=DirectHumanTask();task.record_motion_trace=True;tcp=np.array([.43,0.,.5])
        h=HumanInput(1,np.zeros(3),None,True,0.)
        feedback=Feedback(tcp,visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.))
        task.update(h,feedback,0.,start=True)
        command=task.update(HumanInput(2,np.array([1.,0.,0.]),None,True,.05),feedback,.05)
        fields=diagnostic_fields(task,command)
        self.assertTrue(fields['workspace_clipped'][0])
        self.assertGreater(fields['unclipped_goal'][0],fields['workspace_goal'][0])
        self.assertLessEqual(fields['command_target'][0]-tcp[0],task.config.speed*.05+1e-9)
        task.motion_trace=None
        held=diagnostic_fields(task,TaskCommand('hold'))
        self.assertFalse(held['motion_trace_valid']);self.assertTrue(np.isnan(held['command_target']).all())

    def test_background_video_recording_keeps_frame_source_and_timestamp_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture'
            rec=TestDatasetRecorder(path,dict(synthetic_input=True),queue_size=8)
            frame=np.zeros((120,160,3),np.uint8)
            for i in range(3):
                self.assertTrue(rec.submit(dict(control_frame=np.asarray(i*2),source_seq=np.asarray(i+1),
                    capture_read_end_ns=np.asarray((i+1)*100_000_000,dtype=np.int64)),frame.copy()))
            rec.close();self.assertIsNone(rec.error);self.assertEqual(rec.video_frames,3)
            with np.load(next(path.glob('chunk_*.npz')),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['source_seq'],[1,2,3])
                np.testing.assert_array_equal(z['control_frame'],[0,2,4])
                np.testing.assert_array_equal(z['video_frame_index'],[0,1,2])

    def test_slow_encoder_cannot_block_submit_and_dropped_frames_are_explicit(self):
        import threading
        from unittest.mock import patch
        entered=threading.Event();release=threading.Event()
        class SlowEncoder:
            def isOpened(self):return True
            def write(self,frame):
                entered.set()
                if not release.wait(5):raise TimeoutError('encoder fixture timed out')
            def release(self):pass
        with tempfile.TemporaryDirectory() as folder,patch('live_test_dataset.cv2.VideoWriter',return_value=SlowEncoder()):
            rec=TestDatasetRecorder(Path(folder)/'fixture',dict(synthetic_input=True),queue_size=1)
            def row(i):return dict(control_frame=np.asarray(i),source_seq=np.asarray(i),capture_read_end_ns=np.asarray(i))
            frame=np.zeros((120,160,3),np.uint8)
            try:
                rec.submit(row(1),frame);self.assertTrue(entered.wait(5))
                self.assertTrue(rec.submit(row(2),frame))
                self.assertFalse(rec.submit(row(3),frame))
            finally:
                release.set();rec.close()
            self.assertEqual(rec.dropped_rows,1);self.assertEqual(rec.saved_rows,2)


if __name__=='__main__':unittest.main()
