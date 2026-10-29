"""A trimmed camera replay must decode the indexed source frame, not frame zero."""
import threading,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from recorded_motion_camera import RecordedMotionCamera


class RecordedCameraIndexTests(unittest.TestCase):
    def test_trimmed_and_noncontiguous_frames_use_explicit_video_index(self):
        camera=RecordedMotionCamera.__new__(RecordedMotionCamera)
        camera.root=Path('.');camera.rows=[dict(capture_read_end_ns='1',video_chunk='0',frame_in_chunk=str(i)) for i in (50,51,55)]
        camera.loop=False;camera.closed=False;camera.finished=False;camera.error=None;camera.latest=None;camera.condition=threading.Condition()
        class Capture:
            def __init__(self,*args):self.index=0
            def set(self,prop,value):self.index=int(value)
            def read(self):
                value=self.index;self.index+=1;return True,np.array([value])
            def release(self):pass
        with patch('recorded_motion_camera.cv2.VideoCapture',side_effect=Capture):camera._reader()
        self.assertIsNone(camera.error)
        self.assertEqual(int(camera.latest[3][0]),55)


if __name__=='__main__':unittest.main()
