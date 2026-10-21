"""Exact source frames, including MP4 tails with unreliable random seeking."""
from pathlib import Path

import cv2


class SourceVideoReader:
    def __init__(self, path):
        self.path = Path(path)
        self.capture = None
        self.index = -1
        self.last = None
        self._reopen()

    def _reopen(self):
        self.close()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.close()
            raise FileNotFoundError(self.path)
        self.index = -1
        self.last = None

    def frame(self, index):
        if int(index) != index or index < 0:
            raise ValueError('Source frame index must be a non-negative integer')
        index = int(index)
        if index < self.index:
            self._reopen()
        while self.index < index:
            ok, frame = self.capture.read()
            if not ok:
                raise ValueError(f'Cannot decode measured source frame {index}; stopped at {self.index}')
            self.index += 1
            self.last = frame
        return self.last

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
