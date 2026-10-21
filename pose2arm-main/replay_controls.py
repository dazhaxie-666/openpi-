"""Shared native MuJoCo replay camera and timeline for A and B."""
from dataclasses import dataclass


def configure_replay_camera(viewer, lookat=(0., 0., .30)):
    viewer.cam.lookat[:] = lookat
    viewer.cam.distance = 1.3
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20


@dataclass
class ReplayControls:
    count: int
    index: int = 0
    paused: bool = False
    quit: bool = False
    speed: float = 1.
    ghosts: bool = True

    def key(self, code):
        if code == 32:
            self.paused = not self.paused
        elif code in (ord('Q'), ord('q'), 256, 27):
            self.quit = True
        elif code in (ord('R'), ord('r')):
            self.index = 0
        elif code in (262, ord('D'), ord('d')):
            self.paused = True
            self.index = min(self.count - 1, self.index + 1)
        elif code in (263, ord('A'), ord('a')):
            self.paused = True
            self.index = max(0, self.index - 1)
        elif code in (265, ord('+'), ord('=')):
            self.speed = min(4., self.speed * 2)
        elif code in (264, ord('-')):
            self.speed = max(.125, self.speed / 2)
        elif code in (ord('G'), ord('g')):
            self.ghosts = not self.ghosts

    def advance(self):
        if not self.paused:
            self.index = (self.index + 1) % self.count
