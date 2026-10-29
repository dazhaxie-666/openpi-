"""Operator-controlled aperture hold; never inferred from a recording phase."""
from dataclasses import replace
import numpy as np


class GripperHold:
    def __init__(self):
        self.locked=False;self.aperture=None
        self.resume_after=None;self.resume_last=None;self.resume_count=0

    @property
    def mode(self):
        return 'locked' if self.locked else 'reconnecting' if self.resume_after is not None else 'continuous'

    def lock(self,aperture):
        if not np.isfinite(aperture) or not 0<=aperture<=.044:raise ValueError('invalid held aperture')
        self.locked=True;self.aperture=float(aperture)
        self.resume_after=self.resume_last=None;self.resume_count=0

    def release(self,source_seq):
        self.locked=False;self.resume_after=int(source_seq)
        self.resume_last=None;self.resume_count=0

    def apply(self,command,hand,stage,held_aperture,paused=False):
        if stage!='direct' or paused:
            self.resume_count=0;self.resume_last=None
            return command
        if self.resume_after is not None and (hand is None or not hand['valid']):
            self.resume_last=None;self.resume_count=0
        if command.stage not in ('direct','gripper_reference'):return command
        if self.locked:return replace(command,finger=self.aperture)
        if self.resume_after is None:return command
        if hand is None or not hand['valid']:
            self.resume_last=None;self.resume_count=0
        else:
            seq=int(hand['seq'])
            if seq>self.resume_after and (self.resume_last is None or seq>self.resume_last):
                self.resume_count+=1;self.resume_last=seq
            if self.resume_count>=2:
                self.resume_after=None
                return command
        return replace(command,finger=float(held_aperture))
