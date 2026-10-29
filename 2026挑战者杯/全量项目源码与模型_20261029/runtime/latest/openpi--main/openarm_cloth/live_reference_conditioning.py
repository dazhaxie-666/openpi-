"""Causal execution conditioning; never changes A's calibrated nominal joints.

Source filters advance once per camera observation. Trajectories advance on B's
physics clock, with continuous velocity and bounded acceleration/feedforward.
"""
import numpy as np


class AdaptiveReferenceFilter:
    def __init__(self, min_cutoff=1.5, beta=.35):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.value = self.raw = self.derivative = None
        self.stamp = None

    def update(self, value, stamp):
        value = np.asarray(value, dtype=float)
        if self.stamp is not None and stamp <= self.stamp:
            return self.value.copy()
        if self.stamp is None or stamp-self.stamp > .35:
            self.value = value.copy()
            self.derivative = np.zeros_like(value)
        else:
            dt = float(np.clip(stamp-self.stamp, .005, .2))
            alpha_d = 1.-np.exp(-2.*np.pi*dt)
            self.derivative += alpha_d*((value-self.raw)/dt-self.derivative)
            cutoff = self.min_cutoff+self.beta*np.abs(self.derivative)
            alpha = 1.-np.exp(-2.*np.pi*cutoff*dt)
            self.value += alpha*(value-self.value)
        self.raw, self.stamp = value.copy(), float(stamp)
        return self.value.copy()


class ContinuousPinch:
    """Palm-normalized pinch distance -> each B slide joint in metres.

    .15 and 1.2 are interpolation endpoints, not open/close state thresholds.
    Missing/invalid hands hold the last aperture; gaps rebase filter timing.
    """
    def __init__(self):
        self.filter = AdaptiveReferenceFilter(min_cutoff=2., beta=1.5)
        self.aperture = .044
        self.ratio = np.nan

    def update(self, ratio, stamp):
        if ratio is None or not np.isfinite(ratio) or ratio < 0:
            self.filter.stamp = None
            self.ratio = np.nan
            return self.aperture
        self.ratio = float(ratio)
        openness = float(np.clip((ratio-.15)/1.05, 0., 1.))
        filtered = float(self.filter.update(openness, stamp))
        # Sub-millimetre changes remain continuous, with filtering rather than
        # a hysteresis switch. B's execution path bounds the applied speed.
        self.aperture = float(np.clip(filtered, 0., 1.))*.044
        return self.aperture


class JointReferenceTrajectory:
    def __init__(self, position, speed=None, acceleration=None, limits=None,frequency=20.,filter_enabled=True):
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros_like(self.position)
        self.speed = np.asarray(speed if speed is not None else [2.5]*4+[4.4]*3)
        self.acceleration = np.asarray(acceleration if acceleration is not None else [14.]*4+[24.]*3)
        self.filter = AdaptiveReferenceFilter()
        self.goal = self.position.copy()
        self.limits=None if limits is None else np.asarray(limits,dtype=float).copy()
        self.frequency=float(frequency);self.filter_enabled=bool(filter_enabled);self.source_stamp=None

    def observe(self, goal, capture_s):
        if self.source_stamp is not None and capture_s<=self.source_stamp:return
        self.source_stamp=capture_s
        self.goal = self.filter.update(goal, capture_s) if self.filter_enabled else np.asarray(goal,dtype=float).copy()

    def advance(self, dt, active=True):
        if active:
            # Critically damped servo for the reference, not new motor gains.
            desired = self.frequency**2*(self.goal-self.position)-2*self.frequency*self.velocity
        else:
            # Brake where we are; never snap back to lagging robot encoders.
            desired = -self.velocity/dt
            self.filter.stamp = None
        acceleration = np.clip(desired, -self.acceleration, self.acceleration)
        next_velocity = np.clip(self.velocity+acceleration*dt, -self.speed, self.speed)
        if self.limits is not None:
            # Keep enough distance to brake before B's native bounds. The
            # half-step reserve accounts for trapezoidal integration, including
            # stopping within a physics tick. This changes velocity gradually;
            # clipping position at the bound alone would create a new jump.
            ah=self.acceleration*dt
            upper_distance=np.maximum(0.,self.limits[:,1]-self.position)
            lower_distance=np.maximum(0.,self.position-self.limits[:,0])
            upper_speed=-ah+np.sqrt(np.maximum(0.,ah**2-ah*self.velocity+2*self.acceleration*upper_distance))
            lower_speed=-ah+np.sqrt(np.maximum(0.,ah**2+ah*self.velocity+2*self.acceleration*lower_distance))
            next_velocity=np.clip(next_velocity,-np.maximum(0.,lower_speed),np.maximum(0.,upper_speed))
        self.position += .5*(self.velocity+next_velocity)*dt
        self.velocity = next_velocity
        return self.position.copy(), self.velocity.copy()
