"""Bounded wall-clock physics and backward-compatible episode step timing."""
import numpy as np


class EpisodeRows(list):
    """Bound idle pre-start history; retain all active rows and their clocks."""
    def __init__(self,ready_limit=None):
        super().__init__();self.ready_limit=ready_limit;self.ready_omitted=0;self.started=False

    def append(self,row):
        if str(row['stage'])!='ready':self.started=True
        super().append(row)
        if not self.started and self.ready_limit is not None and len(self)>self.ready_limit:
            removed=len(self)-self.ready_limit;del self[:removed];self.ready_omitted+=removed


class PhysicsClock:
    def __init__(self,timestep,max_interval=.1):
        if not np.isfinite(timestep) or not 0<timestep<=.005:
            raise ValueError('invalid physics timestep')
        self.timestep=float(timestep)
        self.max_steps=int(np.floor(max_interval/self.timestep+1e-9))
        self.remainder=0.
        self.unserved_s=0.

    def advance(self,elapsed):
        if not np.isfinite(elapsed) or elapsed<=0:raise ValueError('invalid wall interval')
        limit=self.max_steps*self.timestep
        self.unserved_s+=max(0.,elapsed-limit)
        budget=min(elapsed,limit)+self.remainder
        steps=min(self.max_steps,max(1,int(np.floor(budget/self.timestep+1e-9))))
        interval=steps*self.timestep
        self.remainder=budget-interval
        return interval


def recorded_step_durations(rows,manifest,count):
    """One duration per PRE-step row; legacy recordings remain fixed at 20 Hz."""
    if manifest.get('physics_clock')=='bounded_wall_time' and 'physics_dt_s' not in rows:
        raise ValueError('wall-clock recording is missing physics_dt_s')
    durations=np.asarray(rows['physics_dt_s'],dtype=float) if 'physics_dt_s' in rows else np.full(count,1/manifest['control_hz'])
    if durations.shape!=(count,) or not np.isfinite(durations).all() or np.any(durations<=0):
        raise ValueError('invalid recorded physics durations')
    timestep=manifest.get('timestep_s')
    if timestep is not None and not np.allclose(durations/timestep,np.round(durations/timestep),rtol=0,atol=1e-7):
        raise ValueError('recorded physics durations are not integral substeps')
    return durations


def runtime_summary(rows,control_hz=20.):
    """Measured host intervals, never label configured Hz as achieved Hz."""
    if not rows:return dict(frames=0)
    elapsed=(float(rows[-1]['host_tick_ns'])-float(rows[0]['host_tick_ns']))/1e9
    simulated=float(rows[-1]['sim_time_s'])-float(rows[0]['sim_time_s'])
    work=np.array([r['control_work_ms'] for r in rows],dtype=float)
    return dict(frames=len(rows),host_elapsed_s=elapsed,sim_elapsed_s=simulated,
        achieved_control_hz=(len(rows)-1)/elapsed if elapsed>0 else None,
        realtime_factor=simulated/elapsed if elapsed>0 else None,
        control_work_ms_p50=float(np.median(work)),control_work_ms_p95=float(np.percentile(work,95)),
        overrun_rows=int(np.count_nonzero(work>1000/control_hz)),
        configured_control_hz=float(control_hz),
        unserved_wall_s=float(rows[-1].get('unserved_wall_s',0.)),
        latency_scope='host control loop; no exposure-to-motion or human accuracy claim')
