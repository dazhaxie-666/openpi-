"""Latest-reference B physics worker, independent of OpenCV/viewer/recording UI."""
import threading
import time
from types import SimpleNamespace
import numpy as np


def capture_control(control,result,first_ns,end_ns,wall_dt,sim_dt,tick):
    return SimpleNamespace(result=result,first_ns=first_ns,end_ns=end_ns,wall_dt=wall_dt,sim_dt=sim_dt,tick=tick,
        qpos=control.data.qpos.copy(),qvel=control.data.qvel.copy(),ctrl=control.data.ctrl.copy(),sim_time=float(control.data.time),
        action=control.control.action.copy(),joints={s:q.copy() for s,q in control.joints.items()},fingers=control.fingers.copy(),
        active_sides=control.active_sides.copy(),active_fingers=control.active_fingers.copy(),
        rejections=dict(control.reference_rejections),goal=np.concatenate([control.trajectories[s].goal for s in ('left','right')]),
        velocity=control.control.target_velocity.copy())


class LivePhysicsWorker:
    def __init__(self,control,hz=40.,max_age=.25,reference_provider=None):
        self.control=control;self.period=1/float(hz);self.max_age=max_age
        self.lock=threading.Lock();self.stop=threading.Event();self.command=(None,{},True,{})
        self.latest=capture_control(control,None,0,0,0.,0.,0)
        self.error=None;self.ticks=0;self.work_ms=[];self.overruns=0;self.unserved_wall_s=0.;self.started=time.perf_counter()
        self.reference_provider=reference_provider
        self.trace=[]
        self.ended=None
        self.thread=threading.Thread(target=self._run,daemon=True,name='b-motion-physics');self.thread.start()

    def submit(self,result,visible,paused=False,finger_visible=None):
        with self.lock:self.command=(result,dict(visible),bool(paused),dict(finger_visible or {}))

    def snapshot(self):
        with self.lock:return self.latest

    def _run(self):
        previous=time.perf_counter()
        try:
            while not self.stop.is_set():
                start=time.perf_counter();now=time.perf_counter_ns()
                if self.reference_provider is not None:result,visible,paused,fingers=self.reference_provider()
                else:
                    with self.lock:result,visible,paused,fingers=self.command
                if result is not None and not 0<=(now-result['capture_ns'])/1e9<=self.max_age:result=None
                wall_dt=start-previous if self.ticks else self.period;previous=start
                sim_dt=float(np.clip(wall_dt,.005,.1))
                # Time lost beyond the bounded catch-up interval is reported,
                # not disguised as another perfectly timed 50 ms control tick.
                self.unserved_wall_s+=max(0.,wall_dt-sim_dt)
                self.control.step(result,visible,paused,sim_dt,finger_visible=fingers)
                end=time.perf_counter_ns();self.ticks+=1
                state=capture_control(self.control,result,now,end,wall_dt,sim_dt,self.ticks)
                self.trace.append(dict(first_ns=np.asarray(now,dtype=np.int64),end_ns=np.asarray(end,dtype=np.int64),
                    wall_dt_s=np.asarray(wall_dt),sim_dt_s=np.asarray(sim_dt),sim_time=np.asarray(state.sim_time),
                    qpos=state.qpos.copy(),qvel=state.qvel.copy(),action=state.action.copy(),
                    source_seq=np.asarray(result['seq'] if result else -1),
                    capture_ns=np.asarray(result['capture_ns'] if result else 0,dtype=np.int64),
                    module_state=np.asarray([[result.get('module_state',{}).get(s,{}).get(m,'LOST') if result else 'LOST' for m in ('torso','arm','palm','pinch','association')] for s in ('left','right')]),
                    actual_arm_valid=np.asarray([state.active_sides[s] for s in ('left','right')]),
                    actual_pinch_valid=np.asarray([state.active_fingers[s] for s in ('left','right')])))
                with self.lock:self.latest=state
                work=time.perf_counter()-start;self.work_ms.append(work*1000)
                if work>self.period:self.overruns+=1
                self.stop.wait(max(0.,self.period-work))
        except Exception as error:self.error=f'{type(error).__name__}: {error}'

    def close(self):
        self.stop.set();self.thread.join(timeout=5)
        self.ended=time.perf_counter()
        if self.thread.is_alive():self.error=self.error or 'physics worker did not stop within 5 s'

    def report(self):
        elapsed=(self.ended or time.perf_counter())-self.started
        return dict(ticks=self.ticks,physics_hz_target=1/self.period,wall_s=elapsed,sim_s=float(self.control.data.time),
                    realtime_factor=float(self.control.data.time)/elapsed if elapsed else None,
                    control_work_ms_p50=float(np.percentile(self.work_ms,50)) if self.work_ms else None,
                    control_work_ms_p95=float(np.percentile(self.work_ms,95)) if self.work_ms else None,
                    overrun_ticks=self.overruns,unserved_wall_s=self.unserved_wall_s,error=self.error,
                    exposure_timestamp_available=False,latency_scope='host-read timestamps; not exposure-to-motion measurement')

    def save_trace(self,path):
        if self.thread.is_alive():raise RuntimeError('stop physics before saving its immutable trace')
        if self.trace:np.savez_compressed(path,**{k:np.stack([r[k] for r in self.trace]) for k in self.trace[0]})
