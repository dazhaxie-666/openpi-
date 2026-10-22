"""Local guided teleoperation recording; labels are prompts, not action truth."""
import numpy as np


def diagnostic_phase(elapsed):
    if elapsed is None:return 'WAIT_START','Press START when right wrist is visible'
    for end,name,prompt in (
        (5,'rest','Hold still at a comfortable center'),
        (15,'left_right','Move wrist LEFT / RIGHT; keep height and depth'),
        (25,'up_down','Move wrist UP / DOWN; keep other axes still'),
        (35,'forward_back','Reach FORWARD / BACK; keep shoulder still'),
        (45,'palm_pinch','Palm toward camera: thumb/index pinch and spread'),
        (55,'back_pinch','Hand BACK toward camera: pinch and spread')):
        if elapsed<end:return name,prompt
    return 'rest_end','Hold still; Q saves early; recording ends automatically'


def diagnostic_fields(task,command):
    trace=getattr(task,'motion_trace',{}) or {}
    result={name:np.asarray(trace[name]).copy() if name in trace else np.full(3,np.nan)
            for name in ('human_anchor','robot_anchor','unclipped_goal','workspace_goal','lead_goal')}
    result['workspace_clipped']=np.asarray(trace.get('workspace_clipped',[False]*3),dtype=bool)
    result['command_target']=np.asarray(command.target).copy() if command.target is not None else np.full(3,np.nan)
    result['motion_trace_valid']=np.asarray(bool(trace))
    return result


def diagnostic_summary(rows,video):
    timing={}
    for name in ('observe_ms','mapping_ms','execute_ms','image_record_ms','physics_ms','ui_ms','control_work_ms'):
        values=np.array([float(r[name]) for r in rows]);values=values[np.isfinite(values)]
        timing[name]=dict(p50=float(np.median(values)),p95=float(np.percentile(values,95))) if len(values) else None
    phases={}
    for phase in sorted({str(r['diagnostic_phase']) for r in rows}):
        group=[r for r in rows if str(r['diagnostic_phase'])==phase]
        phases[phase]=dict(control_rows=len(group),wrist_valid_fraction=float(np.mean([r['human_pose_valid'] for r in group])),
            hand_valid_fraction=float(np.mean([r['human_hand_valid'] for r in group])),
            clipped_axis_control_rows=np.sum([r['workspace_clipped'] for r in group],axis=0).tolist())
    return dict(scope='control timing and operator prompts; not independent gesture accuracy or exposure-to-motion latency',
        frames=len(rows),timing_ms=timing,phases=phases,axis_order='robot x forward, y left, z up',video=video)
