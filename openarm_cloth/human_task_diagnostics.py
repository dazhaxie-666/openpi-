"""Local guided teleoperation recording; labels are prompts, not action truth."""
import numpy as np


def diagnostic_phase(elapsed,protocol='axes'):
    if protocol not in ('axes','response','depth-chain'):raise ValueError('unknown diagnostic protocol')
    if elapsed is None:return 'WAIT_START','Press START when right wrist is visible'
    if protocol=='depth-chain':
        for end,name,prompt in (
            (5,'depth_rest','Hold hand STILL at center; keep shoulder still'),
            (10,'reach_near','Reach TOWARD CAMERA; hold near position for 2 seconds'),
            (15,'return_center','Return hand to CENTER; hold 2 seconds'),
            (20,'reach_far','Pull hand BACK toward body; hold far position 2 seconds'),
            (25,'return_center','Return hand to CENTER; hold 2 seconds'),
            (30,'reach_near_repeat','Repeat TOWARD CAMERA; hold 2 seconds'),
            (35,'reach_far_repeat','Pull hand BACK toward body; hold 2 seconds'),
            (40,'depth_left_right','Move LEFT / RIGHT only; keep depth and height'),
            (45,'depth_up_down','Move UP / DOWN only; keep depth and lateral position'),
            (50,'depth_rest_end','Hold STILL at center; saving at 50 seconds')):
            if elapsed<end:return name,prompt
        return 'depth_end','Finished; saving recording'
    if protocol=='response':
        for end,name,prompt in (
            (5,'settle_half','Hold wrist at center, keep fingers HALF OPEN'),
            (15,'half_left_right','Keep HALF OPEN; move wrist LEFT / RIGHT slowly'),
            (25,'half_up_down','Keep HALF OPEN; move wrist UP / DOWN slowly'),
            (35,'half_forward_back','Keep HALF OPEN; reach FORWARD / BACK slowly'),
            (50,'aperture_steps','Wrist still: pinch / small gap / half / full; hold 2s each'),
            (55,'brief_occlusion','Cover fingers briefly (1s), then show HALF OPEN'),
            (65,'recovery_half','Keep HALF OPEN and still; wait for automatic save')):
            if elapsed<end:return name,prompt
        return 'response_end','Finished; saving recording'
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
    # A previous direct trace must not masquerade as a fresh mapping while held.
    trace=(getattr(task,'motion_trace',{}) or {}) if command.stage=='direct' else {}
    result={name:np.asarray(trace[name]).copy() if name in trace else np.full(3,np.nan)
            for name in ('human_anchor','robot_anchor','unclipped_goal','workspace_goal','lead_goal','workspace_anchor_shift')}
    result['workspace_clipped']=np.asarray(trace.get('workspace_clipped',[False]*3),dtype=bool)
    result['command_target']=np.asarray(command.target).copy() if command.target is not None else np.full(3,np.nan)
    result['motion_trace_valid']=np.asarray(bool(trace))
    return result


def diagnostic_summary(rows,video):
    timing={}
    for name in ('observe_ms','mapping_ms','execute_ms','planning_query_ms','payload_ms','ik_ms',
                 'collision_check_ms','image_record_ms','physics_ms','ui_ms','control_work_ms'):
        values=np.array([float(r.get(name,np.nan)) for r in rows]);values=values[np.isfinite(values)]
        timing[name]=dict(p50=float(np.median(values)),p95=float(np.percentile(values,95))) if len(values) else None
    phases={}
    for phase in sorted({str(r['diagnostic_phase']) for r in rows}):
        group=[r for r in rows if str(r['diagnostic_phase'])==phase]
        phases[phase]=dict(control_rows=len(group),wrist_valid_fraction=float(np.mean([r['human_pose_valid'] for r in group])),
            hand_valid_fraction=float(np.mean([r['human_hand_valid'] for r in group])),
            clipped_axis_control_rows=np.sum([r['workspace_clipped'] for r in group],axis=0).tolist())
    return dict(scope='control timing and operator prompts; not independent gesture accuracy or exposure-to-motion latency',
        frames=len(rows),timing_ms=timing,preview_render_rows=sum(bool(r.get('ui_refreshed',False)) for r in rows),
        phases=phases,axis_order='robot x forward, y left, z up',video=video)
