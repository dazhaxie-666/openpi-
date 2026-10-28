"""Optional pre-step RGB-D sampling; leaves B perception/control cadence intact."""
from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path


class DatasetCapture:
    def __init__(self, hz=10.0, max_trials=0, max_seconds=0.0, reset_seconds=3.0,
                 task_profile='pickplace', hold_seconds=0.0, source_tolerance_m=0.03):
        if not math.isfinite(hz) or not 1 <= hz <= 20:
            raise ValueError('capture Hz must be in 1..20')
        if max_trials < 0 or max_seconds < 0 or not math.isfinite(reset_seconds) or reset_seconds < 0:
            raise ValueError('stop bounds must be nonnegative')
        self.hz = float(hz)
        self.next_due = None
        self.max_trials = int(max_trials)
        self.max_seconds = float(max_seconds)
        self.reset_seconds = float(reset_seconds)
        if task_profile not in ('pickplace', 'hold'):
            raise ValueError('unknown collection task profile')
        if not math.isfinite(hold_seconds) or hold_seconds < 0:
            raise ValueError('hold seconds must be nonnegative')
        self.task_profile = task_profile
        self.hold_seconds = float(hold_seconds)
        self.source_tolerance_m = float(source_tolerance_m)
        self.reset_state = None
        self.reset_count = 0
        self.ready_tcp = np.array([.38, -.06, .58])
        self.collection_context = None

    def describe(self):
        return dict(schema='openarm_dataset_capture_v1', target_camera_hz=self.hz,
                    timing='pre-physics RGB-D snapshot matched to saved control row',
                    perception_cadence_changed=False, max_trials=self.max_trials,
                    max_active_seconds=self.max_seconds,
                    inter_trial_reset_seconds=self.reset_seconds,
                    final_trial_notice_seconds=2.0,
                    automatic_scene_reset=True,
                    task_profile=self.task_profile,
                    required_payload_hold_seconds=self.hold_seconds,
                    return_to_source=self.task_profile == 'hold',
                    source_return_tolerance_m=self.source_tolerance_m,
                    collection_ready_tcp_m=self.ready_tcp.tolist(),
                    reset_scope='instant simulator environment reset between labelled trials; reset rows stay outside training segments',
                    filtering='raw accepted targets preserved; no posthoc action smoothing')

    def configure_task(self, task):
        """Expose collection-only attempt boundaries without changing normal launches."""
        task.trial_limit = self.max_trials
        task.inter_trial_reset_s = self.reset_seconds
        task.final_trial_notice_s = 2.0
        task.automatic_scene_reset = True
        task.collection_task_profile = self.task_profile
        task.collection_hold_seconds = self.hold_seconds
        task.collection_source_tolerance_m = self.source_tolerance_m

    def set_collection_context(self, context):
        self.collection_context = dict(context)

    def initialize_output(self, output):
        if self.collection_context is None:
            raise RuntimeError('collection context was not configured')
        path=Path(output)/'collection.json';pending=path.with_suffix('.json.pending')
        pending.write_text(json.dumps(self.collection_context,ensure_ascii=False,indent=2),encoding='utf-8')
        pending.replace(path)

    def configure_robot(self, robot):
        robot.ready_tcp = self.ready_tcp.copy()

    def bind_runtime(self, robot):
        self.reset_state = robot.capture_collection_reset_state()

    def reset_scene_if_due(self, task, robot, clock):
        if not getattr(task, 'scene_reset_pending', False):
            return False
        if self.reset_state is None:
            raise RuntimeError('collection reset state was not captured')
        robot.restore_collection_reset_state(self.reset_state)
        self.reset_count += 1
        task.scene_reset_completed(clock, self.reset_count)
        return True

    def create_writer(self, output):
        from shared_dataset import SharedFrameWriter

        class DatasetWriter(SharedFrameWriter):
            def close(writer):
                report = super().close()
                report['target_observation_hz'] = self.hz
                return report

        return DatasetWriter(output)

    def due(self, clock):
        if self.next_due is None:
            self.next_due = float(clock)
        if clock + 1e-9 < self.next_due:
            return False
        interval = 1 / self.hz
        missed = max(1, math.floor((clock - self.next_due + 1e-9) / interval) + 1)
        self.next_due += missed * interval
        return True

    def collect_frames(self, robot, clock, observed=False):
        if not self.due(clock):
            return {}
        if observed:
            frames = robot.observer.frames
            if len(frames) == 2 and all(abs(f['time_s'] - clock) < 1e-8 for f in frames.values()):
                return frames
        from calibration_simulation import render_rgb_depth
        frames = {}
        for camera, renderer in (
            ('chest_d435i_preview', robot.observer.chest),
            ('v22_right_d405_rgb', robot.observer.wrist),
        ):
            rgb, depth = render_rgb_depth(renderer, robot.data, camera)
            frames[camera] = dict(rgb=rgb.copy(), depth=depth.copy(), time_s=float(clock))
        return frames

    def should_stop(self, task, clock):
        completed = sum('end_s' in trial for trial in task.trials)
        timed_out = self.max_seconds and task.started is not None and clock - task.started >= self.max_seconds
        reached_limit = bool(self.max_trials and completed >= self.max_trials)
        if reached_limit and getattr(task, 'final_notice_until', None) is not None:
            reached_limit = clock >= task.final_notice_until
        return bool(reached_limit or timed_out)
