import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from capture_extension import DatasetCapture
from dataset_cli import configuration, resolve_session, source_hashes
from dataset_quality import camera_metrics, manual_motion_metrics, robot_arrays


class CaptureTests(unittest.TestCase):
    def test_preflight_hashes_cover_visual_fusion_and_task_state_machine(self):
        hashes = source_hashes()
        self.assertTrue({
            'latest/openpi--main/openarm_cloth/b_scene_fusion.py',
            'latest/openpi--main/openarm_cloth/human_grasp_task.py',
        } <= set(hashes))

    def test_sampling_has_no_drift_or_duplicate_paused_images(self):
        capture = DatasetCapture(10)
        chosen = [i for i in range(301) if capture.due(i / 30)]
        self.assertEqual(chosen, list(range(0, 301, 3)))
        self.assertFalse(capture.due(10.0))
        self.assertTrue(capture.due(11.0))
        self.assertFalse(capture.due(11.0))

    def test_fresh_perception_images_are_reused_without_new_visual_estimation(self):
        capture = DatasetCapture(10)
        frames = {name: dict(time_s=0.) for name in
            ('chest_d435i_preview', 'v22_left_d405_rgb', 'v22_right_d405_rgb')}
        robot = SimpleNamespace(observer=SimpleNamespace(frames=frames))
        self.assertIs(capture.collect_frames(robot, 0., observed=True), frames)
        self.assertEqual(capture.collect_frames(robot, .01, observed=False), {})

    def test_extra_snapshot_never_calls_perception_and_has_current_time(self):
        rgb, depth = np.zeros((4, 5, 3), np.uint8), np.ones((4, 5), np.float32)
        robot = SimpleNamespace(data=object(), observer=SimpleNamespace(chest='a', wrist='b'))
        with patch('calibration_simulation.render_rgb_depth', return_value=(rgb, depth)) as render:
            frames = DatasetCapture(10).collect_frames(robot, 1., observed=False)
        self.assertEqual(render.call_count, 3)
        self.assertEqual(set(frames), {'chest_d435i_preview', 'v22_left_d405_rgb', 'v22_right_d405_rgb'})
        self.assertTrue(all(f['time_s'] == 1. for f in frames.values()))

    def test_attempt_limit_includes_failure_but_not_active_trial(self):
        capture = DatasetCapture(10, max_trials=1)
        task = SimpleNamespace(trials=[dict(status='failed')], started=0.)
        self.assertFalse(capture.should_stop(task, 1.))
        task.trials[0]['end_s'] = 1.
        self.assertTrue(capture.should_stop(task, 1.))

    def test_attempt_limit_waits_for_final_operator_notice(self):
        capture = DatasetCapture(10, max_trials=1, reset_seconds=8)
        task = SimpleNamespace(trials=[dict(status='completed', end_s=1.)], started=0.,
                               final_notice_until=3.)
        self.assertFalse(capture.should_stop(task, 2.9))
        self.assertTrue(capture.should_stop(task, 3.))

    def test_collection_configures_visible_trial_and_reset_contract(self):
        capture = DatasetCapture(10, max_trials=10, reset_seconds=3, task_profile='hold', hold_seconds=3)
        task = SimpleNamespace()
        capture.configure_task(task)
        self.assertEqual(task.trial_limit, 10)
        self.assertEqual(task.inter_trial_reset_s, 3.)
        self.assertEqual(task.final_trial_notice_s, 2.)
        self.assertTrue(task.automatic_scene_reset)
        self.assertEqual(task.collection_task_profile, 'hold')
        self.assertEqual(task.collection_hold_seconds, 3.)

    def test_collection_reset_uses_captured_state_and_farther_ready_tcp(self):
        capture = DatasetCapture(10, max_trials=10, reset_seconds=8)
        robot = SimpleNamespace(ready_tcp=None, restored=None,
            capture_collection_reset_state=lambda: {'state': 'initial'})
        robot.restore_collection_reset_state=lambda state:setattr(robot,'restored',state)
        task = SimpleNamespace(scene_reset_pending=True, completed=None,
            scene_reset_completed=lambda clock,count:setattr(task,'completed',(clock,count)))
        capture.configure_robot(robot);capture.bind_runtime(robot)
        self.assertGreater(np.linalg.norm(robot.ready_tcp-np.array([.43,0.,.429])),.15)
        self.assertTrue(capture.reset_scene_if_due(task,robot,12.))
        self.assertEqual(robot.restored,{'state':'initial'})
        self.assertEqual(task.completed,(12.,1))

    def test_collection_identity_is_written_when_runtime_output_is_created(self):
        capture=DatasetCapture(10);capture.set_collection_context({'dataset_name':'pilot'})
        with tempfile.TemporaryDirectory() as temp:
            capture.initialize_output(temp)
            saved=json.loads((Path(temp)/'collection.json').read_text(encoding='utf-8'))
        self.assertEqual(saved,{'dataset_name':'pilot'})

    def test_writer_reports_capture_rate_and_depth_units(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = DatasetCapture(10).create_writer(Path(temp))
            writer.submit(0, {'c': dict(time_s=0., rgb=np.zeros((3, 4, 3), np.uint8), depth=np.ones((3, 4)))})
            report = writer.close()
            self.assertTrue(report['valid'])
            self.assertEqual(report['target_observation_hz'], 10)
            sample = json.loads((Path(temp) / 'shared_frames.jsonl').read_text())
            with np.load(Path(temp) / sample['depth']) as blob:
                self.assertEqual(blob['depth_m'].dtype, np.float32)


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.settings = configuration()['quality']

    def test_robot_extraction_cannot_include_object_truth_or_torque(self):
        qpos = np.full((2, 30), 9999.)
        qpos[:, :7] = np.arange(7)
        qpos[:, 7:9] = [.044, .044]
        target = np.zeros((2, 18))
        target[:, 9:16] = np.arange(7)
        target[:, 16:18] = [[.044, .044], [0., 0.]]
        state, actions = robot_arrays(dict(qpos=qpos, action_target=target),
             dict(robot_qpos_index=dict(right=list(range(7))), finger_qpos_index=dict(right=[7, 8])))
        self.assertEqual(state.shape, (2, 8))
        np.testing.assert_array_equal(actions[:, -1], [0., 1.])
        self.assertLess(float(state.max()), 9999)

    def test_independent_finger_targets_are_not_silently_averaged(self):
        target = np.zeros((2, 18)); target[:, 17] = .02
        with self.assertRaisesRegex(ValueError, 'independent finger'):
            robot_arrays(dict(qpos=np.zeros((2, 9)), action_target=target),
                         dict(robot_qpos_index=dict(right=list(range(7))), finger_qpos_index=dict(right=[7, 8])))

    def test_five_hz_data_cannot_pass_ten_hz_capture_contract(self):
        samples = [dict(camera=c, control_frame=i * 2, time_s=i * .2)
                   for c in ('chest_d435i_preview', 'v22_left_d405_rgb', 'v22_right_d405_rgb') for i in range(11)]
        _, errors = camera_metrics(samples, np.arange(21) / 10, 0, 21, self.settings)
        self.assertTrue(any('rate below' in e for e in errors))

    def test_jitter_diagnostic_excludes_automatic_phase_motion(self):
        n = 101
        rows = dict(time_s=np.arange(n) / 10, shared_phase=np.full(n, 'close'),
                    action_target=np.tile(np.arange(n)[:, None] % 2, (1, 18)), motion_warning=np.full(n, ''))
        metrics = manual_motion_metrics(rows, 0, n, self.settings)
        self.assertEqual(metrics['target_joint_speed_max_rad_s'], 0.)
        self.assertEqual(metrics['review_flags'], [])
        rows['shared_phase'] = np.full(n, 'manual')
        self.assertGreater(manual_motion_metrics(rows, 0, n, self.settings)['reversal_frames'], 0)

    def test_stage_names_and_training_roles_are_distinct(self):
        config = configuration();stages = config['stages'];hold_stages=config['hold_stages']
        all_stages=[*stages.values(),*hold_stages.values()]
        self.assertEqual(len({s['name'] for s in all_stages}), len(all_stages))
        for stage in ('smoke', 'pilot', 'eval'):
            self.assertFalse(stages[stage]['training_allowed'])
        self.assertEqual(sum(stages[k]['target_successes'] for k in ('core', 'coverage', 'recovery')), 300)
        self.assertEqual(hold_stages['pilot']['target_successes'], 5)
        self.assertEqual(hold_stages['core']['target_successes'], 30)
        self.assertEqual(hold_stages['eval']['target_successes'], 5)

    def test_reversal_check_handles_repeated_targets_between_camera_updates(self):
        target = np.zeros((8, 18))
        target[:, 9] = [0, .01, .01, 0, 0, .01, .01, 0]
        rows = dict(time_s=np.arange(8) / 30, shared_phase=np.full(8, 'manual'),
                    action_target=target, motion_warning=np.full(8, ''))
        self.assertEqual(manual_motion_metrics(rows, 0, 8, self.settings)['reversal_frames'], 3)

    def test_external_sessions_cannot_be_modified_by_review(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'dataset/raw'):
                resolve_session(temp)


if __name__ == '__main__':
    unittest.main()
