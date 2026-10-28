import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dataset_cli import configuration
from dataset_quality import audit_recording, recording_digest


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.settings = configuration()['quality']
        self.manifest = dict(finished=True, synthetic_input=False, shared_segment_schema='full_task_v2',
                             robot_qpos_index=dict(right=list(range(7))), finger_qpos_index=dict(right=[7, 8]),
                             goal=dict(center_xy_m=[.43, -.1]))
        self.context = dict(dataset_name='test', stage='core', training_allowed=True)
        self.trial = dict(trial_id=1, start_frame=0, end_frame_exclusive=21, status='completed',
                          complete_task_segment=True, eligible_success_segment=True,
                          physics=dict(physical_task_success=True))
        action = np.zeros((21, 18)); action[:, 16:18] = .044
        self.rows = dict(time_s=np.arange(21) / 10, paused=np.zeros(21, bool), qpos=np.zeros((21, 16)),
                         action_target=action, shared_phase=np.full(21, 'transport'),
                         motion_warning=np.full(21, ''), evaluation_next=np.tile([.43, 0, .429], (21, 1)))
        self.samples = [dict(camera=c, control_frame=i, time_s=i / 10)
                        for c in ('chest_d435i_preview', 'v22_right_d405_rgb') for i in range(21)]
        self.save()

    def save(self):
        for name, value in [('manifest.json', self.manifest), ('collection.json', self.context),
                            ('shared_trials.json', [self.trial])]:
            (self.path / name).write_text(json.dumps(value), encoding='utf-8')
        np.savez(self.path / 'episode.npz', **self.rows)
        (self.path / 'shared_frames.jsonl').write_text('\n'.join(json.dumps(x) for x in self.samples), encoding='utf-8')

    def audit(self):
        return audit_recording(self.path, self.settings, validate_fn=lambda _: dict(
            recording_valid=True, errors=[], warnings=[]))

    def accept(self):
        report = self.audit()
        (self.path / 'review.json').write_text(json.dumps({'1': dict(decision='accept',
            recording_digest=report['recording_digest'], quality_contract_digest=report['quality_contract_digest'])}))
        (self.path / 'raw_action_replay.json').write_text(json.dumps(dict(verdict='PASS', execution='actions',
            physical_actuator_replay=True, recording_digest=recording_digest(self.path))))

    def test_candidate_is_not_training_approved_without_review_and_replay(self):
        report = self.audit()
        self.assertEqual(report['candidate_successes'], 1)
        self.assertEqual(report['training_approved_successes'], 0)
        self.accept()
        self.assertEqual(self.audit()['training_approved_successes'], 1)

    def test_pilot_and_holdout_cannot_enter_training_even_after_visual_review(self):
        for stage in ('pilot', 'eval'):
            self.context.update(stage=stage, training_allowed=False)
            self.save(); self.accept()
            self.assertEqual(self.audit()['training_approved_successes'], 0)

    def test_synthetic_never_becomes_human_candidate(self):
        self.manifest['synthetic_input'] = True
        self.save(); self.accept()
        report = self.audit()
        self.assertEqual(report['candidate_successes'], 0)
        self.assertEqual(report['training_approved_successes'], 0)

    def test_changed_raw_data_invalidates_prior_approval_and_replay(self):
        self.accept()
        self.rows['qpos'][4, 0] = .001
        self.save()
        report = self.audit()
        self.assertFalse(report['raw_action_replay_verified'])
        self.assertEqual(report['training_approved_successes'], 0)

    def test_long_image_gap_and_pause_reject_otherwise_successful_episode(self):
        self.rows['paused'][3] = True
        self.samples = [x for x in self.samples if not 5 <= x['control_frame'] <= 10]
        self.save()
        report = self.audit()
        self.assertEqual(report['candidate_successes'], 0)
        issues = report['trials'][0]['issues']
        self.assertTrue(any('pause' in x for x in issues))
        self.assertTrue(any('image gap' in x for x in issues))

    def test_already_in_goal_is_not_a_new_transfer_demonstration(self):
        self.rows['evaluation_next'][:, :2] = [.43, -.1]
        self.save()
        self.assertEqual(self.audit()['candidate_successes'], 0)


if __name__ == '__main__':
    unittest.main()
