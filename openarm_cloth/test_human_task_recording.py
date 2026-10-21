"""Integrity checks reject broken images/labels; action replay is independent."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from member_b_actuators import ActuatorController
from member_b_scene import build_scene
from replay_human_task import replay
from validate_human_task import validate


class RecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.initial = build_scene()

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name)
        model = self.model
        data = mujoco.MjData(model)
        data.qpos[:] = self.initial.qpos
        mujoco.mj_forward(model, data)
        control = ActuatorController(model, data)
        control.set_targets({s: data.qpos[spec.qpos].copy() for s, spec in control.specs.items()},
                            dict(left=.044, right=.044))
        # Capture between physical steps, with B's preceding-step gravity cache
        # and solver warmstart, as an actual task recording does.
        for _ in range(200):
            control.apply();mujoco.mj_step(model, data)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        records = []
        for i in range(3):
            control.apply()
            state = np.empty(mujoco.mj_stateSize(model, spec))
            mujoco.mj_getState(model, data, state, spec)
            records.append(dict(time_s=np.array(i * .05), sim_time_s=np.array(data.time),
                                qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                                action_target=control.action.copy(), physics_state=state,
                                human_valid=np.array(False), human_point=np.full(3, np.nan),
                                pinch_ratio=np.array(np.nan), paused=np.array(False)))
            for _ in range(round(.05 / model.opt.timestep)):
                control.apply();mujoco.mj_step(model, data)
        np.savez_compressed(self.path / 'episode.npz', **{k: np.stack([r[k] for r in records]) for k in records[0]})
        mujoco.mj_saveModel(model, str(self.path / 'scene.mjb'))
        self.manifest = dict(schema='openarm_human_task_v1', nq=model.nq, nv=model.nv, nu=model.nu,
                             physics_state_size=len(state), physics_state_spec=int(spec),
                             mujoco_version=mujoco.__version__, cameras=[], image_size=[320, 240],
                             frames=3, control_hz=20, model_file='scene.mjb',
                             model_sha256=hashlib.sha256((self.path / 'scene.mjb').read_bytes()).hexdigest(),
                             source='synthetic_software_fixture', synthetic_input=True, finished=True)
        self.write_manifest()
        (self.path / 'result.json').write_text(json.dumps(dict(frames=3, verdict='FAIL', synthetic_input=True,
                                                             human_validation_performed=False)))
        (self.path / 'events.json').write_text('[]')

    def write_manifest(self):
        (self.path / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_failed_episode_can_be_structurally_valid_but_never_human_training_data(self):
        report = validate(self.path)
        self.assertTrue(report['recording_valid'], report)
        self.assertFalse(report['candidate_for_human_bc_review'])
        self.assertFalse(report['task_success'])

    def test_missing_robot_images_are_detected(self):
        self.manifest['cameras'] = ['chest_d435i_preview']
        self.write_manifest()
        report = validate(self.path)
        self.assertFalse(report['recording_valid'])
        self.assertIn('chest_d435i_preview: missing/extra images', report['errors'])

    def test_changed_snapshot_cannot_be_silently_replayed(self):
        self.manifest['model_sha256'] = 'invalid'
        self.write_manifest()
        self.assertFalse(validate(self.path)['recording_valid'])
        with self.assertRaisesRegex(ValueError, 'model snapshot hash mismatch'):
            replay(self.path, check=True)

    def test_synthetic_episode_cannot_claim_human_validation(self):
        (self.path / 'result.json').write_text(json.dumps(dict(frames=3, verdict='FAIL', synthetic_input=True,
                                                             human_validation_performed=True)))
        self.assertFalse(validate(self.path)['recording_valid'])

    def test_independent_dynamic_action_replay(self):
        report = replay(self.path, execution='actions', check=True)
        self.assertEqual(report['verdict'], 'PASS', report)
        self.assertEqual(report['max_qpos_error'], 0.)
        self.assertEqual(report['max_qvel_error'], 0.)
        self.assertTrue(report['physical_actuator_replay'])
        self.assertFalse(report['human_validation_performed'])


if __name__ == '__main__':
    unittest.main()
