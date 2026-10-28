from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dataset_cli


class BatchTests(unittest.TestCase):
    def test_numbered_batches_have_distinct_paths_under_same_stage(self):
        paths = [dataset_cli.stage_output('core', batch)[1] for batch in (1, 2, 3)]
        self.assertEqual([p.parent.name for p in paths], ['batch_001', 'batch_002', 'batch_003'])
        self.assertEqual(len({p.parent.parent for p in paths}), 1)

    def test_repeated_collection_in_same_batch_cannot_overwrite_a_session(self):
        with patch.object(dataset_cli, 'stamp', return_value='same_timestamp'):
            first = dataset_cli.stage_output('core', 1)[1]
            second = dataset_cli.stage_output('core', 1)[1]
        self.assertEqual(first.parent, second.parent)
        self.assertNotEqual(first, second)

    def test_invalid_batch_numbers_cannot_form_a_path(self):
        for value in (0, -1, 1000000, True, '../2'):
            with self.assertRaises(ValueError):
                dataset_cli.batch_label(value)

    def test_legacy_reports_retain_their_original_path(self):
        context = dict(dataset_name='openarm_pickplace_p02_core100_v1')
        legacy = dataset_cli.workspace_dir('reports', context)
        self.assertEqual(legacy, dataset_cli.ROOT / 'reports' / context['dataset_name'])
        self.assertEqual(dataset_cli.workspace_dir('reports', dict(context, batch_id=2)), legacy / 'batch_002')

    def test_summary_counts_legacy_and_numbered_sessions_once(self):
        config = dataset_cli.configuration()
        with tempfile.TemporaryDirectory() as temp, patch.object(dataset_cli, 'ROOT', Path(temp)), \
                patch.object(dataset_cli, 'configuration', return_value=config):
            name = dataset_cli.configuration()['stages']['core']['name']
            for label, batch_id, candidate in [('old', None, 1), ('new', 2, 2)]:
                context = dict(dataset_name=name)
                if batch_id is not None:
                    context['batch_id'] = batch_id
                session = dataset_cli.workspace_dir('raw', context) / label
                dataset_cli.write_json(session / 'collection.json', context)
                dataset_cli.write_json(dataset_cli.workspace_dir('reports', context) / (label + '.audit.json'),
                                       dict(trials=[{}, {}], candidate_successes=candidate, training_approved_successes=0))
            with redirect_stdout(io.StringIO()):
                dataset_cli.summary()
            report = json.loads(next((Path(temp) / 'reports').glob('summary_*.json')).read_text(encoding='utf-8'))
            core = next(s for s in report if s['dataset_name'] == name)
            self.assertEqual(core['sessions'], 2)
            self.assertEqual(core['trials'], 4)
            self.assertEqual(core['candidate_successes'], 3)
            self.assertEqual(core['batches']['legacy_unbatched']['sessions'], 1)
            self.assertEqual(core['batches']['batch_002']['sessions'], 1)
            self.assertEqual(core['missing_audits'], [])


if __name__ == '__main__':
    unittest.main()
