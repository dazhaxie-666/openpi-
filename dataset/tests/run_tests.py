from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import os
import sys
import tempfile
import unittest

root = Path(__file__).resolve().parents[1]
project = root.parent / 'latest' / 'openpi--main'
if not (project / 'openarm_cloth').is_dir():
    project = root.parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(root / 'scripts'), str(project / 'openarm_cloth')]
(root / 'tmp').mkdir(exist_ok=True)
(root / 'logs').mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
tempfile.tempdir = str(root / 'tmp')
from dataset_cli import Tee

log = root / 'logs' / ('tests_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.log')
with log.open('w', encoding='utf-8') as stream:
    with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_*.py')
        # Existing default entry and current human/robot workflow remain covered.
        suite.addTests(unittest.defaultTestLoader.loadTestsFromNames([
            'test_shared_grasp', 'test_shared_workflow', 'test_recorded_camera_index',
            'test_single_arm_runtime', 'test_shared_interaction_ui', 'test_shared_manual_intent',
        ]))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
print('Test log:', log)
raise SystemExit(0 if result.wasSuccessful() else 1)
