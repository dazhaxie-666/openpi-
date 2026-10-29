"""A completed/failed task stays inspectable without executing any more motion."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import run_human_task


class TaskResultViewTests(unittest.TestCase):
    def test_terminal_result_stays_open_until_user_quits_and_never_steps_physics(self):
        for verdict, reason in (('FAIL', 'task_motion_collision_rejected:follow'),
                                ('PASS', None)):
            with self.subTest(verdict=verdict):
                view = MagicMock()
                view.cam = SimpleNamespace(lookat=np.zeros(3))
                view.is_running.return_value = True
                context = MagicMock()
                context.__enter__.return_value = view
                with (patch('mujoco.viewer.launch_passive', return_value=context),
                      patch('run_human_task.cv2.namedWindow'),
                      patch('run_human_task.cv2.resizeWindow'),
                      patch('run_human_task.cv2.imshow') as display,
                      patch('run_human_task.cv2.waitKey', side_effect=[32, ord('q')]) as keys,
                      patch('run_human_task.cv2.getWindowProperty', return_value=1),
                      patch('run_human_task.cv2.destroyWindow'),
                      patch('run_human_task.time.sleep'),
                      patch('mujoco.mj_step') as step):
                    run_human_task.show_task_result(
                        dict(verdict=verdict, reason=reason, error=None), object(), object())
                self.assertEqual(keys.call_count, 2)
                self.assertEqual(display.call_count, 2)
                step.assert_not_called()


if __name__ == '__main__':
    unittest.main()
