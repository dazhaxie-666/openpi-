import unittest
import numpy as np
from shared_grasp import SharedGraspTask
from human_grasp_task import Feedback
from shared_interaction_ui import shared_interaction_hint
from human_task_preview import compose_task_preview


class SharedInteractionUITests(unittest.TestCase):
    def setUp(self):
        self.task=SharedGraspTask();self.task.stage='direct';self.task.rearm=True;self.task.hand_intent_valid=True
        self.task.request_ratio=1.2
        self.feedback=Feedback(np.array([.43,0,.50]),visual=dict(center=np.array([.43,0,.429]),yaw=0.,time_s=1.))

    def test_green_ready_prompt_matches_grasp_entry_conditions(self):
        status=self.task.grasp_readiness(self.feedback,1.)
        self.assertTrue(status['ready'])
        cue=shared_interaction_hint(self.task,self.feedback,1.)
        self.assertEqual(cue['code'],'grasp_ready');self.assertEqual(cue['theme'],'ready')
        self.assertIn('捏合',cue['title'])

    def test_in_range_prompt_needs_no_extra_open_preparation_but_still_needs_hand_and_view(self):
        self.feedback.tcp[2]=.55
        self.assertEqual(shared_interaction_hint(self.task,self.feedback,1.)['code'],'approach')
        self.feedback.tcp[2]=.50;self.task.rearm=False
        self.assertEqual(shared_interaction_hint(self.task,self.feedback,1.)['code'],'grasp_ready')
        self.assertNotIn('张开右手准备',shared_interaction_hint(self.task,self.feedback,1.)['title'])
        self.task.rearm=True;self.task.hand_intent_valid=False
        self.assertEqual(shared_interaction_hint(self.task,self.feedback,1.)['code'],'show_hand')
        self.task.hand_intent_valid=True;self.feedback.visual=None
        self.assertEqual(shared_interaction_hint(self.task,self.feedback,1.)['code'],'target_hidden')
        self.assertEqual(shared_interaction_hint(self.task,self.feedback,1.,paused=True)['code'],'paused')

    def test_carry_prompt_and_open_progress_do_not_require_another_pinch(self):
        self.task.phase='transport';self.task.grasp_confirmed=True;self.task.release_armed=True
        cue=shared_interaction_hint(self.task,self.feedback,1.)
        self.assertEqual(cue['code'],'carry_ready');self.assertIn('无需再次捏合',cue['detail'])
        self.task.open_since=10.;self.task.open_stamp=10.175
        cue=shared_interaction_hint(self.task,self.feedback,1.)
        self.assertEqual(cue['code'],'release_confirming');self.assertAlmostEqual(cue['progress'],.5)

    def test_large_prompt_is_below_camera_and_does_not_cover_landmarks(self):
        camera=np.full((540,960,3),70,np.uint8)
        cue=shared_interaction_hint(self.task,self.feedback,1.)
        canvas=compose_task_preview(camera,[],'direct',interaction_hint=cue)
        # Existing panel divider straddles x=960; compare the camera interior.
        np.testing.assert_array_equal(canvas[:540,:958],camera[:,:958])
        self.assertGreater(np.std(canvas[548:636,16:944]),10.)
        self.assertGreater(canvas[555,25,1],canvas[555,25,2])

    def test_failed_place_is_not_described_as_missing_hand_or_uncertain_grip(self):
        self.task.phase='transport';self.task.placement_blocked_origin=self.feedback.tcp.copy()
        self.task.hand_intent_valid=False
        cue=shared_interaction_hint(self.task,self.feedback,1.)
        self.assertEqual(cue['code'],'place_blocked');self.assertIn('恢复搬运',cue['title'])

    def test_async_camera_refresh_keeps_the_large_prompt(self):
        from unittest.mock import patch
        from async_task_preview import LatestTaskPreview
        preview=LatestTaskPreview('fixture',render=lambda payload,raw:None);preview.close()
        hint=shared_interaction_hint(self.task,self.feedback,1.)
        payload=dict(lines=[],stage='direct',interaction_hint=hint,mirror=False)
        with patch('async_task_preview.cv2.imshow') as show,patch('async_task_preview.cv2.pollKey',return_value=-1),\
             patch('async_task_preview.cv2.getWindowProperty',return_value=1.):
            preview._draw(payload,(1,0,0,np.full((540,960,3),60,np.uint8)))
        canvas=show.call_args.args[1]
        self.assertGreater(canvas[555,25,1],canvas[555,25,2])
        self.assertGreater(np.std(canvas[548:636,16:944]),10.)


if __name__=='__main__':unittest.main()
