"""The real preview mailbox must not wait for or queue old control frames."""
import threading
import unittest
import time
from unittest.mock import patch
import numpy as np

from async_task_preview import LatestTaskPreview,draw_tracking_overlay


class AsyncPreviewTests(unittest.TestCase):
    def test_old_auxiliary_pose_expires_even_with_fresh_hand_observation(self):
        pose=np.full((33,4),np.nan)
        pose[[12,14,16]]=[[.2,.2,0,1],[.4,.4,0,1],[.6,.2,0,1]]
        hand=np.full((21,3),.7)
        overlay=dict(read_end=1_250_000_000,pose_read_end=1_000_000_000,pose_norm=pose,
            sides=['right'],position_trusted={'right':True},arms={'right':dict(hand_valid=True,hand_norm=hand)})
        camera=np.zeros((540,960,3),np.uint8)
        draw_tracking_overlay(camera,overlay,False,1_280_000_000)
        np.testing.assert_array_equal(camera[162,288],[0,0,0])
        self.assertTrue(camera[378,672].any())

    def test_valid_wrist_cannot_paint_rejected_elbow_chain_green(self):
        pose=np.full((33,4),np.nan)
        pose[[12,14,16]]=[[.2,.2,0,1],[.4,.4,0,1],[.6,.2,0,1]]
        overlay=dict(read_end=1_000_000_000,pose_norm=pose,sides=['right'],
            position_trusted={'right':True},arm_trusted={'right':False},arms={'right':{}})
        camera=np.zeros((540,960,3),np.uint8)
        draw_tracking_overlay(camera,overlay,False,1_050_000_000)
        np.testing.assert_array_equal(camera[162,288],[130,130,130])

    def test_rejected_estimate_is_grey_and_mirror_applies_to_landmarks(self):
        pose=np.full((33,4),np.nan)
        pose[[12,14,16]]=[[.2,.2,0,1],[.4,.4,0,1],[.6,.2,0,1]]
        overlay=dict(read_end=1_000_000_000,pose_norm=pose,sides=['right'],
            position_trusted={'right':False},arms={'right':dict(hand_valid=False,hand_norm=np.zeros((21,3)))})
        camera=np.zeros((540,960,3),np.uint8)
        status=draw_tracking_overlay(camera,overlay,True,1_050_000_000)
        np.testing.assert_array_equal(camera[162,672],[130,130,130])
        np.testing.assert_array_equal(camera[162,288],[0,0,0])
        self.assertIn('WRIST LOST HAND LOST',status)

    def test_new_raw_frame_keeps_recent_landmarks_and_expired_overlay_disappears(self):
        preview=LatestTaskPreview('fixture',render=lambda p,r:None);preview.close()
        pose=np.full((33,4),np.nan)
        pose[[12,14,16]]=[[.2,.2,0,1],[.4,.4,0,1],[.6,.2,0,1]]
        hand=np.full((21,3),.6)
        payload=dict(lines=[],stage='direct',source_seq=10,mirror=False,
            image=np.zeros((540,960,3),np.uint8),overlay=dict(read_end=1_000_000_000,
                pose_norm=pose,sides=['right'],position_trusted={'right':True},
                arms={'right':dict(hand_valid=True,hand_norm=hand)}))
        raw=np.zeros((540,960,3),np.uint8)
        with patch('async_task_preview.cv2.imshow') as show,patch('async_task_preview.cv2.pollKey',return_value=-1),patch(
                'async_task_preview.cv2.getWindowProperty',return_value=1.),patch(
                'async_task_preview.time.perf_counter_ns',return_value=1_080_000_000):
            preview._draw(payload,(11,0,1_060_000_000,raw))
            frame=show.call_args.args[1]
            np.testing.assert_array_equal(frame[162,288], [0,255,0])
            self.assertIn('80',preview.overlay_status)
        with patch('async_task_preview.cv2.imshow') as show,patch('async_task_preview.cv2.pollKey',return_value=-1),patch(
                'async_task_preview.cv2.getWindowProperty',return_value=1.),patch(
                'async_task_preview.time.perf_counter_ns',return_value=1_300_000_000):
            preview._draw(payload,(12,0,1_290_000_000,raw))
            np.testing.assert_array_equal(show.call_args.args[1][162,288], [0,0,0])
            self.assertIn('STALE',preview.overlay_status)
        self.assertFalse(raw.any())

    def test_blocked_display_does_not_block_control_and_uses_latest_payload(self):
        entered=threading.Event();release=threading.Event();seen=[];second=threading.Event()
        def draw(payload,raw):
            seen.append((payload['value'],raw[0]))
            if len(seen)==1:entered.set();release.wait(3)
            else:second.set()
        raw=[1]
        preview=LatestTaskPreview('fixture',hz=30,frame_provider=lambda:(raw[0],0,0,None),render=draw)
        try:
            preview.submit(dict(value=1))
            self.assertTrue(entered.wait(3))
            for value in range(2,101):preview.submit(dict(value=value))
            raw[0]=2;release.set();self.assertTrue(second.wait(3))
        finally:release.set();preview.close()
        self.assertEqual(seen[1],(100,2))
        self.assertIsNone(preview.error)

    def test_camera_refreshes_without_another_control_update(self):
        seen=[];ready=threading.Event()
        def provider():return (len(seen)+1,0,0,None)
        def draw(payload,raw):
            seen.append(raw[0])
            if len(seen)>=3:ready.set()
        preview=LatestTaskPreview('fixture',hz=30,frame_provider=provider,render=draw)
        try:
            preview.submit(dict(value=1));self.assertTrue(ready.wait(3))
        finally:preview.close()
        self.assertGreaterEqual(len(set(seen)),3)
        self.assertGreaterEqual(preview.report()['distinct_camera_frames'],3)

    def test_display_errors_reach_control_instead_of_silently_freezing(self):
        def fail(payload,raw):raise RuntimeError('display failed')
        preview=LatestTaskPreview('fixture',render=fail)
        try:
            preview.submit(dict(value=1));preview.thread.join(3)
            self.assertIn('display failed',preview.error)
        finally:preview.close()

    def test_status_panel_is_reused_while_camera_frames_continue(self):
        import async_task_preview as module
        preview=LatestTaskPreview('fixture',render=lambda p,r:None)
        preview.close()
        payload=dict(lines=['ready'],stage='ready')
        first=np.zeros((180,320,3),np.uint8);second=first.copy()
        with patch.object(module,'compose_task_preview',wraps=module.compose_task_preview) as compose,patch(
                'async_task_preview.cv2.imshow'),patch('async_task_preview.cv2.pollKey',return_value=-1),patch(
                'async_task_preview.cv2.getWindowProperty',return_value=1.):
            preview._draw(payload,(1,0,0,first));preview._draw(payload,(2,0,0,second))
            self.assertEqual(compose.call_count,1)
            preview._draw(dict(payload,paused=True),(2,0,0,second))
            self.assertEqual(compose.call_count,2)


if __name__=='__main__':unittest.main()
