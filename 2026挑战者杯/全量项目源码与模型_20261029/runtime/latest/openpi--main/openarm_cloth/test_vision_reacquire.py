"""Exercise reset while real stream orchestration has inference in flight.

Detectors/camera are controlled fixtures, not evidence of human pose accuracy.
"""
from contextlib import ExitStack
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch
import threading
import time
import unittest

import numpy as np

from live_bimanual_stream import BimanualHumanStream


class VisionReacquireTests(unittest.TestCase):
    def test_single_arm_tracks_crop_audits_two_hands_and_recovers_on_loss(self):
        from mediapipe.tasks.python import vision
        frames=Queue();calls=[];created=[];missing=[False];crop_only=[False]
        class Camera:
            metadata={'synthetic_input':True}
            def __init__(self,*args):pass
            def read_new(self,after):
                seq=frames.get(timeout=5)
                if seq is None:raise EOFError()
                return seq,seq*50_000_000,seq*50_000_000,np.zeros((180,320,3),np.uint8)
            def close(self):frames.put(None)
        class Detector:
            def __init__(self,options,body=False):
                self.body=body;self.count=0 if body else options.num_hands;created.append(self.count)
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def detect_for_video(self,image,stamp):
                calls.append((self.count,stamp))
                if not self.body:
                    if missing[0] or crop_only[0] and self.count==2:return SimpleNamespace(hand_landmarks=[],hand_world_landmarks=[],handedness=[])
                    from test_human_gripper_gestures import hand
                    world=hand();norm=world.copy();norm[:,:2]+=[.25,.6]
                    p=lambda h:[SimpleNamespace(x=x,y=y,z=z) for x,y,z in h]
                    return SimpleNamespace(hand_landmarks=[p(norm)],hand_world_landmarks=[p(world)],
                        handedness=[[SimpleNamespace(category_name='Left',score=.9)]])
                norm=np.zeros((33,4));norm[:,3]=1.
                norm[11,:2]=[.65,.2];norm[12,:2]=[.35,.2]
                norm[13,:2]=[.7,.4];norm[14,:2]=[.3,.4]
                norm[15,:2]=[.75,.6];norm[16,:2]=[.25,.6]
                world=norm.copy();world[23,:3]=[-.1,.8,0];world[24,:3]=[.1,.8,0]
                p=lambda h:[SimpleNamespace(x=x,y=y,z=z,visibility=v) for x,y,z,v in h]
                return SimpleNamespace(pose_landmarks=[p(norm)],pose_world_landmarks=[p(world)])
        with ExitStack() as stack:
            stack.enter_context(patch.object(vision.PoseLandmarker,'create_from_options',side_effect=lambda o:Detector(o,True)))
            stack.enter_context(patch.object(vision.HandLandmarker,'create_from_options',side_effect=Detector))
            # Crop output is already mapped into full-image coordinates here;
            # the test exercises scheduling and real association/confirmation.
            from live_bimanual_stream import full_inference
            stack.enter_context(patch('live_bimanual_stream.roi_inference',side_effect=
                lambda mp,d,rgb,b,t,s:(full_inference(mp,d,mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),t),0.)))
            stream=BimanualHumanStream(camera_factory=Camera,expected_sides=('right',))
            def consume(seq):
                frames.put(seq);deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    if stream.error:self.fail(stream.error)
                    obs=stream.snapshot()
                    if obs is not None and obs['seq']==seq:return obs
                    time.sleep(.002)
                self.fail('fixture timed out')
            try:
                for seq in range(1,9):consume(seq)
                self.assertEqual(created,[0,2,1])
                self.assertTrue(stream.snapshot()['arms']['right']['hand_valid'])
                full_stamps=[stamp for count,stamp in calls if count==2]
                self.assertLess(len(full_stamps),8)
                self.assertGreaterEqual(len(full_stamps),3)
                self.assertTrue(any(count==1 for count,stamp in calls))
                missing[0]=True
                consume(9)
                self.assertFalse(stream.snapshot()['arms']['right']['hand_valid'])
                missing[0]=False
                consume(10)
                self.assertTrue(stream.snapshot()['hand_full_scan'])
                self.assertFalse(stream.snapshot()['arms']['right']['hand_valid'])
                consume(11)
                self.assertTrue(stream.snapshot()['arms']['right']['hand_valid'])
                # A distant hand can be invisible to the full-frame detector.
                # After a long loss, body-guided crops must remain available.
                missing[0]=True
                for seq in range(12,20):consume(seq)
                missing[0]=False;crop_only[0]=True
                consume(20);consume(21)
                self.assertTrue(stream.snapshot()['arms']['right']['hand_valid'])
                crop_only[0]=False
                # Explicit reset also discards the crop and restarts full search.
                stream.request_reacquire();frames.put(22)
                deadline=time.monotonic()+5
                while len(created)<6 and time.monotonic()<deadline:time.sleep(.002)
                consume(23)
                self.assertEqual(stream.snapshot()['vision_generation'],1)
                self.assertTrue(stream.snapshot()['hand_full_scan'])
                self.assertFalse(stream.snapshot()['arms']['right']['hand_valid'])
            finally:frames.put(None);stream.close()
            self.assertFalse(stream.thread.is_alive())

    def test_inflight_old_generation_cannot_republish_after_reacquire(self):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        frames=Queue();entered=threading.Event();release=threading.Event()
        models=[]

        class Camera:
            metadata={'synthetic_input':True}
            def __init__(self,*args):pass
            def read_new(self,previous):
                seq=frames.get(timeout=5)
                if seq is None:raise EOFError()
                return seq,seq*100_000_000,seq*100_000_000,np.zeros((180,320,3),np.uint8)
            def close(self):frames.put(None)

        class Detector:
            def __init__(self,body):
                self.body=body;self.closed=False;self.calls=0;models.append(self)
            def __enter__(self):return self
            def __exit__(self,*args):self.closed=True
            def detect_for_video(self,image,timestamp):
                if self.closed:raise AssertionError('closed detector reused')
                self.calls+=1
                if self.body and timestamp==200:
                    entered.set()
                    if not release.wait(5):raise AssertionError('fixture timed out')
                if not self.body:
                    return SimpleNamespace(hand_landmarks=[],hand_world_landmarks=[],handedness=[])
                norm=np.zeros((33,4));norm[:,3]=1.
                norm[11,:2]=[.65,.2];norm[12,:2]=[.35,.2]
                norm[13,:2]=[.7,.4];norm[14,:2]=[.3,.4]
                norm[15,:2]=[.75,.6];norm[16,:2]=[.25,.6]
                world=norm.copy();world[23,:3]=[-.1,.8,0];world[24,:3]=[.1,.8,0]
                world[:,:3]+=timestamp/1000.
                points=lambda w:[SimpleNamespace(x=x,y=y,z=z,visibility=v) for x,y,z,v in w]
                return SimpleNamespace(pose_landmarks=[points(norm)],pose_world_landmarks=[points(world)])

        stream=None
        def wait_for(predicate):
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                if stream and stream.error:self.fail(stream.error)
                if predicate():return
                time.sleep(.005)
            self.fail('stream fixture did not reach expected state')

        with ExitStack() as stack:
            stack.enter_context(patch.object(vision.PoseLandmarker,'create_from_options',side_effect=lambda _:Detector(True)))
            stack.enter_context(patch.object(vision.HandLandmarker,'create_from_options',side_effect=lambda _:Detector(False)))
            stream=BimanualHumanStream(camera_factory=Camera)
            try:
                frames.put(1);wait_for(lambda:stream.snapshot() is not None)
                self.assertEqual(stream.snapshot().get('vision_generation',0),0)
                frames.put(2);self.assertTrue(entered.wait(5))
                generation=stream.request_reacquire()
                self.assertEqual(generation,1)
                self.assertIsNone(stream.snapshot())
                # Repeated requests during the same recovery must coalesce.
                self.assertEqual(stream.request_reacquire(),generation)
                release.set()
                wait_for(lambda:len(models)==8)
                self.assertTrue(all(m.closed for m in models[:4]))
                self.assertIsNone(stream.snapshot())
                frames.put(3)
                wait_for(lambda:stream.snapshot() is not None)
                result=stream.snapshot()
                self.assertEqual(result['vision_generation'],1)
                self.assertEqual(result['seq'],3)
                # A fresh model must also have fresh smoothing/association state.
                np.testing.assert_allclose(result['pose_filtered'],result['pose_raw'])
                self.assertTrue(all(not a['hand_valid'] for a in result['arms'].values()))
            finally:
                release.set();frames.put(None);stream.close()
            self.assertFalse(stream.thread.is_alive())
            self.assertTrue(all(m.closed for m in models))

    def test_reacquire_pauses_and_resets_mapping_without_changing_pinch_endpoints(self):
        from live_bimanual_stream import apply_vision_reacquire
        from pose2arm_task_mapping import Pose2ArmTaskMapping
        mapper=Pose2ArmTaskMapping(sides=('right',))
        profile=mapper.pinch_calibrations['right'];mapper.fingers['right']=.012
        requested=[]
        stream=SimpleNamespace(request_reacquire=lambda:requested.append(1))
        flags=dict(reacquire=True,paused=False,calibrate='open')
        self.assertTrue(apply_vision_reacquire(flags,stream,mapper))
        self.assertTrue(flags['paused']);self.assertFalse(flags['reacquire'])
        self.assertIsNone(flags['calibrate']);self.assertEqual(requested,[1])
        self.assertIs(mapper.pinch_calibrations['right'],profile)
        self.assertEqual(mapper.fingers['right'],.012)
        self.assertFalse(apply_vision_reacquire(flags,stream,mapper))

    def test_reacquire_cancels_incomplete_pinch_trial_but_retains_accepted_data(self):
        from human_pinch_control import TaskHandControl
        channel=TaskHandControl(initial_aperture=.012)
        calibration=channel.calibration
        channel.begin_pinch_calibration('open')
        channel.reset()
        self.assertIsNone(channel.capture)
        self.assertIs(channel.calibration,calibration)
        self.assertEqual(channel.snapshot()['aperture'],.012)
        self.assertFalse(channel.snapshot()['valid'])

    def test_visible_wrist_with_invalid_hip_reference_is_not_body_missing(self):
        from human_grasp_task import HumanInput
        from run_human_task import tracking_diagnostic
        pose=np.ones((33,4));pose[23:25]=np.nan
        obs=dict(read_end=1_000_000_000,pose_filtered=pose)
        self.assertEqual(tracking_diagnostic(obs,HumanInput(1,None,None,False),1.1)[0],
                         'TORSO_REFERENCE_INVALID')
        pose[23:25]=1.
        self.assertEqual(tracking_diagnostic(obs,HumanInput(1,None,None,False),1.1)[0],
                         'TORSO_REFERENCE_DEGENERATE')

    def test_reset_cancels_a_queued_calibration_while_mapping_is_busy(self):
        from pose2arm_task_mapping import LatestPose2ArmWorker
        from unittest.mock import Mock
        worker=LatestPose2ArmWorker(sides=('right',))
        entered=threading.Event();release=threading.Event();reset_done=threading.Event()
        def busy(obs):
            entered.set()
            if not release.wait(5):raise TimeoutError('mapping fixture timed out')
        reset=worker.mapping.reset
        def reset_mapping():reset();reset_done.set()
        worker.mapping.process=busy;worker.mapping.reset=reset_mapping
        worker.mapping.begin_pinch_calibration=Mock()
        try:
            worker.submit(dict(seq=1,read_end=1_000_000_000))
            self.assertTrue(entered.wait(5))
            worker.begin_pinch_calibration('open');worker.reset();release.set()
            self.assertTrue(reset_done.wait(5))
        finally:
            release.set();worker.close()
        worker.mapping.begin_pinch_calibration.assert_not_called()
        self.assertIsNone(worker.error)

    def test_resuming_after_reacquire_rebases_arm_and_holds_partial_aperture(self):
        from human_grasp_task import DirectHumanTask,Feedback,HumanInput
        from run_human_task import direct_hand_command
        task=DirectHumanTask();feedback=Feedback(np.array([.43,0.,.49]),
            visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.))
        for i in range(8):
            task.update(HumanInput(i,np.zeros(3),1.2,True,i*.05),feedback,i*.05,start=True)
        self.assertEqual(task.stage,'direct')
        # R's pause must hold a partially closed gripper, then discard the
        # sitting-to-standing displacement when the operator resumes.
        hold=task.update(None,feedback,.4,paused=True)
        command=direct_hand_command(hold,None,task.stage,.012,paused=True)
        self.assertEqual(command.stage,'hold');self.assertEqual(command.finger,.012)
        command=task.update(HumanInput(9,np.ones(3),None,True,.45),feedback,.45)
        np.testing.assert_allclose(command.target,feedback.tcp)


if __name__=='__main__':unittest.main()
