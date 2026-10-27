"""Image-only arm input must not fabricate 3D or let elbow errors freeze a wrist."""
import unittest
import numpy as np
from human_body_2d import image_plane_wrist, map_coco_pose,same_person_image
from live_observation_quality import body_hand_consistency
from live_bimanual_stream import roi_bounds
from human_grasp_task import HumanInput
from human_palm_depth import RelativePalmDepth
from run_human_task import palm_depth_input
from test_human_palm_depth import sample


class ImageBodyTests(unittest.TestCase):
    def pose(self):
        p=np.full((33,4),np.nan);p[:,3]=0.
        for i,xy in ((11,(.65,.5)),(12,(.35,.5)),(14,(.3,.7)),(16,(.25,.4))):
            p[i,:2]=xy;p[i,3]=.9
        return p

    def test_image_wrist_axes_are_scale_invariant_and_never_measure_depth(self):
        p=self.pose();a=image_plane_wrist(p,'right',(1000,500))
        self.assertEqual(a[0],0.)
        p[16,0]+=.03;p[16,1]-=.06
        np.testing.assert_allclose(image_plane_wrist(p,'right',(1000,500))-a,[0,.04,.04])
        q=self.pose();q[:,:2]=(q[:,:2]-.5)*.7+.5
        np.testing.assert_allclose(image_plane_wrist(q,'right',(1000,500)),a)

    def test_outside_elbow_is_rejected_separately_from_observable_wrist(self):
        p=self.pose();p[14,1]=1.17
        trusted,reason=body_hand_consistency(p,[],require_elbow=True)
        self.assertFalse(trusted['right']);self.assertIn('elbow outside',reason['right'])
        trusted,_=body_hand_consistency(p,[],require_elbow=False)
        self.assertTrue(trusted['right'])
        p[14,:3]=np.nan
        self.assertIsNotNone(image_plane_wrist(p,'right'))
        self.assertIsNotNone(roi_bounds(p,'right',1280,720,allow_missing_elbow=True))

    def test_coco_to_image_pose_never_provides_fake_world_or_normalized_z(self):
        xy=np.zeros((17,2));score=np.ones(17)*.9
        xy[8]=[448,600]
        p=map_coco_pose(xy,score,(1280,720))
        np.testing.assert_allclose(p[14,:2],[.35,600/720])
        self.assertTrue(np.isnan(p[:,2]).all())
        p[12,3]=.1
        self.assertIsNone(image_plane_wrist(p,'right'))

    def test_image_identity_can_keep_palm_reference_without_any_pose_depth(self):
        p=self.pose();tracker=RelativePalmDepth()
        for i in range(10):
            h,n=sample();p[16,:2]=n[0,:2]
            obs=dict(read_end=(i+1)*50_000_000,position_norm=p,arms={'right':dict(hand_valid=True,hand_raw=h,hand_norm=n)})
            body=HumanInput(i,np.array([0.,.1,.2]),None,True,(i+1)*.05)
            hybrid,result=palm_depth_input(body,obs,tracker)
        self.assertTrue(result['valid']);self.assertIsNotNone(tracker.identity_stamp)
        np.testing.assert_allclose(hybrid.point[1:],[.1,.2])

    def test_background_or_another_face_cannot_authorize_the_2d_body(self):
        p=self.pose();p[0]=[.5,.25,np.nan,.9];reference=p.copy()
        self.assertTrue(same_person_image(reference,p))
        self.assertFalse(same_person_image(None,p))
        reference[0,0]=.9
        self.assertFalse(same_person_image(reference,p))

    def test_native_rtmpose_score_is_not_mediapipe_visibility(self):
        p=self.pose();p[12,3]=.5
        self.assertIsNone(image_plane_wrist(p))
        self.assertIsNotNone(image_plane_wrist(p,min_confidence=.45))
        self.assertFalse(body_hand_consistency(p,[])[0]['right'])
        self.assertTrue(body_hand_consistency(p,[],min_confidence=.45)[0]['right'])

    def test_live_stream_preserves_raw_bad_elbow_but_uses_current_image_position(self):
        from queue import Queue
        from contextlib import ExitStack
        from types import SimpleNamespace
        from unittest.mock import patch
        import time
        from mediapipe.tasks.python import vision
        from live_bimanual_stream import BimanualHumanStream
        queue=Queue();p=self.pose();p[0]=[.5,.2,0.,1.];p[:,2]=0.
        p[14,1]=1.17
        def landmarks(a):return [SimpleNamespace(x=x,y=y,z=z,visibility=v) for x,y,z,v in a]
        class Camera:
            metadata={'synthetic_input':True}
            def __init__(self,*args):pass
            def read_new(self,after):
                seq=queue.get(timeout=5)
                if seq is None:raise EOFError()
                return seq,seq*100_000_000,seq*100_000_000,np.zeros((180,320,3),np.uint8)
            def close(self):queue.put(None)
        class Body:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def detect_for_video(self,*args):return SimpleNamespace(pose_landmarks=[landmarks(p)],pose_world_landmarks=[landmarks(p)])
        class Hand(Body):
            def detect_for_video(self,*args):return SimpleNamespace(hand_landmarks=[],hand_world_landmarks=[],handedness=[])
        class ImageDetector:
            sha256='fixture'
            def detect(self,image,context=None):
                correct=p.copy();correct[:,2]=np.nan;correct[14,1]=.7
                return dict(norm=correct,score=np.ones(17),ms=1.)
        with ExitStack() as stack:
            stack.enter_context(patch.object(vision.PoseLandmarker,'create_from_options',return_value=Body()))
            stack.enter_context(patch.object(vision.HandLandmarker,'create_from_options',side_effect=lambda _:Hand()))
            stack.enter_context(patch('human_body_2d.ImageBodyDetector',return_value=ImageDetector()))
            stream=BimanualHumanStream(camera_factory=Camera,expected_sides=('right',),position_frame='upper-body',body_position_source='rtmpose')
            try:
                queue.put(1);deadline=time.monotonic()+5
                while stream.snapshot() is None and time.monotonic()<deadline:
                    if stream.error:self.fail(stream.error)
                    time.sleep(.005)
                obs=stream.snapshot();self.assertIsNotNone(obs)
                self.assertAlmostEqual(obs['pose_norm'][14,1],1.17)
                self.assertAlmostEqual(obs['position_norm'][14,1],.7)
                self.assertFalse(obs['pose_trusted']['right']);self.assertTrue(obs['display_trusted']['right'])
                self.assertTrue(obs['position_trusted']['right'])
                self.assertEqual(obs['arms']['right']['position_point'][0],0.)
                self.assertIsNone(obs['arms']['right']['pose'])
            finally:queue.put(None);stream.close()


if __name__=='__main__':unittest.main()
