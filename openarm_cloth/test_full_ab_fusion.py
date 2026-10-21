"""Full source-clock mapping and B-vision integration, not tiny-curve transfer."""
from unittest.mock import patch
import unittest
import numpy as np
import mujoco

from full_video_source import FullVideoSource
from b_scene_fusion import BSceneFusionController,VisionTaskPlanner,TaskGeometry,NearObjectAssist
from member_b_scene import build_scene
from teleop_towel_dataset import build_arm_specs
from test_human_gripper_gestures import hand
from live_motion_scene import initialize_motion_posture
from source_video_reader import SourceVideoReader


class ExactSourceFrameTests(unittest.TestCase):
    def test_tail_skips_duplicates_and_restart_never_depend_on_random_seek(self):
        class Capture:
            def __init__(self, path): self.index=0
            def isOpened(self): return True
            def set(self, *args): raise AssertionError('MP4 random seeking is unreliable')
            def read(self):
                if self.index==6: return False,None
                value=np.full((2,2,3),self.index,np.uint8);self.index+=1
                return True,value
            def release(self): pass
        with patch('source_video_reader.cv2.VideoCapture',Capture):
            reader=SourceVideoReader('fixture.mp4');self.addCleanup(reader.close)
            for index in (0,2,2,5,0,5):
                np.testing.assert_array_equal(reader.frame(index),np.full((2,2,3),index,np.uint8))
            with self.assertRaisesRegex(ValueError,'Cannot decode measured source frame 6'):
                reader.frame(6)


def source_blob(n=90):
    world=np.zeros((n,33,4));world[:,:,3]=1
    for i in range(n):
        world[i,11,:3]=[-.2,0,0];world[i,12,:3]=[.2,0,0]
        world[i,23,:3]=[-.15,-.5,0];world[i,24,:3]=[.15,-.5,0]
        world[i,13,:3]=[-.25,-.25,0];world[i,15,:3]=[-.25,-.25,-.25]
        world[i,14,:3]=[.25,-.25,0];world[i,16,:3]=[.25+.08*np.sin(i/10),-.25,-.25]
    hands=np.tile(np.stack([hand(0),hand(4)])[None],(n,1,1,1))
    return dict(fps=np.array(30.),pose_world=world,pose_norm=world.copy(),hand_world=hands,
                hand_norm=np.full((n,2,21,2),.5),hand_valid=np.ones((n,2),bool),
                closure=np.column_stack([np.zeros(n),np.linspace(0,1,n)]),image_size=np.array([1280,720]))


class FullSourceTests(unittest.TestCase):
    def test_full_timeline_and_both_arms_survive_without_phase_selection(self):
        source=FullVideoSource(source_blob())
        observations=[source.at(t) for t in np.arange(60)*.05]
        self.assertEqual(observations[0]['seq'],0)
        self.assertGreater(observations[-1]['seq'],85)
        self.assertIsNone(source.at(3.))
        self.assertTrue(all(set(o['arms'])=={'left','right'} for o in observations))
        self.assertGreater(np.linalg.norm(observations[0]['arms']['right']['pose']['fore']-
                                         observations[20]['arms']['right']['pose']['fore']),.01)

    def test_source_capture_clock_uses_video_time_not_processing_time(self):
        source=FullVideoSource(source_blob())
        for t in (0.,.05,1.25,2.95):
            o=source.at(t)
            self.assertGreaterEqual(source.now_ns(t)-o['read_end'],0)
            self.assertLess(source.now_ns(t)-o['read_end'],34_000_000)

    def test_missing_left_landmarks_do_not_invent_a_pose_or_stop_right(self):
        blob=source_blob();blob['pose_world'][:,13,3]=0
        source=FullVideoSource(blob)
        self.assertIsNone(source.at(1.)['arms']['left']['pose'])
        self.assertIsNotNone(source.at(1.)['arms']['right']['pose'])

    def test_full_hand_orientation_is_present_even_when_pinch_is_open(self):
        source=FullVideoSource(source_blob())
        self.assertEqual(source.at(0.)['arms']['left']['closure'],0.)
        self.assertEqual(source.at(0.)['arms']['left']['hand_filtered'].shape,(21,3))
        self.assertTrue(source.at(0.)['arms']['left']['pose']['palm'])


class TaskFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model,cls.data=build_scene();cls.specs=build_arm_specs(cls.model)

    def planner(self):
        planner=VisionTaskPlanner(self.model,self.data,self.specs)
        self.addCleanup(planner.close)
        return planner

    def test_hidden_object_state_never_enters_collision_queries(self):
        p=self.planner();p.visual=dict(center=np.array([.43,0,.43]),yaw=.2)
        data=mujoco.MjData(self.model);data.qpos[:]=self.data.qpos
        a=p.query_state(data).qpos.copy();adr=p.object_address
        data.qpos[adr:adr+7]=[.9,.9,.9,0,0,0,1]
        np.testing.assert_allclose(p.query_state(data).qpos,a)
        p.visual=dict(center=np.array([.45,0,.43]),yaw=.2)
        self.assertAlmostEqual(p.query_state(data).qpos[adr]-a[adr],.02)

    def test_without_a_visual_target_both_arms_hold(self):
        p=self.planner();before={s:self.data.qpos[self.specs[s].qpos].copy() for s in ('left','right')}
        goals={s:before[s]+.05 for s in before}
        result=p.coordinate(before,goals,self.data,np.full(14,.125))
        for side in before: np.testing.assert_allclose(result[side],before[side])
        self.assertEqual(p.status['mode'],'vision_missing')

    def test_B_follow_posture_is_safe_in_full_table_scene_without_moving_object(self):
        p=self.planner();data=mujoco.MjData(self.model);data.qpos[:]=self.data.qpos
        adr=p.object_address;object_pose=data.qpos[adr:adr+7].copy()
        initialize_motion_posture(self.model,data,self.specs)
        p.visual=dict(center=object_pose[:3],yaw=0.)
        p.geometry.data.qpos[:]=p.query_state(data).qpos
        self.assertTrue(p.geometry.is_safe(data.qpos[p.qpos]))
        np.testing.assert_allclose(data.qpos[adr:adr+7],object_pose)

    def test_grasp_contact_permission_is_only_the_selected_two_fingers(self):
        a=TaskGeometry(self.model,self.specs);b=TaskGeometry(self.model,self.specs,'right')
        self.assertEqual(len(a.allowed),0);self.assertEqual(len(b.allowed),2)
        obj=int(self.model.geom('droid_object_geom').id)
        for pair in b.allowed:
            self.assertIn(obj,pair)
            other=next(g for g in pair if g!=obj)
            self.assertIn('openarm_right_',mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,other))

    def test_free_close_gesture_far_from_object_does_not_trigger_b_autopilot(self):
        c=BSceneFusionController(self.model,self.data);self.addCleanup(c.close)
        assist=NearObjectAssist(c);obs=FullVideoSource(source_blob()).at(.5)
        with patch('visual_grasp_planner.plan_face_grasp') as plan:
            joints,fingers=assist.update(obs,dict(center=np.array([.9,.9,1.]),yaw=0.),
                                        dict(left=0.,right=1.),.5)
        plan.assert_not_called();self.assertEqual(joints,{})
        self.assertEqual(fingers,{});self.assertIsNone(assist.side)


if __name__=='__main__': unittest.main()
