"""Idle recording and elbow display recovery must not block or alter control."""
from concurrent.futures import Future
import unittest
import numpy as np
from single_arm_runtime import EpisodeRows
from elbow_recovery import BackgroundElbowRecovery,select_display_pose


class RuntimeRecoveryTests(unittest.TestCase):
    def test_elbow_repair_cannot_change_control_input_or_current_shoulders_and_wrist(self):
        pose=np.zeros((33,4));pose[:,3]=1.
        pose[0,:2]=[.5,.1];pose[[11,12],:2]=[[.4,.3],[.6,.3]]
        pose[[14,16],:2]=[[.8,1.1],[.7,.6]]
        before=pose.copy();other=pose.copy();other[14,:2]=[.8,.5]
        cache=dict(norm=other,valid={'right':True},capture_s=1.)
        display,stamp,repaired=select_display_pose(pose,cache,1.1,(1280,720),['right'])
        self.assertTrue(repaired);self.assertEqual(stamp,1.)
        np.testing.assert_array_equal(pose,before)
        np.testing.assert_array_equal(display[[0,11,12,16]],before[[0,11,12,16]])
        np.testing.assert_array_equal(display[14,:2],other[14,:2])
        self.assertIs(select_display_pose(pose,cache,1.3,(1280,720),['right'])[0],pose)
        other[0,:2]=[.1,.9]
        self.assertIs(select_display_pose(pose,cache,1.1,(1280,720),['right'])[0],pose)

    def test_long_ready_wait_is_bounded_but_complete_active_operation_is_kept(self):
        rows=EpisodeRows(ready_limit=40)
        for i in range(90000):rows.append(dict(stage='ready',seq=i))
        self.assertEqual(len(rows),40);self.assertEqual(rows.ready_omitted,89960)
        for i in range(300):rows.append(dict(stage='direct',seq=90000+i))
        self.assertEqual(len(rows),340);self.assertEqual(rows[-1]['seq'],90299)
        self.assertEqual(rows[0]['seq'],89960)
        complete=EpisodeRows()
        for i in range(100):complete.append(dict(stage='ready',seq=i))
        self.assertEqual(len(complete),100)

    def test_background_elbow_work_never_queues_or_returns_old_generation(self):
        class Pool:
            def __init__(self):self.jobs=[]
            def submit(self,*args):
                future=Future();self.jobs.append(future);return future
        pool=Pool();worker=BackgroundElbowRecovery(pool,detector=object())
        pose=np.ones((33,4))*.5;pose[:,3]=1.
        frame=np.zeros((20,30,3),np.uint8)
        self.assertIsNone(worker.observe(frame,pose,1.,0,True))
        for t in (1.01,1.02,1.03):self.assertIsNone(worker.observe(frame,pose,t,0,True))
        self.assertEqual(len(pool.jobs),1)
        pool.jobs[0].set_result(dict(norm=pose,valid={'right':True},ms=10.))
        self.assertIsNotNone(worker.observe(frame,pose,1.05,0,True))
        self.assertIsNone(worker.observe(frame,pose,1.4,1,True))
        self.assertEqual(len(pool.jobs),2)
        pool.jobs[1].set_result(dict(norm=pose,valid={'right':True},ms=10.))
        self.assertIsNone(worker.observe(frame,pose,1.8,1,True))
        pool.jobs[2].set_exception(Exception('auxiliary detector failed'))
        self.assertIsNone(worker.observe(frame,pose,1.9,1,True))
        self.assertIn('auxiliary detector failed',worker.error)


if __name__=='__main__':unittest.main()
