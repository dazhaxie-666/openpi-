import unittest
from types import SimpleNamespace
import numpy as np
from human_grasp_task import HumanInput
from shared_human_control import SharedWristContinuity
from shared_dataset import SharedEpisodeSegments


class SharedWristTests(unittest.TestCase):
    def setUp(self):
        self.control=SharedWristContinuity()
        self.tracker=SimpleNamespace(scale=256.,nominal_span=.4,pending_recenter=False)
        self.hand=np.zeros((21,3));self.hand[:,:2]=[.4,.5]
        self.hand[[5,9,13,17],:2]=[[.38,.46],[.4,.45],[.42,.46],[.44,.47]]
        self.i=0

    def tick(self,valid=False,hand=True,reason='body shoulders outside frame',generation=0,age=0.,gap=.05):
        self.i+=1;stamp=getattr(self,'stamp',0.)+gap;self.stamp=stamp
        obs=dict(read_end=int(stamp*1e9),vision_generation=generation,position_quality=dict(right=reason),
            arms=dict(right=dict(hand_valid=hand,hand_norm=self.hand.copy())))
        human=HumanInput(self.i,np.array([0.,.1,.2]) if valid else None,None,valid,stamp)
        return self.control.update(human,obs,self.tracker,stamp+age)

    def test_current_associated_hand_bridges_shoulders_and_preserves_measured_displacement(self):
        self.tick(True)
        for _ in range(20):
            self.hand[:,0]+=.001;result=self.tick()
            self.assertTrue(result.valid)
        self.assertAlmostEqual(result.point[1]-.1,.02*1280*.4/256,places=8)
        self.assertEqual(self.control.reason,'IMAGE_ASSOCIATED_HAND_TRACKING')

    def test_initial_body_calibration_and_reassociation_after_long_hand_loss_are_required(self):
        self.assertFalse(self.tick().valid)
        self.tick(True);self.tick(hand=False,gap=.5)
        self.assertFalse(self.tick().valid)
        self.assertTrue(self.tick(True).valid)

    def test_stale_mismatched_identity_generation_change_and_hand_jump_never_fill_position(self):
        self.tick(True)
        self.assertFalse(self.tick(age=.4).valid)
        self.assertFalse(self.tick(reason='body/hand wrist disagreement').valid)
        self.assertFalse(self.tick(generation=1).valid)
        self.tick(True,generation=1);self.hand[:,0]+=.2
        self.assertFalse(self.tick(generation=1).valid)

    def test_body_hand_switches_preserve_motion_without_repeated_rebasing(self):
        self.tick(True)
        for _ in range(10):
            self.hand[:,0]+=.001;self.assertTrue(self.tick().valid)
            self.hand[:,0]+=.001;result=self.tick(True)
        self.assertAlmostEqual(result.point[1]-.1,.02*1280*.4/256,places=8)
        self.assertFalse(self.tracker.pending_recenter)


class SharedSegmentTests(unittest.TestCase):
    def setUp(self):
        self.segments=SharedEpisodeSegments();self.rows=[]
        self.task=SimpleNamespace(stage='direct',phase='manual',trials=[],trial_id=0)

    def step(self,phase='manual',end=None,request=False):
        t=len(self.rows)*.05;self.task.phase=phase
        if request:
            self.task.trial_id+=1
            self.task.trials.append(dict(trial_id=self.task.trial_id,start_s=t,status='active'))
        if end:self.task.trials[-1].update(end_s=t,status=end)
        began=self.segments.update(self.task,self.rows,t)
        self.rows.append(dict(time_s=t,shared_phase=phase,shared_episode_id=self.segments.episode_id))
        return began

    def test_complete_segments_include_approach_and_keep_retries_disjoint(self):
        self.assertTrue(self.step());self.step()
        self.step('plan',request=True)
        for phase in ('approach','descend','close','lift','transport','release','verify'):self.step(phase)
        self.step('manual',end='completed')
        self.assertTrue(self.step());self.step('plan',request=True);self.step('recover',end='cancelled')
        self.segments.finalize(self.task,self.rows)
        a,b=self.task.trials
        self.assertEqual((a['start_frame'],a['request_frame']), (0,2))
        self.assertTrue(a['complete_task_segment']);self.assertFalse(b['complete_task_segment'])
        self.assertLessEqual(a['end_frame_exclusive'],b['start_frame'])
        for trial in self.task.trials:
            self.assertTrue(all(r['shared_episode_id']==trial['trial_id'] for r in self.rows[trial['start_frame']:trial['end_frame_exclusive']]))

    def test_approach_without_any_grasp_request_is_retained_as_interrupted(self):
        self.step();self.step();self.segments.finalize(self.task,self.rows)
        trial=self.task.trials[0]
        self.assertEqual(trial['status'],'interrupted');self.assertIsNone(trial['request_frame'])
        self.assertEqual(trial['end_frame_exclusive'],2);self.assertFalse(trial['complete_task_segment'])

    def test_immediate_request_and_manual_verify_row_cannot_fake_an_approach(self):
        self.step('plan',request=True)
        for phase in ('close','lift','transport','release','verify'):self.step(phase)
        self.step('manual',end='completed');self.segments.finalize(self.task,self.rows)
        self.assertFalse(self.task.trials[0]['includes_human_approach'])
        self.assertFalse(self.task.trials[0]['complete_task_segment'])

    def test_reset_and_final_notice_rows_do_not_create_a_phantom_trial(self):
        self.step();self.step('plan',request=True);self.step('verify')
        self.step('manual',end='completed')
        self.step('reset');self.step('reset');self.step('session_complete')
        self.segments.finalize(self.task,self.rows)
        self.assertEqual(len(self.task.trials),1)
        self.assertEqual(self.task.trials[0]['status'],'completed')


if __name__=='__main__':unittest.main()
