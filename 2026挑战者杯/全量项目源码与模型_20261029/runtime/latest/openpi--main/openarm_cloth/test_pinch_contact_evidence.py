"""Regression at the direct wrist channel: reject brief near-contact collapses."""
import unittest
from unittest.mock import patch
import numpy as np
from human_pinch_control import TaskHandControl,PinchCalibration


class PinchContactEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.calibration=PinchCalibration()
        for _ in range(10):
            self.calibration.add_trial('closed',[dict(r3=.2+v,r2=.1+v) for v in np.linspace(-.02,.02,8)])
            self.calibration.add_trial('open',[dict(r3=1.2+v,r2=1.5+v) for v in np.linspace(-.02,.02,8)])
        loader=patch('human_pinch_control.load_personal_calibration',return_value={'right':self.calibration.export()})
        loader.start();self.addCleanup(loader.stop)
        self.channel=TaskHandControl();self.seq=-1;self.stamp=-.05

    def sample(self,r3=.7,r2=.8,dt=.05):
        self.seq+=1;self.stamp+=dt
        return self.channel.process_features(self.seq,self.stamp,dict(r3=r3,r2=r2))

    def half(self):
        for _ in range(12):result=self.sample()
        self.assertAlmostEqual(result['aperture']/.044,.5,places=3)
        return result['aperture']

    def test_recorded_pattern_short_collapse_cannot_issue_nearly_closed_reference(self):
        held=self.half()
        # Three low measurements occupy only 111ms in the recorded flip.
        for r3,r2,dt in ((.31,.12,.095),(.31,.12,.032),(.19,.17,.079)):
            result=self.sample(r3,r2,dt)
            self.assertAlmostEqual(result['aperture'],held)
            self.assertFalse(result['valid'])
        result=self.sample(.7,.8)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['aperture'],held)

    def test_sustained_real_contact_remains_reachable_without_an_aperture_floor(self):
        self.half()
        for _ in range(18):result=self.sample(.2,.1)
        self.assertTrue(result['valid'])
        self.assertLess(result['aperture']/.044,.005)

    def test_projected_overlap_does_not_erase_a_measured_3d_gap(self):
        self.half()
        # Projected overlap can also be an occluded half-open hand.
        for _ in range(18):result=self.sample(.35,.08)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['aperture']/.044,.15,places=3)

    def test_confirmation_needs_three_fresh_samples_even_at_slow_capture_rate(self):
        self.half()
        self.assertFalse(self.sample(.2,.1)['valid'])
        self.assertFalse(self.sample(.2,.1,dt=.2)['valid'])
        self.assertTrue(self.sample(.2,.1)['valid'])

    def test_visible_small_gap_is_not_snapped_to_contact(self):
        self.half()
        for _ in range(18):result=self.sample(.35,.38)
        self.assertGreater(result['aperture']/.044,.19)
        self.assertLess(result['aperture']/.044,.21)

    def test_duplicates_and_loss_cannot_finish_contact_confirmation(self):
        self.half();self.sample(.2,.1)
        for _ in range(10):
            result=self.channel.process_features(self.seq,self.stamp,dict(r3=.2,r2=.1))
            self.assertFalse(result['valid'])
        self.seq+=1;self.stamp+=.05;self.channel.process_features(self.seq,self.stamp,None)
        result=self.sample(.2,.1,dt=.2)
        self.assertFalse(result['valid']);self.assertGreater(result['aperture']/.044,.49)

    def test_non_contact_middle_motion_keeps_continuous_updates(self):
        self.half()
        for fraction in (.45,.35,.65,.8):
            result=self.sample(.2+fraction,.1+1.4*fraction)
            self.assertTrue(result['valid'])

    def test_confirmed_near_contact_values_remain_continuous(self):
        self.half()
        for _ in range(10):self.sample(.25,.17)
        result=self.sample(.32,.268)
        self.assertTrue(result['valid'])
        result=self.sample(.32,.268)
        self.assertGreater(result['aperture']/.044,.05)
        self.assertLess(result['aperture']/.044,.12)

    def test_isolated_middle_aperture_outlier_cannot_move_a_steady_gripper(self):
        held=self.half()
        for r3,r2 in ((.4,.38),(.7,.8)):
            result=self.sample(r3,r2)
            self.assertTrue(result['valid'])
            self.assertAlmostEqual(result['aperture'],held)

    def test_sustained_pinch_reaches_near_closed_within_350ms(self):
        self.half();begin=self.stamp
        for _ in range(7):result=self.sample(.2,.1)
        self.assertLessEqual(self.stamp-begin,.350001)
        self.assertLess(result['aperture']/.044,.05)

    def test_b_dynamic_fingers_hold_through_spike_then_close_on_sustained_pinch(self):
        from member_b_scene import build_scene
        from human_task_robot import HumanTaskRobot
        from human_grasp_task import TaskCommand
        from run_human_task import direct_hand_command
        model,data=build_scene()
        with patch('human_task_robot.BVisualObserver'):
            robot=HumanTaskRobot(model,data)
        self.addCleanup(robot.close);robot.initialize()
        robot.visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.)

        def execute(result):
            robot.visual['time_s']=self.stamp
            command=direct_hand_command(TaskCommand('hold'),result,'direct',robot.finger_targets['right'])
            robot.execute(command,self.stamp);robot.step(.05)
            self.assertIsNone(robot.error);self.assertIsNone(robot.warning)

        for _ in range(18):execute(self.sample())
        held=robot.finger_targets['right']
        for r3,r2 in ((.31,.12),(.31,.12),(.19,.17)):
            execute(self.sample(r3,r2))
            self.assertAlmostEqual(robot.finger_targets['right'],held)
        self.assertAlmostEqual(float(np.mean(data.qpos[robot.fingers[0]])),.022,delta=.002)
        for _ in range(20):execute(self.sample(.2,.1))
        self.assertLess(robot.finger_targets['right'],.001)
        self.assertLess(float(np.mean(data.qpos[robot.fingers[0]])),.003)


if __name__=='__main__':unittest.main()
