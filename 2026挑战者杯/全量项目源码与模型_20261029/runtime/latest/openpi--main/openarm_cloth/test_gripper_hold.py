"""Explicit aperture hold must preserve arm targets and safe resume behavior."""
import unittest
import numpy as np
from human_grasp_task import TaskCommand
from human_gripper_hold import GripperHold


class GripperHoldTests(unittest.TestCase):
    def hand(self,seq,aperture,valid=True):
        return dict(seq=seq,aperture=aperture,valid=valid)

    def test_explicit_hold_is_constant_despite_drifting_hand_but_arm_still_moves(self):
        hold=GripperHold();hold.lock(.022)
        for i,aperture in enumerate((0.,.044,.01,.03)):
            target=np.array([.43+i*.002,0.,.5])
            result=hold.apply(TaskCommand('direct',target,aperture),self.hand(i,aperture),'direct',.022)
            self.assertEqual(result.finger,.022)
            np.testing.assert_array_equal(result.target,target)
            self.assertEqual(result.stage,'direct')

    def test_default_never_locks_from_hand_or_arm_motion(self):
        hold=GripperHold()
        for i,aperture in enumerate((.01,.02,.03)):
            c=TaskCommand('direct',np.array([.4,.01*i,.5]),aperture)
            self.assertIs(hold.apply(c,self.hand(i,aperture),'direct',.02),c)
        self.assertFalse(hold.locked)

    def test_unlock_requires_two_new_valid_sources_and_does_not_reuse_held_image(self):
        hold=GripperHold();hold.lock(.022);hold.release(10)
        c=TaskCommand('direct',np.ones(3),.044)
        for seq in (10,10,11,11):
            self.assertEqual(hold.apply(c,self.hand(seq,.044),'direct',.022).finger,.022)
        self.assertEqual(hold.apply(c,self.hand(12,.044),'direct',.022).finger,.044)

    def test_missing_reference_breaks_resume_confirmation(self):
        hold=GripperHold();hold.lock(.022);hold.release(10)
        c=TaskCommand('gripper_reference',finger=.044)
        hold.apply(c,self.hand(11,.044),'direct',.022)
        hold.apply(c,self.hand(12,.044,False),'direct',.022)
        self.assertEqual(hold.apply(c,self.hand(13,.044),'direct',.022).finger,.022)
        self.assertEqual(hold.apply(c,self.hand(14,.044),'direct',.022).finger,.044)

    def test_pause_terminal_and_loss_hold_do_not_inject_a_new_gripper_move(self):
        hold=GripperHold();hold.lock(.022)
        c=TaskCommand('hold',finger=.017)
        for stage,paused in (('ready',False),('finished',False),('direct',True),('direct',False)):
            self.assertIs(hold.apply(c,None,stage,.017,paused),c)

    def test_invalid_lock_apertures_are_rejected(self):
        for value in (-.1,.05,float('nan')):
            with self.assertRaises(ValueError):GripperHold().lock(value)

    def test_hold_command_breaks_resume_confirmation_without_moving_fingers(self):
        hold=GripperHold();hold.lock(.022);hold.release(10)
        c=TaskCommand('direct',np.ones(3),.044)
        hold.apply(c,self.hand(11,.044),'direct',.022)
        missing=TaskCommand('hold',finger=.022)
        self.assertIs(hold.apply(missing,None,'direct',.022),missing)
        self.assertEqual(hold.apply(c,self.hand(13,.044),'direct',.022).finger,.022)

    def test_half_hold_reaches_b_actuators_while_arm_targets_continue(self):
        from unittest.mock import patch
        from member_b_scene import build_scene
        from human_task_robot import HumanTaskRobot
        model,data=build_scene()
        with patch('human_task_robot.BVisualObserver'):
            robot=HumanTaskRobot(model,data)
        self.addCleanup(robot.close);robot.initialize()
        robot.visual=dict(center=np.array([.43,0.,.429]),yaw=0.,time_s=0.)
        before=data.qpos[robot.spec.qpos].copy();origin=robot.feedback(0.).tcp.copy()
        hold=GripperHold();hold.lock(.022)
        for i in range(30):
            robot.visual['time_s']=i*.05
            command=TaskCommand('direct',origin+np.array([.002,0.,.002]),.044 if i%2 else 0.)
            command=hold.apply(command,self.hand(i,command.finger),'direct',robot.finger_targets['right'])
            robot.execute(command,i*.05);robot.step(.05)
            self.assertIsNone(robot.error);self.assertIsNone(robot.warning)
        self.assertAlmostEqual(robot.finger_targets['right'],.022)
        self.assertAlmostEqual(float(np.mean(data.qpos[robot.fingers[0]])),.022,delta=.002)
        self.assertGreater(float(np.linalg.norm(data.qpos[robot.spec.qpos]-before)),.001)


if __name__=='__main__':unittest.main()
