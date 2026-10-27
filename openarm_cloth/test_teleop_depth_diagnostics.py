"""Layer traces must preserve missing evidence, source clocks and axis meaning."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diagnose_teleop_depth import pose_depth_features, export_episode
from human_task_diagnostics import diagnostic_phase


class TeleopDepthDiagnosticTests(unittest.TestCase):
    def test_pose_depth_features_are_raw_observations_not_absolute_range(self):
        pose=np.zeros((33,4));world=pose.copy();pose[:,3]=world[:,3]=1.
        pose[11,:2]=[.3,.5];pose[12,:2]=[.7,.5];pose[16]=[.6,.4,-.12,.9]
        world[16,:3]=[.2,-.1,-.3]
        f=pose_depth_features(pose,world,(1000,500))
        self.assertAlmostEqual(f['shoulder_px'],400.)
        self.assertAlmostEqual(f['wrist_img_z'],-.12)
        self.assertAlmostEqual(f['wrist_world_z'],-.3)
        self.assertTrue(f['pose_feature_visible'])
        pose[16,3]=.2
        f=pose_depth_features(pose,world,(1000,500))
        self.assertFalse(f['pose_feature_visible'])
        self.assertAlmostEqual(f['wrist_img_z'],-.12)  # raw value kept, never authorized
        f=pose_depth_features(None,None,(1000,500))
        self.assertFalse(f['pose_feature_visible']);self.assertTrue(np.isnan(f['shoulder_px']))

    def test_depth_protocol_separates_held_reaches_and_stationary_noise(self):
        self.assertEqual(diagnostic_phase(None,'depth-chain')[0],'WAIT_START')
        self.assertEqual([diagnostic_phase(t,'depth-chain')[0] for t in range(0,51,5)],
            ['depth_rest','reach_near','return_center','reach_far','return_center',
             'reach_near_repeat','reach_far_repeat','depth_left_right','depth_up_down','depth_rest_end','depth_end'])

    def test_historical_export_never_invents_ik_success_or_controller_ack(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'episode';p.mkdir();out=Path(folder)/'report'
            pose=np.zeros((3,33,4));pose[:,:,3]=1.;pose[:,11,0]=.3;pose[:,12,0]=.7
            np.savez_compressed(p/'episode.npz',time_s=[0.,.05,.1],stage=['direct']*3,
                source_seq=[10,10,11],capture_s=[1.,1.,1.08],source_read_end_ns=[100,100,180],
                human_pose_norm=pose,human_pose=pose,human_point=np.zeros((3,3)),
                tcp=np.tile([.43,0.,.54],(3,1)),human_pose_valid=[1,1,1],
                human_depth_valid=[1,1,0],depth_reason=['DEPTH_TRACKING','DEPTH_TRACKING','DEPTH_HAND_MISSING'])
            (p/'manifest.json').write_text(json.dumps(dict(camera_metadata=dict(reported_width=1000,reported_height=500))))
            report=export_episode(p,out,plot=False)
            self.assertEqual(report['active_control_rows'],3)
            self.assertEqual(report['fresh_active_sources'],2)
            self.assertIsNone(report['ik']['accepted_fraction'])
            self.assertFalse(report['availability']['command_target'])
            with (out/'chain.csv').open(newline='',encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
            self.assertEqual([r['fresh_source'] for r in rows],['True','False','True'])
            self.assertEqual(rows[0]['controller_ack'],'unavailable_simulation')
            self.assertEqual(rows[0]['ik_success'],'unknown')
            self.assertEqual(rows[0]['mapped_robot_x'],'nan')

    def test_recorded_ik_attempts_are_distinguished_from_holds(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'episode';p.mkdir();out=Path(folder)/'report'
            np.savez_compressed(p/'episode.npz',time_s=[0.,.05,.1],stage=['direct']*3,
                source_seq=[10,11,12],capture_s=[1.,1.05,1.1],human_depth_valid=[1,1,0],
                command_target=[[.43,0,.54],[np.nan]*3,[np.nan]*3],
                ik_attempted=[True,False,False],ik_accepted=[True,False,False],
                ik_position_error_m=[.001,np.nan,np.nan])
            (p/'manifest.json').write_text('{}')
            report=export_episode(p,out,plot=False)
            self.assertEqual(report['ik']['attempts'],1)
            self.assertEqual(report['ik']['accepted_fraction'],1.)
            with (out/'chain.csv').open(newline='',encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
            self.assertEqual([r['ik_success'] for r in rows],['True','not_attempted','not_attempted'])


if __name__=='__main__':unittest.main()
