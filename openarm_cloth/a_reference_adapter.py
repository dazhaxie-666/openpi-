"""Preserve A's existing trajectory/RTMW grip; verify compatibility with B arms."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from member_b_scene import ROOT

sys.path.insert(0,str(ROOT.parent/'pose2arm-main'))
from member_b_export import adapt
from ik_solver import OpenArmIK

SIDES=('left','right')
DEFAULT_REFERENCE=ROOT.parent/'pose2arm-main/data/p2a_test1_rtmw_hybrid_sim.npz'


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


@dataclass
class AReference:
    q:np.ndarray
    closure:np.ndarray
    source_time:np.ndarray
    source_frame:np.ndarray
    trusted:np.ndarray
    source_fps:float
    reference_path:Path
    reference_hash:str
    video_hash:str


def resample_reference(blob,hz=20.):
    # A's original/mirror split, source clock and normalized grip contract.
    sampled=adapt(blob,hz=float(blob['fps']) if hz is None else hz,clip='orig')
    q=sampled['joint_angles']
    if str(sampled['joint_order'])=='right-first':q=np.column_stack((q[:,7:],q[:,:7]))
    closure=sampled.get('gripper_position',sampled.get('gripper_cmd'))
    if str(sampled['gripper_order'])=='right-first':closure=closure[:,::-1]
    frames=sampled['source_frame_indices']
    trusted=np.asarray(blob.get('source_arm_trusted',np.ones((int(frames.max())+1,2),bool)))
    if trusted.ndim!=2 or trusted.shape[1]!=2 or frames.max()>=len(trusted):
        raise ValueError('A source-arm confidence does not cover the original clip')
    return sampled,q,closure,trusted[frames].astype(bool)


def load_reference(reference,video,binding_path=None,hz=None):
    reference=Path(reference).resolve()
    reference_hash,video_hash=file_hash(reference),file_hash(video)
    metadata=json.loads(Path(binding_path or ROOT/'a_reference_bindings.json').read_text(encoding='utf-8'))
    if not any(entry['reference_sha256']==reference_hash and entry['source_sha256']==video_hash
               for entry in metadata['references']):
        raise ValueError('A reference/video pair is unbound: supply verified --reference-bindings JSON; '
                         'never use the p2a_test1 trajectory for a different video')
    with np.load(reference,allow_pickle=False) as z:
        sampled,q,closure,trusted=resample_reference({k:z[k] for k in z.files},hz=hz)
    return AReference(q,closure,sampled['timestamps_s'],sampled['source_frame_indices'],trusted,
                      float(sampled['source_fps']),reference,reference_hash,video_hash)


def verify_kinematics(model,specs):
    """Check A's canonical arm points across poses, not its obsolete fingers."""
    data=mujoco.MjData(model);result={}
    for side in SIDES:
        ik=OpenArmIK(side)
        data.qpos[specs[side].qpos]=0.
        mujoco.mj_forward(model,data)
        anchor=(data.xanchor[model.joint(f'openarm_{side}_joint1').id]
                -ik.forward_points(np.zeros(7))['joint_positions'][0]).copy()
        errors=[]
        # Some imported anchors move ALONG their joint axis. Compare the
        # coincident canonical 1/3/5/7 points, not dissimilar visual landmarks.
        for q in (np.zeros(7),np.array([.2,-.1 if side=='left' else .1,.3,.8,0,.2,0]),
                  np.array([-.3,-.3 if side=='left' else .3,-.2,1.,.4,-.3,.2])):
            data.qpos[specs[side].qpos]=q;mujoco.mj_forward(model,data)
            points=ik.forward_points(q)['joint_positions']
            for j in (0,2,4,6):
                actual=data.xanchor[model.joint(f'openarm_{side}_joint{j+1}').id]
                errors.append(np.linalg.norm(actual-points[j]-anchor))
        result[side]=float(max(errors))
    if max(result.values())>1e-5:
        raise ValueError(f'A/B canonical arm geometry differs; direct joint reuse rejected: {result}')
    return result
