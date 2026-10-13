"""Day 2/3 geometry self-test with synthetic keypoints -- no video required.

Checks (a) the body frame is a valid rotation matrix, (b) elbow_angle
responds the way geometry says it should for hand-crafted straight/bent
poses, and (c) rigidly translating+rotating the whole body in world space
leaves the body-frame-relative outputs unchanged (this is the Day-2 fix for
"walking across frame looks like swinging your arm").

Run: python3 scripts/smoke_test_retarget_geometry.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retarget import OPENARM_BODY_TO_ROBOT, Retargeter, compute_arm_frame, compute_body_frame, to_body_frame

L_SH, R_SH = np.array([-0.2, 1.4, 0.0]), np.array([0.2, 1.4, 0.0])
L_HIP, R_HIP = np.array([-0.15, 0.9, 0.0]), np.array([0.15, 0.9, 0.0])
origin, R_body = compute_body_frame(L_SH, R_SH, L_HIP, R_HIP)
assert np.allclose(R_body.T @ R_body, np.eye(3), atol=1e-9), "body frame is not orthonormal"

assert np.allclose(OPENARM_BODY_TO_ROBOT @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0]), (
    "human/body right must map to robot -Y"
)
assert np.allclose(OPENARM_BODY_TO_ROBOT @ np.array([-1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]), (
    "human/body left must map to robot +Y"
)
assert np.allclose(OPENARM_BODY_TO_ROBOT @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]), (
    "human/body up must map to robot +Z"
)
assert np.allclose(OPENARM_BODY_TO_ROBOT @ np.array([0.0, 0.0, -1.0]), [1.0, 0.0, 0.0]), (
    "human forward must map to robot +X"
)

# Straight arm pointing forward from the right shoulder.
S = to_body_frame(R_SH, origin, R_body)
E_straight = to_body_frame(R_SH + np.array([0.0, 0.0, -0.3]), origin, R_body)
W_straight = to_body_frame(R_SH + np.array([0.0, 0.0, -0.6]), origin, R_body)
_, angle_straight, _, normal_seed = compute_arm_frame(
    S, E_straight, W_straight, prev_normal=np.array([1.0, 0.0, 0.0]))
assert angle_straight > np.deg2rad(170), f"expected near-straight elbow, got {np.degrees(angle_straight):.1f} deg"

# Bent ~90 degrees: forearm turns up instead of continuing forward.
E_bent = E_straight
W_bent = to_body_frame(R_SH + np.array([0.0, 0.0, -0.3]) + np.array([0.0, 0.3, 0.0]), origin, R_body)
_, angle_bent, _, _ = compute_arm_frame(S, E_bent, W_bent, prev_normal=normal_seed)
assert abs(np.degrees(angle_bent) - 90) < 5, f"expected ~90 deg bent elbow, got {np.degrees(angle_bent):.1f} deg"

# Translation/rotation invariance: move the whole rig and re-derive body frame.
shift = np.array([5.0, -2.0, 3.0])
yaw = 0.7
Rz = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])


def world_move(p):
    return Rz @ p + shift


origin2, R_body2 = compute_body_frame(world_move(L_SH), world_move(R_SH), world_move(L_HIP), world_move(R_HIP))
S2 = to_body_frame(world_move(R_SH), origin2, R_body2)
E2 = to_body_frame(world_move(R_SH + np.array([0.0, 0.0, -0.3])), origin2, R_body2)
W2 = to_body_frame(world_move(R_SH + np.array([0.0, 0.0, -0.3]) + np.array([0.0, 0.3, 0.0])), origin2, R_body2)
_, angle_bent2, _, _ = compute_arm_frame(S2, E2, W2, prev_normal=normal_seed)
assert abs(angle_bent - angle_bent2) < 1e-6, (
    f"elbow angle changed after rigidly moving the whole body "
    f"({np.degrees(angle_bent):.2f} vs {np.degrees(angle_bent2):.2f} deg) -- body-frame invariance is broken"
)

# Retargeter end-to-end with the bent-arm calibration frame.
rt = Retargeter()
rt.calibrate([(S, E_bent, W_bent)])
target_pos, target_R, elbow_angle = rt.frame_to_target(S, E_bent, W_bent)
assert np.allclose(target_R.T @ target_R, np.eye(3), atol=1e-9), "target orientation is not a valid rotation matrix"

print(f"elbow_angle_straight={np.degrees(angle_straight):.1f} deg")
print(f"elbow_angle_bent={np.degrees(angle_bent):.1f} deg (frame-invariant: {np.degrees(angle_bent2):.1f} deg)")
print(f"calibrated scale={rt.scale:.3f}")
print(f"target_position={np.round(target_pos, 3)}")
print("All retarget.py geometry self-tests passed.")
