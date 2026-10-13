"""OpenArm joint metadata shared by playback, tests, and live control."""

import numpy as np

# Physical arm geometry (OpenArm v2.0)
OPENARM_ARM_LENGTH = 0.455          # shoulder-to-wrist reach, metres
OPENARM_ANCHOR = np.zeros(3)        # robot-frame shoulder anchor (origin)
OPENARM_BASE_OFFSETS = {            # bimanual base mount offsets
    "left":  np.array([0.0,  0.031, 0.0], dtype=float),
    "right": np.array([0.0, -0.031, 0.0], dtype=float),
}

# ── Target observation layout (18D total, 9D per arm) ─────────────────────────
# Per arm: upper_dir(3)  forearm_dir(3)  arm_plane(3)  — all unit vectors
#   upper_dir   : shoulder→elbow direction (scale-invariant)
#   forearm_dir : elbow→wrist direction (scale-invariant)
#   arm_plane   : normal to shoulder-elbow-wrist plane (elbow swivel / null-space)
TARGET_OBS_DIM = 18

T_UPPER_L = slice(0, 3)
T_FORE_L  = slice(3, 6)
T_PLANE_L = slice(6, 9)

T_UPPER_R = slice( 9, 12)
T_FORE_R  = slice(12, 15)
T_PLANE_R = slice(15, 18)

RIGHT_JOINT_NAMES = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]

LEFT_JOINT_NAMES = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

JOINT_NAMES = RIGHT_JOINT_NAMES


def joint_names_for_side(side):
    side = side.lower()
    if side == "right":
        return RIGHT_JOINT_NAMES
    if side == "left":
        return LEFT_JOINT_NAMES
    raise ValueError(f"unknown OpenArm side {side!r}")

RIGHT_JOINT_LIMITS = np.array([
    [-1.3963, 3.4907],
    [-0.17453, 3.3161],
    [-1.5708, 1.5708],
    [0.0, 2.4435],
    [-1.5708, 1.5708],
    [-0.7854, 0.7854],
    [-0.7854, 0.7854],
], dtype=float)

LEFT_JOINT_LIMITS = np.array([
    [-3.4907, 1.3963],
    [-3.3161, 0.17453],
    [-1.5708, 1.5708],
    [0.0, 2.4435],
    [-1.5708, 1.5708],
    [-0.7854, 0.7854],
    [-0.7854, 0.7854],
], dtype=float)

JOINT_LIMITS = RIGHT_JOINT_LIMITS


def joint_limits_for_side(side):
    side = side.lower()
    if side == "right":
        return RIGHT_JOINT_LIMITS
    if side == "left":
        return LEFT_JOINT_LIMITS
    raise ValueError(f"unknown OpenArm side {side!r}")

# OpenArm v2.0 URDF velocity limits, rad/s.
JOINT_VELOCITY_LIMITS = np.array([
    16.755,
    16.755,
    5.4454,
    5.4454,
    20.944,
    20.944,
    20.944,
], dtype=float)


def clip_to_limits(q, limits=JOINT_LIMITS):
    q = np.asarray(q, dtype=float)
    limits = np.asarray(limits, dtype=float)
    clipped = np.clip(q, limits[:, 0], limits[:, 1])
    n_clipped = int(np.count_nonzero(np.abs(clipped - q) > 1e-9))
    return clipped, n_clipped
