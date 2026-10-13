"""OpenArm 7-DOF pose-likeness IK (SciPy / Levenberg-Marquardt)."""

import numpy as np
from scipy.optimize import least_squares

from openarm_common import joint_limits_for_side

NUM_ARM_JOINTS = 7
RIGHT_OPENARM_JOINT_ORIGINS = np.array([
    [0.0, -0.0625, 0.0],
    [0.0, -0.0600000000000001, 0.0],
    [0.0, 0.0, -0.0662500000000008],
    [0.0, 0.0, -0.15375],
    [0.0, 0.0, -0.0955000000000005],
    [0.0, 0.0, -0.120499999999998],
    [0.0, 0.0, 0.0],
], dtype=float)
LEFT_OPENARM_JOINT_ORIGINS = np.array([
    [0.0, 0.0625, 0.0],
    [0.0, 0.0600000000000001, 0.0],
    [0.0, 0.0, -0.0662500000000008],
    [0.0, 0.0, -0.15375],
    [0.0, 0.0, -0.0955000000000005],
    [0.0, 0.0, -0.120499999999998],
    [0.0, 0.0, 0.0],
], dtype=float)
RIGHT_OPENARM_JOINT_AXES = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=float)
LEFT_OPENARM_JOINT_AXES = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=float)
OPENARM_JOINT_ORIGINS = RIGHT_OPENARM_JOINT_ORIGINS
OPENARM_JOINT_AXES = RIGHT_OPENARM_JOINT_AXES
OPENARM_TOOL_TRANSLATION = np.array([0.0205, 0.0, 0.0], dtype=float)
OPENARM_ELBOW_JOINT_INDEX = 3  # zero-based joint4 origin, visually closest to the OpenArm elbow.
POSE_LIKENESS_MAX_ELBOW_ANGLE = np.deg2rad(140.0)
POSE_LIKENESS_LIMIT_MARGIN = np.deg2rad(15.0)
POSE_LIKENESS_SHOULDER_LIMIT_MARGIN = np.deg2rad(5.0)
POSE_LIKENESS_MAX_JOINT_STEP = np.array([0.70, 0.70, 0.90, 0.90, 1.20, 1.20, 1.20], dtype=float)
POSE_LIKENESS_WEIGHTS = {
    "upper_dir": 20.0,
    "forearm_dir": 30.0,
    "wrist_position": 2.0,
    "arm_plane": 10.0,
    "elbow_angle": 8.0,
    "wrist_dir": 5.0,
    "smooth": 1.0,
    "step_margin": 4.0,
    "shoulder_home": 1.2,
    "home": 0.05,
    "limit_margin": 6.0,
}

# All-zero is close to a straight-arm singular branch. Always warm-start from
# a bent-elbow home pose, or from the previous frame's solution.
RIGHT_OPENARM_HOME_SEED = np.array([0.0, 0.6, 0.0, 0.9, 0.0, 0.0, 0.0])
LEFT_OPENARM_HOME_SEED = np.array([0.0, -0.6, 0.0, 0.9, 0.0, 0.0, 0.0])
OPENARM_HOME_SEED = RIGHT_OPENARM_HOME_SEED


def _joint_metadata_for_side(side):
    side = side.lower()
    if side == "right":
        return RIGHT_OPENARM_JOINT_ORIGINS, RIGHT_OPENARM_JOINT_AXES, joint_limits_for_side(side)
    if side == "left":
        return LEFT_OPENARM_JOINT_ORIGINS, LEFT_OPENARM_JOINT_AXES, joint_limits_for_side(side)
    raise ValueError(f"unknown OpenArm side {side!r}")


def _rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=float)


def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return v / n


def _cosine_vector_residual(actual, target):
    actual_u = _unit(actual)
    target_u = _unit(target)
    if actual_u is None or target_u is None:
        return np.zeros(3)
    return actual_u - target_u


def _angle_between(a, b):
    au = _unit(a)
    bu = _unit(b)
    if au is None or bu is None:
        return 0.0
    return float(np.arccos(np.clip(np.dot(au, bu), -1.0, 1.0)))


def _joint_limit_margin_residual(q, limits, margin=POSE_LIKENESS_LIMIT_MARGIN):
    q = np.asarray(q, dtype=float)
    limits = np.asarray(limits, dtype=float)
    lower_margin = q - limits[:, 0]
    upper_margin = limits[:, 1] - q
    lower = np.maximum(0.0, margin - lower_margin) / margin
    upper = np.maximum(0.0, margin - upper_margin) / margin
    return np.concatenate([lower, upper])


def _joint_step_margin_residual(q, seed, max_step=POSE_LIKENESS_MAX_JOINT_STEP):
    q = np.asarray(q, dtype=float)
    seed = np.asarray(seed, dtype=float)
    max_step = np.asarray(max_step, dtype=float)
    return np.maximum(0.0, np.abs(q - seed) - max_step) / max_step


class OpenArmIK:
    """Pose-likeness IK for one OpenArm 7-DOF arm.

    Coordinates are in the side-specific OpenArm base frame.
    """

    def __init__(self, side="right"):
        self.side = side.lower()
        self.joint_origins, self.joint_axes, self.joint_limits = _joint_metadata_for_side(self.side)
        self.home_seed = LEFT_OPENARM_HOME_SEED if self.side == "left" else RIGHT_OPENARM_HOME_SEED
        self._last_solution = None
        self.last_reject_reason = ""

    def solve_pose_likeness(self, pose_target, weights=None, max_nfev=80):
        seed = self._last_solution if self._last_solution is not None else self.home_seed
        seed = np.clip(seed, self.joint_limits[:, 0], self.joint_limits[:, 1])
        had_previous_solution = self._last_solution is not None

        def residual(q):
            return self.pose_likeness_residual(q, pose_target, seed=seed, weights=weights)

        result = least_squares(
            residual,
            seed,
            bounds=(self.joint_limits[:, 0], self.joint_limits[:, 1]),
            max_nfev=max_nfev,
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
        )
        self._last_solution = self._accept_solution(np.asarray(result.x, dtype=float), seed, had_previous_solution)
        return self._last_solution

    def _accept_solution(self, q, seed, had_previous_solution):
        self.last_reject_reason = ""
        q = np.asarray(q, dtype=float)
        seed = np.asarray(seed, dtype=float)
        if not np.isfinite(q).all():
            self.last_reject_reason = "nonfinite"
            raise ValueError("IK solution is not finite")
        if np.any(q < self.joint_limits[:, 0] - 1e-6) or np.any(q > self.joint_limits[:, 1] + 1e-6):
            self.last_reject_reason = "outside_limits"
            raise ValueError("IK solution exceeds joint limits")

        shoulder = q[:2]
        shoulder_limits = self.joint_limits[:2]
        shoulder_lower_margin = shoulder - shoulder_limits[:, 0]
        shoulder_upper_margin = shoulder_limits[:, 1] - shoulder
        if (
            np.any(shoulder_lower_margin < POSE_LIKENESS_SHOULDER_LIMIT_MARGIN)
            or np.any(shoulder_upper_margin < POSE_LIKENESS_SHOULDER_LIMIT_MARGIN)
        ):
            self.last_reject_reason = "shoulder_limit_margin"
            raise ValueError("IK solution is too close to a shoulder joint limit")

        if had_previous_solution and np.any(np.abs(q - seed) > POSE_LIKENESS_MAX_JOINT_STEP):
            self.last_reject_reason = "joint_step"
            raise ValueError("IK solution jumps too far from previous frame")
        return q

    def pose_likeness_residual(self, q, pose_target, seed=None, weights=None):
        weights = {**POSE_LIKENESS_WEIGHTS, **(weights or {})}
        seed = self.home_seed if seed is None else np.asarray(seed, dtype=float)
        target_elbow_angle = min(float(pose_target.elbow_angle), POSE_LIKENESS_MAX_ELBOW_ANGLE)
        points = self.forward_points(q)
        wrist_position_residual = np.zeros(3)
        if hasattr(pose_target, "wrist_position"):
            wrist_position_residual = points["tool"] - pose_target.wrist_position
        parts = [
            np.sqrt(weights["upper_dir"]) * _cosine_vector_residual(points["upper_dir"], pose_target.upper_dir),
            np.sqrt(weights["forearm_dir"]) * _cosine_vector_residual(points["forearm_dir"], pose_target.forearm_dir),
            np.sqrt(weights["wrist_dir"]) * _cosine_vector_residual(points["wrist_dir"], pose_target.wrist_dir),
            np.sqrt(weights["wrist_position"]) * wrist_position_residual,
            np.sqrt(weights["arm_plane"]) * _cosine_vector_residual(points["arm_plane"], pose_target.arm_plane),
            np.array([np.sqrt(weights["elbow_angle"]) * (points["elbow_angle"] - target_elbow_angle)]),
            np.sqrt(weights["smooth"]) * (q - seed),
            np.sqrt(weights["step_margin"]) * _joint_step_margin_residual(q, seed),
            np.sqrt(weights["shoulder_home"]) * (q[:2] - self.home_seed[:2]),
            np.sqrt(weights["home"]) * (q - self.home_seed),
            np.sqrt(weights["limit_margin"]) * _joint_limit_margin_residual(q, self.joint_limits),
        ]
        return np.concatenate(parts)

    def forward_kinematics(self, arm_angles):
        T = self._forward_transform(arm_angles)
        return T[:3, 3], T[:3, :3]

    def forward_points(self, arm_angles):
        T, joint_positions = self._forward_transform(arm_angles, return_joint_positions=True)
        tool = T[:3, 3]
        shoulder = np.zeros(3, dtype=float)
        elbow = joint_positions[OPENARM_ELBOW_JOINT_INDEX]
        upper = elbow - shoulder
        forearm = tool - elbow
        wrist = tool - shoulder
        plane = np.cross(upper, forearm)
        return {
            "shoulder": shoulder,
            "elbow": elbow,
            "tool": tool,
            "joint_positions": np.asarray(joint_positions),
            "upper_dir": upper,
            "forearm_dir": forearm,
            "wrist_dir": wrist,
            "arm_plane": plane,
            "elbow_angle": _angle_between(-upper, forearm),
        }

    def _forward_transform(self, arm_angles, return_joint_positions=False):
        q = np.asarray(arm_angles, dtype=float)
        T = np.eye(4)
        joint_positions = []
        for origin, axis, angle in zip(self.joint_origins, self.joint_axes, q):
            T[:3, 3] += T[:3, :3] @ origin
            joint_positions.append(T[:3, 3].copy())
            T[:3, :3] = T[:3, :3] @ _rotation_matrix(axis, angle)

        T[:3, 3] += T[:3, :3] @ OPENARM_TOOL_TRANSLATION
        if return_joint_positions:
            return T, joint_positions
        return T

    def reset(self):
        self._last_solution = None
