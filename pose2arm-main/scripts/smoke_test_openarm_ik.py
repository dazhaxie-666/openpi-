"""Smoke tests for the Gazebo OpenArm IK backend.

Run:
    source venv/bin/activate
    python3 scripts/smoke_test_openarm_ik.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ik_solver import OpenArmIK
from openarm_common import joint_limits_for_side


def test_openarm_fk_home():
    expected_by_side = {
        "right": np.array([0.0205, -0.1225, -0.436]),
        "left": np.array([0.0205, 0.1225, -0.436]),
    }
    for side, expected in expected_by_side.items():
        ik = OpenArmIK(side=side)
        pos, _ = ik.forward_kinematics([0.0] * 7)
        err = np.linalg.norm(pos - expected)
        print(f"[OpenArm FK zero {side}] pos={np.round(pos, 4)} err={err:.2e} m")
        assert err < 1e-9


def test_openarm_pose_likeness_self_target():
    source_by_side = {
        "right": np.array([0.2, 0.9, -0.3, 1.0, 0.2, 0.1, 0.0]),
        "left": np.array([-0.2, -0.9, 0.3, 1.0, -0.2, -0.1, 0.0]),
    }
    for side, source_q in source_by_side.items():
        ik = OpenArmIK(side=side)
        points = ik.forward_points(source_q)

        ik.reset()
        target = SimpleNamespace(
            upper_dir=points["upper_dir"],
            forearm_dir=points["forearm_dir"],
            wrist_dir=points["wrist_dir"],
            arm_plane=points["arm_plane"],
            elbow_angle=points["elbow_angle"],
        )
        solved = ik.solve_pose_likeness(target)
        solved_points = ik.forward_points(solved)
        upper_err = 0.5 - 0.5 * np.dot(
            solved_points["upper_dir"] / np.linalg.norm(solved_points["upper_dir"]),
            points["upper_dir"] / np.linalg.norm(points["upper_dir"]),
        )
        forearm_err = 0.5 - 0.5 * np.dot(
            solved_points["forearm_dir"] / np.linalg.norm(solved_points["forearm_dir"]),
            points["forearm_dir"] / np.linalg.norm(points["forearm_dir"]),
        )
        print(f"[OpenArm pose-likeness {side}] upper_err={upper_err:.2e} forearm_err={forearm_err:.2e}")
        assert upper_err < 5e-4
        assert forearm_err < 5e-4


def test_generated_openarm_angles_within_limits(path="data/joint_angles_openarm_smoothed.npz"):
    if not os.path.exists(path):
        print(f"[OpenArm limits] skipped; {path} does not exist")
        return
    data = np.load(path)
    q = data["joint_angles"]
    limits = joint_limits_for_side("right")
    below = q < limits[:, 0]
    above = q > limits[:, 1]
    print(f"[OpenArm limits] frames={len(q)} below={below.sum()} above={above.sum()}")
    assert not below.any()
    assert not above.any()


if __name__ == "__main__":
    test_openarm_fk_home()
    test_openarm_pose_likeness_self_target()
    test_generated_openarm_angles_within_limits()
    print("OpenArm IK smoke tests passed.")
