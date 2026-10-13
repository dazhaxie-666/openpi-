"""Live MediaPipe pose-world landmarks -> OpenArm Gazebo forward-position control."""

import threading
import time
from collections import deque

import numpy as np
from scipy.optimize import least_squares

from ik_solver import OpenArmIK
from openarm_common import JOINT_VELOCITY_LIMITS, clip_to_limits, joint_limits_for_side
from pose_constants import LANDMARK
from retarget import (
    ARM_BETA,
    ARM_MIN_CUTOFF,
    OPENARM_ARM_LENGTH,
    OPENARM_ANCHOR,
    ROOT_BETA,
    ROOT_MIN_CUTOFF,
    Retargeter,
    compute_body_frame,
    openarm_body_to_robot_for_side,
    to_body_frame,
)
from openarm_common import OPENARM_BASE_OFFSETS
from smoothing import VectorOneEuroFilter
DUAL_ARM_SAFE_DISTANCE = 0.055
DUAL_ARM_COLLISION_WEIGHT = 45.0
DUAL_ARM_WARNING_DISTANCE = 0.060
DUAL_ARM_COUPLED_MAX_NFEV = 20
DUAL_ARM_FALLBACK_CONSECUTIVE_FRAMES = 2


class OpenArmLiveController:
    """Online equivalent of retarget.run_pipeline for one frame at a time."""

    def __init__(self, fps, n_calib_frames=10, side="right"):
        side = side.lower()
        self.root_filter = VectorOneEuroFilter(12, fps, min_cutoff=ROOT_MIN_CUTOFF, beta=ROOT_BETA)
        self.arm_filter = VectorOneEuroFilter(9, fps, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA)
        self.retargeter = Retargeter(
            robot_arm_length=OPENARM_ARM_LENGTH,
            robot_anchor=OPENARM_ANCHOR,
            body_to_robot=openarm_body_to_robot_for_side(side),
        )
        self.ik = OpenArmIK(side=side)
        self.joint_limits = joint_limits_for_side(side)
        self.side = side.upper()
        self.side_label = side
        self.n_calib_frames = max(1, int(n_calib_frames))
        self.calib_frames = []
        self.last_world_lm = None
        self.last_q = None
        self._bad_ik_frames = 0

    def process(self, world_lm):
        pose_target, status = self.pose_target_from_world(world_lm)
        if pose_target is None:
            return self.last_q, status

        return self.solve_target(pose_target)

    def solve_target(self, pose_target):
        try:
            q = self.ik.solve_pose_likeness(pose_target)
        except ValueError:
            self._bad_ik_frames += 1
            if self._bad_ik_frames >= 3:
                self.ik.reset()
                self._bad_ik_frames = 0
            reason = self.ik.last_reject_reason or "ik"
            return self.last_q, f"holding:{reason}"

        self._bad_ik_frames = 0
        return self.set_command(q), "tracking"

    def pose_target_from_world(self, world_lm):
        if world_lm is not None:
            self.last_world_lm = world_lm
        elif self.last_world_lm is None:
            return None, "no pose"
        else:
            world_lm = self.last_world_lm

        root = np.concatenate([
            world_lm[LANDMARK["LEFT_SHOULDER"], :3],
            world_lm[LANDMARK["RIGHT_SHOULDER"], :3],
            world_lm[LANDMARK["LEFT_HIP"], :3],
            world_lm[LANDMARK["RIGHT_HIP"], :3],
        ])
        if not np.isfinite(root).all():
            return None, "holding"

        l_sh, r_sh, l_hip, r_hip = self.root_filter.filter(root).reshape(4, 3)
        origin, body_R = compute_body_frame(l_sh, r_sh, l_hip, r_hip)

        S = to_body_frame(world_lm[LANDMARK[f"{self.side}_SHOULDER"], :3], origin, body_R)
        E = to_body_frame(world_lm[LANDMARK[f"{self.side}_ELBOW"], :3], origin, body_R)
        W = to_body_frame(world_lm[LANDMARK[f"{self.side}_WRIST"], :3], origin, body_R)
        if not np.isfinite(np.concatenate([S, E, W])).all():
            return None, "holding"

        S, E, W = self.arm_filter.filter(np.concatenate([S, E, W])).reshape(3, 3)
        if self.retargeter.scale is None:
            self.calib_frames.append((S, E, W))
            if len(self.calib_frames) < self.n_calib_frames:
                return None, f"calibrating {len(self.calib_frames)}/{self.n_calib_frames}"
            self.retargeter.calibrate(self.calib_frames)

        try:
            pose_target = self.retargeter.frame_to_pose_target(S, E, W)
        except ValueError:
            return None, "holding"

        return pose_target, "target"

    def set_command(self, q):
        q, _ = clip_to_limits(np.asarray([q], dtype=float), self.joint_limits)
        self.last_q = q[0]
        return self.last_q


def _segment_distance(a0, a1, b0, b1):
    a0 = np.asarray(a0, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    if denom < 1e-12:
        s = 0.0
        t = np.clip(e / max(c, 1e-12), 0.0, 1.0)
    else:
        s = np.clip((b * e - c * d) / denom, 0.0, 1.0)
        t = np.clip((a * e - b * d) / denom, 0.0, 1.0)
    pa = a0 + s * u
    pb = b0 + t * v
    return float(np.linalg.norm(pa - pb))


def _arm_collision_segments(ik, q, side):
    points = ik.forward_points(q)
    offset = OPENARM_BASE_OFFSETS[side]
    shoulder = points["shoulder"] + offset
    elbow = points["elbow"] + offset
    tool = points["tool"] + offset
    upper_mid = shoulder + 0.35 * (elbow - shoulder)
    return [(upper_mid, elbow), (elbow, tool)]


def _dual_arm_collision_residual(left_ik, left_q, right_ik, right_q,
                                 safe_distance=DUAL_ARM_SAFE_DISTANCE):
    left_segments = _arm_collision_segments(left_ik, left_q, "left")
    right_segments = _arm_collision_segments(right_ik, right_q, "right")
    residuals = []
    for left_seg in left_segments:
        for right_seg in right_segments:
            distance = _segment_distance(left_seg[0], left_seg[1], right_seg[0], right_seg[1])
            residuals.append(max(0.0, safe_distance - distance) / safe_distance)
    return np.asarray(residuals, dtype=float)


class DualOpenArmLiveController:
    """Coordinates both arms so pose-likeness IK avoids inter-arm capsule overlap."""

    def __init__(self, fps, n_calib_frames=10):
        self.controllers = {
            "left": OpenArmLiveController(fps, n_calib_frames=n_calib_frames, side="left"),
            "right": OpenArmLiveController(fps, n_calib_frames=n_calib_frames, side="right"),
        }
        self.last_mode = "init"
        self._near_frames = 0

    def process(self, world_lm):
        targets = {}
        statuses = {}
        for side, controller in self.controllers.items():
            target, status = controller.pose_target_from_world(world_lm)
            targets[side] = target
            statuses[side] = status

        if targets["left"] is None or targets["right"] is None:
            return {
                side: (controller.last_q, statuses[side])
                for side, controller in self.controllers.items()
            }

        left_q, left_status = self.controllers["left"].solve_target(targets["left"])
        right_q, right_status = self.controllers["right"].solve_target(targets["right"])
        if left_q is None or right_q is None or left_status != "tracking" or right_status != "tracking":
            self.last_mode = "hold"
            return {
                "left": (left_q, left_status),
                "right": (right_q, right_status),
            }

        min_distance = self.min_inter_arm_distance(left_q, right_q)
        if min_distance < DUAL_ARM_SAFE_DISTANCE:
            self._near_frames = DUAL_ARM_FALLBACK_CONSECUTIVE_FRAMES
        elif min_distance < DUAL_ARM_WARNING_DISTANCE:
            self._near_frames += 1
        else:
            self._near_frames = 0
        if self._near_frames < DUAL_ARM_FALLBACK_CONSECUTIVE_FRAMES:
            self.last_mode = "independent"
            return {
                "left": (left_q, "tracking"),
                "right": (right_q, "tracking"),
            }

        left_q, right_q = self._solve_coupled(targets)
        min_distance = self.min_inter_arm_distance(left_q, right_q)
        self.last_mode = "coupled"
        status = "tracking" if min_distance >= DUAL_ARM_SAFE_DISTANCE else "near"
        return {
            "left": (left_q, status),
            "right": (right_q, status),
        }

    def _solve_coupled(self, targets):
        left = self.controllers["left"]
        right = self.controllers["right"]
        left_seed = left.ik._last_solution if left.ik._last_solution is not None else left.ik.home_seed
        right_seed = right.ik._last_solution if right.ik._last_solution is not None else right.ik.home_seed
        left_seed = np.clip(left_seed, left.joint_limits[:, 0], left.joint_limits[:, 1])
        right_seed = np.clip(right_seed, right.joint_limits[:, 0], right.joint_limits[:, 1])
        seed = np.concatenate([left_seed, right_seed])

        def residual(x):
            left_q = x[:7]
            right_q = x[7:]
            collision = _dual_arm_collision_residual(left.ik, left_q, right.ik, right_q)
            return np.concatenate([
                left.ik.pose_likeness_residual(left_q, targets["left"], seed=left_seed),
                right.ik.pose_likeness_residual(right_q, targets["right"], seed=right_seed),
                np.sqrt(DUAL_ARM_COLLISION_WEIGHT) * collision,
            ])

        lower = np.concatenate([left.joint_limits[:, 0], right.joint_limits[:, 0]])
        upper = np.concatenate([left.joint_limits[:, 1], right.joint_limits[:, 1]])
        result = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=DUAL_ARM_COUPLED_MAX_NFEV,
            ftol=1e-4,
            xtol=1e-4,
            gtol=1e-4,
        )
        try:
            left_candidate = left.ik._accept_solution(result.x[:7], left_seed, had_previous_solution=True)
            right_candidate = right.ik._accept_solution(result.x[7:], right_seed, had_previous_solution=True)
        except ValueError:
            return left.last_q, right.last_q

        left_q = left.set_command(left_candidate)
        right_q = right.set_command(right_candidate)
        left.ik._last_solution = left_q
        right.ik._last_solution = right_q
        return left_q, right_q

    def min_inter_arm_distance(self, left_q=None, right_q=None):
        left = self.controllers["left"]
        right = self.controllers["right"]
        left_q = left.last_q if left_q is None else left_q
        right_q = right.last_q if right_q is None else right_q
        if left_q is None or right_q is None:
            return float("inf")
        distances = []
        for left_seg in _arm_collision_segments(left.ik, left_q, "left"):
            for right_seg in _arm_collision_segments(right.ik, right_q, "right"):
                distances.append(_segment_distance(left_seg[0], left_seg[1], right_seg[0], right_seg[1]))
        return min(distances) if distances else float("inf")


class GazeboForwardPositionPublisher:
    def __init__(self, side="right", wait_time=5.0, queue_depth=1):
        import rclpy
        from std_msgs.msg import Float64MultiArray

        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(f"pose2arm_live_{side}_forward_demo")
        topic = f"/{side}_forward_position_controller/commands"
        self.pub = self.node.create_publisher(Float64MultiArray, topic, int(queue_depth))
        self.joint_velocity_limits = JOINT_VELOCITY_LIMITS
        self._last_q = None
        deadline = self.node.get_clock().now() + self._duration(wait_time)
        while self.pub.get_subscription_count() == 0 and self.node.get_clock().now() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.pub.get_subscription_count() == 0:
            self.node.get_logger().warn(f"no subscribers on {topic}; continuing anyway")

    @staticmethod
    def _duration(seconds):
        from rclpy.duration import Duration
        return Duration(seconds=float(seconds))

    def _clamp_velocity(self, q, dt):
        q = np.asarray(q, dtype=float)
        if self._last_q is None:
            self._last_q = q
            return q
        max_delta = self.joint_velocity_limits * max(float(dt), 1e-6)
        q_limited = self._last_q + np.clip(q - self._last_q, -max_delta, max_delta)
        self._last_q = q_limited
        return q_limited

    def publish(self, q, dt):
        from std_msgs.msg import Float64MultiArray

        q = self._clamp_velocity(q, dt)
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q]
        self.pub.publish(msg)

    def spin_once(self):
        if self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        if self.rclpy.ok():
            self.node.destroy_node()


class OpenArmIKForwarder:
    """Latest-only pose mailbox feeding OpenArm IK and Gazebo commands.

    This intentionally is not a FIFO queue: if IK is busy, newer pose results
    overwrite older ones. The worker never catches up on stale skeletons.
    """

    def __init__(self, fps, side="both", profile=False, profile_interval=1.0):
        self.frame_period = 1.0 / max(1e-6, float(fps))
        self.profile = bool(profile)
        self.profile_interval = float(profile_interval)
        self.sides = ["right", "left"] if side == "both" else [side]
        self.controllers = {}
        self.publishers = {}
        self.dual_controller = None
        if side == "both":
            self.dual_controller = DualOpenArmLiveController(fps)
        else:
            self.controllers[side] = OpenArmLiveController(fps, side=side)
        for arm_side in self.sides:
            self.publishers[arm_side] = GazeboForwardPositionPublisher(side=arm_side)

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest = None
        self._last_processed_frame = None
        self._status_text = ""
        self._profile = {
            "submitted": 0,
            "processed": 0,
            "dropped": 0,
            "input_age_ms": deque(maxlen=240),
            "solve_publish_ms": deque(maxlen=240),
            "publish_period_ms": deque(maxlen=240),
            "independent": 0,
            "coupled": 0,
        }
        self._last_report_time = time.monotonic()
        self._last_publish_time = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="pose2arm_ik_forwarder", daemon=True)
        self._thread.start()

    @property
    def status_text(self):
        with self._lock:
            return self._status_text

    @property
    def profile_text(self):
        if not self.profile:
            return ""
        with self._lock:
            return self._format_profile_locked()

    def submit_latest_pose(self, world_lm, frame_idx, timestamp_ms, command_dt=None):
        if world_lm is None:
            return
        submitted_at = time.monotonic()
        item = {
            "world_lm": np.asarray(world_lm, dtype=float).copy(),
            "frame_idx": int(frame_idx),
            "timestamp_ms": float(timestamp_ms),
            "command_dt": self.frame_period if command_dt is None else float(command_dt),
            "submitted_at": submitted_at,
        }
        with self._condition:
            if self._latest is not None and self._latest["frame_idx"] != self._last_processed_frame:
                self._profile["dropped"] += 1
            self._profile["submitted"] += 1
            self._latest = item
            self._condition.notify()

    def _run(self):
        last_command_timestamp_ms = None
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or (
                        self._latest is not None
                        and self._latest["frame_idx"] != self._last_processed_frame
                    )
                )
                if self._stop:
                    break
                item = self._latest
                self._last_processed_frame = item["frame_idx"]

            started_at = time.monotonic()
            if last_command_timestamp_ms is None:
                command_dt = item["command_dt"]
            else:
                command_dt = max(
                    item["command_dt"],
                    (item["timestamp_ms"] - last_command_timestamp_ms) / 1000.0,
                )
            status_text, solve_mode = self._publish(item["world_lm"], command_dt)
            finished_at = time.monotonic()
            last_command_timestamp_ms = item["timestamp_ms"]
            self._spin_publishers()
            with self._lock:
                self._profile["processed"] += 1
                self._profile["input_age_ms"].append(1000.0 * max(0.0, started_at - item["submitted_at"]))
                self._profile["solve_publish_ms"].append(1000.0 * max(0.0, finished_at - started_at))
                if self._last_publish_time is not None:
                    self._profile["publish_period_ms"].append(1000.0 * max(0.0, finished_at - self._last_publish_time))
                if solve_mode in ("independent", "coupled"):
                    self._profile[solve_mode] += 1
                self._last_publish_time = finished_at
                self._status_text = status_text
                self._maybe_report_profile_locked(finished_at)

    def _publish(self, world_lm, command_dt):
        status_parts = []
        solve_mode = ""
        if self.dual_controller is not None:
            results = self.dual_controller.process(world_lm)
            solve_mode = self.dual_controller.last_mode
            for arm_side in self.sides:
                q, arm_status = results[arm_side]
                status_parts.append(f"{arm_side[0]}:{arm_status}")
                if q is not None:
                    self.publishers[arm_side].publish(q, command_dt)
        else:
            for arm_side, controller in self.controllers.items():
                q, arm_status = controller.process(world_lm)
                status_parts.append(f"{arm_side[0]}:{arm_status}")
                if q is not None:
                    self.publishers[arm_side].publish(q, command_dt)
        return " ".join(status_parts), solve_mode

    def _spin_publishers(self):
        for arm_publisher in self.publishers.values():
            arm_publisher.spin_once()

    @staticmethod
    def _stats(values):
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return float(np.mean(arr)), float(np.percentile(arr, 95)), float(np.max(arr))

    def _format_profile_locked(self):
        parts = [
            f"ik submitted={self._profile['submitted']}",
            f"processed={self._profile['processed']}",
            f"dropped={self._profile['dropped']}",
            f"ind={self._profile['independent']}",
            f"coupled={self._profile['coupled']}",
        ]
        for label, key in [
            ("age", "input_age_ms"),
            ("solve", "solve_publish_ms"),
            ("period", "publish_period_ms"),
        ]:
            stats = self._stats(self._profile[key])
            if stats is not None:
                mean, p95, max_v = stats
                parts.append(f"{label} mean/p95/max={mean:.1f}/{p95:.1f}/{max_v:.1f}ms")
        return " ".join(parts)

    def _maybe_report_profile_locked(self, now):
        if not self.profile or now - self._last_report_time < self.profile_interval:
            return
        print("[ik_profile]", self._format_profile_locked(), flush=True)
        self._last_report_time = now

    def close(self):
        with self._condition:
            self._stop = True
            self._condition.notify()
        self._thread.join(timeout=1.0)

        if self.dual_controller is not None:
            for arm_side, controller in self.dual_controller.controllers.items():
                if controller.last_q is not None:
                    self.publishers[arm_side].publish(controller.last_q, 0.25)
        else:
            for arm_side, controller in self.controllers.items():
                if controller.last_q is not None:
                    self.publishers[arm_side].publish(controller.last_q, 0.25)
        for arm_publisher in self.publishers.values():
            arm_publisher.close()
        if self.profile:
            with self._lock:
                print("[ik_profile_final]", self._format_profile_locked(), flush=True)
