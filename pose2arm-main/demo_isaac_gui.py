"""Live demo: webcam → MediaPipe → RL policy → Isaac Sim dual-arm visualization.

Latency optimizations vs prior version:
- MediaPipe GPU delegate (CPU ~50ms → GPU ~10ms; CPU fallback if unavailable)
- Policy runs at 60 Hz — DECIMATION removed
- Pre-allocated GPU obs_buf; in-place torch.sin/cos(out=) — no per-step tensor alloc
- Pinned memory + non_blocking async H2D for target transfer
- Pre-alloc _world_buf/_target_buf in MediaPipe thread — no per-frame numpy alloc
- Pre-resize frames inside MediaPipe thread; double-buffer bytearray with cv2.cvtColor(dst=)
  eliminates bytearray(tobytes()) and cv2.resize from the sim hot loop

Usage:
    source ~/isaac_venv/bin/activate
    python3 demo_isaac_gui.py                           # webcam index 0
    python3 demo_isaac_gui.py --source 1                # webcam index 1
    python3 demo_isaac_gui.py --source videos/demo.mp4  # pre-recorded video
"""

# ── cv2 and numpy must be imported before AppLauncher injects Isaac Sim's
#    bundled packages into sys.path (which shadows the correct RL-venv versions).
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2        # load from RL venv before AppLauncher modifies sys.path
import numpy as np

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--source",     default="videos/sample_0p25x_24fps.mp4",
                     help="Webcam index (int) or video file path")
_parser.add_argument("--checkpoint", default="",
                     help="Path to best_agent.pt (auto-detected if blank)")
_parser.add_argument("--scale", type=float, default=None,
                     help="Display scale override. Auto-computed from screen size if omitted.")
_args, _remaining = _parser.parse_known_args()

# ── AppLauncher must be instantiated before any Isaac Sim imports ─────────────
from isaaclab.app import AppLauncher
_launcher = AppLauncher(headless=False)
simulation_app = _launcher.app

# ── All other imports after Isaac Sim is live ─────────────────────────────────
# Inject pose-pipeline venv so mediapipe can be found. cv2/numpy are already
# cached in sys.modules from the pre-AppLauncher imports above.
_POSE2ARM_SITE = str(
    Path(__file__).resolve().parent / "venv" / "lib" /
    f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
)
if _POSE2ARM_SITE not in sys.path:
    sys.path.append(_POSE2ARM_SITE)

import mediapipe as mp
import torch
import torch.nn as nn
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from extract_pose import POSE_MODEL_PATH
from openarm_common import (
    T_UPPER_L, T_FORE_L, T_PLANE_L,
    T_UPPER_R, T_FORE_R, T_PLANE_R,
)
from openarm_live import OpenArmLiveController
from pose_constants import POSE_CONNECTIONS
from rl.robot_cfg import OPENARM_BI_CFG



def _detect_video_size(source: str) -> tuple[int, int]:
    """Return (width, height) of the video source without consuming frames."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w, h) if w > 0 and h > 0 else (640, 360)


def _auto_display_scale(vid_w: int, vid_h: int) -> float:
    """Scale so the display width is ~480px (landscape-friendly, safe for Isaac Sim UI)."""
    return 480.0 / vid_w


# ── Auto-detect latest checkpoint ────────────────────────────────────────────
def _find_checkpoint() -> str:
    if _args.checkpoint:
        return _args.checkpoint
    log_root   = REPO_ROOT / "logs" / "skrl" / "openarm_retarget"
    candidates = sorted(log_root.glob("*/checkpoints/best_agent.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No best_agent.pt found. Train first: python3 train_retarget.py"
        )
    return str(candidates[-1])


# ── Policy (inference-only, same architecture as skrl GaussianMixin) ──────────
class _RetargetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(92, 512), nn.ELU(),   # 18D target_t + 18D target_t+1 + 14D sin + 14D cos + 14D q_vel + 14D prev_action
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.head = nn.Linear(128, 14)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))


def _load_policy(ckpt_path: str, device: torch.device) -> _RetargetNet:
    sd  = torch.load(ckpt_path, map_location=device, weights_only=False)["policy"]
    net = _RetargetNet().to(device)
    net.trunk[0].weight.data = sd["net_container.0.weight"]
    net.trunk[0].bias.data   = sd["net_container.0.bias"]
    net.trunk[2].weight.data = sd["net_container.2.weight"]
    net.trunk[2].bias.data   = sd["net_container.2.bias"]
    net.trunk[4].weight.data = sd["net_container.4.weight"]
    net.trunk[4].bias.data   = sd["net_container.4.bias"]
    net.head.weight.data     = sd["net_container.6.weight"]
    net.head.bias.data       = sd["net_container.6.bias"]
    net.eval()
    return net


# ── Skeleton overlay helpers ─────────────────────────────────────────────────
def _draw_arm_axes(frame, lms, w, h, sh_idx, el_idx, pt) -> None:
    """Draw RGB axis arrows at shoulder/elbow from the PoseTarget body-frame vectors.

    The robot-frame directions are converted back to image space via the inverse of
    OPENARM_BODY_TO_ROBOT: body_x = -d[1], body_y = d[2]; then image_y = -body_y.
    Result: image_dx = -d[1], image_dy = -d[2] (Z = depth, invisible in projection).
    """
    sh, el = lms[sh_idx], lms[el_idx]
    if sh.visibility < 0.4 or el.visibility < 0.4:
        return
    arrow_len = max(25, w // 12)
    sh_px = (int(sh.x * w), int(sh.y * h))
    el_px = (int(el.x * w), int(el.y * h))

    def _arrow(anchor, d, color):
        dx, dy = -float(d[1]), -float(d[2])
        n = (dx * dx + dy * dy) ** 0.5
        if n < 1e-6:
            return
        ex = int(anchor[0] + dx / n * arrow_len)
        ey = int(anchor[1] + dy / n * arrow_len)
        cv2.arrowedLine(frame, anchor, (ex, ey), color, 2, tipLength=0.3)

    _arrow(sh_px, pt.upper_dir,   (0,   0,   255))  # red  – upper arm direction
    _arrow(sh_px, pt.arm_plane,   (255, 0,   0  ))  # blue – arm plane normal
    _arrow(el_px, pt.forearm_dir, (0,   255, 0  ))  # green – forearm direction


# ── Thread-safe latest-only mailbox ──────────────────────────────────────────
class _Mailbox:
    __slots__ = ("_lock", "_value", "_status", "_new")

    def __init__(self):
        self._lock   = threading.Lock()
        self._value  = None
        self._status = "waiting for camera…"
        self._new    = False

    def put(self, value, status: str):
        with self._lock:
            self._value  = value
            self._status = status
            self._new    = True

    def get_if_new(self):
        """Return (value, status, is_new). is_new flag clears after first read."""
        with self._lock:
            v, s, n   = self._value, self._status, self._new
            self._new = False
        return v, s, n

    def peek(self):
        """Return (value, status) without clearing the new flag."""
        with self._lock:
            return self._value, self._status


# ── MediaPipe background thread ───────────────────────────────────────────────
class MediaPipeThread(threading.Thread):
    """
    Reads frames, runs MediaPipe pose detection, retargets both arms,
    and writes the 24-D target observation into a mailbox.

    All per-frame buffers are pre-allocated at init time to keep _on_result
    and run() allocation-free in steady state.
    """

    def __init__(self, source: str, mailbox: _Mailbox, frame_mailbox: _Mailbox,
                 vid_w: int, vid_h: int, fps: float = 30.0):
        super().__init__(daemon=True, name="mediapipe")
        self.source        = source
        self.mailbox       = mailbox
        self.frame_mailbox = frame_mailbox
        self.vid_w         = vid_w
        self.vid_h         = vid_h
        self.fps           = fps
        self.running       = False
        self._ctrl_L = OpenArmLiveController(fps, side="left")
        self._ctrl_R = OpenArmLiveController(fps, side="right")
        self._frame_idx = 0
        self._lm_lock   = threading.Lock()
        self._img_landmarks = None

        self._pt_L: "PoseTarget | None" = None  # arm pose for left;  protected by _lm_lock
        self._pt_R: "PoseTarget | None" = None  # arm pose for right; protected by _lm_lock

        # Pre-allocated CPU buffers — no per-frame heap allocation in hot path
        self._world_buf  = np.zeros((33, 4), dtype=np.float32)  # MediaPipe landmarks
        self._target_buf = np.zeros(18, dtype=np.float32)        # 18D target obs

        # Double-buffer for video frames: MediaPipe writes into back buffer,
        # sim loop always peeks the most recently completed front buffer.
        # cv2.cvtColor(dst=) writes directly into the bytearray via numpy view.
        self._vid_bufs = [
            bytearray(vid_w * vid_h * 4),
            bytearray(vid_w * vid_h * 4),
        ]
        self._vid_np = [
            np.frombuffer(self._vid_bufs[0], dtype=np.uint8).reshape(vid_h, vid_w, 4),
            np.frombuffer(self._vid_bufs[1], dtype=np.uint8).reshape(vid_h, vid_w, 4),
        ]
        self._back = 0  # index of the buffer currently being written

    @staticmethod
    def _draw_skeleton(frame: np.ndarray, landmarks, pt_L=None, pt_R=None) -> None:
        h, w = frame.shape[:2]
        for idx1, idx2 in POSE_CONNECTIONS:
            lm1, lm2 = landmarks[idx1], landmarks[idx2]
            if lm1.visibility > 0.4 and lm2.visibility > 0.4:
                cv2.line(frame,
                         (int(lm1.x * w), int(lm1.y * h)),
                         (int(lm2.x * w), int(lm2.y * h)),
                         (0, 230, 80), 2)
        for lm in landmarks:
            if lm.visibility > 0.4:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 120, 0), -1)
        if pt_L is not None:
            _draw_arm_axes(frame, landmarks, w, h, 11, 13, pt_L)
        if pt_R is not None:
            _draw_arm_axes(frame, landmarks, w, h, 12, 14, pt_R)

    def _on_result(self, result, _image, _timestamp_ms):
        img_lm = result.pose_landmarks[0] if result.pose_landmarks else None

        if not result.pose_world_landmarks:
            if img_lm is not None:
                with self._lm_lock:
                    self._img_landmarks = img_lm
            self.mailbox.put(None, "no pose detected")
            return

        # Write into pre-allocated buffer — no list comprehension, no heap alloc
        buf = self._world_buf
        for i, lm in enumerate(result.pose_world_landmarks[0]):
            buf[i, 0] = lm.x
            buf[i, 1] = lm.y
            buf[i, 2] = lm.z
            buf[i, 3] = lm.visibility

        pt_L, status_L = self._ctrl_L.pose_target_from_world(buf)
        pt_R, status_R = self._ctrl_R.pose_target_from_world(buf)

        # Single lock acquisition: update image landmarks and arm poses atomically
        with self._lm_lock:
            if img_lm is not None:
                self._img_landmarks = img_lm
            if pt_L is not None:
                self._pt_L = pt_L
            if pt_R is not None:
                self._pt_R = pt_R

        status = f"L:{status_L}  R:{status_R}"
        if pt_L is None or pt_R is None:
            self.mailbox.put(None, status)
            return

        # Fill 18D target in-place — all unit vectors, no scaling needed
        t = self._target_buf
        t[T_UPPER_L] = pt_L.upper_dir
        t[T_FORE_L]  = pt_L.forearm_dir
        t[T_PLANE_L] = pt_L.arm_plane
        t[T_UPPER_R] = pt_R.upper_dir
        t[T_FORE_R]  = pt_R.forearm_dir
        t[T_PLANE_R] = pt_R.arm_plane

        self.mailbox.put(t.copy(), "tracking")  # 72-byte copy: safe for concurrent read

    def run(self):
        # Try GPU delegate first, fall back to CPU
        for delegate in (BaseOptions.Delegate.GPU, BaseOptions.Delegate.CPU):
            try:
                options = vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=POSE_MODEL_PATH,
                        delegate=delegate,
                    ),
                    running_mode=vision.RunningMode.LIVE_STREAM,
                    num_poses=1,
                    result_callback=self._on_result,
                )
                landmarker = vision.PoseLandmarker.create_from_options(options)
                print(f"[demo] MediaPipe: {delegate.name} delegate", flush=True)
                break
            except Exception as e:
                if delegate == BaseOptions.Delegate.CPU:
                    raise
                print(f"[demo] MediaPipe GPU delegate failed ({e}), retrying CPU", flush=True)

        src = int(self.source) if self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[demo] ERROR: cannot open video source '{self.source}'", flush=True)
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS)
        if detected_fps > 0:
            self.fps = detected_fps
            print(f"[demo] video FPS detected: {detected_fps:.1f}", flush=True)

        self.running = True
        period = 1.0 / self.fps
        t_next = time.monotonic()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                if self.source.isdigit():
                    time.sleep(0.01)
                    continue
                break

            # Cap MediaPipe input to 1280px on the long edge — avoids OOM on 4K sources
            _mp_max = 1280
            _h, _w = frame.shape[:2]
            if max(_h, _w) > _mp_max:
                _s = _mp_max / max(_h, _w)
                mp_frame = cv2.resize(frame, (int(_w * _s), int(_h * _s)))
            else:
                mp_frame = frame
            rgb    = cv2.cvtColor(mp_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_img, int(self._frame_idx * 1000 / self.fps))

            # Draw skeleton on original frame, resize to panel dims, then
            # write BGR→RGBA directly into back bytearray via numpy view (no alloc)
            with self._lm_lock:
                lm   = self._img_landmarks
                pt_L = self._pt_L
                pt_R = self._pt_R
            if lm is not None:
                self._draw_skeleton(frame, lm, pt_L, pt_R)
            if frame.shape[1] != self.vid_w or frame.shape[0] != self.vid_h:
                frame = cv2.resize(frame, (self.vid_w, self.vid_h))
            back = self._back
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA, dst=self._vid_np[back])
            self.frame_mailbox.put(self._vid_bufs[back], "frame")
            self._back = 1 - back   # swap to other buffer

            self._frame_idx += 1
            t_next += period
            wait = t_next - time.monotonic()
            if wait > 0:
                time.sleep(wait)

        cap.release()
        landmarker.close()
        self.running = False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, render_interval=1))
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    robot_cfg = OPENARM_BI_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    sim.reset()
    robot.reset()

    try:
        import carb
        import omni.kit.viewport.utility as vu
        from pxr import Gf
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
        vp  = vu.get_active_viewport()
        cam = ViewportCameraState("/OmniverseKit_Persp", vp)
        cam.set_position_world(Gf.Vec3d(2.0, 0.0, 0.8), True)
        cam.set_target_world(Gf.Vec3d(0.0, 0.0, 0.3), True)
        print("[demo] RTX + front camera set")
    except Exception as e:
        print(f"[demo] camera/render setup skipped: {e}")

    left_ids,  _ = robot.find_joints("openarm_left_joint[1-7]")
    right_ids, _ = robot.find_joints("openarm_right_joint[1-7]")
    arm_ids = left_ids + right_ids

    limits        = robot.data.soft_joint_pos_limits[0, arm_ids, :]
    q_low         = limits[:, 0].to(device)
    q_range_half  = (0.5 * (limits[:, 1] - limits[:, 0])).to(device)

    def _unscale(a: torch.Tensor) -> torch.Tensor:
        return q_low + (a + 1.0) * q_range_half

    ckpt_path = _find_checkpoint()
    policy    = _load_policy(ckpt_path, device)
    print(f"[demo] policy loaded  → {ckpt_path}")
    try:
        policy = torch.compile(policy, mode='reduce-overhead')
        print("[demo] policy compiled (reduce-overhead)")
    except Exception:
        print("[demo] torch.compile unavailable, using eager mode")
    with torch.no_grad():
        policy(torch.zeros(1, 92, device=device))
    print("[demo] policy warmed up")

    native_w, native_h = _detect_video_size(_args.source)
    scale = _args.scale if _args.scale is not None else _auto_display_scale(native_w, native_h)
    vid_w = max(1, int(native_w * scale))
    vid_h = max(1, int(native_h * scale))
    print(f"[demo] video size: {native_w}×{native_h} → display {vid_w}×{vid_h} (scale={scale:.3f})")

    import omni.ui as ui
    video_window = ui.Window("Reference Video", width=vid_w, height=vid_h + 30,
                             flags=ui.WINDOW_FLAGS_NO_SCROLLBAR)
    with video_window.frame:
        image_provider = ui.ByteImageProvider()
        ui.ImageWithProvider(image_provider, width=vid_w, height=vid_h)

    mailbox       = _Mailbox()
    frame_mailbox = _Mailbox()
    mp_thread = MediaPipeThread(source=_args.source, mailbox=mailbox,
                                frame_mailbox=frame_mailbox,
                                vid_w=vid_w, vid_h=vid_h)
    mp_thread.start()
    print(f"[demo] camera thread  → source='{_args.source}'")
    print("[demo] entering sim loop. Stand in front of the camera to control the robot.\n")

    # ── Pre-allocated GPU buffers — zero tensor allocations in the hot loop ──
    q_current = torch.zeros(1, 14, device=device)
    q_target  = torch.zeros(1, 14, device=device)
    # obs layout: [target_t(18)|target_t+1(18)|sin(q)(14)|cos(q)(14)|q_vel/20.944(14)|prev_action(14)] = 92D
    # In live demo next_target mirrors current_target (no lookahead from dataset)
    obs_buf = torch.zeros(1, 92, device=device)
    # Pinned host memory for async H2D — numpy view writes directly into it
    target_pinned    = torch.zeros(1, 18, dtype=torch.float32).pin_memory()
    target_pinned_np = target_pinned.numpy()   # zero-copy view of pinned buffer
    prev_action = torch.zeros(1, 14, device=device)   # action-rate obs slot
    has_target = False

    step       = 0
    status     = "waiting…"
    t_last_log = time.monotonic()

    while simulation_app.is_running():
        # New target available? — async copy 96 bytes CPU→GPU
        target_np, status, is_new = mailbox.get_if_new()
        if is_new and target_np is not None:
            np.copyto(target_pinned_np[0], target_np)
            obs_buf[:, :18].copy_(target_pinned, non_blocking=True)   # target_t
            obs_buf[:, 18:36].copy_(target_pinned, non_blocking=True)  # target_t+1 ≈ target_t
            has_target = True

        # Policy at 60 Hz — in-place sin/cos + q_vel, no tensor allocation
        if has_target:
            torch.sin(q_current, out=obs_buf[:, 36:50])
            torch.cos(q_current, out=obs_buf[:, 50:64])
            obs_buf[:, 64:78] = robot.data.joint_vel[:, arm_ids] * (1.0 / 20.944)
            obs_buf[:, 78:92] = prev_action
            with torch.no_grad():
                raw_action = policy(obs_buf).clamp(-1.0, 1.0)
                q_target = _unscale(raw_action)
                prev_action.copy_(raw_action)

        robot.set_joint_position_target(q_target, joint_ids=arm_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)
        q_current[:] = robot.data.joint_pos[:, arm_ids]

        # Video panel: only upload when a new frame has arrived (~30 Hz vs 60 Hz sim)
        frame_bytes, _, frame_is_new = frame_mailbox.get_if_new()
        if frame_is_new and frame_bytes is not None:
            image_provider.set_bytes_data(
                frame_bytes, [vid_w, vid_h], ui.TextureFormat.RGBA8_UNORM
            )

        simulation_app.update()

        if step % 300 == 0 and step > 0:
            now = time.monotonic()
            print(f"[demo] step {step:7d} | {status} | "
                  f"sim {300 / (now - t_last_log):.1f} Hz", flush=True)
            t_last_log = now
        step += 1

    mp_thread.running = False
    simulation_app.close()


if __name__ == "__main__":
    main()
