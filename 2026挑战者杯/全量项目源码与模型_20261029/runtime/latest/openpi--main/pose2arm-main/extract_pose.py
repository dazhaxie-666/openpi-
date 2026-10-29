"""Per-frame body keypoint extraction from a pre-recorded video file using
the MediaPipe Tasks API.

Deliberately uses the VIDEO running mode (`detect_for_video`, fed real frame
timestamps from the source file) rather than the IMAGE mode looped frame by
frame: VIDEO mode enables MediaPipe's own cross-frame tracking, which is
where most of the "is this jittery" difference between MediaPipe and a
per-frame detector (e.g. YOLO-pose) comes from. See retarget.py/openarm_live.py
for the rest of the pipeline -- this module only produces raw landmark arrays.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from pose_constants import LANDMARK, NUM_POSE_LANDMARKS, POSE_CONNECTIONS

_HERE = Path(__file__).resolve().parent
_MODEL_DIR_CANDIDATES = [
    _HERE / "models",
    _HERE.parent / "models",
    Path.cwd() / "models",
]


def _model_path(filename):
    for model_dir in _MODEL_DIR_CANDIDATES:
        path = model_dir / filename
        if path.exists():
            return str(path)
    return str(_MODEL_DIR_CANDIDATES[0] / filename)


# heavy, not lite: this project processes pre-recorded video offline, so the
# heavy model's ~3x slower inference doesn't matter, and it measurably
# reduces frame-to-frame jitter on real clips; see FAILURE_MODES.md.
POSE_MODEL_PATH = _model_path("pose_landmarker_heavy.task")

def _delegate_from_name(name):
    if name == "cpu":
        return BaseOptions.Delegate.CPU
    if name == "gpu":
        return BaseOptions.Delegate.GPU
    raise ValueError(f"unsupported MediaPipe delegate: {name}")


def _make_pose_landmarker(model_path=POSE_MODEL_PATH, delegate="cpu",
                          running_mode=vision.RunningMode.VIDEO, result_callback=None):
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=_delegate_from_name(delegate)),
        running_mode=running_mode,
        num_poses=1,
        result_callback=result_callback,
    )
    try:
        return vision.PoseLandmarker.create_from_options(options)
    except Exception:
        if delegate != "gpu":
            raise
        print("MediaPipe GPU delegate unavailable; falling back to CPU delegate", flush=True)
        cpu_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
            running_mode=running_mode,
            num_poses=1,
            result_callback=result_callback,
        )
        return vision.PoseLandmarker.create_from_options(cpu_options)


def extract_from_video(video_path, delegate="cpu"):
    """Returns a dict of per-frame arrays:
      - "pose": (T, 33, 4) array of (x, y, z, visibility), normalized image coords
                (x, y in [0,1] relative to frame size; z is MediaPipe's relative-depth
                estimate, NOT metric). Used only for drawing overlays on the source
                video -- pixel positions need image-space coordinates.
      - "pose_world": (T, 33, 4) array of (x, y, z, visibility), REAL-WORLD METRIC
                coordinates in meters, origin at the hip midpoint. This is what
                retarget.py actually uses for all geometry: BlazePose's world-landmark
                head is trained against Google's GHUM statistical body model, so it
                carries population-level anatomical proportions as an implicit prior --
                empirically ~80% less frame-to-frame jitter than deriving the same
                geometry from the normalized "pose" field, with zero added latency
                (same forward pass, different output head). See FAILURE_MODES.md.
      - "pose_present": (T,) bool, False on frames where no person was detected
      - "fps": float, the source video's frame rate (frames are processed in VIDEO mode
        using this for the timestamp clock MediaPipe's tracker expects)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pose_landmarker = _make_pose_landmarker(delegate=delegate)

    poses, poses_world, pose_present = [], [], []

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            timestamp_ms = int(round((frame_idx / fps) * 1000))
            mp_image = _to_mp_image(frame_bgr)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_result.pose_landmarks and pose_result.pose_world_landmarks:
                lm, wlm = pose_result.pose_landmarks[0], pose_result.pose_world_landmarks[0]
                poses.append([[p.x, p.y, p.z, p.visibility] for p in lm])
                poses_world.append([[p.x, p.y, p.z, p.visibility or 0.0] for p in wlm])
                pose_present.append(True)
            else:
                poses.append(np.full((NUM_POSE_LANDMARKS, 4), np.nan).tolist())
                poses_world.append(np.full((NUM_POSE_LANDMARKS, 4), np.nan).tolist())
                pose_present.append(False)

            frame_idx += 1
    finally:
        cap.release()
        pose_landmarker.close()

    out = {
        "pose": np.array(poses, dtype=float),
        "pose_world": np.array(poses_world, dtype=float),
        "pose_present": np.array(pose_present, dtype=bool),
        "fps": fps,
    }
    return out


def _to_mp_image(frame_bgr):
    import mediapipe as mp
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def write_overlay_video(video_path, pose_array, pose_present, fps, out_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx < len(pose_present) and pose_present[frame_idx]:
            lm = pose_array[frame_idx]
            pts = [(int(x * width), int(y * height)) for x, y, _, _ in lm]
            for a, b in POSE_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for x, y in pts:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        else:
            cv2.putText(frame, "NO POSE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="path to write keypoints .npz")
    parser.add_argument("--overlay-video", default=None, help="optional skeleton-overlay mp4 for visual sanity check")
    parser.add_argument("--delegate", choices=["cpu", "gpu"], default="cpu", help="MediaPipe inference delegate")
    args = parser.parse_args()

    result = extract_from_video(args.input, delegate=args.delegate)
    detected = int(result["pose_present"].sum())
    total = len(result["pose_present"])
    print(f"pose detected on {detected}/{total} frames ({100 * detected / max(total, 1):.1f}%)")

    np.savez(args.output, **result)
    print(f"wrote {args.output}")

    if args.overlay_video:
        write_overlay_video(args.input, result["pose"], result["pose_present"], result["fps"], args.overlay_video)
        print(f"wrote {args.overlay_video}")


if __name__ == "__main__":
    main()
