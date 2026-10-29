"""A/B MediaPipe Hand Landmarker vs RTMPose-Hand on the same frames.

This is a diagnostic script. It does not drive the robot and does not modify
the production hand backend. Both models see every source frame independently;
their keypoints, pinch ratios, latency and detection gaps are saved for review.

Run with the separate pose3d environment:

    python scripts/ab_hand_backends.py --camera 0 --seconds 20
    python scripts/ab_hand_backends.py --video backhand.mp4

RTMPose defaults to --rtm-device auto: DirectML if available, otherwise CPU.
On this laptop DirectML uses the RTX 4060 through onnxruntime-directml.

Outputs default to test_outputs/hand_ab/<timestamp>/:
    raw.mp4, mediapipe.mp4, rtmpose.mp4, side_by_side.mp4
    per_frame.csv, keypoints.npz, summary.json

Both keypoint sets use the 21-point OpenPose/MediaPipe order:
0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from rtmlib import Hand

SCRIPT = Path(__file__).resolve()
POSE2ARM = SCRIPT.parents[1]
PROJECT = SCRIPT.parents[4]
DEFAULT_MODEL = POSE2ARM / "models" / "hand_landmarker.task"

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# MediaPipe/OpenPose hand5 order used by both backends.
THUMB_TIP, INDEX_TIP = 4, 8
INDEX_MCP, PINKY_MCP = 5, 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="recorded video file")
    source.add_argument("--camera", type=int, help="USB camera index, e.g. 0")
    parser.add_argument("--seconds", type=float, default=20.,
                        help="camera recording duration (default 20 s)")
    parser.add_argument("--start", type=float, default=0.,
                        help="video start offset in seconds")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--mp-confidence", type=float, default=.5,
                        help="MediaPipe detection/presence/tracking threshold")
    parser.add_argument("--rtm-device", default="auto",
                        choices=("auto", "cpu", "dml", "cuda"),
                        help="ONNX Runtime provider: auto, cpu, dml (DirectML), or cuda")
    parser.add_argument("--close-ratio", type=float, default=.35)
    parser.add_argument("--open-ratio", type=float, default=.80)
    parser.add_argument("--dwell", type=float, default=.15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-video", action="store_true",
                        help="do not write MP4 files")
    parser.add_argument("--preview", action="store_true",
                        help="show a live side-by-side window")
    return parser.parse_args()


def open_source(args: argparse.Namespace):
    if args.video:
        path = args.video.expanduser().resolve()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps)
        if args.start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.)
        return cap, path.stem, fps, False
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"camera {args.camera} could not be opened")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, f"camera{args.camera}", float(args.fps), True


def make_mediapipe(model_path: Path, confidence: float) -> vision.HandLandmarker:
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def resolve_rtm_device(device: str) -> str:
    if device != "auto":
        return device
    import onnxruntime as ort
    providers = ort.get_available_providers()
    return "dml" if "DmlExecutionProvider" in providers else "cpu"


def make_rtmpose(device: str) -> Hand:
    if device == "dml":
        import onnxruntime as ort
        from rtmlib.tools.base import RTMLIB_SETTINGS
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "DmlExecutionProvider unavailable; install onnxruntime-directml")
        RTMLIB_SETTINGS["onnxruntime"]["dml"] = "DmlExecutionProvider"
    # Do not use to_openpose=True here: rtmlib's body COCO->OpenPose
    # conversion inserts a neck point and returns 22 keypoints for a hand.
    # The native hand5 order is already wrist, thumb, index, middle, ring,
    # pinky, matching MediaPipe/OpenPose hand order.
    return Hand(to_openpose=False, backend="onnxruntime", device=device)


def mediapipe_hands(result, width: int, height: int) -> list[dict]:
    candidates = []
    for index, hand in enumerate(result.hand_landmarks):
        points = np.array([[p.x * width, p.y * height] for p in hand], dtype=float)
        if points.shape != (21, 2) or not np.isfinite(points).all():
            continue
        world = None
        if index < len(result.hand_world_landmarks):
            world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[index]],
                             dtype=float)
        score = 1.0
        if index < len(result.handedness) and result.handedness[index]:
            score = float(result.handedness[index][0].score)
        candidates.append(dict(points=points, world=world, score=score))
    return candidates


def rtmpose_hands(hand_model: Hand, frame: np.ndarray) -> list[dict]:
    keypoints, scores = hand_model(frame)
    keypoints = np.asarray(keypoints, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if keypoints.size == 0:
        return []
    if keypoints.ndim == 2:
        keypoints = keypoints[None]
        scores = scores[None]
    candidates = []
    for points, point_scores in zip(keypoints, scores):
        if points.shape != (21, 2) or not np.isfinite(points).all():
            continue
        candidates.append(dict(points=points, world=None,
                               score=float(np.mean(point_scores)),
                               point_scores=point_scores))
    return candidates


def bbox_area(points: np.ndarray) -> float:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    return float(max(0., high[0] - low[0]) * max(0., high[1] - low[1]))


def select_hand(candidates: list[dict], previous: np.ndarray | None,
                width: int, height: int) -> dict | None:
    """Pick one hand. Prefer temporal continuity, then the largest detection."""
    usable = []
    margin = 0.08
    for candidate in candidates:
        points = candidate["points"]
        if (np.any(points[:, 0] < -margin * width)
                or np.any(points[:, 0] > (1 + margin) * width)
                or np.any(points[:, 1] < -margin * height)
                or np.any(points[:, 1] > (1 + margin) * height)):
            continue
        usable.append(candidate)
    if not usable:
        return None
    if previous is not None and np.isfinite(previous).all():
        return min(usable, key=lambda c: float(np.linalg.norm(c["points"][0] - previous)))
    return max(usable, key=lambda c: bbox_area(c["points"]) * max(c["score"], .05))


def pinch_ratio(points: np.ndarray) -> float:
    palm = float(np.linalg.norm(points[INDEX_MCP] - points[PINKY_MCP]))
    pinch = float(np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP]))
    if palm < 1e-6 or not np.isfinite(pinch):
        return float("nan")
    return pinch / palm


def pinch_ratio_3d(world: np.ndarray | None) -> float:
    if world is None or world.shape != (21, 3) or not np.isfinite(world).all():
        return float("nan")
    palm = float(np.linalg.norm(world[INDEX_MCP] - world[PINKY_MCP]))
    pinch = float(np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP]))
    return pinch / palm if palm > 1e-9 else float("nan")


def draw_hand(image: np.ndarray, points: np.ndarray | None, color,
              title: str, score: float = float("nan"), ratio: float = float("nan"),
              ms: float = float("nan")) -> None:
    height, width = image.shape[:2]
    if points is not None and points.shape == (21, 2):
        p = np.round(points).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(image, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)
        for index, point in enumerate(p):
            cv2.circle(image, tuple(point), 3 if index not in (4, 8) else 5,
                       color, -1, cv2.LINE_AA)
    lines = [title,
             f"score {score:.2f}" if np.isfinite(score) else "score -",
             f"pinch {ratio:.3f}" if np.isfinite(ratio) else "pinch -",
             f"{ms:.1f} ms" if np.isfinite(ms) else "time -"]
    for index, text in enumerate(lines):
        y = 22 + 24 * index
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    color, 1, cv2.LINE_AA)


def make_panel(frame: np.ndarray, points, title, color, score, ratio, ms,
               panel_size=(640, 360)) -> np.ndarray:
    panel = frame.copy()
    draw_hand(panel, points, color, title, score, ratio, ms)
    return cv2.resize(panel, panel_size, interpolation=cv2.INTER_AREA)


def ratio_events(values: np.ndarray, times: np.ndarray, close: float,
                 open_: float, dwell: float) -> list[dict]:
    events = []
    state = None
    candidate = None
    since = None
    for value, stamp in zip(values, times):
        if not np.isfinite(value):
            candidate = None
            continue
        label = "closed" if value <= close else "open" if value >= open_ else None
        if label is None:
            candidate = None
            continue
        if label != candidate:
            candidate = label
            since = stamp
        elif label != state and stamp - since >= dwell:
            state = label
            events.append(dict(time_s=float(stamp), state=label, ratio=float(value)))
    return events


def longest_missing(present: np.ndarray) -> int:
    best = run = 0
    for value in present:
        run = 0 if value else run + 1
        best = max(best, run)
    return int(best)


def model_summary(name: str, rows: list[dict], times: np.ndarray,
                  args: argparse.Namespace) -> dict:
    present = np.array([row[f"{name}_present"] for row in rows], dtype=bool)
    ratios = np.array([row[f"{name}_ratio"] for row in rows], dtype=float)
    latencies = np.array([row[f"{name}_ms"] for row in rows], dtype=float)
    events = ratio_events(ratios, times, args.close_ratio, args.open_ratio, args.dwell)
    return dict(
        detection_rate=float(present.mean()) if len(present) else 0.,
        missing_frames=int((~present).sum()),
        longest_missing_frames=longest_missing(present),
        pinch_valid_rate=float(np.isfinite(ratios).mean()) if len(ratios) else 0.,
        pinch_median=float(np.nanmedian(ratios)) if np.isfinite(ratios).any() else None,
        pinch_p05=float(np.nanpercentile(ratios, 5)) if np.isfinite(ratios).any() else None,
        pinch_p95=float(np.nanpercentile(ratios, 95)) if np.isfinite(ratios).any() else None,
        latency_ms_p50=float(np.nanpercentile(latencies, 50)) if len(latencies) else None,
        latency_ms_p95=float(np.nanpercentile(latencies, 95)) if len(latencies) else None,
        events=events,
    )


def match_events(left: list[dict], right: list[dict], tolerance: float = .5) -> dict:
    used = set()
    deltas = []
    matched = 0
    for event in left:
        best = None
        for index, other in enumerate(right):
            if index in used or other["state"] != event["state"]:
                continue
            delta = abs(other["time_s"] - event["time_s"])
            if delta <= tolerance and (best is None or delta < best[0]):
                best = (delta, index)
        if best is not None:
            used.add(best[1])
            matched += 1
            deltas.append(best[0])
    return dict(matched=matched, left_events=len(left), right_events=len(right),
                median_abs_delta_s=float(np.median(deltas)) if deltas else None)


def main() -> int:
    args = parse_args()
    if args.seconds <= 0 and args.camera is not None:
        raise ValueError("--seconds must be positive")
    if args.close_ratio >= args.open_ratio:
        raise ValueError("--close-ratio must be smaller than --open-ratio")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or (PROJECT / "test_outputs" / "hand_ab" / stamp)).resolve()
    output.mkdir(parents=True, exist_ok=False)

    cap, source_name, source_fps, live = open_source(args)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid source resolution")

    mp_model = make_mediapipe(DEFAULT_MODEL, args.mp_confidence)
    rtm_device = resolve_rtm_device(args.rtm_device)
    print(f"RTMPose device: {rtm_device}", flush=True)
    rtm_model = make_rtmpose(rtm_device)
    writers = {}
    panel_size = (640, 360)
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers["raw"] = cv2.VideoWriter(
            str(output / "raw.mp4"), fourcc, source_fps, (width, height))
        writers["mediapipe"] = cv2.VideoWriter(
            str(output / "mediapipe.mp4"), fourcc, source_fps, panel_size)
        writers["rtmpose"] = cv2.VideoWriter(
            str(output / "rtmpose.mp4"), fourcc, source_fps, panel_size)
        writers["side_by_side"] = cv2.VideoWriter(
            str(output / "side_by_side.mp4"), fourcc, source_fps, (1280, 360))
        for name, writer in writers.items():
            if not writer.isOpened():
                raise RuntimeError(f"video writer failed: {name}")

    rows: list[dict] = []
    mp_points, mp_world, mp_scores, mp_ratios, mp_ratios3d = [], [], [], [], []
    rtm_points, rtm_scores, rtm_ratios = [], [], []
    previous_mp = previous_rtm = None
    missing_mp = missing_rtm = 0
    last_ts = -1
    started = time.perf_counter()
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
            if live:
                elapsed = time.perf_counter() - started
                if elapsed >= args.seconds:
                    break
                stamp_s = elapsed
            else:
                stamp_s = frame_index / max(source_fps, 1e-6)
            ts_ms = max(last_ts + 1, int(round(stamp_s * 1000)))
            last_ts = ts_ms

            mp_start = time.perf_counter()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_result = mp_model.detect_for_video(mp_image, ts_ms)
            mp_ms = (time.perf_counter() - mp_start) * 1000.
            mp_candidates = mediapipe_hands(mp_result, width, height)
            mp_choice = select_hand(mp_candidates, previous_mp, width, height)
            if mp_choice is None:
                missing_mp += 1
                previous_mp = None if missing_mp > 15 else previous_mp
            else:
                missing_mp = 0
                previous_mp = mp_choice["points"][0].copy()

            rtm_start = time.perf_counter()
            rtm_candidates = rtmpose_hands(rtm_model, frame)
            rtm_ms = (time.perf_counter() - rtm_start) * 1000.
            rtm_choice = select_hand(rtm_candidates, previous_rtm, width, height)
            if rtm_choice is None:
                missing_rtm += 1
                previous_rtm = None if missing_rtm > 15 else previous_rtm
            else:
                missing_rtm = 0
                previous_rtm = rtm_choice["points"][0].copy()

            mp_ratio = pinch_ratio(mp_choice["points"]) if mp_choice else float("nan")
            mp_ratio3d = pinch_ratio_3d(mp_choice["world"]) if mp_choice else float("nan")
            rtm_ratio = pinch_ratio(rtm_choice["points"]) if rtm_choice else float("nan")
            mp_score = mp_choice["score"] if mp_choice else float("nan")
            rtm_score = rtm_choice["score"] if rtm_choice else float("nan")
            mp_wrist = tuple(mp_choice["points"][0]) if mp_choice else (float("nan"),) * 2
            rtm_wrist = tuple(rtm_choice["points"][0]) if rtm_choice else (float("nan"),) * 2

            rows.append(dict(
                frame=frame_index, time_s=stamp_s,
                mp_present=mp_choice is not None, mp_score=mp_score,
                mp_ratio=mp_ratio, mp_ratio3d=mp_ratio3d, mp_ms=mp_ms,
                mp_wrist_x=mp_wrist[0], mp_wrist_y=mp_wrist[1],
                rtm_present=rtm_choice is not None, rtm_score=rtm_score,
                rtm_ratio=rtm_ratio, rtm_ms=rtm_ms,
                rtm_wrist_x=rtm_wrist[0], rtm_wrist_y=rtm_wrist[1],
            ))
            mp_points.append(mp_choice["points"] if mp_choice else np.full((21, 2), np.nan))
            mp_world.append(mp_choice["world"] if mp_choice and mp_choice["world"] is not None
                            else np.full((21, 3), np.nan))
            mp_scores.append(mp_score)
            mp_ratios.append(mp_ratio)
            mp_ratios3d.append(mp_ratio3d)
            rtm_points.append(rtm_choice["points"] if rtm_choice else np.full((21, 2), np.nan))
            rtm_scores.append(rtm_score)
            rtm_ratios.append(rtm_ratio)

            panel_mp = make_panel(frame, mp_choice["points"] if mp_choice else None,
                                  "MediaPipe", (0, 220, 255), mp_score, mp_ratio,
                                  mp_ms, panel_size)
            panel_rtm = make_panel(frame, rtm_choice["points"] if rtm_choice else None,
                                   "RTMPose-Hand", (80, 255, 80), rtm_score, rtm_ratio,
                                   rtm_ms, panel_size)
            side = np.hstack([panel_mp, panel_rtm])
            if writers:
                writers["raw"].write(frame)
                writers["mediapipe"].write(panel_mp)
                writers["rtmpose"].write(panel_rtm)
                writers["side_by_side"].write(side)
            if args.preview:
                cv2.imshow("HAND A/B | MP vs RTMPose | Q exit", side)
                code = cv2.waitKey(1) & 0xFF
                if code in (27, ord("q")):
                    break
            frame_index += 1
            if frame_index % 30 == 0:
                print(f"{frame_index} frames | MP {mp_ratio:.2f} | RTM {rtm_ratio:.2f}",
                      flush=True)
    finally:
        cap.release()
        mp_model.close()
        for writer in writers.values():
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    times = np.array([row["time_s"] for row in rows], dtype=float)
    keypoint_arrays = dict(
        times=times,
        mp_points=np.asarray(mp_points), mp_world=np.asarray(mp_world),
        mp_scores=np.asarray(mp_scores), mp_ratios=np.asarray(mp_ratios),
        mp_ratios3d=np.asarray(mp_ratios3d),
        rtm_points=np.asarray(rtm_points), rtm_scores=np.asarray(rtm_scores),
        rtm_ratios=np.asarray(rtm_ratios),
        mp_present=np.array([row["mp_present"] for row in rows], dtype=bool),
        rtm_present=np.array([row["rtm_present"] for row in rows], dtype=bool),
    )
    np.savez_compressed(output / "keypoints.npz", **keypoint_arrays)

    if rows:
        with (output / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    mp_summary = model_summary("mp", rows, times, args)
    rtm_summary = model_summary("rtm", rows, times, args)
    present_mp = keypoint_arrays["mp_present"]
    present_rtm = keypoint_arrays["rtm_present"]
    summary = dict(
        source=source_name, live=live, frames=len(rows),
        width=width, height=height, source_fps=source_fps,
        mediapipe=mp_summary, rtmpose=rtm_summary,
        agreement=dict(
            both_present=float(np.mean(present_mp & present_rtm)) if len(rows) else 0.,
            mp_only=float(np.mean(present_mp & ~present_rtm)) if len(rows) else 0.,
            rtm_only=float(np.mean(~present_mp & present_rtm)) if len(rows) else 0.,
            neither=float(np.mean(~present_mp & ~present_rtm)) if len(rows) else 0.,
            events=match_events(mp_summary["events"], rtm_summary["events"]),
        ),
        settings=dict(mp_confidence=args.mp_confidence, close_ratio=args.close_ratio,
                      open_ratio=args.open_ratio, dwell=args.dwell,
                      rtm_device=rtm_device),
        model=DEFAULT_MODEL.name,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
