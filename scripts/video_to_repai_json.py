#!/usr/bin/env python3
"""Convert exercise videos into REP-AI session JSON via MediaPipe.

This is the bridge that makes every public *video* dataset usable.  None of
them ship the 12 biomechanical angles REP-AI trains on -- they ship frames.
Running the same geometry the Kotlin `FeatureExtractor` runs on the phone
turns any labelled video folder into files the trainer can read directly.

    pip install mediapipe opencv-python
    python scripts/video_to_repai_json.py \
        --videos  ~/kaggle/squat/shallow \
        --exercise squat --label shallow_squats \
        --out data/raw/DATASET_COLLECTION/SQUATS/shallow_squats/normal

Accuracy warning: the angles here are computed from MediaPipe's *2D* image
coordinates, exactly like the phone, so they inherit the same view dependence.
Videos shot from an angle the phone app would reject (heavy foreshortening,
body cropped) produce plausible-looking but wrong angles.  Use --preview to
eyeball a few before converting a whole folder, and keep converted external
data in its own label folder so it can be ablated out later.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

# BlazePose landmark indices (33-point model), matching the phone pipeline.
L = dict(
    nose=0,
    l_shoulder=11, r_shoulder=12,
    l_elbow=13, r_elbow=14,
    l_wrist=15, r_wrist=16,
    l_hip=23, r_hip=24,
    l_knee=25, r_knee=26,
    l_ankle=27, r_ankle=28,
)


def angle(a, b, c) -> float:
    """Angle ABC in degrees, B the vertex -- the same cos^-1 dot-product
    formulation as section 4.2.5 of the FYP-1 report."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1) or 1e-6
    n2 = math.hypot(*v2) or 1e-6
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def vertical_angle(a, b) -> float:
    """Angle of segment A->B from the image vertical, in degrees."""
    return math.degrees(math.atan2(abs(b[0] - a[0]), abs(b[1] - a[1]) or 1e-6))


def frame_features(lm) -> dict:
    """Reproduce the 12 channels in repai.schema.FRAME_FEATURES."""
    p = {k: (lm[i].x, lm[i].y) for k, i in L.items()}
    mid_sh = ((p["l_shoulder"][0] + p["r_shoulder"][0]) / 2,
              (p["l_shoulder"][1] + p["r_shoulder"][1]) / 2)
    mid_hip = ((p["l_hip"][0] + p["r_hip"][0]) / 2,
               (p["l_hip"][1] + p["r_hip"][1]) / 2)

    l_elbow = angle(p["l_shoulder"], p["l_elbow"], p["l_wrist"])
    r_elbow = angle(p["r_shoulder"], p["r_elbow"], p["r_wrist"])
    return {
        "knee_angle": round(angle(p["l_hip"], p["l_knee"], p["l_ankle"]), 2),
        "hip_angle": round(angle(p["l_shoulder"], p["l_hip"], p["l_knee"]), 2),
        "elbow_angle": round(l_elbow, 2),
        "right_elbow_angle": round(r_elbow, 2),
        # back_angle: torso against vertical, reported as 90 deg when upright
        # so it matches the collected data's ~90 baseline.
        "back_angle": round(90.0 - vertical_angle(mid_hip, mid_sh), 2),
        "left_shoulder_angle": round(angle(p["l_elbow"], p["l_shoulder"], p["l_hip"]), 2),
        "right_shoulder_angle": round(angle(p["r_elbow"], p["r_shoulder"], p["r_hip"]), 2),
        "avg_elbow_angle": round((l_elbow + r_elbow) / 2, 2),
        "left_right_asymmetry": round(abs(l_elbow - r_elbow), 2),
        # filled in by _add_derivatives once the whole clip is known
        "velocity": 0.0, "angular_velocity": 0.0, "angular_acceleration": 0.0,
        "phase": "up",
    }


def add_derivatives(frames: list[dict], fps: int, key: str = "avg_elbow_angle") -> None:
    """Finite-difference the driving angle into velocity/acceleration and label
    the concentric/eccentric phase.  The shipped collection has this stage
    broken (99.7 % of frames say 'up'); it is fixed here."""
    dt = 1.0 / max(fps, 1)
    vals = [f["features"][key] for f in frames]
    for i, f in enumerate(frames):
        prev, nxt = vals[max(i - 1, 0)], vals[min(i + 1, len(vals) - 1)]
        av = (nxt - prev) / (2 * dt) if len(vals) > 1 else 0.0
        f["features"]["angular_velocity"] = round(av, 2)
        f["features"]["velocity"] = round(av / 100.0, 4)
        f["features"]["phase"] = "up" if av >= 0 else "down"
    for i, f in enumerate(frames):
        prev = frames[max(i - 1, 0)]["features"]["angular_velocity"]
        nxt = frames[min(i + 1, len(frames) - 1)]["features"]["angular_velocity"]
        f["features"]["angular_acceleration"] = round((nxt - prev) / (2 * dt), 2)


def segment_reps(frames: list[dict], key: str = "avg_elbow_angle",
                 min_len: int = 12, prominence: float = 25.0) -> list[list[dict]]:
    """Split a clip into repetitions at the local maxima of the driving angle.

    Relative-peak detection, not a fixed threshold -- the same fix the FYP-1
    report describes under 'Rep Segmentation Stabilization'.  A rigid angle
    threshold breaks the moment a different body or camera distance shifts the
    baseline, which is guaranteed with third-party video.
    """
    vals = [f["features"][key] for f in frames]
    if len(vals) < 2 * min_len:
        return [frames] if len(frames) >= min_len else []
    lo, hi = min(vals), max(vals)
    if hi - lo < prominence:
        return []                       # no real movement: idle clip
    mid = (hi + lo) / 2
    boundaries, above = [], vals[0] > mid
    for i, v in enumerate(vals):
        now = v > mid
        if now and not above and i - (boundaries[-1] if boundaries else 0) >= min_len:
            boundaries.append(i)
        above = now
    if not boundaries:
        return []
    # Segments between boundaries are whole reps.  The fragment before the
    # first boundary and the one after the last are partial by construction --
    # the clip started or stopped mid-rep.  Keeping them is the "giant first
    # repetition" contamination described in the FYP-1 report, so they are
    # only kept when long enough to plausibly be complete.
    interior = [frames[a:b] for a, b in zip(boundaries, boundaries[1:])
                if b - a >= min_len]
    if not interior:
        return []
    typical = sorted(len(r) for r in interior)[len(interior) // 2]
    head, tail = frames[:boundaries[0]], frames[boundaries[-1]:]
    reps = list(interior)
    if len(head) >= 0.7 * typical:
        reps.insert(0, head)
    if len(tail) >= 0.7 * typical:
        reps.append(tail)
    return reps


def rep_summary(rep: list[dict], fps: int) -> dict:
    e = [f["features"]["avg_elbow_angle"] for f in rep]
    back = [f["features"]["back_angle"] for f in rep]
    asym = [f["features"]["left_right_asymmetry"] for f in rep]
    acc = [f["features"]["angular_acceleration"] for f in rep]
    avg_back = sum(back) / len(back)
    return {
        "max_elbow_angle": round(max(e), 2),
        "min_elbow_angle": round(min(e), 2),
        "rom_degrees": round(max(e) - min(e), 2),
        "max_left_right_asymmetry": round(max(asym), 2),
        "avg_back_angle": round(avg_back, 2),
        "max_back_deviation": round(max(abs(b - avg_back) for b in back), 2),
        "curl_rom_degrees": round(max(e) - min(e), 2),
        "rep_duration_ms": int(1000 * len(rep) / max(fps, 1)),
        "motion_smoothness": round(sum(a * a for a in acc) / len(acc), 2),
    }


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")


class PoseBackend:
    """Wraps whichever MediaPipe pose API is installed.

    MediaPipe 1.0 removed `mp.solutions.pose`; the replacement is the Tasks
    API, which needs a .task model file downloaded separately.  Older releases
    only have the legacy API.  Both are supported so the script does not
    depend on which version the machine happens to have.
    """

    def __init__(self, model_path: str | None = None):
        import mediapipe as mp
        self.mp = mp
        self.legacy = hasattr(mp, "solutions")
        if self.legacy:
            self.pose = mp.solutions.pose.Pose(
                model_complexity=1, min_detection_confidence=0.5,
                min_tracking_confidence=0.5, static_image_mode=False)
            return

        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        path = pathlib.Path(model_path or "pose_landmarker.task")
        if not path.exists():
            raise SystemExit(
                f"the Tasks API needs a model file and {path} is missing.\n"
                f"download it once with:\n  curl -sSL -o {path} {MODEL_URL}\n"
                f"then re-run, or pass --model <path>.")
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO))

    def landmarks(self, rgb, timestamp_ms: int):
        """Return the 33 landmarks for one frame, or None if no person."""
        if self.legacy:
            res = self.pose.process(rgb)
            return res.pose_landmarks.landmark if res.pose_landmarks else None
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.pose.detect_for_video(image, timestamp_ms)
        return res.pose_landmarks[0] if res.pose_landmarks else None

    def close(self):
        self.pose.close()


def convert(path: pathlib.Path, exercise: str, label: str, pose, stride: int) -> dict | None:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 30)) or 30
    frames, idx, kept = [], 0, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            ts = int(1000 * idx / fps)
            lm = pose.landmarks(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ts)
            if lm is not None:
                frames.append({
                    "frame_id": kept,
                    "timestamp": ts,
                    "features": frame_features(lm),
                })
                kept += 1
        idx += 1
    cap.release()
    if len(frames) < 24:
        return None

    eff_fps = max(fps // stride, 1)
    add_derivatives(frames, eff_fps)
    reps = segment_reps(frames)
    if not reps:
        return None
    return {
        "session_id": f"ext_{path.stem}_{int(time.time() * 1000) % 10 ** 9}",
        "user_id": f"ext_{path.stem[:12]}",
        "exercise": exercise,
        "label": label,
        "fps": eff_fps,
        "source": "external_video",
        "source_file": path.name,
        "reps": [{"rep_id": i + 1, "frame_count": len(r), "frames": r,
                  "rep_summary": rep_summary(r, eff_fps)}
                 for i, r in enumerate(reps)],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="folder of video files")
    ap.add_argument("--exercise", required=True, choices=["squat", "bicep_curl", "shoulder_press"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--model", help="path to pose_landmarker.task (MediaPipe >= 1.0 only)")
    args = ap.parse_args()

    try:
        import mediapipe  # noqa: F401
        import cv2  # noqa: F401
    except ImportError:
        print("required:  pip install mediapipe opencv-python", file=sys.stderr)
        return 1

    videos = sorted(pathlib.Path(args.videos).glob(args.glob))
    if not videos:
        print(f"no files matching {args.glob} in {args.videos}", file=sys.stderr)
        return 1
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pose = PoseBackend(args.model)
    print(f"pose backend: {'legacy solutions API' if pose.legacy else 'Tasks API'}")
    ok = skipped = total_reps = 0
    for v in videos:
        session = convert(v, args.exercise, args.label, pose, args.stride)
        if session is None:
            skipped += 1
            print(f"  skip  {v.name}  (no pose / too short / no clean reps)")
            continue
        dest = out / f"{args.exercise}_{args.label}_{v.stem}.json"
        dest.write_text(json.dumps(session, indent=1))
        ok += 1
        total_reps += len(session["reps"])
        print(f"  ok    {v.name}  -> {len(session['reps'])} reps")
    pose.close()
    print(f"\nconverted {ok}/{len(videos)} videos, {total_reps} reps, {skipped} skipped")
    print(f"now run:  python scripts/audit_dataset.py {args.out.split('/DATASET')[0]}"
          "/DATASET_COLLECTION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
