#!/usr/bin/env python3
"""Generate SYNTHETIC squat sessions to smoke-test the squat pipeline.

The shipped collection contains zero squat repetitions, so nothing downstream
of the loader has ever executed for that exercise.  This script produces
kinematically plausible squat angle trajectories -- correct, shallow, and
knee-forward -- so the loader, windower, trainer and export path can be run
and debugged today, before any real squat recording exists.

WHAT THIS IS NOT
    It is not squat data.  A model trained on it has learned the parametric
    curves written below, not human movement, and its accuracy is a property
    of this file.  Every generated session carries
    ``"source": "synthetic_kinematic"`` and ``"synthetic": true``; the trainer
    prints a warning whenever such reps are present.  If a number produced
    with this data appears in a report or a demo, say on the slide that the
    squat class is synthetic.  Delete the folder and re-run the audit before
    collecting the real thing.

    python scripts/make_squat_smoketest.py --sessions 20
    rm -rf data/raw/DATASET_COLLECTION/SQUATS/*/normal/squat_synth_*.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

# Per-label kinematic envelope: (knee_min, knee_max, hip_min, torso_lean_deg).
# Read off the biomechanics literature ranges quoted in the FYP-1 report, not
# measured from a person.
PROFILES = {
    "correct":        dict(knee=(72, 176), hip=(70, 176), lean=18, knee_travel=0.10),
    "shallow_squats": dict(knee=(118, 176), hip=(122, 176), lean=14, knee_travel=0.08),
    "knee_forward":   dict(knee=(66, 176), hip=(96, 176), lean=34, knee_travel=0.34),
}


def rep_curve(n: int, lo: float, hi: float, rng, asym: float = 0.0) -> np.ndarray:
    """Half-cosine descent/ascent between hi and lo with a short bottom hold."""
    t = np.linspace(0.0, 1.0, n)
    # skew controls eccentric/concentric balance; real squats descend slower
    skew = 0.5 + asym
    down = t < skew
    phase = np.empty_like(t)
    phase[down] = t[down] / skew
    phase[~down] = 1.0 - (t[~down] - skew) / (1.0 - skew)
    curve = hi - (hi - lo) * (0.5 - 0.5 * np.cos(math.pi * phase))
    return curve + rng.normal(0.0, 0.8, n)


def make_session(label: str, idx: int, rng, fps: int = 30, n_reps: int = 8) -> dict:
    prof = PROFILES[label]
    reps = []
    for r in range(n_reps):
        n = int(rng.integers(38, 62))
        knee = rep_curve(n, *prof["knee"], rng, asym=float(rng.normal(0.03, 0.03)))
        hip = rep_curve(n, *prof["hip"], rng, asym=float(rng.normal(0.03, 0.03)))
        depth = (prof["knee"][1] - knee) / (prof["knee"][1] - prof["knee"][0])
        back = 90.0 - prof["lean"] * depth + rng.normal(0.0, 1.2, n)
        # arms hang; elbow stays near-extended through a squat
        elbow_l = 168.0 + rng.normal(0.0, 3.0, n)
        elbow_r = elbow_l + rng.normal(0.0, 2.5, n)
        sh_l = 12.0 + 6.0 * depth + rng.normal(0.0, 2.0, n)
        sh_r = sh_l + rng.normal(0.0, 2.0, n)

        dt = 1.0 / fps
        av = np.gradient(knee, dt)
        aa = np.gradient(av, dt)

        frames = []
        for i in range(n):
            frames.append({
                "frame_id": i,
                "timestamp": int(1000 * i / fps),
                "features": {
                    "knee_angle": round(float(knee[i]), 2),
                    "hip_angle": round(float(hip[i]), 2),
                    "elbow_angle": round(float(elbow_l[i]), 2),
                    "right_elbow_angle": round(float(elbow_r[i]), 2),
                    "back_angle": round(float(back[i]), 2),
                    "left_shoulder_angle": round(float(sh_l[i]), 2),
                    "right_shoulder_angle": round(float(sh_r[i]), 2),
                    "velocity": round(float(av[i]) / 100.0, 4),
                    "phase": "down" if av[i] < 0 else "up",
                    "avg_elbow_angle": round(float((elbow_l[i] + elbow_r[i]) / 2), 2),
                    "left_right_asymmetry": round(float(abs(elbow_l[i] - elbow_r[i])), 2),
                    "angular_velocity": round(float(av[i]), 2),
                    "angular_acceleration": round(float(aa[i]), 2),
                },
            })
        avg_back = float(back.mean())
        reps.append({
            "rep_id": r + 1,
            "frame_count": n,
            "frames": frames,
            "rep_summary": {
                "max_elbow_angle": round(float(knee.max()), 2),
                "min_elbow_angle": round(float(knee.min()), 2),
                "rom_degrees": round(float(knee.max() - knee.min()), 2),
                "max_left_right_asymmetry": round(float(np.abs(elbow_l - elbow_r).max()), 2),
                "avg_back_angle": round(avg_back, 2),
                "max_back_deviation": round(float(np.abs(back - avg_back).max()), 2),
                "curl_rom_degrees": round(float(hip.max() - hip.min()), 2),
                "rep_duration_ms": int(1000 * n / fps),
                "motion_smoothness": round(float((aa ** 2).mean()), 2),
            },
        })
    return {
        "session_id": f"synth_squat_{label}_{idx:03d}",
        "user_id": f"synthetic_{idx % 5:02d}",
        "exercise": "squat",
        "label": label,
        "fps": fps,
        "synthetic": True,
        "source": "synthetic_kinematic",
        "reps": reps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/DATASET_COLLECTION/SQUATS")
    ap.add_argument("--sessions", type=int, default=20, help="sessions per label")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = pathlib.Path(args.out)
    n = 0
    for label in PROFILES:
        folder = root / label / "normal"
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.sessions):
            session = make_session(label, i + 1, rng)
            (folder / f"squat_synth_{label}_{i + 1:03d}.json").write_text(
                json.dumps(session, indent=1))
            n += 1
    print(f"wrote {n} SYNTHETIC squat sessions under {root}")
    print("these are not recordings.  disclose them wherever a squat number is shown.")
    print(f"remove with:  rm -rf {root}/*/normal/squat_synth_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
