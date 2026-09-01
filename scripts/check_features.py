#!/usr/bin/env python3
"""Compare a session's feature ranges against the training data.

Run this on anything converted from third-party video BEFORE trusting a
prediction from it.  The model was trained on angles produced by the app's
Kotlin FeatureExtractor; `scripts/video_to_repai_json.py` reimplements that
geometry in Python, and if any channel uses a different zero point or unit,
every number on that channel shifts.  The model still returns a confident
answer -- it is just answering a different question.

That failure is invisible in the prediction output, which is why this exists.

    python scripts/check_features.py --session data/new/from_youtube.json

Reading the output: `z` is how many training standard deviations the new
session's mean sits from the training mean, per channel.  Under 2 is normal
person-to-person variation.  Over 4 on a single channel almost always means a
convention mismatch, not an unusual athlete -- check that channel's formula
first.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from repai.dataset import load_reps  # noqa: E402
from repai.schema import FRAME_FEATURES  # noqa: E402


def stats(reps) -> tuple[np.ndarray, np.ndarray]:
    allf = np.concatenate([r.frames for r in reps], axis=0)
    return allf.mean(axis=0), allf.std(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="a single session JSON to check")
    ap.add_argument("--dir", help="a folder of session JSONs to check")
    ap.add_argument("--train", default="data/raw/DATASET_COLLECTION",
                    help="the collection the model was trained on")
    ap.add_argument("--exercise", default="bicep_curl")
    ap.add_argument("--warn", type=float, default=2.0)
    ap.add_argument("--fail", type=float, default=4.0)
    args = ap.parse_args()

    if not args.session and not args.dir:
        ap.error("give --session or --dir")

    train = load_reps(args.train, exercise=args.exercise)
    if not train:
        print(f"no training reps for {args.exercise} under {args.train}")
        return 1
    tmean, tstd = stats(train)

    root = pathlib.Path(args.dir) if args.dir else pathlib.Path(args.session).parent
    new = load_reps(root, drop_duplicates=False)
    if args.session:
        want = pathlib.Path(args.session).name
        new = [r for r in new if pathlib.Path(r.source_file).name == want]
    if not new:
        print(f"no repetitions found under {root}")
        return 1
    nmean, nstd = stats(new)

    z = np.abs(nmean - tmean) / np.maximum(tstd, 1e-6)

    print("-" * 74)
    print(f"training: {len(train)} reps from {args.train}")
    print(f"checking: {len(new)} reps from {root}")
    print("-" * 74)
    print(f"{'channel':<24}{'train mean':>12}{'new mean':>12}{'z':>8}   verdict")

    bad, warn = [], []
    for i, name in enumerate(FRAME_FEATURES):
        if z[i] >= args.fail:
            verdict, bucket = "MISMATCH", bad
        elif z[i] >= args.warn:
            verdict, bucket = "check", warn
        else:
            verdict, bucket = "ok", None
        if bucket is not None:
            bucket.append(name)
        print(f"{name:<24}{tmean[i]:>12.2f}{nmean[i]:>12.2f}{z[i]:>8.2f}   {verdict}")

    print("-" * 74)
    if bad:
        print(f"!! {len(bad)} channel(s) far outside the training range: "
              f"{', '.join(bad)}")
        print("   Predictions from this session are not trustworthy.  Most likely")
        print("   the converter measures these differently from the phone --")
        print("   check the zero point and units in scripts/video_to_repai_json.py")
        print("   against the Kotlin FeatureExtractor before reading any result.")
        return 2
    if warn:
        print(f"   {len(warn)} channel(s) drifting: {', '.join(warn)}")
        print("   Plausible for a different person or camera distance.  Predictions")
        print("   are usable but treat a low-confidence answer as genuinely unsure.")
        return 0
    print("   all channels within the training range -- safe to predict on this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
