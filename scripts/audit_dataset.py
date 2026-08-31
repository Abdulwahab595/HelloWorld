#!/usr/bin/env python3
"""Forensic audit of a DATASET_COLLECTION tree.

Run this before every training run.  It answers the questions a supervisor or
an external examiner will ask first: how much data is really there, how much
of it is duplicated, and how many distinct people are in it.

    python scripts/audit_dataset.py data/raw/DATASET_COLLECTION
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from repai.schema import EXERCISE_DIRS, FRAME_FEATURES  # noqa: E402


def audit(root: pathlib.Path) -> dict:
    files = sorted(p for p in root.rglob("*.json"))
    by_class = collections.defaultdict(lambda: {"files": 0, "reps": 0, "frames": 0})
    digests = collections.defaultdict(list)
    users, exercises, json_labels = collections.Counter(), collections.Counter(), collections.Counter()
    phases = collections.Counter()
    rep_lengths, fps_values = [], collections.Counter()
    missing_features = collections.Counter()

    for path in files:
        rel = path.relative_to(root)
        # <EXERCISE>/<label>/<speed>/<file>.json
        exercise = EXERCISE_DIRS.get(rel.parts[0], rel.parts[0])
        label, speed = rel.parts[1], rel.parts[2]
        key = f"{exercise}/{label}/{speed}"

        raw = path.read_bytes()
        digests[hashlib.md5(raw).hexdigest()].append(str(rel))
        session = json.loads(raw)

        users[session.get("user_id")] += 1
        exercises[session.get("exercise")] += 1
        json_labels[session.get("label")] += 1
        fps_values[session.get("fps")] += 1

        by_class[key]["files"] += 1
        for rep in session["reps"]:
            by_class[key]["reps"] += 1
            by_class[key]["frames"] += rep["frame_count"]
            rep_lengths.append(rep["frame_count"])
            for frame in rep["frames"]:
                feats = frame["features"]
                phases[feats.get("phase")] += 1
                for name in FRAME_FEATURES:
                    if name not in feats:
                        missing_features[name] += 1

    empty_dirs = sorted(
        str(d.relative_to(root))
        for d in root.rglob("*")
        if d.is_dir() and not any(d.rglob("*.json"))
    )
    duplicates = {h: p for h, p in digests.items() if len(p) > 1}
    n_dupe_files = sum(len(p) - 1 for p in duplicates.values())

    return {
        "files": len(files),
        "unique_files": len(digests),
        "duplicate_files": n_dupe_files,
        "duplicate_groups": duplicates,
        "by_class": dict(by_class),
        "empty_dirs": empty_dirs,
        "users": dict(users),
        "exercises": dict(exercises),
        "json_labels": dict(json_labels),
        "fps": dict(fps_values),
        "phases": dict(phases),
        "rep_lengths": rep_lengths,
        "missing_features": dict(missing_features),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/raw/DATASET_COLLECTION")
    ap.add_argument("--json", help="also write the report as JSON to this path")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 1
    r = audit(root)

    print("=" * 78)
    print(f"REP-AI DATASET AUDIT  --  {root}")
    print("=" * 78)

    print("\n[1] VOLUME")
    total_reps = sum(v["reps"] for v in r["by_class"].values())
    total_frames = sum(v["frames"] for v in r["by_class"].values())
    print(f"    session files      {r['files']}")
    print(f"    unique files       {r['unique_files']}  "
          f"({r['duplicate_files']} byte-identical duplicates)")
    print(f"    repetitions        {total_reps}")
    print(f"    frames             {total_frames}")
    if r["rep_lengths"]:
        rl = r["rep_lengths"]
        print(f"    rep length         min {min(rl)}  median {st.median(rl):.0f}  "
              f"max {max(rl)}  mean {st.mean(rl):.1f}")

    print("\n[2] CLASS BALANCE")
    print(f"    {'exercise/label/speed':<48}{'files':>7}{'reps':>7}{'frames':>9}")
    for key in sorted(r["by_class"]):
        v = r["by_class"][key]
        print(f"    {key:<48}{v['files']:>7}{v['reps']:>7}{v['frames']:>9}")

    print("\n[3] SUBJECT DIVERSITY")
    for user, n in sorted(r["users"].items()):
        print(f"    {user:<20}{n} sessions")
    if len(r["users"]) < 2:
        print("    !! single subject -- no held-out person exists, so any reported")
        print("       accuracy measures memorisation of one body, not generalisation.")

    print("\n[4] DEGENERATE FIELDS")
    total_phase = sum(r["phases"].values()) or 1
    for phase, n in sorted(r["phases"].items(), key=lambda kv: -kv[1]):
        print(f"    phase={str(phase):<10}{n:>8}  ({100 * n / total_phase:5.1f} %)")
    if r["phases"] and max(r["phases"].values()) / total_phase > 0.95:
        print("    !! `phase` is effectively constant -- the eccentric/concentric")
        print("       detector is not firing.  Excluded from the model input.")
    if r["missing_features"]:
        print(f"    missing frame features: {r['missing_features']}")

    print("\n[5] EMPTY DIRECTORIES (declared but never collected)")
    for d in r["empty_dirs"]:
        print(f"    {d}")
    if not r["empty_dirs"]:
        print("    none")

    print("\n[6] BYTE-IDENTICAL DUPLICATES")
    if not r["duplicate_groups"]:
        print("    none")
    for h, paths in sorted(r["duplicate_groups"].items(), key=lambda kv: -len(kv[1])):
        print(f"    {h[:12]}  x{len(paths)}")
        for p in sorted(paths):
            print(f"        {p}")

    print("\n[7] METADATA CONSISTENCY")
    print(f"    exercise field : {r['exercises']}")
    print(f"    label field    : {r['json_labels']}")
    print(f"    fps            : {r['fps']}")

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
