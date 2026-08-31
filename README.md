# REP-AI — Temporal exercise form-error classification

Iteration 3 of the REP-AI FYP: a Temporal Convolutional Network that reads
sequences of biomechanical joint angles and classifies **which form error** a
repetition contains.

Abdul Wahab (22i-1714) · Shafiq Ullah Khan (22i-2556)
FAST-NUCES Islamabad · Supervisor Dr. Muhammad Rizwan · Co-supervisor Mr. Mohsin Khan

---

## Result

Bicep curl, 3 classes, 5-fold cross-validation **grouped by recording
session** so no two windows from one recording straddle the split:

| | Accuracy | Macro-F1 |
|---|---|---|
| Window level (30 frames) | 0.881 | 0.884 ± 0.043 |
| **Repetition level** | **0.927** | **0.928** |

| truth ↓ / predicted → | correct | elbow_moving | incomplete_extension |
|---|---|---|---|
| **correct** | 156 | 5 | 23 |
| **elbow_moving** | 1 | 159 | 0 |
| **incomplete_extension** | 6 | 1 | 145 |

51,795 parameters · 61-frame receptive field · 61 KB as ONNX · 496 repetitions
from 62 sessions.

The same code with a random window split reports **0.998**. That number is
wrong, and reproducing the gap is the point — see
[`docs/JUGAAR_STRATEGY.md`](docs/JUGAAR_STRATEGY.md).

> **One subject.** All recordings are `user_01`, so this measures whether the
> movement classes are separable — not whether the model transfers to a new
> body. Stated up front because it is the first thing an examiner should ask.

## Quick start

```bash
pip install -r requirements.txt

python scripts/audit_dataset.py data/raw/DATASET_COLLECTION   # what is really there
python -m repai.train --exercise bicep_curl                   # ~2 min on CPU
python -m repai.export --ckpt reports/bicep_curl_tcn.pt       # ONNX + norm + labels
```

Useful flags:

```bash
--leaky              random window split, for the ablation table only
--mode rep           whole rep resampled to 64 frames (0.944) instead of windows
--aug 0              disable augmentation
--folds 10           more folds
```

## Layout

```
repai/
  schema.py     feature order, label taxonomy, coaching cues — the contract
                the Kotlin FeatureExtractor must match
  dataset.py    JSON → (N, T, F) tensors; session-grouped splitting
  augment.py    time warp / jitter / mirror / ROM scale, with derivative rescaling
  model.py      dilated causal TCN (Bai, Kolter & Koltun 2018)
  train.py      grouped k-fold, class weighting, rep-level aggregation
  export.py     ONNX + normalisation constants + TFLite conversion notes
scripts/
  audit_dataset.py         volume, duplicates, subjects, empty folders
  video_to_repai_json.py   MediaPipe bridge: third-party video → REP-AI JSON
  make_squat_smoketest.py  synthetic squats to exercise the untested code path
docs/
  DATASET_AUDIT.md     what the collection contains, and its seven defects
  DATASET_SOURCES.md   EC3D / Fitness-AQA / Kaggle / Roboflow / UCF101, ranked
  JUGAAR_STRATEGY.md   the five-day plan to a defensible iteration 3
data/raw/DATASET_COLLECTION/   79 session files as delivered
```

## How the data works

The Android app already runs MediaPipe BlazePose, computes joint angles and
segments repetitions. What it stores is **12 pre-computed biomechanical
channels per frame** — not video, not keypoints:

```
knee_angle · hip_angle · elbow_angle · right_elbow_angle · back_angle
left_shoulder_angle · right_shoulder_angle · velocity · avg_elbow_angle
left_right_asymmetry · angular_velocity · angular_acceleration
```

`phase` is present in the JSON but excluded: it reads `"up"` in 99.7 % of
frames, so the eccentric/concentric detector is not working.

This is why no public dataset drops in unmodified — they ship frames, REP-AI
consumes angles. `scripts/video_to_repai_json.py` is that bridge.

## Status

| Exercise | Classes with data | Trainable |
|---|---|---|
| bicep_curl | correct, elbow_moving, incomplete_extension | **yes** |
| bicep_curl | fast_swing (1 session) | no — cannot be split across folds |
| shoulder_press | correct only | no — needs the 3 error classes |
| squat | none | no — 480 reps to record |

`docs/JUGAAR_STRATEGY.md` has the plan. The short version: two evenings of
recording with the app you already built beats two weeks of dataset hunting.
