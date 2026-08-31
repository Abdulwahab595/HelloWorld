# Dataset audit — `DATASET_COLLECTION`

Reproduce everything below with:

```bash
python scripts/audit_dataset.py data/raw/DATASET_COLLECTION
```

## 1. What the collection actually is

The single most important fact, and the one that decides every downstream
choice: **this is not a video dataset and it is not a keypoint dataset.** The
Android app already ran MediaPipe BlazePose, already computed joint angles,
and already segmented repetitions. What ships is 79 JSON files of
**pre-computed biomechanical features**, 12 numeric channels per frame,
grouped into reps.

```
session
 └─ reps[8]
     ├─ frames[N]  → features{ knee_angle, hip_angle, elbow_angle,
     │                         right_elbow_angle, back_angle,
     │                         left_shoulder_angle, right_shoulder_angle,
     │                         velocity, avg_elbow_angle,
     │                         left_right_asymmetry, angular_velocity,
     │                         angular_acceleration, phase }
     └─ rep_summary{ max/min_elbow_angle, rom_degrees, curl_rom_degrees,
                     max_left_right_asymmetry, avg_back_angle,
                     max_back_deviation, rep_duration_ms, motion_smoothness }
```

Every consequence in this document follows from that. In particular: no
public video dataset can be dropped in as-is. It has to be pushed through the
same MediaPipe → angle geometry first. That converter is
`scripts/video_to_repai_json.py`.

## 2. Volume

| | Value |
|---|---|
| Session files | 79 |
| **Unique** session files | **68** (11 byte-identical duplicates) |
| Repetitions | 632 (8 per file, exactly, in all 79) |
| Frames | 30,730 |
| Rep length | 20 – 120 frames, median 47 (≈1.6 s at 30 fps) |
| Subjects | **1** (`user_01` in all 79 files) |
| fps | 30, consistent |

## 3. Class coverage

| Exercise | Label | Files | Reps | Usable |
|---|---|---|---|---|
| bicep_curl | correct | 23 | 184 | yes |
| bicep_curl | elbow_moving | 20 | 160 | yes |
| bicep_curl | incomplete_extension | 20 | 160 | yes (19 unique) |
| bicep_curl | fast_swing | **1** | 8 | **no** — one session cannot be split across folds |
| shoulder_press | correct | 15 | 120 | **no** — only 5 unique, and no error class to contrast against |
| shoulder_press | uneven_press / back_arch / incomplete_extension | 0 | 0 | **absent** |
| squat | correct / shallow_squats / knee_forward | 0 | 0 | **absent** |

Only **one exercise is trainable**: bicep curl, 3 classes, 62 sessions,
496 reps.

## 4. Defects, in order of how much they cost you

**(a) One subject.** All 79 sessions are `user_01`. There is no held-out
person, so no experiment on this data can measure generalisation to a new
body. Joint-angle features are less subject-dependent than raw pixels, which
helps, but limb-length ratios and habitual movement style still leak. Any
accuracy reported from this collection is an *upper bound*. This is the
defect an examiner is most likely to name, and the cheapest one to fix —
three friends, one evening.

**(b) Squats are entirely missing.** The proposal, the presentation and the
FYP-1 report all list squat as a supported exercise, and `SQUATS/` contains
nine empty folders. Squat is also the exercise with the best public dataset
support (see `DATASET_SOURCES.md`), so this is recoverable.

**(c) 11 duplicate files.** Five shoulder-press files are byte-identical to
each other, plus three more groups. `shoulder_press/correct` is 15 files but
only 5 distinct recordings. `bicep_curl/incomplete_extension` 011 and 012 are
the same file. Left in, these triple the weight of a few recordings and leak
across any split. `load_reps(..., drop_duplicates=True)` removes them by
default.

**(d) `phase` is broken.** 30,627 frames say `"up"`, 103 say `"down"` — 99.7 %
constant. The eccentric/concentric detector that Module 2 is specified to
provide is not firing. The field is excluded from the model input
(`schema.DEGENERATE_FEATURES`); it needs fixing in the Kotlin
`FeatureExtractor` before iteration 4, because the feedback engine is supposed
to say *when* in the rep the error occurred.

**(e) Tempo buckets never filled.** Each label folder has `slow/`, `normal/`,
`fast/`; only `normal/` was used. 26 of the 36 empty directories are this.
`repai/augment.py` synthesises the missing tempos by time-warping real reps —
the joint-angle path stays real, only speed along it changes.

**(f) Shoulder-press features are bicep-curl features.** Shoulder-press
sessions carry `curl_rom_degrees` and an elbow-centric `rep_summary`. The
extractor was written for curls and applied unchanged. It still produces
usable numbers (shoulder angle sweeps 30°→179°, clearly a press), but the
summary field names are wrong and `max_back_deviation` is not measuring what
back-arch detection will need.

**(g) Label field disagrees with folder.** Folder `incomplete_extension`
contains JSON with `"label": "partial_range"`. The loader trusts the folder.
Pick one name and make the app emit it.

## 5. Discrepancy with the FYP-1 report

Section 4.5.3 of `FYP1_FINAL_REPORT.pdf` states:

> Total Sessions: 240 · Total Repetitions: 1920 · Total Temporal Frames:
> Approximately 42,000

and gives per-label tables of 160 reps each across squat (5 labels),
bicep curl (5 labels) and shoulder press (4 labels) — 2,240 reps.

The delivered archive contains 632 reps in 5 label-classes, with squat at
zero. Slide 19 of the mid-evaluation deck additionally claims *"500 exercise
videos collected"*, *"15k–25k temporal sequences"* and *"multiple
participants"*.

Either a large part of the collection was not included in the zip, or the
report describes the intended protocol rather than what exists. **Resolve
this before the next evaluation.** If the data exists, add it and re-run the
audit. If it does not, correct the tables — a panel that asks to see the 1,920
reps and is shown 632 will discount everything else in the report, including
the parts that are solid.

## 6. What survives

The pipeline itself is genuinely good, and that is the real deliverable of
iteration 2. Rep segmentation is stable (every file yields exactly 8 clean
reps), the angle geometry is correct, temporal ordering holds, and the classes
are biomechanically separable in the way the report predicts:

| rep_summary feature | correct | elbow_moving | incomplete_extension |
|---|---|---|---|
| `rom_degrees` | 106.6 ± 9.2 | 95.6 ± 11.9 | **77.7 ± 11.0** |
| `curl_rom_degrees` | 110.9 ± 12.8 | 86.5 ± 15.2 | 75.8 ± 12.8 |
| `max_back_deviation` | 4.6 ± 2.1 | **7.7 ± 4.4** | 3.5 ± 3.1 |
| mean `left_shoulder_angle` | 8.2 | **15.0** | 9.8 |

Reduced range of motion identifies `incomplete_extension`; sustained shoulder
drift identifies `elbow_moving`. Both match the mechanism claimed in section
4.2.6.2. The distributions overlap enough that a threshold rule would be
mediocre — which is the honest argument for the TCN, and a better answer to
*"why not just use if-statements?"* than any of the slides currently give.
