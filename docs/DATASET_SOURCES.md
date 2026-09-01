# External dataset survey

Assessed against one question: **how many hours until it produces labelled
`correct` / `error-type` repetitions in REP-AI's 12-channel format?**

The hard constraint, restated: REP-AI trains on pre-computed joint angles.
Nothing public ships that format. Every source below has to cross a bridge:

| Source ships | Bridge needed | Tool |
|---|---|---|
| Video | MediaPipe → angles → rep segmentation | `scripts/video_to_repai_json.py` |
| 2D/3D keypoints | angles → rep segmentation | adapt `frame_features()` to read arrays |
| Angles | none | direct |

## The four links you sent

**`kaggle.com/datasets`** — the generic gym-video sets
([workout/fitness video][k1], [gym workout/exercises video][k2]) are
**exercise-type** classification: folders named `squat`, `bicep curl`,
`push-up`. They are all nominally correct form. There is no error taxonomy,
so they cannot supply your `shallow_squats` / `elbow_moving` / `back_arch`
classes. Useful for one thing only: an exercise-*recognition* head, or as raw
footage you re-label yourself.

**`public.roboflow.com`** and Roboflow Universe — object detection and
per-image keypoints. Frame-level annotation, **no temporal dimension and no
form labels**. The [`bicep-curl`][r1] and [`fitness_models`][r2] projects are
bounding boxes / keypoints on stills. Structurally wrong for a TCN: your model
consumes 30-frame sequences, these datasets have no notion of a sequence.
Skip.

**`crcv.ucf.edu/.../ucf101`** — 101 action classes, 13,320 clips, 320×240 at
25 fps. Contains `BodyWeightSquats`, `BenchPress`, `PullUps`, `Lunges`. It is
an **action recognition** benchmark: labels say *what* exercise, never *how
well*. Additionally the resolution and frequent body cropping make MediaPipe
unreliable on it. Cite it as related work, do not train on it.

**`github.com/jinwchoi/awesome-action-recognition`** — a curated link list,
not a dataset. Good for the related-work section of the FYP-2 report. Its
datasets are almost all action recognition, so the same objection applies.

**Conclusion on your four links: none of them contain exercise-form error
labels.** That is not an oversight on your part — form-error data is rare
because it needs a trainer to annotate. The sets that do have it are below.

## Sources that actually carry form-error labels

### EC3D — the best fit, and it fills your worst gap

*Zhao, Kiciroglu, Wang, Salzmann, Fua — "3D Pose Based Feedback for Physical
Exercises", ACCV 2022.* [Code + data][ec3d] · [paper][ec3dpaper]

- 3 exercises, 4 subjects. **132 squat**, 127 lunge, 103 plank sequences.
- 11 instruction labels. Squat labels: **Correct, Not low enough, Knees
  inward, Feet too wide, Front bent.**
- Ships `data_3D.pickle` — 3D joint coordinates, `(29789, 3, 25)`, 25 skeletal
  nodes. **Coordinates, not video.**

Why this is the one to take:

1. Squat is the exercise you have *zero* data for.
2. `Not low enough` ≈ your `shallow_squats`; `Front bent` ≈ a back-bend fault.
   Two of your three squat classes map almost directly.
3. 3D coordinates mean **no MediaPipe step** — feed them straight into the
   angle geometry. Shorter bridge than any video source, and no pose-estimation
   error added.
4. **4 subjects.** This is the only cheap way to put more than one body in
   your training set, which is your most-criticisable weakness.

Cost: ~1 day to write the coordinate→angle adapter. Caveat: it is MoCap-grade
3D, while the phone gives noisy 2D — so a model trained on EC3D alone will
over-trust clean input. Train jointly with your own data, and add the jitter
augmentation.

### Fitness-AQA — right labels, slow access

*Parmar, Gharat, Rhodin — "Domain Knowledge-Informed Self-Supervised
Representations for Workout Form Assessment", ECCV 2022.* [repo][faqa]

- BackSquat, BarbellRow, **OverheadPress** — the last is your shoulder press.
- Real gym footage from Instagram/YouTube, **annotated by expert trainers for
  specific errors**. Exactly the taxonomy you want.
- Access is by [Google Form][faqaform] request. That gate is the problem: you
  cannot plan a deadline around someone else's inbox.

**Submit the request today anyway** — it costs five minutes, and if it lands
before FYP-2 it is the strongest citation available for the shoulder-press
classes. Do not make it load-bearing.

### Fit3D — large, clean, wrong labels

611 multi-view sequences, 2.96 M images, 11 subjects, 37 exercises, SMPL-X
ground truth. [Site][fit3d] · registration required.

Every rep is performed *correctly*. No error labels. Enormous download for
data that only gives you a `correct` class you already have. Skip for now;
mention in related work as the reference for high-quality fitness MoCap.

### InfiniteRep — synthetic, useful for pretraining

Open-source synthetic video of avatars exercising, with 3D joint angles,
keypoints, segmentation masks, rep counts, and deliberate variation in body
type, lighting and camera angle. [Announcement][irep]

No form-error labels either, but it is the one source that directly attacks
your **single-subject** problem: varied synthetic bodies. Worth it only if you
have spare time — pretrain on it, fine-tune on your 496 real reps.

## Ranked verdict

| Source | Form-error labels | Bridge cost | Fills which gap | Verdict |
|---|---|---|---|---|
| **Record it yourself** | you define them | ~2 h/exercise | all of them | **do this first** |
| EC3D | yes (squat) | ~1 day, no video | squat + 4 subjects | **do this second** |
| Fitness-AQA | yes (press, squat) | request + video pipeline | shoulder press | request now, don't wait |
| InfiniteRep | no | video pipeline | subject diversity | optional |
| Kaggle gym videos | no | video pipeline | none | only if self-labelling |
| Fit3D | no | registration + huge | none | cite only |
| UCF101 | no | video pipeline | none | cite only |
| Roboflow Universe | no | n/a — not temporal | none | skip |

## The finding that should change your plan

You spent this search assuming the bottleneck is *finding data*. It is not.
Your app **is a dataset collection instrument** — it emits training-ready
JSON, correctly segmented, at 8 reps a session. Two people with a tripod
produce roughly 200 labelled reps an hour.

Every external source above costs *at least* a day of engineering before it
yields its first usable rep, and most of them then yield reps in the wrong
label space. Recording squats yourself costs one evening and yields exactly
the classes your report already promises.

**Use the app. It is the shortcut.** The external sources are for the FYP-2
related-work section and for the subject diversity you cannot fake — in that
order.

[k1]: https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video
[k2]: https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video
[r1]: https://universe.roboflow.com/gym-exercise-correction/bicep-curl
[r2]: https://universe.roboflow.com/detekgerak-9rw99/fitness_models
[ec3d]: https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises
[ec3dpaper]: https://openaccess.thecvf.com/content/ACCV2022/papers/Zhao_3D_Pose_Based_Feedback_For_Physical_Exercises_ACCV_2022_paper.pdf
[faqa]: https://github.com/ParitoshParmar/Fitness-AQA
[faqaform]: https://forms.gle/PbPTX1eVxGpa3QG88
[fit3d]: https://fit3d.imar.ro/
[irep]: https://medium.com/infinity-ai/infiniterep-an-open-source-synthetic-dataset-for-remote-fitness-and-pt-applications-906946643e74

## Field note: what happened on the first real video

A 6.3 s YouTube clip (`Dumbbell Bicep Curl Side View`, 790x786, static
tripod, plain wall, full body in frame) was pushed through the whole
pipeline. It is worth recording what went right and what did not, because
the failure is instructive and will recur.

**Worked:** MediaPipe found the pose on every frame, rep segmentation
returned 3 repetitions, and `check_features.py` put all 12 channels inside
the training range (largest z = 0.95). So the Python geometry agrees with
the phone's Kotlin extractor on real footage, not just on a self-test.

**Failed:** the model called 2 of the 3 reps `elbow_moving`. The man's form
is visibly correct -- elbows pinned to the ribs throughout.

**Cause:** the clip is a *pure* side profile. The far arm is occluded for
the entire video, so MediaPipe is estimating the far-side joints rather than
seeing them. Per-channel comparison against the class means:

| channel | correct | elbow_moving | this clip |
|---|---|---|---|
| left_shoulder_angle (near, visible) | 8.2 | 15.0 | **9.1** |
| right_shoulder_angle (far, occluded) | 5.6 | 15.8 | **12.7** |

The visible side reads as correct form; the occluded side reads as the error
class. `left_right_asymmetry` is one of the 12 input channels and
`right_shoulder_angle` is another, so occlusion noise on one arm is enough
to flip the verdict.

**Not a tempo problem.** The subject curls at 1.9 s/rep against a 4.6 s
training mean, so the first hypothesis was that the model had learned tempo
as a proxy for form. Widening the time-warp augmentation to 0.55x-2.2x was
tried and *rejected*: the misclassifications became more confident (0.60 ->
0.70) and cross-validated rep accuracy fell 0.927 -> 0.919. The change was
reverted. Recording it here so the experiment is not repeated.

**Consequences for the protocol:**

1. Prefer a three-quarter view (roughly 30-45 degrees off-axis) over a pure
   profile, for evaluation clips and for the app's own guidance. Both arms
   need to be visible for the symmetry channels to mean anything.
2. MediaPipe reports a `visibility` score per landmark. The converter
   discards it. Carrying it through and gating the far-side channels on it
   -- or falling back to near-side-only features when a limb is occluded --
   is the principled fix, and belongs in the phone pipeline too.
3. One clip with 3 repetitions is not an evaluation. This is a diagnostic
   that produced a concrete, testable cause; it is not a transfer result.
