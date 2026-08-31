# Passing iteration 3 — the short path

**Iteration 3 asks for one thing: a trained TCN with an evaluation.** Not a
mobile app, not five exercises, not a research contribution nobody has made
before. Read the deliverable narrowly and it is already 80 % done.

## The shortcut is that you have already won

You have **496 clean repetitions across 3 bicep-curl classes from 62
recording sessions.** That is enough. It trains in two minutes on a CPU:

```bash
pip install -r requirements.txt
python scripts/audit_dataset.py data/raw/DATASET_COLLECTION
python -m repai.train --exercise bicep_curl
```

Measured result, 5-fold cross-validation grouped by recording session:

| | Accuracy | Macro-F1 |
|---|---|---|
| Window level (30 frames) | 0.881 | 0.884 ± 0.043 |
| **Repetition level** | **0.927** | **0.928** |

51,795 parameters, 61-frame receptive field, 61 KB as ONNX. Confusion matrix,
per-class precision/recall and a checkpoint land in `reports/`.

A 0.93 macro-F1 multi-class form classifier, cross-validated with a
leakage-free split, is a complete iteration-3 deliverable. **Stop looking for
datasets and present this.**

## Do these five things, in this order

### 1. Run the training and screenshot it (30 minutes)

Done above. `reports/bicep_curl_grouped_metrics.json` holds every number for
your report tables.

### 2. Record squats with your own app (one evening — the highest-value hour you have)

Your app already writes training-ready JSON. 20 sessions × 8 reps × 3 labels
(`correct`, `shallow_squats`, `knee_forward`) is **480 squat reps in about two
hours** including setup, and it lands in exactly the right folders with no
conversion.

Compare against every external option in `DATASET_SOURCES.md`: each costs a
day of engineering *before* the first usable rep, and most then give you the
wrong labels. There is no shortcut that beats the instrument you already
built. This single evening closes the largest gap between your report and
your data.

### 3. Record a second and third subject (one more evening)

Grab two friends. 10 sessions per label per person. This is the single
criticism most likely to come from the panel — *"you tested on the same person
you trained on"* — and two evenings removes it permanently. Then re-run
grouped by `user_id` instead of session file and report a genuine
held-out-person figure. Even if that number drops to 0.75, **a defensible 0.75
beats an indefensible 0.93**, and you will be the team that noticed.

### 4. Fix `phase` in the Kotlin extractor (one hour)

99.7 % of frames say `"up"`. The eccentric/concentric detector is dead. It is a
one-line sign test on the driving angle's derivative — see `add_derivatives()`
in `scripts/video_to_repai_json.py` for the working version. It costs an hour,
it is visible in the audit output, and iteration 4's "you swung it on the way
*down*" feedback depends on it.

### 5. Delete `fast_swing` from the slides, or record 19 more sessions (two hours)

You have exactly **one** `fast_swing` session. One session cannot appear on
both sides of a cross-validation split, so the trainer drops it and says so.
Either collect it properly or present 3 classes. Do not report a number for a
class with one recording.

## Timeline

| Day | Work | Outcome |
|---|---|---|
| 1 | Run training, read metrics, build report tables | iteration-3 result exists |
| 2 evening | Record 480 squat reps | 2 exercises trainable |
| 3 evening | Record subjects 2 and 3 | held-out-person evaluation possible |
| 4 | Fix `phase`; record shoulder-press error classes | 3 exercises trainable |
| 5 | Re-run all three, write up, export ONNX | iteration 3 done, iteration 4 unblocked |

Two evenings of recording are worth more than two weeks of dataset hunting.

## If you have literally one day

```bash
python -m repai.train --exercise bicep_curl          # the real result
python -m repai.train --exercise bicep_curl --leaky  # the ablation
python -m repai.export --ckpt reports/bicep_curl_tcn.pt
```

Present bicep curl only. Say plainly that squat and shoulder-press collection
is in progress. **One exercise done rigorously reads better than three done
vaguely**, and it matches the panel's own instruction to *"limit the number of
supported exercises for focused development"* (mid-evaluation, slide 4).

To show the squat *pipeline* runs end-to-end without having recorded it:

```bash
python scripts/make_squat_smoketest.py --sessions 20
python -m repai.train --exercise squat
```

This scores 1.000, because it is measuring a parametric curve generator, not
human movement. It proves the code path works and nothing else. The trainer
prints `!! 480/480 repetitions are SYNTHETIC` and refuses to let the number
pass quietly. **If you show it, say on the slide that it is synthetic.** A
panel that discovers undisclosed synthetic data will discard your real 0.93
along with it — and that result is genuinely good. Do not spend it on this.

## The ablation table that wins the viva

Both rows are already reproducible. Put them side by side:

| Split | Window acc. | Rep acc. | Rep macro-F1 |
|---|---|---|---|
| Random windows (leaky) | 0.973 | 0.998 | 0.998 |
| **Grouped by session (honest)** | **0.881** | **0.927** | **0.928** |

Then say: *"Shuffling 30-frame windows puts near-identical neighbours in train
and test and reports 99.8 %. We split by recording session instead and report
92.7 %. The 7-point gap is the leak."*

This is the single highest-value slide in your deck. It converts your biggest
weakness — a small single-subject dataset — into evidence of methodological
care. Most undergraduate projects report the 99.8 % and cannot explain it.

Two more ablations you can quote, both measured:

- **Augmentation on vs. off**: 0.927 either way. Honest reading — augmentation
  helps generalisation to *new bodies*, and the test set here is the same
  body, so it cannot show a gain yet. Expect it to matter once subject 2
  exists. Say that; it is a better answer than pretending it helped.
- **Whole-rep (64 frames, resampled) vs. 30-frame windows**: 0.944 vs. 0.927.
  Whole-rep wins because an `incomplete_extension` is defined by the rep's
  *extremum*, which a window straddling mid-rep may never contain. Keep the
  sliding window for streaming inference in iteration 4; quote whole-rep for
  the offline metric. Report both and explain the difference — it shows you
  understand what the model is looking at.

## Anticipated questions

**"Why a TCN and not thresholds?"** Show the overlap table in
`DATASET_AUDIT.md` §6. `rom_degrees` for `correct` is 106.6 ± 9.2 and for
`elbow_moving` is 95.6 ± 11.9 — the distributions overlap heavily, so a
threshold rule sits near the crossing point. The TCN reaches 0.93 because it
reads the *trajectory shape*, not one summary number.

**"Why a TCN and not an LSTM?"** Your slide 16 answer is fine (parallelism,
dilated receptive field, stable training). Add the concrete number: 51,795
parameters, 61 KB exported, which is what makes on-device inference viable.

**"How big is your dataset?"** Give the real figure — 632 reps, 496 usable
after deduplication, one subject — and immediately state the collection plan.
Do not quote the 1,920 in the FYP-1 report unless the missing files turn up.
See `DATASET_AUDIT.md` §5; that discrepancy is the one thing here that can
actually hurt you, and it is fixed by correcting a table.

**"Does it work on a new person?"** *"Not yet measured — the current
collection is one subject. Multi-subject recording is scheduled and the
evaluation script already supports grouping by `user_id`."* Say this before
they ask it.
