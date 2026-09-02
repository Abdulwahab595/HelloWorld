"""Train and honestly evaluate the REP-AI TCN.

    python -m repai.train --exercise bicep_curl
    python -m repai.train --exercise bicep_curl --leaky   # the inflated number

Evaluation protocol
-------------------
Grouped k-fold over *session files*.  Every window and every repetition from
one recording stays on one side of the split.  The alternative -- shuffling
windows -- puts near-identical neighbours in train and test and is reported
here as `--leaky` purely so the gap is visible; it is not a result.

Because the shipped collection contains exactly one subject, even the grouped
number is an upper bound on real-world accuracy.  It answers "can the model
separate these movement patterns at all", not "will it work on a new user".
Both numbers are printed with that caveat attached.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .augment import Augmenter
from .dataset import build_samples, class_counts, load_reps, windows_of
from .schema import FRAME_FEATURES
from .model import TCN


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, mean, std, augmenter: Augmenter | None = None):
        self.X, self.y = X, y
        self.mean, self.std = mean, std
        self.aug = augmenter

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.aug is not None:
            x = self.aug(x)
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])


def drop_rare_classes(reps, min_sessions: int):
    """Remove classes recorded in too few sessions to appear on both sides of
    a grouped split.  Keeping them produces folds where a class is absent from
    train or from test, and every metric for it becomes meaningless."""
    sessions = collections.defaultdict(set)
    for r in reps:
        sessions[r.label].add(r.group)
    keep = {lab for lab, s in sessions.items() if len(s) >= min_sessions}
    dropped = {lab: len(s) for lab, s in sessions.items() if lab not in keep}
    return [r for r in reps if r.label in keep], dropped


def train_fold(Xtr, ytr, Xte, yte, n_classes, args, device,
               features=None) -> tuple[np.ndarray, dict]:
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-6

    aug = (Augmenter(strength=args.aug, seed=args.seed, features=features)
           if args.aug > 0 else None)
    tr = WindowDataset(Xtr, ytr, mean, std, aug)
    te = WindowDataset(Xte, yte, mean, std, None)
    ltr = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=False)
    lte = torch.utils.data.DataLoader(te, batch_size=256, shuffle=False)

    model = TCN(Xtr.shape[-1], n_classes, dropout=args.dropout).to(device)

    # Class weights: the collection is unbalanced (23 correct sessions vs 20
    # of each error), and after dropping duplicates it gets worse.
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weights = torch.tensor((counts.sum() / (n_classes * np.maximum(counts, 1))), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for _ in range(args.epochs):
        model.train()
        for xb, yb in ltr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in lte:
            probs.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    state = {"model": model.state_dict(), "mean": mean, "std": std}
    return probs, state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw/DATASET_COLLECTION")
    ap.add_argument("--exercise", default="bicep_curl")
    ap.add_argument("--mode", default="window", choices=["window", "rep"])
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--aug", type=float, default=1.0, help="augmentation strength; 0 disables")
    ap.add_argument("--min-sessions", type=int, default=3)
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--leaky", action="store_true",
                    help="shuffle windows instead of grouping by session (inflated)")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="sanity control: randomly permute labels within the "
                         "dataset.  A correct pipeline must collapse to chance "
                         "(1/n_classes).  Anything above that is a leak or a bug.")
    ap.add_argument("--exclude", default="",
                    help="comma-separated frame features to drop, e.g. "
                         "back_angle,left_shoulder_angle -- for ablations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reps = load_reps(args.data, exercise=args.exercise,
                     drop_duplicates=not args.keep_duplicates)
    if not reps:
        print(f"no repetitions found for exercise={args.exercise!r} under {args.data}")
        print("if this is squat: the shipped collection has zero squat sessions.")
        return 1

    reps, dropped = drop_rare_classes(reps, args.min_sessions)
    dropped_feats = [f.strip() for f in args.exclude.split(",") if f.strip()]
    unknown = [f for f in dropped_feats if f not in FRAME_FEATURES]
    if unknown:
        print(f"unknown feature(s) {unknown}; valid names: {FRAME_FEATURES}")
        return 1
    kept_feats = [f for f in FRAME_FEATURES if f not in dropped_feats]
    X, y, groups, classes = build_samples(reps, mode=args.mode,
                                          window=args.window, stride=args.stride,
                                          features=kept_feats)

    if args.shuffle_labels:
        # Permute labels at the *session* level, so every window of a recording
        # keeps one consistent (wrong) label.  Shuffling per window would leave
        # the majority-vote structure intact and muddy the control.
        rng = np.random.default_rng(args.seed)
        sess = sorted(set(groups.tolist()))
        lab_of = {g: y[groups == g][0] for g in sess}
        shuffled = rng.permutation([lab_of[g] for g in sess])
        remap = dict(zip(sess, shuffled))
        y = np.array([remap[g] for g in groups.tolist()], dtype=np.int64)

    print("=" * 78)
    print(f"REP-AI TCN  --  exercise={args.exercise}  mode={args.mode}  "
          f"split={'RANDOM (leaky)' if args.leaky else 'grouped by session'}")
    print("=" * 78)
    print(f"sessions {len({r.group for r in reps})}   repetitions {len(reps)}   "
          f"samples {len(X)}   shape {X.shape[1:]}")
    if dropped_feats:
        print(f"excluded {dropped_feats}  ({len(kept_feats)} features kept)")
    print(f"classes  {classes}")
    print(f"balance  {class_counts(y, classes)}")
    if dropped:
        print(f"dropped  {dropped}  (< {args.min_sessions} sessions -- cannot be "
              f"split across folds; collect more before claiming a number for them)")
    synth = sum(r.synthetic for r in reps)
    if synth:
        print(f"!! {synth}/{len(reps)} repetitions are SYNTHETIC -- disclose this "
              f"in any reported result")

    if len(classes) < 2:
        print(f"\nrefusing to train: only {len(classes)} class present "
              f"({classes}).  A single-class 'classifier' scores 100 % by "
              f"predicting the only label\nit knows, which is not a result.  "
              f"Collect the missing error classes for {args.exercise} first "
              f"(see docs/JUGAAR_STRATEGY.md).")
        return 1

    probe = TCN(X.shape[-1], len(classes))
    print(f"model    TCN {probe.n_params():,} params, receptive field "
          f"{probe.receptive_field} frames vs window {X.shape[1]}")

    splitter = (StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
                if args.leaky else
                StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed))
    split_args = (X, y) if args.leaky else (X, y, groups)

    oof = np.zeros((len(X), len(classes)), dtype=np.float32)
    fold_f1, best = [], (None, -1.0)
    t0 = time.time()
    for k, (itr, ite) in enumerate(splitter.split(*split_args), 1):
        probs, state = train_fold(X[itr], y[itr], X[ite], y[ite],
                                  len(classes), args, device, features=kept_feats)
        oof[ite] = probs
        f1 = f1_score(y[ite], probs.argmax(1), average="macro", zero_division=0)
        fold_f1.append(f1)
        if f1 > best[1]:
            best = (state, f1)
        print(f"  fold {k}/{args.folds}  test sessions "
              f"{len(set(groups[ite])) if not args.leaky else '-':>3}  "
              f"macro-F1 {f1:.3f}")

    pred = oof.argmax(1)
    print(f"\nwindow-level  accuracy {(pred == y).mean():.3f}   "
          f"macro-F1 {f1_score(y, pred, average='macro', zero_division=0):.3f}   "
          f"(fold spread {np.mean(fold_f1):.3f} +/- {np.std(fold_f1):.3f})")
    print("\n" + classification_report(y, pred, target_names=classes, digits=3,
                                       zero_division=0))

    # Rep-level verdict: average the window probabilities inside each rep.
    # This is what the phone will actually show the user.
    rep_index, cursor = [], 0
    for r in reps:
        n = 1 if args.mode == "rep" else sum(
            1 for _ in windows_of(r.frames, args.window, args.stride))
        rep_index.append((cursor, cursor + n, r.label))
        cursor += n
    rep_true = np.array([classes.index(lab) for _, _, lab in rep_index])
    rep_pred = np.array([oof[a:b].mean(axis=0).argmax() for a, b, _ in rep_index])
    print(f"rep-level     accuracy {(rep_pred == rep_true).mean():.3f}   "
          f"macro-F1 {f1_score(rep_true, rep_pred, average='macro', zero_division=0):.3f}"
          f"   ({len(rep_true)} repetitions)")

    cm = confusion_matrix(rep_true, rep_pred, labels=range(len(classes)))
    width = max(len(c) for c in classes) + 2
    print("\nrep-level confusion matrix (rows = truth)")
    print(" " * width + "".join(f"{c[:10]:>12}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<{width}}" + "".join(f"{v:>12}" for v in cm[i]))

    if args.leaky:
        print("\n!! --leaky was used.  Windows from one recording appeared in both "
              "train and test.\n   This number is for the ablation table only; do "
              "not report it as a result.")
    elif synth == len(reps):
        print("\n!! every repetition was SYNTHETIC.  This number describes the "
              "generator in\n   scripts/make_squat_smoketest.py, not human "
              "movement.  It confirms the\n   pipeline runs end-to-end; it is "
              "not a result.")
    else:
        subjects = {r.session_id.split("_")[0] for r in reps}
        n_subj = len({s for s in subjects}) if len(subjects) > 1 else 1
        print(f"\nNote: the split holds out recordings, not people.  This "
              f"collection has one\nsubject, so the number is an upper bound: "
              f"it shows the classes are separable,\nnot that the model "
              f"transfers to a new body.  Record a second subject and\nre-run "
              f"with groups = user_id for a deployment-grade figure.")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.exercise}_{'leaky' if args.leaky else 'grouped'}"
    (out / f"{tag}_metrics.json").write_text(json.dumps({
        "exercise": args.exercise, "classes": classes, "leaky": args.leaky,
        "sessions": len({r.group for r in reps}), "reps": len(reps),
        "samples": int(len(X)), "dropped_classes": dropped,
        "window_accuracy": float((pred == y).mean()),
        "window_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "rep_accuracy": float((rep_pred == rep_true).mean()),
        "rep_macro_f1": float(f1_score(rep_true, rep_pred, average="macro", zero_division=0)),
        "fold_macro_f1": [float(v) for v in fold_f1],
        "confusion_matrix": cm.tolist(),
        "params": probe.n_params(), "receptive_field": probe.receptive_field,
        "args": vars(args),
    }, indent=2))

    if best[0] is not None and not args.leaky:
        ckpt = out / f"{args.exercise}_tcn.pt"
        torch.save({"state_dict": best[0]["model"], "mean": best[0]["mean"],
                    "std": best[0]["std"], "classes": classes,
                    "n_features": X.shape[-1], "window": args.window,
                    "feature_names": kept_feats,
                    # so repai.predict can warn when asked to score files this
                    # model has already seen
                    "train_files": sorted({r.source_file for r in reps})}, ckpt)
        print(f"\nsaved {ckpt}  ({best[1]:.3f} macro-F1 fold)")
    print(f"wrote {out / f'{tag}_metrics.json'}   [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
