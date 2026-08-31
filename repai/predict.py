"""Run a trained model over recordings and show its verdict per repetition.

This is the honest test.  Cross-validation tells you the model separates the
recordings it was built from; this tells you what it does with a session it
has never seen, recorded after training finished.  Record a fresh set with the
app, run it through here, and compare.

    # one file
    python -m repai.predict --session data/new/curl_test_001.json

    # a folder, scored against the label in each file
    python -m repai.predict --dir data/new/ --score

    # where to put the confidence threshold
    python -m repai.predict --dir data/new/ --sweep

Confidence gating
-----------------
The model always outputs *some* answer.  Below a confidence threshold it is
better to say nothing than to say something wrong -- a coach who invents
corrections gets ignored.  `--threshold` sets the bar; reps below it are
reported as `unsure` and, on the phone, would produce no voice prompt at all.
`--sweep` prints how accuracy trades against how many reps you keep, so the
threshold is a measured choice rather than a guess.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from .dataset import load_reps, windows_of
from .model import TCN
from .schema import COACHING_CUES

BAR = "-" * 72


def load_model(ckpt_path: str | pathlib.Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TCN(ckpt["n_features"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_rep(model, ckpt, frames: np.ndarray) -> np.ndarray:
    """Average the window probabilities across one repetition."""
    chunks = list(windows_of(frames, ckpt["window"], stride=10))
    x = (np.stack(chunks) - ckpt["mean"]) / ckpt["std"]
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x.astype(np.float32))), dim=1)
    return probs.mean(dim=0).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="reports/bicep_curl_tcn.pt")
    ap.add_argument("--session", help="a single session JSON")
    ap.add_argument("--dir", help="a folder of session JSONs")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="below this confidence the model says 'unsure'")
    ap.add_argument("--score", action="store_true",
                    help="compare against the label recorded in each file")
    ap.add_argument("--sweep", action="store_true",
                    help="print the accuracy/coverage trade-off across thresholds")
    args = ap.parse_args()

    if not args.session and not args.dir:
        ap.error("give --session or --dir")
    ckpt_path = pathlib.Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"no checkpoint at {ckpt_path}\nrun:  python -m repai.train --exercise bicep_curl")
        return 1

    model, ckpt = load_model(ckpt_path)
    classes = ckpt["classes"]

    root = pathlib.Path(args.dir) if args.dir else pathlib.Path(args.session).parent
    reps = load_reps(root, drop_duplicates=False)
    if args.session:
        want = pathlib.Path(args.session).name
        reps = [r for r in reps if pathlib.Path(r.source_file).name == want]
    if not reps:
        print(f"no repetitions found under {root}")
        return 1

    print(BAR)
    print(f"model {ckpt_path.name}   classes {classes}   threshold {args.threshold}")
    print(BAR)

    rows, current = [], None
    for rep in reps:
        if rep.source_file != current:
            current = rep.source_file
            print(f"\n{pathlib.Path(current).name}"
                  + (f"   recorded as: {rep.label}" if args.score else ""))
        p = predict_rep(model, ckpt, rep.frames)
        idx = int(p.argmax())
        conf = float(p[idx])
        confident = conf >= args.threshold
        verdict = classes[idx] if confident else "unsure"

        mark = ""
        if args.score and rep.label in classes:
            mark = ("  ok" if classes[idx] == rep.label else "  MISS") if confident else "  --"
        print(f"  rep {rep.rep_id:>2}   {verdict:<22} {conf:.2f}{mark}")
        if confident and classes[idx] != "correct":
            print(f"           -> \"{COACHING_CUES.get(classes[idx], '')}\"")
        rows.append((rep.label, idx, conf, confident))

    # Compare on basename: `source_file` is relative to whatever root was
    # passed, so the trainer's paths and these will not match as written.
    seen = {pathlib.Path(f).name for f in ckpt.get("train_files", [])}
    here = {pathlib.Path(r.source_file).name for r in reps}
    overlap = (here & seen) if seen else set()

    print("\n" + BAR)
    n = len(rows)
    kept = sum(1 for *_, c in rows if c)
    print(f"{n} repetitions   {kept} confident ({100 * kept / n:.0f} %)   "
          f"{n - kept} unsure")

    if args.score:
        known = [(t, i, c, ok) for t, i, c, ok in rows if t in classes]
        if known:
            hit = sum(1 for t, i, _, ok in known if ok and classes[i] == t)
            kept_k = sum(1 for *_, ok in known if ok)
            print(f"agreement with the recorded label: {hit}/{kept_k} confident "
                  f"reps ({100 * hit / max(kept_k, 1):.0f} %)")
            allhit = sum(1 for t, i, _, _ in known if classes[i] == t)
            print(f"                    ignoring confidence: {allhit}/{len(known)} "
                  f"({100 * allhit / len(known):.0f} %)")

            if overlap:
                frac = 100 * len(overlap) / len(here)
                print(f"\n!! {len(overlap)} of these session files ({frac:.0f} %) "
                      f"were in this model's training data.")
                print("   The agreement above is the model recognising recordings "
                      "it has already\n   memorised -- it is not an accuracy.  "
                      "Record new sessions and score those.")
                print("   The cross-validated figure is the one to report: see "
                      "reports/*_metrics.json.")
            else:
                print("\n   none of these files were in the training data -- "
                      "this is a genuine\n   held-out result.")

    if args.sweep and args.score:
        known = [(t, i, c) for t, i, c, _ in rows if t in classes]
        print(f"\n{'threshold':>10}{'kept':>8}{'coverage':>11}{'accuracy':>11}")
        for th in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            sel = [(t, i) for t, i, c in known if c >= th]
            if not sel:
                print(f"{th:>10.2f}{0:>8}{'0 %':>11}{'-':>11}")
                continue
            acc = sum(1 for t, i in sel if classes[i] == t) / len(sel)
            print(f"{th:>10.2f}{len(sel):>8}"
                  f"{100 * len(sel) / len(known):>10.0f} %{100 * acc:>10.0f} %")
        print("\npick the lowest threshold whose accuracy you would say out loud "
              "to a user.\nhigher threshold = fewer prompts, but the ones you give "
              "are more often right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
