"""Export a trained TCN for on-device inference.

    python -m repai.export --ckpt reports/bicep_curl_tcn.pt

Writes:
  * ``<name>.onnx``            portable graph, and the input to most converters
  * ``<name>_norm.json``       the train-split mean/std the phone must apply
  * ``<name>_labels.json``     class order + the coaching cue per class

Getting to .tflite
------------------
TensorFlow Lite cannot read a PyTorch checkpoint.  Two routes, in order of
how much can go wrong:

1. ONNX -> TensorFlow -> TFLite (``pip install onnx onnx2tf``), then
   ``ai_edge_torch`` or ``tf.lite.TFLiteConverter``.  Works, but dilated
   causal Conv1d sometimes lands on a Conv2d with an awkward transpose.
2. Re-implement this 52 k-parameter architecture in Keras and load the
   weights.  Tedious, but the graph is then native and quantises cleanly.

Whichever route: after converting, run ``verify_parity`` below on at least
200 real windows.  A converter that silently reorders the channel axis is the
classic way a model that scored 0.93 offline behaves randomly on the phone,
and it will not show up in any accuracy metric you compute on the desktop.

The normalisation constants matter as much as the weights.  They are computed
on the training split only; hard-code the exported values into the Kotlin
side rather than recomputing per session, or the first few reps of every
workout will be scored against garbage statistics.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from .model import TCN
from .schema import COACHING_CUES, FRAME_FEATURES


def load(ckpt_path: str | pathlib.Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TCN(ckpt["n_features"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def verify_parity(model, runner, X: np.ndarray, mean, std, atol: float = 1e-3) -> dict:
    """Compare PyTorch logits against any other runtime on the same windows.

    `runner` takes an (N, T, F) float32 array and returns (N, C) logits.
    Call this after every conversion, not once.
    """
    xb = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(xb)).numpy()
    got = np.asarray(runner(xb), dtype=np.float32)
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    return {"max_abs_diff": float(np.abs(ref - got).max()),
            "argmax_agreement": agree,
            "passed": bool(agree >= 0.99 and np.abs(ref - got).max() < atol * 100)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="reports/bicep_curl_tcn.pt")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model, ckpt = load(args.ckpt)
    name = pathlib.Path(args.ckpt).stem
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    window, n_feat = ckpt["window"], ckpt["n_features"]
    dummy = torch.randn(1, window, n_feat)

    onnx_path = out / f"{name}.onnx"
    torch.onnx.export(
        model, (dummy,), str(onnx_path),
        input_names=["window"], output_names=["logits"],
        dynamic_axes={"window": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )

    (out / f"{name}_norm.json").write_text(json.dumps({
        "feature_order": FRAME_FEATURES,
        "mean": [float(v) for v in np.asarray(ckpt["mean"]).ravel()],
        "std": [float(v) for v in np.asarray(ckpt["std"]).ravel()],
        "window": window,
        "note": "apply (x - mean) / std per channel before inference",
    }, indent=2))

    (out / f"{name}_labels.json").write_text(json.dumps({
        "classes": ckpt["classes"],
        "cues": {c: COACHING_CUES.get(c, "") for c in ckpt["classes"]},
    }, indent=2))

    size_kb = onnx_path.stat().st_size / 1024
    print(f"input  (batch, {window}, {n_feat})   classes {ckpt['classes']}")
    print(f"params {model.n_params():,}")
    print(f"wrote  {onnx_path}  ({size_kb:.0f} KB)")
    print(f"wrote  {out / f'{name}_norm.json'}")
    print(f"wrote  {out / f'{name}_labels.json'}")
    print("\nnext: convert to .tflite, then run repai.export.verify_parity on "
          "real windows\nbefore trusting anything the phone prints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
