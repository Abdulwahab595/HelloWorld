"""Load REP-AI JSON sessions into model-ready tensors.

Two windowing modes are supported:

``window``  the 30-frame sliding window the proposal specifies.  A rep of
            length L yields ``ceil((L - W) / S) + 1`` windows, each inheriting
            the rep's label.  At inference the phone predicts per window and
            the per-rep verdict is the mean of the window probabilities.

``rep``     one sample per repetition, linearly resampled to a fixed length.
            Fewer samples but no label noise from windows that straddle a
            phase boundary.  Useful as a sanity check.

Splitting is *always* by session, never by window or rep.  Windows from one
recording are near-duplicates of each other; letting them straddle the split
inflates accuracy by tens of points and is the single most common way a form-
classification result turns out to be meaningless.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import pathlib
from typing import Iterator, Sequence

import numpy as np

from .schema import EXERCISE_DIRS, FRAME_FEATURES, LABEL_ALIASES, MIRROR_PAIRS


@dataclasses.dataclass
class Rep:
    """One repetition, already reduced to a (T, F) float32 array."""
    exercise: str
    label: str
    speed: str
    session_id: str
    source_file: str
    rep_id: int
    frames: np.ndarray          # (T, len(FRAME_FEATURES))
    summary: dict
    synthetic: bool = False

    @property
    def group(self) -> str:
        """Split key.  All reps recorded in one sitting share it."""
        return f"{self.source_file}"


def _frames_to_array(frames: Sequence[dict]) -> np.ndarray:
    out = np.empty((len(frames), len(FRAME_FEATURES)), dtype=np.float32)
    for i, frame in enumerate(frames):
        feats = frame["features"]
        for j, name in enumerate(FRAME_FEATURES):
            out[i, j] = feats.get(name, 0.0)
    return out


def load_reps(
    root: str | pathlib.Path,
    exercise: str | None = None,
    drop_duplicates: bool = True,
    min_frames: int = 8,
) -> list[Rep]:
    """Read every session under `root` and return a flat list of `Rep`.

    `drop_duplicates` removes byte-identical session files, which the shipped
    collection contains 11 of.  Keeping them silently triples the weight of a
    handful of shoulder-press recordings.
    """
    root = pathlib.Path(root)
    reps: list[Rep] = []
    seen: set[str] = set()

    for path in sorted(root.rglob("*.json")):
        raw = path.read_bytes()
        digest = hashlib.md5(raw).hexdigest()
        if drop_duplicates and digest in seen:
            continue
        seen.add(digest)

        rel = path.relative_to(root)
        session = json.loads(raw)

        # The canonical layout is <EXERCISE>/<label>/<speed>/<file>.json.  When
        # `root` points partway into that tree the folder names are no longer
        # where we expect them, so fall back to the fields inside the file --
        # otherwise the label silently becomes a filename.
        if len(rel.parts) >= 4:
            ex = EXERCISE_DIRS.get(rel.parts[0], rel.parts[0])
            label = rel.parts[1]
            speed = rel.parts[2]
        else:
            ex = session.get("exercise", "unknown")
            label = session.get("label", "unknown")
            speed = rel.parts[-2] if len(rel.parts) > 1 else "normal"
        label = LABEL_ALIASES.get(label, label)

        if exercise is not None and ex != exercise:
            continue
        for rep in session["reps"]:
            arr = _frames_to_array(rep["frames"])
            if len(arr) < min_frames:
                continue
            reps.append(
                Rep(
                    exercise=ex,
                    label=label,
                    speed=speed,
                    session_id=session.get("session_id", str(rel)),
                    source_file=str(rel),
                    rep_id=rep.get("rep_id", -1),
                    frames=arr,
                    summary=rep.get("rep_summary", {}),
                    synthetic=bool(session.get("synthetic", False)),
                )
            )
    return reps


def resample(seq: np.ndarray, length: int) -> np.ndarray:
    """Linear time-resampling of a (T, F) sequence to (length, F)."""
    t_src = np.linspace(0.0, 1.0, len(seq), dtype=np.float32)
    t_dst = np.linspace(0.0, 1.0, length, dtype=np.float32)
    out = np.empty((length, seq.shape[1]), dtype=np.float32)
    for j in range(seq.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, seq[:, j])
    return out


def windows_of(seq: np.ndarray, size: int, stride: int) -> Iterator[np.ndarray]:
    """Yield `size`-frame windows.  Reps shorter than `size` are edge-padded
    once so that short (e.g. incomplete) reps are not silently dropped -- that
    would delete exactly the class we want to detect."""
    if len(seq) < size:
        pad = np.repeat(seq[-1:], size - len(seq), axis=0)
        yield np.concatenate([seq, pad], axis=0)
        return
    for start in range(0, len(seq) - size + 1, stride):
        yield seq[start:start + size]
    tail = seq[-size:]
    if (len(seq) - size) % stride:
        yield tail


def build_samples(
    reps: Sequence[Rep],
    mode: str = "window",
    window: int = 30,
    stride: int = 10,
    rep_length: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return `(X, y, groups, classes)`.

    X       (N, T, F) float32
    y       (N,) int64 class index
    groups  (N,) object, the session file each sample came from
    classes sorted list of label names, index-aligned with `y`
    """
    classes = sorted({r.label for r in reps})
    index = {c: i for i, c in enumerate(classes)}

    xs, ys, gs = [], [], []
    for rep in reps:
        if mode == "rep":
            chunks = [resample(rep.frames, rep_length)]
        elif mode == "window":
            chunks = list(windows_of(rep.frames, window, stride))
        else:
            raise ValueError(f"unknown mode {mode!r}")
        for chunk in chunks:
            xs.append(chunk)
            ys.append(index[rep.label])
            gs.append(rep.group)

    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    groups = np.asarray(gs, dtype=object)
    return X, y, groups, classes


def mirror(seq: np.ndarray) -> np.ndarray:
    """Swap the left/right feature channels of a (T, F) sequence."""
    out = seq.copy()
    for a, b in MIRROR_PAIRS:
        ia, ib = FRAME_FEATURES.index(a), FRAME_FEATURES.index(b)
        out[:, [ia, ib]] = out[:, [ib, ia]]
    return out


def class_counts(y: np.ndarray, classes: Sequence[str]) -> dict[str, int]:
    counter = collections.Counter(y.tolist())
    return {c: counter.get(i, 0) for i, c in enumerate(classes)}
