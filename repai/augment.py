"""Temporal augmentation for biomechanical angle sequences.

The collected tree declares ``slow/``, ``normal/`` and ``fast/`` tempo buckets
but only ``normal/`` was ever filled.  Time-warping a normal-tempo rep to
0.7x and 1.4x duration synthesises the two missing buckets from real motion,
which is far more defensible than fabricating trajectories from scratch: the
joint-angle *path* is real, only the speed along it changes.

Every transform here is applied on-the-fly to the training split only.  None
of it ever touches validation or test data -- augmenting the test set is a
silent way to report an accuracy that does not exist.

Angles are in degrees; velocity/acceleration channels are derivatives, so a
time warp has to rescale them (x1/k and x1/k^2) or the augmented sample stops
being physically consistent.  That rescaling is the part most implementations
get wrong.
"""
from __future__ import annotations

import numpy as np

from .schema import FRAME_FEATURES, MIRROR_PAIRS

# Channel roles, by name.  Indices are resolved against whatever feature
# subset is actually in use -- hardcoding positions breaks silently the
# moment an ablation drops a channel, and the resulting IndexError is the
# lucky case; a subset that happens to stay in range would corrupt the
# wrong column instead.
_ANGLE_NAMES = ["knee_angle", "hip_angle", "elbow_angle", "right_elbow_angle",
                "back_angle", "left_shoulder_angle", "right_shoulder_angle",
                "avg_elbow_angle"]
_RATE1_NAMES = ["velocity", "angular_velocity"]
_RATE2_NAMES = ["angular_acceleration"]


def channel_roles(features: list[str] | None = None):
    """Resolve (angle, rate1, rate2, mirror-pair) indices for a feature list."""
    names = list(features) if features else list(FRAME_FEATURES)
    pos = {n: i for i, n in enumerate(names)}
    angles = [pos[n] for n in _ANGLE_NAMES if n in pos]
    rate1 = [pos[n] for n in _RATE1_NAMES if n in pos]
    rate2 = [pos[n] for n in _RATE2_NAMES if n in pos]
    mirror = [(pos[a], pos[b]) for a, b in MIRROR_PAIRS if a in pos and b in pos]
    return angles, rate1, rate2, mirror


_ANGLE_CHANNELS, _RATE1_CHANNELS, _RATE2_CHANNELS, _MIRROR_IDX = channel_roles()


def time_warp(seq: np.ndarray, factor: float, length: int | None = None,
              rate1=None, rate2=None) -> np.ndarray:
    """Replay the rep `factor`x faster (factor > 1) or slower (factor < 1).

    The output is resampled back to `length` frames (default: unchanged) so it
    still fits a fixed-size model input, and the derivative channels are
    rescaled to stay consistent with the new tempo.
    """
    length = len(seq) if length is None else length
    t_src = np.linspace(0.0, 1.0, len(seq), dtype=np.float32)
    t_dst = np.linspace(0.0, 1.0, length, dtype=np.float32)
    out = np.empty((length, seq.shape[1]), dtype=np.float32)
    for j in range(seq.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, seq[:, j])
    out[:, _RATE1_CHANNELS if rate1 is None else rate1] *= factor
    out[:, _RATE2_CHANNELS if rate2 is None else rate2] *= factor ** 2
    return out


def jitter(seq: np.ndarray, sigma_deg: float, rng: np.random.Generator,
           angles=None) -> np.ndarray:
    """Additive noise on the angle channels only, standing in for the
    frame-to-frame MediaPipe landmark flicker reported in the FYP-1 report."""
    out = seq.copy()
    a = _ANGLE_CHANNELS if angles is None else angles
    noise = rng.normal(0.0, sigma_deg, size=(len(seq), len(a)))
    out[:, a] += noise.astype(np.float32)
    return out


def scale_rom(seq: np.ndarray, factor: float, angles=None) -> np.ndarray:
    """Scale each angle channel about its own mean.

    Simulates a taller/shorter subject and a slightly different camera
    distance.  Deliberately mild: push it too far and a `correct` rep turns
    into an `incomplete_extension` one, which relabels the sample.
    """
    out = seq.copy()
    a = _ANGLE_CHANNELS if angles is None else angles
    block = out[:, a]
    mean = block.mean(axis=0, keepdims=True)
    out[:, a] = mean + (block - mean) * factor
    return out


def offset(seq: np.ndarray, deg: float, angles=None) -> np.ndarray:
    """Constant angular bias -- a camera mounted a few degrees off-axis."""
    out = seq.copy()
    out[:, _ANGLE_CHANNELS if angles is None else angles] += deg
    return out


def mirror(seq: np.ndarray, pairs=None) -> np.ndarray:
    out = seq.copy()
    for ia, ib in (_MIRROR_IDX if pairs is None else pairs):
        out[:, [ia, ib]] = out[:, [ib, ia]]
    return out


def crop_shift(seq: np.ndarray, rng: np.random.Generator, max_frac: float = 0.1,
               rate1=None, rate2=None) -> np.ndarray:
    """Trim up to `max_frac` off one end and stretch back to full length,
    modelling the rep-boundary jitter the report calls 'giant first-repetition
    contamination'."""
    n = len(seq)
    k = int(n * max_frac)
    if k < 1:
        return seq
    lo = rng.integers(0, k + 1)
    hi = n - rng.integers(0, k + 1)
    if hi - lo < 4:
        return seq
    return time_warp(seq[lo:hi], 1.0, length=n, rate1=rate1, rate2=rate2)


class Augmenter:
    """Randomised augmentation pipeline.

    `strength` scales every magnitude at once so ablations are one number.
    Setting it to 0 disables augmentation entirely.
    """

    def __init__(
        self,
        strength: float = 1.0,
        p_time: float = 0.6,
        p_jitter: float = 0.7,
        p_scale: float = 0.4,
        p_offset: float = 0.3,
        p_mirror: float = 0.3,
        p_crop: float = 0.3,
        seed: int = 0,
        features: list[str] | None = None,
    ):
        self.angles, self.rate1, self.rate2, self.pairs = channel_roles(features)
        self.strength = strength
        self.p = dict(time=p_time, jitter=p_jitter, scale=p_scale,
                      offset=p_offset, mirror=p_mirror, crop=p_crop)
        self.rng = np.random.default_rng(seed)

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        if self.strength <= 0:
            return seq
        s, rng = self.strength, self.rng
        out = seq
        if rng.random() < self.p["time"]:
            out = time_warp(out, float(rng.uniform(1 - 0.3 * s, 1 + 0.4 * s)),
                            rate1=self.rate1, rate2=self.rate2)
        if rng.random() < self.p["crop"]:
            out = crop_shift(out, rng, max_frac=0.10 * s,
                             rate1=self.rate1, rate2=self.rate2)
        if rng.random() < self.p["scale"]:
            out = scale_rom(out, float(rng.uniform(1 - 0.08 * s, 1 + 0.08 * s)),
                            angles=self.angles)
        if rng.random() < self.p["offset"]:
            out = offset(out, float(rng.uniform(-4.0 * s, 4.0 * s)), angles=self.angles)
        if rng.random() < self.p["mirror"]:
            out = mirror(out, pairs=self.pairs)
        if rng.random() < self.p["jitter"]:
            out = jitter(out, 1.5 * s, rng, angles=self.angles)
        return out.astype(np.float32)


def expand_tempo_buckets(seq: np.ndarray, factors=(0.7, 1.0, 1.4)) -> list[np.ndarray]:
    """Deterministic version used to materialise the empty slow/fast folders.

    0.7 -> the `slow/` bucket, 1.4 -> the `fast/` bucket.
    """
    return [time_warp(seq, f) for f in factors]
