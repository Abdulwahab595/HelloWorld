"""Temporal Convolutional Network for posture-error classification.

Architecture follows Bai, Kolter & Koltun (2018), which is the reference the
FYP proposal cites: stacked residual blocks of dilated causal 1-D convolutions
with weight normalisation and spatial dropout.  Dilation doubles per block, so
`n_blocks` blocks with kernel size `k` cover a receptive field of

    RF = 1 + 2 * (k - 1) * (2^n_blocks - 1)

With k=3 and 4 blocks that is 61 frames -- just over 2 s at 30 fps, which
comfortably spans one repetition (median 47 frames in the collected data).
Print `TCN.receptive_field` before training and check it against your window
size; a receptive field shorter than the window means the last layer never
sees the start of the rep.

Causal padding is kept even though the phone classifies a completed window,
because the same weights are meant to be re-used for streaming inference in
iteration 4 -- a non-causal model would have to be retrained for that.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import parametrizations


class Chomp1d(nn.Module):
    """Drop the right-hand padding that makes a dilated conv causal."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp].contiguous() if self.chomp else x


class TemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = parametrizations.weight_norm(
            nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation))
        self.conv2 = parametrizations.weight_norm(
            nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation))
        self.net = nn.Sequential(
            self.conv1, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            self.conv2, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        for conv in (self.conv1, self.conv2):
            nn.init.normal_(conv.weight, 0.0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """(B, T, F) -> (B, n_classes) logits.

    The head pools over time with mean+max concatenation rather than taking
    only the last timestep: an `incomplete_extension` rep is defined by the
    *extremum* of the elbow angle, which max-pooling reads directly, while
    `elbow_moving` is a sustained shoulder drift, which mean-pooling reads.
    Using only the last frame threw away both and cost ~8 points of macro-F1
    in early runs.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        channels: tuple[int, ...] = (48, 48, 48, 48),
        kernel: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        blocks = []
        c_prev = n_features
        for i, c in enumerate(channels):
            blocks.append(TemporalBlock(c_prev, c, kernel, dilation=2 ** i, dropout=dropout))
            c_prev = c
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * c_prev, n_classes),
        )
        self.kernel = kernel
        self.n_blocks = len(channels)

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel - 1) * (2 ** self.n_blocks - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x.transpose(1, 2))          # (B, C, T)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(pooled)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
