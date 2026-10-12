"""CRNN: sequence recognition for words and lines (research extension).

Why this is separate from the character pipeline
------------------------------------------------
A single-character classifier answers "which one character is this?" It has no
mechanism to (a) discover *how many* characters are present, (b) segment them, or
(c) enforce consistency across a sequence. Unconstrained handwriting recognition
therefore needs a **sequence** model that maps a variable-width image to a variable
length of characters - this is the classic CRNN formulation:

    CNN feature extractor -> recurrent sequence model -> per-frame character logits
    -> CTC loss (alignment-free) -> greedy/beam decoding.

The blank symbol required by CTC is class index ``0``; real characters occupy
``1..num_classes``.

Status
------
This module is **implemented but not trained here**. It requires a word/line-level
dataset such as the IAM Handwriting Database, which is distributed under a
research license and requires registration. No word- or sentence-level accuracy is
claimed anywhere in this repository until this model is trained and evaluated on
such a dataset with CER/WER metrics. See ``docs/architecture.md``.

The CRNN is intentionally *not* part of the character-classifier registry used by
the training CLI, because its output is a sequence, not a single class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from handwriting_intelligence.config import ModelConfig

#: CTC blank symbol index.
CTC_BLANK = 0


class CRNN(nn.Module):
    """CNN + BiLSTM + CTC sequence recognizer.

    Args:
        num_classes: Number of *character* classes (excluding the CTC blank).
        cfg: Model configuration (channel widths, RNN width, depth, dropout).
        input_height: Height the CNN expects; features are pooled to height 1.
    """

    def __init__(self, num_classes: int, cfg: ModelConfig, *, input_height: int = 32) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        self.num_classes = num_classes
        self.input_channels = 1
        self.input_size = input_height
        self.num_outputs = num_classes + 1  # +1 for the CTC blank at index 0

        channels = list(cfg.channels) or [32, 64]
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=in_ch,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(2 * cfg.rnn_hidden, self.num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map an image batch to per-timestep log-probabilities.

        Args:
            x: Tensor of shape ``(B, 1, H, W)``.

        Returns:
            ``log_probs`` of shape ``(T, B, num_classes + 1)`` suitable for
            :func:`torch.nn.functional.ctc_loss`, where ``T`` is the downsampled width.
        """
        features = self.cnn(x)  # (B, C, H', W')
        features = self.height_pool(features).squeeze(2)  # (B, C, W')
        sequence = features.permute(0, 2, 1)  # (B, W', C)
        encoded, _ = self.rnn(sequence)  # (B, W', 2H)
        logits = self.fc(self.dropout(encoded))  # (B, W', num_outputs)
        return logits.log_softmax(dim=2).permute(1, 0, 2)  # (T, B, num_outputs)

    @staticmethod
    def decode_greedy(log_probs: torch.Tensor, *, blank: int = CTC_BLANK) -> list[list[int]]:
        """Greedy CTC decoding: argmax per frame, collapse repeats, drop blanks.

        Returns one list of character indices (1-based, blank excluded) per sample.
        """
        best = log_probs.argmax(dim=2)  # (T, B)
        best = best.transpose(0, 1)  # (B, T)
        decoded: list[list[int]] = []
        for sequence in best.tolist():
            collapsed: list[int] = []
            previous: int | None = None
            for token in sequence:
                if token != previous and token != blank:
                    collapsed.append(token)
                previous = token
            decoded.append(collapsed)
        return decoded


@dataclass(slots=True)
class CRNNPrediction:
    """A decoded sequence plus its average per-frame confidence."""

    indices: list[int]
    confidence: float


def build_crnn(num_classes: int, cfg: ModelConfig, *, input_height: int = 32) -> CRNN:
    """Construct a :class:`CRNN` for ``num_classes`` characters (blank added internally)."""
    return CRNN(num_classes, cfg, input_height=input_height)
