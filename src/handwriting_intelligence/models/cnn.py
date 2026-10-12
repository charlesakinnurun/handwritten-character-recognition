"""Convolutional character classifiers.

Two families are provided, both returning raw logits of shape ``(B, num_classes)``:

* :class:`CNNBaseline` - the compact, well-understood reference model. Two
  convolutional blocks with batch norm and max-pooling, then a small MLP head.
  It is the yardstick every later change is measured against.
* :class:`CNNImproved` - deeper stack of residual blocks with global average
  pooling, stronger regularization and a wider head. Every addition here is an
  *experimentally motivated* change (see docs/experiments.md), not decoration.

The flat feature dimension is inferred at construction time with a dummy forward
pass, so the same code supports 28x28 and 32x32 inputs without hard-coded sizes.
"""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn as nn

from handwriting_intelligence.config import ModelConfig
from handwriting_intelligence.models.base import CharacterClassifier, register_model


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, optionally followed by 2x2 max-pool and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        pool: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a projection shortcut when the shape changes."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CNNBaseline(CharacterClassifier):
    """Compact CNN baseline: ``channels`` conv blocks + MLP head."""

    def __init__(self, num_classes: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = 1
        self.input_size = 28
        channels = list(cfg.channels)
        if not channels:
            raise ValueError("CNNBaseline requires at least one channel width")

        blocks: list[nn.Module] = [ConvBlock(1, channels[0], pool=True)]
        for prev, curr in pairwise(channels):
            blocks.append(ConvBlock(prev, curr, pool=True))
        self.features = nn.Sequential(*blocks)

        flat_dim = self._infer_flat_dim(channels[-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(flat_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.hidden_dim, num_classes),
        )

    def _infer_flat_dim(self, last_channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_size, self.input_size)
            return int(self.features(dummy).reshape(1, -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNImproved(CharacterClassifier):
    """Deeper residual CNN with global average pooling and a wider head."""

    def __init__(self, num_classes: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = 1
        self.input_size = 28
        channels = list(cfg.channels)
        if not channels:
            raise ValueError("CNNImproved requires at least one channel width")

        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        in_ch = channels[0]
        for index, out_ch in enumerate(channels):
            stride = 1 if index == 0 else 2
            blocks.append(ResidualBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(channels[-1], cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(x))
        return self.classifier(self.pool(features))


@register_model("cnn_baseline")
def build_cnn_baseline(num_classes: int, cfg: ModelConfig) -> CNNBaseline:
    """Factory for the baseline CNN."""
    return CNNBaseline(num_classes, cfg)


@register_model("cnn_improved")
def build_cnn_improved(num_classes: int, cfg: ModelConfig) -> CNNImproved:
    """Factory for the improved residual CNN."""
    return CNNImproved(num_classes, cfg)
