"""Assemble datasets into ready-to-iterate :class:`torch.utils.data.DataLoader`s.

This is the only place that knows about batching, shuffling, worker counts and
pinned memory, so training, evaluation and tests all get identically configured
loaders. ``pin_memory`` is enabled only for CUDA devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from handwriting_intelligence.classes import ClassSet
from handwriting_intelligence.config import Config
from handwriting_intelligence.data.datasets import build_datasets
from handwriting_intelligence.data.transforms import build_dataset_transform
from handwriting_intelligence.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DataLoaders:
    """Bundle of loaders plus class-set and dataset metadata."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_set: ClassSet
    meta: dict[str, object] = field(default_factory=dict)


def build_dataloaders(cfg: Config, *, pin_memory: bool = False) -> DataLoaders:
    """Build train/val/test loaders from a validated configuration.

    Args:
        cfg: Validated configuration.
        pin_memory: Set True when training on CUDA to speed host->device copies.

    Returns:
        A :class:`DataLoaders` bundle.
    """

    def transform_factory(train: bool):
        return build_dataset_transform(cfg.preprocessing, cfg.training, train=train)

    bundle = build_datasets(cfg, transform_factory=transform_factory)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    common: dict[str, Any] = {
        "num_workers": cfg.data.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": cfg.data.num_workers > 0,
    }
    train_loader = DataLoader(
        bundle.train,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        bundle.val, batch_size=cfg.data.eval_batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        bundle.test, batch_size=cfg.data.eval_batch_size, shuffle=False, **common
    )

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_set=bundle.class_set,
        meta=bundle.meta,
    )
