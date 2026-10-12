"""Data package: datasets, transforms and DataLoader assembly."""

from __future__ import annotations

from handwriting_intelligence.data.datamodule import DataLoaders, build_dataloaders
from handwriting_intelligence.data.datasets import (
    ArrayDataset,
    DatasetBundle,
    build_datasets,
    load_mnist,
)

__all__ = [
    "ArrayDataset",
    "DataLoaders",
    "DatasetBundle",
    "build_dataloaders",
    "build_datasets",
    "load_mnist",
]
