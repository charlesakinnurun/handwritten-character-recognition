"""Shared pytest fixtures.

The whole suite runs against **synthetic fixtures** by default: no dataset download
and no full training. Tests that need real data are marked ``dataset`` and skipped
unless explicitly selected.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from handwriting_intelligence.classes import ClassSet, build_class_set
from handwriting_intelligence.config import Config

NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Low-level synthetic data helpers
# ---------------------------------------------------------------------------
def write_idx_images(path: Path, images: np.ndarray) -> Path:
    """Write a NumPy ``(N, H, W)`` uint8 array as an IDX image file."""
    count, rows, cols = images.shape
    with path.open("wb") as fh:
        fh.write(struct.pack(">IIII", 2051, count, rows, cols))
        fh.write(images.astype(np.uint8).tobytes())
    return path


def write_idx_labels(path: Path, labels: np.ndarray) -> Path:
    """Write a NumPy ``(N,)`` label array as an IDX label file."""
    with path.open("wb") as fh:
        fh.write(struct.pack(">II", 2049, len(labels)))
        fh.write(labels.astype(np.uint8).tobytes())
    return path


def make_character_images(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create ``n`` structured 28x28 images with digit-like patterns and labels.

    Each class gets a distinct blob pattern so a tiny model can actually learn
    something in an integration test.
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 28, 28), dtype=np.uint8)
    labels = rng.integers(0, NUM_CLASSES, size=n).astype(np.int64)
    yy, xx = np.mgrid[0:28, 0:28]
    for i, label in enumerate(labels):
        cx = 6 + (label % 5) * 4
        cy = 6 + (label // 5) * 8 + 6
        radius = 4 + (label % 3)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        images[i][mask] = 255
        images[i] += (rng.random((28, 28)) * 30).astype(np.uint8)
    return images, labels


@pytest.fixture
def synthetic_arrays() -> tuple[np.ndarray, np.ndarray]:
    return make_character_images(120, seed=1)


@pytest.fixture
def mnist_like_dir(tmp_path: Path) -> Path:
    """A directory containing tiny synthetic MNIST-shaped IDX files."""
    train_x, train_y = make_character_images(160, seed=2)
    test_x, test_y = make_character_images(40, seed=3)
    write_idx_images(tmp_path / "train-images-idx3-ubyte", train_x)
    write_idx_labels(tmp_path / "train-labels-idx1-ubyte", train_y)
    write_idx_images(tmp_path / "t10k-images-idx3-ubyte", test_x)
    write_idx_labels(tmp_path / "t10k-labels-idx1-ubyte", test_y)
    return tmp_path


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------
def make_config(root: Path, **overrides) -> Config:
    """Build a small, fast MNIST config rooted at ``root``."""
    data = {
        "experiment": {"name": "tiny", "output_dir": str(root / "runs")},
        "seed": 7,
        "data": {
            "name": "mnist",
            "root": str(root),
            "download": False,
            "val_split": 0.2,
            "batch_size": 16,
            "eval_batch_size": 32,
            "num_workers": 0,
            "cache": False,
        },
        "preprocessing": {"image_size": 28, "invert": "auto", "normalize": True},
        "model": {
            "name": "cnn_baseline",
            "num_classes": None,
            "dropout": 0.1,
            "channels": [8, 16],
            "hidden_dim": 16,
        },
        "training": {
            "epochs": 1,
            "optimizer": "adam",
            "lr": 0.01,
            "scheduler": "none",
            "early_stopping": False,
            "augment": False,
            "device": "cpu",
            "log_every": 0,
        },
    }
    for section, values in overrides.items():
        data[section].update(values)
    return Config.from_dict(data).resolved()


@pytest.fixture
def tiny_config(mnist_like_dir: Path) -> Config:
    return make_config(mnist_like_dir)


# ---------------------------------------------------------------------------
# Model / artifact fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_class_set() -> ClassSet:
    return build_class_set("mnist")


@pytest.fixture
def tiny_artifact(tmp_path: Path, tiny_config: Config, tiny_class_set: ClassSet) -> Path:
    """Serialize a small but real inference artifact and return its path."""
    from handwriting_intelligence.models import build_model
    from handwriting_intelligence.training.checkpoints import save_artifact

    torch.manual_seed(0)
    model = build_model(tiny_config.model, tiny_class_set.num_classes)
    artifact_path = tmp_path / "artifact.pt"
    save_artifact(
        artifact_path,
        model=model,
        class_set=tiny_class_set,
        config=tiny_config,
        environment={"test": True},
        training_summary={"note": "synthetic fixture"},
    )
    return artifact_path


@pytest.fixture
def png_bytes() -> bytes:
    """A small PNG encoding of a synthetic character."""
    import io

    from PIL import Image

    images, _ = make_character_images(1, seed=11)
    buffer = io.BytesIO()
    Image.fromarray(images[0]).save(buffer, format="PNG")
    return buffer.getvalue()
