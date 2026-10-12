"""Unit tests for dataset loading, splits and the EMNIST corpus loader."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from handwriting_intelligence.config import Config
from handwriting_intelligence.data.datasets import (
    ArrayDataset,
    build_datasets,
    load_emnist_corpus,
    load_mnist,
    locate_emnist_corpus,
    locate_mnist,
    make_split_indices,
    read_idx_images,
    read_idx_labels,
)
from handwriting_intelligence.exceptions import DatasetError, DatasetNotFoundError
from tests.conftest import make_character_images, write_idx_images, write_idx_labels


def test_idx_roundtrip(tmp_path: Path) -> None:
    images, labels = make_character_images(20, seed=0)
    img_path = write_idx_images(tmp_path / "images", images)
    lbl_path = write_idx_labels(tmp_path / "labels", labels)
    assert np.array_equal(read_idx_images(img_path), images)
    assert np.array_equal(read_idx_labels(lbl_path), labels)


def test_idx_bad_magic_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 16)
    with pytest.raises(DatasetError, match="magic"):
        read_idx_images(bad)


def test_idx_size_mismatch_raises(tmp_path: Path) -> None:
    images, _ = make_character_images(4, seed=0)
    path = write_idx_images(tmp_path / "images", images)
    data = bytearray(path.read_bytes())
    path.write_bytes(bytes(data[:-10]))  # truncate
    with pytest.raises(DatasetError, match="size mismatch"):
        read_idx_images(path)


def test_make_split_indices_disjoint_and_deterministic() -> None:
    a = make_split_indices(100, val_split=0.2, test_split=0.1, seed=3)
    b = make_split_indices(100, val_split=0.2, test_split=0.1, seed=3)
    for key in ("train", "val", "test"):
        assert np.array_equal(a[key], b[key])
    combined = np.concatenate([a["train"], a["val"], a["test"]])
    assert len(np.unique(combined)) == 100
    assert len(a["val"]) == 20 and len(a["test"]) == 10


def test_make_split_indices_caps() -> None:
    out = make_split_indices(
        100, val_split=0.2, test_split=0.2, seed=1, max_train=5, max_val=3, max_test=2
    )
    assert len(out["train"]) == 5
    assert len(out["val"]) == 3
    assert len(out["test"]) == 2


def test_make_split_indices_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError):
        make_split_indices(10, val_split=0.6, test_split=0.6, seed=0)


def test_locate_and_load_mnist(mnist_like_dir: Path) -> None:
    paths = locate_mnist(mnist_like_dir)
    assert paths is not None and len(paths) == 4
    arrays = load_mnist(mnist_like_dir, download=False)
    assert arrays["train_images"].shape == (160, 28, 28)
    assert arrays["test_images"].shape == (40, 28, 28)


def test_load_mnist_missing_without_download(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError):
        load_mnist(tmp_path, download=False)


def test_build_datasets_bundle(mnist_like_dir: Path, tiny_config: Config) -> None:
    bundle = build_datasets(tiny_config)
    assert bundle.class_set.num_classes == 10
    assert bundle.meta["official_test"] is True
    assert len(bundle.train) + len(bundle.val) == 160
    assert len(bundle.test) == 40


def test_array_dataset_transform_applied() -> None:
    images, labels = make_character_images(8, seed=0)
    dataset = ArrayDataset(images, labels, transform=lambda a: a.astype(np.float32))
    item, label = dataset[0]
    assert item.shape == (28, 28)
    assert label == int(labels[0])
    assert dataset.raw_image(0).shape == (28, 28)


def _make_emnist_corpus(root: Path) -> Path:
    """Create a minimal 47-class EMNIST-style corpus with one image per class."""
    corpus = root / "EMNIST" / "EMNIST"
    corpus.mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(0)
    for label in range(47):
        class_dir = corpus / str(label)
        class_dir.mkdir()
        arr = np.zeros((28, 28), dtype=np.uint8)
        arr += (rng.random((28, 28)) * 10).astype(np.uint8)
        arr[0, :] = 255  # a horizontal bar; transposed it becomes vertical
        name = f"char_{label}.jpeg"
        Image.fromarray(arr).save(class_dir / name)
        rows.append({"img_name": f"{label}/{name}", "label": label})
    with (corpus / "emnist.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["img_name", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return corpus


def test_locate_emnist_corpus(tmp_path: Path) -> None:
    corpus = _make_emnist_corpus(tmp_path)
    assert locate_emnist_corpus(tmp_path) == corpus


def test_load_emnist_corpus_and_transpose(tmp_path: Path) -> None:
    _make_emnist_corpus(tmp_path)
    arrays = load_emnist_corpus(locate_emnist_corpus(tmp_path), cache_dir=None, use_cache=False)
    assert arrays["images"].shape == (47, 28, 28)
    assert arrays["labels"].tolist() == list(range(47))
    # The horizontal bar becomes a vertical bar after the documented transpose.
    assert arrays["images"][0][20, 0] == 255


def test_load_emnist_corpus_rejects_wrong_class_count(tmp_path: Path) -> None:
    corpus = tmp_path / "EMNIST" / "EMNIST"
    corpus.mkdir(parents=True)
    (corpus / "0").mkdir()
    Image.fromarray(np.zeros((28, 28), dtype=np.uint8)).save(corpus / "0" / "a.jpeg")
    (corpus / "emnist.csv").write_text("img_name,label\n0/a.jpeg,0\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="47-class"):
        load_emnist_corpus(corpus, cache_dir=None, use_cache=False)


def test_load_emnist_corpus_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="manifest"):
        load_emnist_corpus(tmp_path, cache_dir=None, use_cache=False)
