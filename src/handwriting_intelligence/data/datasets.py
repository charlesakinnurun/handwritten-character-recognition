"""Dataset loading for MNIST and the EMNIST splits.

Design goals
------------
* **One interface.** Every dataset is exposed as a ``torch.utils.data.Dataset``
  returning ``(uint8 HxW array, int label)``. Transform application is a separate
  concern handled by :mod:`handwriting_intelligence.data.transforms`.
* **Deterministic splits.** The validation split is derived from the training data
  with a seeded RNG and sorted indices, so the *same* split is used across
  experiments. The official test set is only ever touched for final evaluation.
* **Integrity, not assumptions.** MNIST IDX files are validated by magic number and
  expected byte size; the local EMNIST corpus is validated against its CSV manifest.
* **Loud, actionable failures.** Missing or corrupt data raises
  :class:`~handwriting_intelligence.exceptions.DatasetError` with the exact path and
  the command that fixes it - never a silent substitution of another dataset.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import struct
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from handwriting_intelligence.classes import ClassSet, build_class_set
from handwriting_intelligence.exceptions import DatasetError, DatasetNotFoundError
from handwriting_intelligence.utils.logging import get_logger

logger = get_logger(__name__)

#: Official MNIST mirror (the original yann.lecun.com host is frequently offline).
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}

#: EMNIST ``balanced`` corpus size as distributed (112800 training samples).
EMNIST_BALANCED_TRAIN_SIZE = 112_800

_IDX_MAGIC_IMAGES = 2051  # 0x00000803
_IDX_MAGIC_LABELS = 2049  # 0x00000801


# ---------------------------------------------------------------------------
# IDX format
# ---------------------------------------------------------------------------
def read_idx_images(path: str | Path) -> np.ndarray:
    """Read an IDX image file into a ``uint8`` array of shape ``(N, H, W)``."""
    data = _read_idx_bytes(path)
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != _IDX_MAGIC_IMAGES:
        raise DatasetError(f"{path}: bad IDX image magic {magic} (expected {_IDX_MAGIC_IMAGES})")
    expected = 16 + count * rows * cols
    if len(data) != expected:
        raise DatasetError(
            f"{path}: size mismatch - expected {expected} bytes for {count} {rows}x{cols} "
            f"images, found {len(data)}"
        )
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def read_idx_labels(path: str | Path) -> np.ndarray:
    """Read an IDX label file into an ``int64`` array of shape ``(N,)``."""
    data = _read_idx_bytes(path)
    magic, count = struct.unpack(">II", data[:8])
    if magic != _IDX_MAGIC_LABELS:
        raise DatasetError(f"{path}: bad IDX label magic {magic} (expected {_IDX_MAGIC_LABELS})")
    expected = 8 + count
    if len(data) != expected:
        raise DatasetError(
            f"{path}: size mismatch - expected {expected} bytes for {count} labels, "
            f"found {len(data)}"
        )
    return np.frombuffer(data, dtype=np.uint8, offset=8).astype(np.int64)


def _read_idx_bytes(path: str | Path) -> bytes:
    p = Path(path)
    if not p.exists():
        raise DatasetNotFoundError(f"dataset file not found: {p}")
    raw = p.read_bytes()
    if raw[:2] == b"\x1f\x8b":  # gzip-compressed
        raw = gzip.decompress(raw)
    return raw


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------
def _find_idx_file(directory: Path, keywords: Sequence[str]) -> Path | None:
    """Return the first file in ``directory`` whose name contains all keywords."""
    if not directory.is_dir():
        return None
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name.lower()
        # Skip truncated/placeholder files (e.g. a 1-byte stub left by a failed
        # download) while still accepting small but valid label files.
        if all(k in name for k in keywords) and entry.stat().st_size > 8:
            return entry
    return None


def _mnist_candidate_dirs(root: Path) -> list[Path]:
    """Candidate directories that may contain the MNIST IDX files."""
    dirs = [root, root / "MNIST", root / "MNIST" / "raw", root / "mnist"]
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and ("mnist" in child.name.lower() or "minst" in child.name.lower()):
                dirs.extend([child, child / "raw"])
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        rd = d.resolve() if d.exists() else d
        if rd not in seen:
            seen.add(rd)
            unique.append(d)
    return unique


def locate_mnist(root: str | Path) -> dict[str, Path] | None:
    """Locate the four MNIST IDX files below ``root`` (any common layout)."""
    root = Path(root)
    found: dict[str, Path] = {}
    for key, keywords in {
        "train_images": ("train-images", "idx3", "ubyte"),
        "train_labels": ("train-labels", "idx1", "ubyte"),
        "test_images": ("t10k-images", "idx3", "ubyte"),
        "test_labels": ("t10k-labels", "idx1", "ubyte"),
    }.items():
        for directory in _mnist_candidate_dirs(root):
            match = _find_idx_file(directory, keywords)
            if match is not None:
                found[key] = match
                break
    return found if len(found) == 4 else None


def download_mnist(root: str | Path) -> dict[str, Path]:
    """Download (and MD5-verify) the MNIST IDX files into ``root/mnist/raw``.

    Kept under a dedicated subdirectory so we never clobber a user's existing copy.
    """
    root = Path(root)
    target = root / "mnist" / "raw"
    target.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for key, (filename, md5) in MNIST_FILES.items():
        gz_path = target / filename
        idx_name = filename.replace(".gz", "")
        idx_path = target / idx_name
        if not idx_path.exists():
            if not gz_path.exists():
                _download(f"{MNIST_MIRROR}{filename}", gz_path)
            _verify_md5(gz_path, md5)
            with gzip.open(gz_path, "rb") as src, open(idx_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        resolved[key] = idx_path
    # Validate immediately so corruption surfaces here, not mid-training.
    read_idx_images(resolved["train_images"])
    read_idx_labels(resolved["train_labels"])
    return resolved


def _download(url: str, destination: Path) -> None:
    logger.info("downloading dataset", extra={"url": url})
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(destination, "wb") as fh:
            shutil.copyfileobj(response, fh)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise DatasetError(
            f"failed to download {url}: {exc}\n"
            "Check your network connection, or place the MNIST IDX files under "
            "`data/` manually (see docs/data.md)."
        ) from exc


def _verify_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    if digest != expected:
        path.unlink(missing_ok=True)
        raise DatasetError(f"MD5 mismatch for {path}: expected {expected}, got {digest}")


def load_mnist(root: str | Path, *, download: bool = True) -> dict[str, np.ndarray]:
    """Load MNIST as arrays. Uses local files when present, else downloads."""
    paths = locate_mnist(root)
    if paths is None:
        if not download:
            raise DatasetNotFoundError(
                f"MNIST not found under {Path(root).resolve()} and download=False.\n"
                "Run `python -m handwriting_intelligence.cli.download --config configs/mnist_cnn.yaml`."
            )
        paths = download_mnist(root)
    return {
        "train_images": read_idx_images(paths["train_images"]),
        "train_labels": read_idx_labels(paths["train_labels"]),
        "test_images": read_idx_images(paths["test_images"]),
        "test_labels": read_idx_labels(paths["test_labels"]),
    }


# ---------------------------------------------------------------------------
# EMNIST local JPEG corpus
# ---------------------------------------------------------------------------
def locate_emnist_corpus(root: str | Path) -> Path | None:
    """Locate a local ``EMNIST`` JPEG corpus (numeric class dirs + ``emnist.csv``)."""
    root = Path(root)
    candidates = [
        root / "EMNIST" / "EMNIST",
        root / "EMNIST" / "emnist_jpeg" / "EMNIST",
        root / "EMNIST",
        root / "emnist",
    ]
    for candidate in candidates:
        if (candidate / "emnist.csv").is_file():
            return candidate
    return None


def _is_numeric_class_dir(path: Path) -> bool:
    return path.is_dir() and path.name.isdigit()


def load_emnist_corpus(
    corpus_dir: str | Path,
    *,
    expected_name: str = "emnist_balanced",
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, np.ndarray]:
    """Load a local EMNIST JPEG corpus into arrays, with an on-disk NumPy cache.

    The corpus ships a single folder per class plus an ``emnist.csv`` manifest
    mapping ``img_name`` (``"<class>/<file>"``) to integer ``label``. We read the
    manifest, decode every referenced image, and store the result in a compressed
    ``.npz`` cache so subsequent runs are fast.
    """
    corpus_dir = Path(corpus_dir)
    manifest = corpus_dir / "emnist.csv"
    if not manifest.is_file():
        raise DatasetError(f"EMNIST manifest not found: {manifest}")

    class_dirs = sorted(
        (d for d in corpus_dir.iterdir() if _is_numeric_class_dir(d)),
        key=lambda d: int(d.name),
    )
    if not class_dirs:
        raise DatasetError(f"no numeric class directories found under {corpus_dir}")

    cache_path = None
    if use_cache and cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{expected_name}_corpus.npz"
        if cache_path.is_file():
            logger.info("loading EMNIST corpus from cache", extra={"path": str(cache_path)})
            cached = np.load(cache_path)
            return {"images": cached["images"], "labels": cached["labels"]}

    rows = _read_emnist_manifest(manifest)
    images = np.zeros((len(rows), 28, 28), dtype=np.uint8)
    labels = np.zeros(len(rows), dtype=np.int64)
    for i, (rel_path, label) in enumerate(rows):
        image_path = corpus_dir / rel_path
        if not image_path.is_file():
            raise DatasetError(f"EMNIST manifest references a missing file: {image_path}")
        with Image.open(image_path) as img:
            arr = np.asarray(img.convert("L"))
        if arr.shape != (28, 28):
            raise DatasetError(f"expected 28x28 EMNIST image, got {arr.shape} at {image_path}")
        # EMNIST images are distributed in a transposed orientation; without this the
        # glyphs are rotated 90 degrees (verified visually: class 1 becomes a '1').
        images[i] = np.ascontiguousarray(arr.T)
        labels[i] = label

    num_classes = int(np.max(labels)) + 1
    if num_classes != 47:
        raise DatasetError(
            f"local corpus has {num_classes} classes; this loader expects the 47-class "
            "EMNIST balanced split"
        )
    if cache_path is not None:
        np.savez_compressed(cache_path, images=images, labels=labels)
        logger.info("wrote EMNIST cache", extra={"path": str(cache_path)})
    return {"images": images, "labels": labels}


def _read_emnist_manifest(manifest: Path) -> list[tuple[str, int]]:
    import csv

    rows: list[tuple[str, int]] = []
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "img_name" not in reader.fieldnames:
            raise DatasetError(
                f"unexpected EMNIST manifest header in {manifest}: {reader.fieldnames}"
            )
        for record in reader:
            rows.append((record["img_name"], int(record["label"])))
    return rows


def load_emnist_torchvision(
    root: str | Path, split: str, *, download: bool = True
) -> dict[str, np.ndarray]:
    """Load an EMNIST split via torchvision (used for letters/digits/byclass).

    torchvision distributes EMNIST as a ~500 MB archive. This path is only used for
    the splits that are not available as a local JPEG corpus. Images are transposed
    to the upright orientation, matching the local corpus loader.
    """
    try:
        from torchvision.datasets import EMNIST
    except ImportError as exc:  # pragma: no cover - torchvision is a hard dependency
        raise DatasetError("torchvision is required to download EMNIST") from exc

    try:
        train = EMNIST(str(root), split=split, train=True, download=download)
        test = EMNIST(str(root), split=split, train=False, download=download)
    except Exception as exc:
        raise DatasetError(
            f"could not obtain EMNIST split '{split}' from torchvision: {exc}\n"
            "EMNIST is distributed as a large archive; see docs/data.md."
        ) from exc

    def to_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
        images = dataset.data.squeeze(1).numpy().astype(np.uint8)
        labels = dataset.targets.numpy().astype(np.int64)
        return np.ascontiguousarray(np.transpose(images, (0, 2, 1))), labels

    train_x, train_y = to_arrays(train)
    test_x, test_y = to_arrays(test)
    return {
        "train_images": train_x,
        "train_labels": train_y,
        "test_images": test_x,
        "test_labels": test_y,
    }


# ---------------------------------------------------------------------------
# torch.utils.data datasets
# ---------------------------------------------------------------------------
class ArrayDataset(Dataset):
    """An in-memory ``(images, labels)`` dataset with an optional transform.

    ``images`` is a ``uint8`` array of shape ``(N, H, W)``. ``indices`` selects a
    subset (used for deterministic splits). ``transform`` maps a uint8 HxW array to
    the tensor the model expects.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        indices: np.ndarray | None = None,
        transform: Callable[[np.ndarray], object] | None = None,
    ) -> None:
        if len(images) != len(labels):
            raise DatasetError(f"images ({len(images)}) and labels ({len(labels)}) length mismatch")
        self._images = images
        self._labels = labels
        self._indices = (
            np.arange(len(labels), dtype=np.int64) if indices is None else np.asarray(indices)
        )
        self._transform = transform

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> tuple[object, int]:
        real = int(self._indices[index])
        image = self._images[real]
        label = int(self._labels[real])
        if self._transform is not None:
            return self._transform(image), label
        return image, label

    @property
    def labels(self) -> np.ndarray:
        """Labels of the selected subset (used for class-distribution reporting)."""
        return self._labels[self._indices]

    def raw_image(self, index: int) -> np.ndarray:
        """Return the raw ``uint8`` image at subset position ``index`` (no transform)."""
        return self._images[int(self._indices[index])]


@dataclass(slots=True)
class DatasetBundle:
    """Datasets plus the metadata needed to describe and reproduce a run."""

    train: Dataset
    val: Dataset
    test: Dataset
    class_set: ClassSet
    meta: dict[str, object] = field(default_factory=dict)


def make_split_indices(
    num_samples: int,
    *,
    val_split: float,
    test_split: float,
    seed: int,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
) -> dict[str, np.ndarray]:
    """Return deterministic, disjoint train/val/test index arrays.

    The split is a seeded permutation so it is stable across processes and
    machines, and the resulting index arrays are sorted so iteration order is
    reproducible regardless of RNG state.
    """
    if not 0.0 <= val_split + test_split < 1.0:
        raise ValueError("val_split + test_split must be in [0, 1)")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_samples)
    n_val = round(num_samples * val_split)
    n_test = round(num_samples * test_split)
    val_idx = permutation[:n_val]
    test_idx = permutation[n_val : n_val + n_test]
    train_idx = permutation[n_val + n_test :]

    def cap(indices: np.ndarray, limit: int | None) -> np.ndarray:
        return indices[:limit] if limit is not None else indices

    return {
        "train": np.sort(cap(train_idx, max_train)),
        "val": np.sort(cap(val_idx, max_val)),
        "test": np.sort(cap(test_idx, max_test)),
    }


def build_datasets(
    cfg,
    *,
    transform_factory: Callable[[bool], Callable[[np.ndarray], object]] | None = None,
) -> DatasetBundle:
    """Build the train/val/test :class:`DatasetBundle` described by ``cfg``.

    Args:
        cfg: A validated :class:`~handwriting_intelligence.config.Config`.
        transform_factory: Optional callable ``train_flag -> transform``. When
            omitted, raw ``uint8`` arrays are returned (useful for tests).

    The validation split always comes from the training data. The test set is the
    official one when available (MNIST); for the local EMNIST corpus, which ships
    only training data, a documented held-out split is carved from the training set
    and ``meta["official_test"]`` is set to ``False``.
    """
    name = cfg.data.name
    root = Path(cfg.data.root)
    seed = cfg.seed

    if name == "mnist":
        arrays = load_mnist(root, download=cfg.data.download)
        train_images = arrays["train_images"]
        train_labels = arrays["train_labels"]
        official_test = True
        source = "idx"
        splits = make_split_indices(
            len(train_labels),
            val_split=cfg.data.val_split,
            test_split=0.0,
            seed=seed,
            max_train=cfg.data.max_train_samples,
            max_val=cfg.data.max_val_samples,
        )
        test_images = arrays["test_images"]
        test_labels = arrays["test_labels"]
        test_idx = _capped_indices(len(test_labels), cfg.data.max_test_samples)
    elif name in {"emnist_balanced", "emnist_bymerge"}:
        corpus = locate_emnist_corpus(root)
        if corpus is None:
            raise DatasetNotFoundError(
                "EMNIST corpus not found. Expected an `emnist.csv` under one of "
                "data/EMNIST/..., data/emnist/.... See docs/data.md for acquisition "
                "options, or use dataset 'mnist'."
            )
        arrays = load_emnist_corpus(
            corpus,
            expected_name=name,
            cache_dir=root / "cache" if cfg.data.cache else None,
            use_cache=cfg.data.cache,
        )
        train_images = arrays["images"]
        train_labels = arrays["labels"]
        # The shipped corpus is the 112800-sample *training* split only. We carve a
        # held-out test set from it and mark the split as non-official.
        official_test = False
        source = "local_corpus"
        splits = make_split_indices(
            len(train_labels),
            val_split=cfg.data.val_split,
            test_split=cfg.data.val_split,
            seed=seed,
            max_train=cfg.data.max_train_samples,
            max_val=cfg.data.max_val_samples,
            max_test=cfg.data.max_test_samples,
        )
        test_images = train_images
        test_labels = train_labels
        test_idx = splits["test"]
    else:
        # EMNIST letters / digits / byclass are not shipped as a local corpus, so we
        # obtain them through torchvision's downloader (large archive; opt-in).
        arrays = load_emnist_torchvision(
            root, _torchvision_split_name(name), download=cfg.data.download
        )
        train_images = arrays["train_images"]
        train_labels = arrays["train_labels"]
        official_test = True
        source = "emnist_idx"
        splits = make_split_indices(
            len(train_labels),
            val_split=cfg.data.val_split,
            test_split=0.0,
            seed=seed,
            max_train=cfg.data.max_train_samples,
            max_val=cfg.data.max_val_samples,
        )
        test_images = arrays["test_images"]
        test_labels = arrays["test_labels"]
        test_idx = _capped_indices(len(test_labels), cfg.data.max_test_samples)

    class_set = build_class_set(name)
    train_transform = transform_factory(True) if transform_factory else None
    eval_transform = transform_factory(False) if transform_factory else None

    bundle = DatasetBundle(
        train=ArrayDataset(
            train_images, train_labels, indices=splits["train"], transform=train_transform
        ),
        val=ArrayDataset(
            train_images, train_labels, indices=splits["val"], transform=eval_transform
        ),
        test=ArrayDataset(test_images, test_labels, indices=test_idx, transform=eval_transform),
        class_set=class_set,
        meta={
            "dataset": name,
            "source": source,
            "official_test": official_test,
            "num_classes": class_set.num_classes,
            "train_size": len(splits["train"]),
            "val_size": len(splits["val"]),
            "test_size": len(test_idx),
            "val_split": cfg.data.val_split,
            "split_seed": seed,
            "root": str(root.resolve()),
        },
    )
    logger.info("datasets ready", extra=dict(bundle.meta))
    return bundle


def _capped_indices(num_samples: int, limit: int | None) -> np.ndarray:
    indices = np.arange(num_samples, dtype=np.int64)
    return indices[:limit] if limit is not None else indices


def _torchvision_split_name(dataset_name: str) -> str:
    """Map our dataset name to torchvision's EMNIST ``split`` argument."""
    return {
        "emnist_letters": "letters",
        "emnist_digits": "digits",
        "emnist_byclass": "byclass",
        "emnist_bymerge": "bymerge",
        "emnist_balanced": "balanced",
    }[dataset_name]
