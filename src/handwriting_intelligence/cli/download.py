"""``hi-download-data`` - download and verify the datasets a config requires.

This command never silently substitutes a different dataset: if a requested dataset
cannot be obtained it fails with the reason and a pointer to docs/data.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from handwriting_intelligence.cli._common import add_common_options
from handwriting_intelligence.config import Config
from handwriting_intelligence.data.datasets import (
    load_emnist_corpus,
    load_emnist_torchvision,
    load_mnist,
    locate_emnist_corpus,
    locate_mnist,
)
from handwriting_intelligence.exceptions import DatasetNotFoundError
from handwriting_intelligence.utils.logging import configure_logging, get_logger

logger = get_logger("hi.download")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hi-download-data",
        description="Download and verify the dataset described by a config.",
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    add_common_options(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_format)
    cfg = Config.load(args.config)
    root = Path(cfg.data.root)
    name = cfg.data.name

    if name == "mnist":
        existing = locate_mnist(root)
        print(f"MNIST: {'found locally' if existing else 'downloading'}")
        arrays = load_mnist(root, download=cfg.data.download)
        print(
            f"  train={arrays['train_images'].shape} "
            f"test={arrays['test_images'].shape} "
            f"labels={sorted(set(arrays['train_labels'].tolist()))}"
        )
    elif name in {"emnist_balanced", "emnist_bymerge"}:
        corpus = locate_emnist_corpus(root)
        if corpus is None:
            raise DatasetNotFoundError(
                f"no local EMNIST corpus under {root.resolve()}. "
                "Provide an EMNIST JPEG corpus with an emnist.csv manifest "
                "(see docs/data.md)."
            )
        arrays = load_emnist_corpus(
            corpus,
            expected_name=name,
            cache_dir=root / "cache" if cfg.data.cache else None,
            use_cache=cfg.data.cache,
        )
        print(
            f"EMNIST ({name}): corpus={corpus} images={arrays['images'].shape} "
            f"classes={int(arrays['labels'].max()) + 1}"
        )
    else:
        print(f"EMNIST ({name}): downloading via torchvision (large archive)...")
        arrays = load_emnist_torchvision(root, _split(name), download=cfg.data.download)
        print(f"  train={arrays['train_images'].shape} test={arrays['test_images'].shape}")

    print("Dataset ready.")
    return 0


def _split(name: str) -> str:
    return {
        "emnist_letters": "letters",
        "emnist_digits": "digits",
        "emnist_byclass": "byclass",
    }[name]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
