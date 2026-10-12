"""Shared CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from handwriting_intelligence.config import Config
from handwriting_intelligence.exceptions import ConfigurationError


def add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add options shared by every CLI command."""
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument(
        "--log-format",
        default="text",
        choices=["text", "json"],
        help="Log output format (default: text).",
    )


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply CLI overrides onto a validated config and re-validate.

    Overrides are applied to the serialized dict and re-validated, so an invalid
    combination (e.g. changing the dataset without updating num_classes) is caught
    here rather than mid-training.
    """
    data = cfg.to_dict()
    if getattr(args, "output_dir", None):
        data["experiment"]["output_dir"] = args.output_dir
    if getattr(args, "seed", None) is not None:
        data["seed"] = args.seed
    if getattr(args, "epochs", None) is not None:
        data["training"]["epochs"] = args.epochs
    if getattr(args, "device", None):
        data["training"]["device"] = args.device
    if getattr(args, "max_train_samples", None) is not None:
        data["data"]["max_train_samples"] = args.max_train_samples
    if getattr(args, "max_val_samples", None) is not None:
        data["data"]["max_val_samples"] = args.max_val_samples
    if getattr(args, "max_test_samples", None) is not None:
        data["data"]["max_test_samples"] = args.max_test_samples
    if getattr(args, "no_download", False):
        data["data"]["download"] = False
    # model.num_classes is always re-derived for consistency.
    data["model"]["num_classes"] = None
    return Config.from_dict(data).resolved()


def resolve_resume_path(value: str | None, run_dir: Path) -> Path | None:
    """Resolve ``--resume``: ``"auto"`` picks ``last.pt`` in the run directory."""
    if not value:
        return None
    if value == "auto":
        candidate = run_dir / "last.pt"
        if not candidate.is_file():
            raise ConfigurationError(f"no checkpoint to resume from at {candidate}")
        return candidate
    path = Path(value)
    if not path.is_file():
        raise ConfigurationError(f"resume checkpoint not found: {path}")
    return path
