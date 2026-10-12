"""Reproducibility and provenance helpers.

Every experiment records the random seed, the git revision and the versions of the
libraries that could change results. If a number in the README cannot be traced
back to a run with this metadata, it does not belong in the README.
"""

from __future__ import annotations

import importlib
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TRACKED_PACKAGES = ("torch", "torchvision", "numpy", "scikit-learn", "Pillow")


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch RNGs.

    Args:
        seed: The seed to apply.
        deterministic: When True, request deterministic cuDNN kernels. This can be
            slower and may error for some ops; it is opt-in.

    Returns:
        The applied seed (handy for logging).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
    return seed


def git_commit(short: bool = True) -> str:
    """Return the current git revision, or ``"unknown"`` outside a repository."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    """Return versions of the libraries that influence numerical results."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in _TRACKED_PACKAGES:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:  # pragma: no cover - optional
            versions[name] = "not-installed"
    return versions


def environment_info() -> dict[str, Any]:
    """Return a provenance dictionary persisted with each run."""
    info: dict[str, Any] = {
        "versions": package_versions(),
        "git_commit": git_commit(),
        "platform": sys.platform,
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:  # pragma: no cover
        info["cuda_available"] = False
    return info
