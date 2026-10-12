"""Device selection and mixed-precision capability checks.

CUDA availability is *never* assumed. Device choice flows through a single
function so training, evaluation and inference all agree on where tensors live.
"""

from __future__ import annotations

from typing import Literal

import torch

DevicePreference = Literal["auto", "cpu", "cuda", "mps"]


def resolve_device(preference: DevicePreference = "auto") -> torch.device:
    """Resolve a device preference into a concrete :class:`torch.device`.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU. An explicit preference that
    is unavailable raises ``RuntimeError`` rather than silently falling back.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return torch.device("cuda")
    if preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("device='mps' requested but the MPS backend is not available")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_supported(device: torch.device) -> bool:
    """Return True when CUDA automatic mixed precision is worthwhile here.

    AMP only accelerates CUDA. On CPU it is either unsupported or a pessimisation,
    so it is deliberately disabled everywhere else.
    """
    return device.type == "cuda" and torch.cuda.is_available()
