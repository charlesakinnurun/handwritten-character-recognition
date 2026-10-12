"""Model registry and shared model utilities.

Architectures are looked up by name so that a configuration string is the only
thing that selects a model. Unknown names fail with a list of valid options rather
than a traceback deep inside ``__init__``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from handwriting_intelligence.exceptions import ModelError

if TYPE_CHECKING:  # pragma: no cover
    from handwriting_intelligence.config import ModelConfig

#: name -> factory ``(num_classes, cfg) -> nn.Module``.
MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator that registers a model factory under ``name``."""

    def decorator(factory: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in MODEL_REGISTRY:
            raise ModelError(f"model '{name}' is already registered")
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def available_models() -> list[str]:
    """Return the sorted names of all registered architectures."""
    _ensure_builtin_models_registered()
    return sorted(MODEL_REGISTRY)


def _ensure_builtin_models_registered() -> None:
    """Import the built-in model modules for their registration side effects."""
    # Imported lazily to avoid a circular import (cnn/crnn import this module).
    from handwriting_intelligence.models import cnn, crnn  # noqa: F401


def build_model(model_cfg: ModelConfig, num_classes: int | None = None) -> nn.Module:
    """Instantiate a model from a :class:`ModelConfig`.

    Args:
        model_cfg: Validated model configuration.
        num_classes: Explicit class count; defaults to ``model_cfg.num_classes``.
            Passing it explicitly lets the caller use the dataset's authoritative
            count.

    Raises:
        ModelError: if the name is unknown or the class count cannot be determined.
    """
    _ensure_builtin_models_registered()
    name = model_cfg.name
    if name not in MODEL_REGISTRY:
        raise ModelError(f"unknown model '{name}'; available: {', '.join(available_models())}")

    resolved_classes = num_classes if num_classes is not None else model_cfg.num_classes
    if resolved_classes is None:
        raise ModelError(
            "num_classes is required: set model.num_classes in the config or pass it "
            "explicitly (it is derived from the dataset during training)."
        )
    model = MODEL_REGISTRY[name](int(resolved_classes), model_cfg)
    if not isinstance(model, nn.Module):  # pragma: no cover - defensive
        raise ModelError(f"model factory for '{name}' did not return an nn.Module")
    return model


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Count model parameters (trainable by default)."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def model_size_bytes(model: nn.Module) -> int:
    """Approximate on-disk size of the model's parameters and buffers, in bytes."""
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total


def describe_model(model: nn.Module) -> dict[str, Any]:
    """Return a JSON-serializable summary used in checkpoints and the model card."""
    return {
        "architecture": type(model).__name__,
        "num_classes": int(getattr(model, "num_classes", 0)),
        "input_channels": int(getattr(model, "input_channels", 1)),
        "input_size": int(getattr(model, "input_size", 0)),
        "parameters": count_parameters(model),
        "size_bytes": model_size_bytes(model),
    }


class CharacterClassifier(nn.Module):
    """Base class for single-character classifiers.

    Concrete models must set ``num_classes``, ``input_channels`` and ``input_size``
    and implement :meth:`forward` returning **raw logits** of shape ``(B, num_classes)``.
    Returning logits (not probabilities) keeps ``CrossEntropyLoss`` correct and lets
    callers apply softmax explicitly.
    """

    num_classes: int
    input_channels: int = 1
    input_size: int = 28

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for a batch of inputs."""
        return torch.softmax(self.forward(x), dim=1)
