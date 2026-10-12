"""Model package: registry, CNN classifiers and the CRNN sequence extension."""

from __future__ import annotations

from handwriting_intelligence.models.base import (
    CharacterClassifier,
    available_models,
    build_model,
    count_parameters,
    describe_model,
    model_size_bytes,
    register_model,
)
from handwriting_intelligence.models.cnn import CNNBaseline, CNNImproved

__all__ = [
    "CNNBaseline",
    "CNNImproved",
    "CharacterClassifier",
    "available_models",
    "build_model",
    "count_parameters",
    "describe_model",
    "model_size_bytes",
    "register_model",
]
