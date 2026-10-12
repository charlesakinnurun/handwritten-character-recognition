"""Training package: trainer, checkpoints and metrics."""

from __future__ import annotations

from handwriting_intelligence.training.checkpoints import (
    load_checkpoint,
    model_from_checkpoint,
    save_artifact,
    save_checkpoint,
)
from handwriting_intelligence.training.metrics import (
    ClassificationMetrics,
    compute_metrics,
    majority_class_accuracy,
    predict_logits,
)
from handwriting_intelligence.training.trainer import Trainer, TrainingResult

__all__ = [
    "ClassificationMetrics",
    "Trainer",
    "TrainingResult",
    "compute_metrics",
    "load_checkpoint",
    "majority_class_accuracy",
    "model_from_checkpoint",
    "predict_logits",
    "save_artifact",
    "save_checkpoint",
]
