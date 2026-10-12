"""Evaluation metrics for multiclass character classification.

All metrics are computed with :mod:`sklearn.metrics` using explicit averaging
strategies, because a bare "accuracy" hides exactly the failure modes that matter
for character recognition:

* **macro** averaging weights every class equally, so a model that is excellent on
  frequent digits but poor on rare letters is penalised. This is the headline
  number when classes matter equally.
* **weighted** averaging weights classes by support and approximates overall
  sample-level performance.

Both are reported so readers can see the gap. ``zero_division=0`` keeps undefined
metrics (e.g. a class never predicted) from raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def topk_accuracy(logits: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    """Return top-k accuracy for a batch of logits.

    Args:
        logits: Array of shape ``(N, C)``.
        targets: Integer targets of shape ``(N,)``.
        k: Number of top predictions to consider.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected 2D logits, got shape {logits.shape}")
    if targets.size == 0:
        return 0.0
    k = min(k, logits.shape[1])
    top_k = np.argsort(-logits, axis=1)[:, :k]
    hits = (top_k == targets[:, None]).any(axis=1)
    return float(np.mean(hits))


@dataclass(slots=True)
class ClassificationMetrics:
    """Container for a full evaluation of one dataset split."""

    num_samples: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    labels: list[str]
    top5_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "top5_accuracy": self.top5_accuracy,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[str],
    logits: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute the full metric set for predictions.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        labels: Ordered display labels; index ``i`` is the name of class ``i``.
        logits: Optional ``(N, C)`` logits, enabling top-k accuracy.

    Returns:
        A populated :class:`ClassificationMetrics`.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    class_indices = list(range(len(labels)))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_indices, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_indices, average="weighted", zero_division=0
    )
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_indices, average=None, zero_division=0
    )
    per_class = {
        labels[i]: {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f_cls[i]),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    }

    cm = confusion_matrix(y_true, y_pred, labels=class_indices)

    top5 = None
    if logits is not None and logits.shape[1] >= 5:
        top5 = topk_accuracy(logits, y_true, k=5)

    return ClassificationMetrics(
        num_samples=int(y_true.size),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_w),
        recall_weighted=float(recall_w),
        f1_weighted=float(f1_w),
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        labels=list(labels),
        top5_accuracy=top5,
    )


def majority_class_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Accuracy of a trivial "always predict the most frequent training class" model.

    This is the honest lower bound any real model must beat.
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if y_train.size == 0 or y_test.size == 0:
        return {"majority_class": -1.0, "accuracy": 0.0}
    counts = np.bincount(y_train)
    majority = int(np.argmax(counts))
    return {"majority_class": float(majority), "accuracy": float(np.mean(y_test == majority))}


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a model over a loader and return ``(logits, targets)`` as NumPy arrays.

    The model is switched to eval mode; the caller is responsible for switching it
    back if training continues.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(np.asarray(targets))
    if not all_logits:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)
