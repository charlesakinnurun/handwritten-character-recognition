"""The evaluation module.

Training answers "did the loss go down?". This module answers the questions that
actually decide whether a model is usable: how accurate is it per class, where does
it fail, how fast is it, and how big is it?

All functions take plain arrays or a :class:`~handwriting_intelligence.inference.predictor.Predictor`
plus a loader, so they can be unit-tested on synthetic data without a real dataset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from handwriting_intelligence.training.metrics import (
    ClassificationMetrics,
    compute_metrics,
    majority_class_accuracy,
    predict_logits,
)
from handwriting_intelligence.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.data import DataLoader


logger = get_logger(__name__)


@dataclass(slots=True)
class EvaluationResult:
    """Metrics plus the raw predictions needed for error analysis."""

    metrics: ClassificationMetrics
    y_true: np.ndarray
    y_pred: np.ndarray
    logits: np.ndarray
    inference_ms_per_image: float
    model_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        data = self.metrics.to_dict()
        data["inference_ms_per_image"] = self.inference_ms_per_image
        data["model_size_bytes"] = self.model_size_bytes
        return data


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    labels: list[str],
    device: torch.device,
    model_size_bytes: int = 0,
) -> EvaluationResult:
    """Evaluate ``model`` over ``loader`` and return a full :class:`EvaluationResult`."""
    started = time.perf_counter()
    logits, y_true = predict_logits(model, loader, device)
    elapsed = time.perf_counter() - started
    if logits.size == 0:
        raise ValueError("evaluation loader produced no samples")
    y_pred = logits.argmax(axis=1)
    metrics = compute_metrics(y_true, y_pred, labels=labels, logits=logits)

    seen = loader.batch_size or len(loader.dataset)  # type: ignore[arg-type]
    num_batches = max(1, len(loader))
    approx_images = seen * num_batches
    per_image_ms = (elapsed * 1000.0) / max(1, approx_images)
    return EvaluationResult(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        logits=logits,
        inference_ms_per_image=round(per_image_ms, 4),
        model_size_bytes=model_size_bytes,
    )


def format_metrics_table(metrics: ClassificationMetrics, *, top_n_classes: int = 0) -> str:
    """Render a compact human-readable summary of the metrics."""
    lines = [
        f"samples        : {metrics.num_samples}",
        f"accuracy       : {metrics.accuracy:.4f}",
        f"f1 (macro)     : {metrics.f1_macro:.4f}",
        f"f1 (weighted)  : {metrics.f1_weighted:.4f}",
        f"precision (macro/weighted): {metrics.precision_macro:.4f} / {metrics.precision_weighted:.4f}",
        f"recall    (macro/weighted): {metrics.recall_macro:.4f} / {metrics.recall_weighted:.4f}",
    ]
    if metrics.top5_accuracy is not None:
        lines.append(f"top-5 accuracy : {metrics.top5_accuracy:.4f}")

    if top_n_classes > 0:
        worst = sorted(metrics.per_class.items(), key=lambda kv: kv[1]["f1"])[:top_n_classes]
        lines.append("worst classes by F1:")
        for label, stats in worst:
            lines.append(
                f"    {label:>4}  f1={stats['f1']:.3f}  "
                f"p={stats['precision']:.3f}  r={stats['recall']:.3f}  n={stats['support']}"
            )
    return "\n".join(lines)


def baseline_report(train_labels: np.ndarray, test_labels: np.ndarray) -> dict[str, float]:
    """Return the majority-class baseline accuracy for comparison in reports."""
    return majority_class_accuracy(train_labels, test_labels)


def save_confusion_matrix_png(
    metrics: ClassificationMetrics, path: str | Path, *, title: str = "Confusion matrix"
) -> Path:
    """Render the confusion matrix to a PNG (uses a non-interactive backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(metrics.confusion_matrix)
    labels = metrics.labels
    size = max(6.0, len(labels) * 0.28)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if len(labels) <= 62:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_examples_grid(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    path: str | Path,
    *,
    correct: bool,
    count: int = 16,
    columns: int = 8,
    title: str | None = None,
) -> Path | None:
    """Save a grid of correctly or incorrectly classified examples.

    Args:
        images: ``(N, H, W)`` uint8 array aligned with ``y_true``/``y_pred``.
        correct: If True, select correct predictions; otherwise misclassifications.
        count: Maximum number of examples to draw.
        columns: Grid width.

    Returns:
        The written path, or ``None`` if there are no examples of that kind.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = (y_pred == y_true) if correct else (y_pred != y_true)
    indices = np.flatnonzero(mask)[:count]
    if indices.size == 0:
        logger.info("no examples to plot", extra={"correct": correct})
        return None

    rows = int(np.ceil(indices.size / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.3, rows * 1.5))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, indices, strict=False):
        ax.imshow(images[idx], cmap="gray")
        color = "green" if correct else "red"
        ax.set_title(
            f"t={_safe(y_true[idx])} p={_safe(y_pred[idx])}\n{probabilities[idx]:.2f}",
            fontsize=7,
            color=color,
        )
    fig.suptitle(title or ("correctly classified" if correct else "misclassified"))
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _safe(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def confidence_analysis(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Summarize the confidence distribution of correct vs incorrect predictions.

    This surfaces over-confidence on errors: a well-calibrated model is *less*
    confident when it is wrong.
    """
    confidences = probabilities[np.arange(len(y_pred)), y_pred]
    correct = y_pred == y_true
    result: dict[str, Any] = {
        "mean_confidence_all": float(confidences.mean()),
        "mean_confidence_correct": float(confidences[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": (
            float(confidences[~correct].mean()) if (~correct).any() else None
        ),
        "num_correct": int(correct.sum()),
        "num_incorrect": int((~correct).sum()),
    }
    # Fraction of predictions above common confidence thresholds.
    for threshold in (0.5, 0.9, 0.99):
        result[f"frac_above_{threshold}"] = float((confidences >= threshold).mean())
    return result
