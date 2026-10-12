"""Unit tests for metrics, baselines and prediction collection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from handwriting_intelligence.classes import build_class_set
from handwriting_intelligence.training.metrics import (
    ClassificationMetrics,
    compute_metrics,
    majority_class_accuracy,
    predict_logits,
    topk_accuracy,
)


def test_perfect_predictions() -> None:
    labels = list(build_class_set("mnist").labels)
    y_true = np.arange(10)
    metrics = compute_metrics(y_true, y_true.copy(), labels=labels)
    assert metrics.accuracy == 1.0
    assert metrics.f1_macro == pytest.approx(1.0)
    assert metrics.num_samples == 10


def test_macro_vs_weighted_differ_when_classes_imbalanced() -> None:
    labels = list(build_class_set("mnist").labels)
    # Class 0 dominates and is predicted perfectly; class 9 is always wrong.
    y_true = np.array([0, 0, 0, 0, 9])
    y_pred = np.array([0, 0, 0, 0, 0])
    metrics = compute_metrics(y_true, y_pred, labels=labels)
    assert metrics.f1_weighted > metrics.f1_macro


def test_confusion_matrix_shape() -> None:
    labels = list(build_class_set("mnist").labels)
    metrics = compute_metrics(np.array([0, 1]), np.array([1, 1]), labels=labels)
    assert np.asarray(metrics.confusion_matrix).shape == (10, 10)


def test_to_dict_is_serializable() -> None:
    import json

    labels = list(build_class_set("mnist").labels)
    metrics = compute_metrics(np.array([0, 1]), np.array([0, 1]), labels=labels)
    assert isinstance(metrics, ClassificationMetrics)
    json.dumps(metrics.to_dict())


def test_majority_class_baseline() -> None:
    y_train = np.array([3, 3, 3, 3, 1, 2])
    y_test = np.array([3, 1, 3, 3])
    report = majority_class_accuracy(y_train, y_test)
    assert report["majority_class"] == 3.0
    assert report["accuracy"] == pytest.approx(0.75)


def test_topk_accuracy() -> None:
    logits = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    targets = np.array([1, 2])
    assert topk_accuracy(logits, targets, k=1) == pytest.approx(0.5)
    assert topk_accuracy(logits, targets, k=3) == pytest.approx(1.0)


def test_predict_logits_shapes() -> None:
    class ToyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 3)

    loader = [(torch.zeros(4, 1, 28, 28), torch.tensor([0, 1, 2, 2]))]
    logits, targets = predict_logits(ToyModel(), loader, torch.device("cpu"))
    assert logits.shape == (4, 3)
    assert targets.shape == (4,)
