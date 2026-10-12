"""Unit tests for the evaluation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from handwriting_intelligence.classes import build_class_set
from handwriting_intelligence.config import Config
from handwriting_intelligence.data.datasets import ArrayDataset
from handwriting_intelligence.data.transforms import build_dataset_transform
from handwriting_intelligence.evaluation import (
    confidence_analysis,
    evaluate_model,
    format_metrics_table,
    save_confusion_matrix_png,
    save_examples_grid,
)
from handwriting_intelligence.models import build_model
from tests.conftest import make_character_images


def _loader(batch_size: int = 16) -> DataLoader:
    images, labels = make_character_images(48, seed=5)
    cfg = Config.from_dict({}).resolved()
    transform = build_dataset_transform(cfg.preprocessing, cfg.training, train=False)
    dataset = ArrayDataset(images, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _model() -> torch.nn.Module:
    cfg = Config.from_dict({"model": {"channels": [8], "hidden_dim": 8}}).resolved()
    return build_model(cfg.model, build_class_set("mnist").num_classes)


def test_evaluate_model_returns_metrics() -> None:
    result = evaluate_model(
        _model(),
        _loader(),
        labels=list(build_class_set("mnist").labels),
        device=torch.device("cpu"),
        model_size_bytes=1234,
    )
    assert result.metrics.num_samples == 48
    assert 0.0 <= result.metrics.accuracy <= 1.0
    assert result.model_size_bytes == 1234
    assert result.inference_ms_per_image >= 0.0


def test_format_metrics_table_mentions_accuracy() -> None:
    result = evaluate_model(
        _model(),
        _loader(),
        labels=list(build_class_set("mnist").labels),
        device=torch.device("cpu"),
    )
    table = format_metrics_table(result.metrics, top_n_classes=3)
    assert "accuracy" in table
    assert "worst classes" in table


def test_save_confusion_matrix_png(tmp_path: Path) -> None:
    result = evaluate_model(
        _model(),
        _loader(),
        labels=list(build_class_set("mnist").labels),
        device=torch.device("cpu"),
    )
    out = save_confusion_matrix_png(result.metrics, tmp_path / "cm.png")
    assert out.is_file() and out.stat().st_size > 0


def test_save_examples_grids(tmp_path: Path) -> None:
    images, _ = make_character_images(32, seed=9)
    y_true = np.arange(32) % 10
    y_pred = y_true.copy()
    y_pred[0] = (y_pred[0] + 1) % 10  # force one error
    probs = np.full(32, 0.9)
    correct = save_examples_grid(images, y_true, y_pred, probs, tmp_path / "ok.png", correct=True)
    wrong = save_examples_grid(images, y_true, y_pred, probs, tmp_path / "bad.png", correct=False)
    assert correct is not None and correct.is_file()
    assert wrong is not None and wrong.is_file()


def test_confidence_analysis() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 0])
    probs = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.4, 0.35, 0.25]])
    report = confidence_analysis(y_true, y_pred, probs)
    assert report["num_correct"] == 2
    assert report["num_incorrect"] == 1
    assert report["mean_confidence_correct"] > report["mean_confidence_incorrect"]
