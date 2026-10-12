"""Integration test: a short end-to-end training run on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from handwriting_intelligence.data import build_dataloaders
from handwriting_intelligence.models import build_model
from handwriting_intelligence.training.checkpoints import load_checkpoint, model_from_checkpoint
from handwriting_intelligence.training.trainer import Trainer
from tests.conftest import make_config

pytestmark = pytest.mark.integration


def _train(data_dir: Path, run_dir: Path, epochs: int = 2):
    cfg = make_config(data_dir, training={"epochs": epochs, "log_every": 0})
    loaders = build_dataloaders(cfg)
    model = build_model(cfg.model, loaders.class_set.num_classes)
    trainer = Trainer(model, cfg, loaders.class_set, output_dir=run_dir)
    result = trainer.fit(loaders.train, loaders.val)
    return cfg, loaders, trainer, result


def test_short_training_run(tmp_path: Path, mnist_like_dir: Path) -> None:
    _, _, trainer, result = _train(mnist_like_dir, tmp_path / "run", epochs=2)
    assert len(result.history) == 2
    assert result.best_checkpoint.is_file()
    assert result.last_checkpoint.is_file()
    assert (trainer.output_dir / "metrics.jsonl").is_file()
    lines = (trainer.output_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert {"epoch", "train_loss", "val_loss", "val_accuracy"} <= set(record)


def test_checkpoint_can_be_reloaded_and_resumed(tmp_path: Path, mnist_like_dir: Path) -> None:
    cfg, loaders, _, result = _train(mnist_like_dir, tmp_path / "run", epochs=2)
    payload = load_checkpoint(result.last_checkpoint)
    assert payload["epoch"] == 1
    assert model_from_checkpoint(payload) is not None

    new_model = build_model(cfg.model, loaders.class_set.num_classes)
    resumed = Trainer(new_model, cfg, loaders.class_set, output_dir=tmp_path / "run")
    resumed.resume(result.last_checkpoint)
    assert resumed.start_epoch == 2


def test_training_reduces_loss_on_fixture(tmp_path: Path, mnist_like_dir: Path) -> None:
    _, _, _, result = _train(mnist_like_dir, tmp_path / "run", epochs=3)
    assert result.history[-1]["train_loss"] <= result.history[0]["train_loss"]
