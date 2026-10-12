"""Unit tests for checkpoint and artifact serialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from handwriting_intelligence.classes import build_class_set
from handwriting_intelligence.config import Config
from handwriting_intelligence.exceptions import ArtifactError, CheckpointError
from handwriting_intelligence.models import build_model
from handwriting_intelligence.training.checkpoints import (
    load_artifact,
    load_checkpoint,
    model_from_checkpoint,
    save_artifact,
    save_checkpoint,
)


def _config(num_classes: int = 10) -> Config:
    return Config.from_dict(
        {
            "data": {"name": "mnist", "download": False},
            "model": {
                "name": "cnn_baseline",
                "channels": [8],
                "hidden_dim": 8,
                "num_classes": None,
            },
        }
    ).resolved()


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = _config()
    class_set = build_class_set("mnist")
    model = build_model(cfg.model, class_set.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    path = save_checkpoint(
        tmp_path / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=3,
        global_step=42,
        best_metric=0.123,
        monitor="val_loss",
        config=cfg,
        class_set=class_set,
        history=[{"epoch": 0.0, "val_loss": 1.0}],
        environment={"test": True},
    )
    payload = load_checkpoint(path)
    assert payload["epoch"] == 3
    assert payload["global_step"] == 42
    assert payload["model_name"] == "cnn_baseline"
    restored = model_from_checkpoint(payload)
    for a, b in zip(model.state_dict().values(), restored.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_checkpoint_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")


def test_load_artifact_rejects_checkpoint(tmp_path: Path) -> None:
    cfg = _config()
    class_set = build_class_set("mnist")
    model = build_model(cfg.model, class_set.num_classes)
    ckpt = save_checkpoint(
        tmp_path / "last.pt",
        model=model,
        optimizer=None,
        epoch=0,
        global_step=0,
        best_metric=0.0,
        monitor="val_loss",
        config=cfg,
        class_set=class_set,
        history=[],
        environment={},
    )
    with pytest.raises(ArtifactError, match="not a 'artifact'"):
        load_artifact(ckpt)


def test_artifact_roundtrip_and_inference_equivalence(tmp_path: Path) -> None:
    cfg = _config()
    class_set = build_class_set("mnist")
    torch.manual_seed(0)
    model = build_model(cfg.model, class_set.num_classes)
    path = save_artifact(
        tmp_path / "artifact.pt",
        model=model,
        class_set=class_set,
        config=cfg,
        environment={"test": True},
        training_summary={"best_metric": 0.5},
    )
    payload = load_artifact(path)
    assert payload["class_set"]["name"] == "mnist"
    restored = model_from_checkpoint(
        {
            "model_config": payload["model_config"],
            "model_state": payload["model_state"],
            "class_set": payload["class_set"],
        }
    )
    x = torch.randn(2, 1, 28, 28)
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), restored(x), atol=1e-6)


def test_artifact_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not found"):
        load_artifact(tmp_path / "missing.pt")


def test_future_format_version_rejected(tmp_path: Path) -> None:
    cfg = _config()
    class_set = build_class_set("mnist")
    model = build_model(cfg.model, class_set.num_classes)
    path = save_artifact(
        tmp_path / "artifact.pt", model=model, class_set=class_set, config=cfg, environment={}
    )
    payload = torch.load(path, weights_only=False)
    payload["format_version"] = 999
    torch.save(payload, path)
    with pytest.raises(ArtifactError, match="newer than this build"):
        load_artifact(path)


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    cfg = _config()
    class_set = build_class_set("mnist")
    model = build_model(cfg.model, class_set.num_classes)
    save_artifact(
        tmp_path / "artifact.pt", model=model, class_set=class_set, config=cfg, environment={}
    )
    assert not (tmp_path / "artifact.pt.tmp").exists()
