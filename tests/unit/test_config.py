"""Unit tests for configuration validation and IO."""

from __future__ import annotations

from pathlib import Path

import pytest

from handwriting_intelligence.config import Config, load_config
from handwriting_intelligence.exceptions import ConfigurationError


def test_defaults() -> None:
    cfg = Config.from_dict({})
    assert cfg.data.name == "mnist"
    assert cfg.num_classes == 10
    assert cfg.training.epochs == 5


def test_invalid_dataset_name() -> None:
    with pytest.raises(ConfigurationError, match="unknown dataset"):
        Config.from_dict({"data": {"name": "not_a_dataset"}})


def test_extra_key_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        Config.from_dict({"data": {"nope": 1}})


def test_num_classes_mismatch_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="does not match dataset"):
        Config.from_dict({"data": {"name": "mnist"}, "model": {"num_classes": 5}})


def test_num_classes_matches_dataset_is_accepted() -> None:
    cfg = Config.from_dict({"data": {"name": "emnist_balanced"}, "model": {"num_classes": 47}})
    assert cfg.num_classes == 47


def test_out_of_range_values_rejected() -> None:
    with pytest.raises(ConfigurationError):
        Config.from_dict({"training": {"epochs": 0}})
    with pytest.raises(ConfigurationError):
        Config.from_dict({"data": {"val_split": 1.5}})


def test_resolved_fills_num_classes() -> None:
    cfg = Config.from_dict({"data": {"name": "emnist_balanced"}})
    assert cfg.model.num_classes is None
    assert cfg.resolved().model.num_classes == 47


def test_run_dir() -> None:
    cfg = Config.from_dict({"experiment": {"name": "abc", "output_dir": "runs"}})
    assert cfg.run_dir == Path("runs") / "abc"


def test_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = Config.from_dict({"experiment": {"name": "roundtrip"}})
    path = cfg.dump(tmp_path / "cfg.yaml")
    reloaded = load_config(path)
    assert reloaded.experiment.name == "roundtrip"
    assert reloaded.data.name == cfg.data.name


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="empty"):
        load_config(path)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_repo_configs_are_valid() -> None:
    from handwriting_intelligence.utils.paths import project_root

    for name in ("mnist_cnn.yaml", "mnist_cnn_improved.yaml", "emnist_cnn.yaml"):
        cfg = load_config(project_root() / "configs" / name)
        assert cfg.model.name in {"cnn_baseline", "cnn_improved"}
