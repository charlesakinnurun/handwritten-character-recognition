"""Smoke tests: command-line entry points parse and run."""

from __future__ import annotations

from pathlib import Path

import pytest

from handwriting_intelligence.cli import download, evaluate, predict, serve, train

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "module",
    [download, train, evaluate, predict, serve],
)
def test_cli_help_exits_zero(module) -> None:
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0


def test_download_cli_runs_on_fixture(tmp_path: Path, mnist_like_dir: Path) -> None:
    from tests.conftest import make_config

    cfg = make_config(mnist_like_dir)
    config_path = cfg.dump(tmp_path / "cfg.yaml")
    assert download.main(["--config", str(config_path)]) == 0


def test_train_cli_help_mentions_config() -> None:
    parser = train.build_parser()
    args = parser.parse_args(["--config", "configs/mnist_cnn.yaml"])
    assert args.config.endswith("mnist_cnn.yaml")
