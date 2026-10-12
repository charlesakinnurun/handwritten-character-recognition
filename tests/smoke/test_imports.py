"""Smoke tests: the package imports and public entry points exist."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.smoke

MODULES = [
    "handwriting_intelligence",
    "handwriting_intelligence.classes",
    "handwriting_intelligence.config",
    "handwriting_intelligence.evaluation",
    "handwriting_intelligence.data.datasets",
    "handwriting_intelligence.data.transforms",
    "handwriting_intelligence.data.datamodule",
    "handwriting_intelligence.models.base",
    "handwriting_intelligence.models.cnn",
    "handwriting_intelligence.models.crnn",
    "handwriting_intelligence.training.trainer",
    "handwriting_intelligence.training.checkpoints",
    "handwriting_intelligence.training.metrics",
    "handwriting_intelligence.inference.predictor",
    "handwriting_intelligence.inference.preprocessing",
    "handwriting_intelligence.api.main",
    "handwriting_intelligence.api.schemas",
    "handwriting_intelligence.api.dependencies",
    "handwriting_intelligence.utils.logging",
    "handwriting_intelligence.utils.reproducibility",
    "handwriting_intelligence.utils.device",
    "handwriting_intelligence.utils.paths",
    "handwriting_intelligence.cli.train",
    "handwriting_intelligence.cli.evaluate",
    "handwriting_intelligence.cli.predict",
    "handwriting_intelligence.cli.download",
    "handwriting_intelligence.cli.serve",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_version_exposed() -> None:
    import handwriting_intelligence

    assert handwriting_intelligence.__version__


def test_api_app_is_created_at_import() -> None:
    from handwriting_intelligence.api.main import app

    assert app is not None
