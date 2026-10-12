"""Smoke test: the API starts and degrades gracefully without a trained model."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from handwriting_intelligence.api.dependencies import Settings
from handwriting_intelligence.api.main import _iter_route_paths, create_app

pytestmark = pytest.mark.smoke


def test_app_starts_without_artifact(tmp_path: Path) -> None:
    settings = Settings(artifact_path=tmp_path / "missing.pt", log_format="text")
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200

        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["ready"] is False
        assert "not found" in ready.json()["detail"].lower()

        info = client.get("/v1/model/info")
        assert info.status_code == 503
        assert info.json()["error"] == "model_unavailable"


def test_expected_routes_are_registered() -> None:
    app = create_app(Settings(log_format="text"))
    paths = set(_iter_route_paths(app))
    assert {
        "/health",
        "/ready",
        "/v1/model/info",
        "/v1/predict",
        "/v1/predict/batch",
    } <= paths
