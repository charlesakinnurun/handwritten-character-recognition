"""Integration tests for the FastAPI inference service."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from handwriting_intelligence.api.dependencies import Settings
from handwriting_intelligence.api.main import create_app
from handwriting_intelligence.inference.predictor import load_predictor

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tiny_artifact: Path):
    predictor = load_predictor(tiny_artifact, device="cpu")
    settings = Settings(artifact_path=tiny_artifact, max_batch_size=2, log_format="text")
    app = create_app(settings, predictor=predictor)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["num_classes"] == 10


def test_model_info(client: TestClient) -> None:
    response = client.get("/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["num_classes"] == 10
    assert body["supports_word_recognition"] is False


def test_request_id_header_is_returned(client: TestClient) -> None:
    response = client.get("/health", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"


def test_predict_single(client: TestClient, png_bytes: bytes) -> None:
    response = client.post("/v1/predict", files={"file": ("char.png", png_bytes, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in {str(i) for i in range(10)}
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_k"]) == 5
    assert body["request_id"]


def test_predict_rejects_non_image(client: TestClient) -> None:
    response = client.post(
        "/v1/predict", files={"file": ("broken.png", b"not an image", "image/png")}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_image"


def test_predict_batch_within_limit(client: TestClient, png_bytes: bytes) -> None:
    files = [
        ("files", ("a.png", png_bytes, "image/png")),
        ("files", ("b.png", png_bytes, "image/png")),
    ]
    response = client.post("/v1/predict/batch", files=files)
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_predict_batch_over_limit(client: TestClient, png_bytes: bytes) -> None:
    files = [("files", (f"{i}.png", png_bytes, "image/png")) for i in range(3)]
    response = client.post("/v1/predict/batch", files=files)
    assert response.status_code == 400
    assert "exceeds the limit" in response.json()["detail"]


def test_batch_reports_offending_index(client: TestClient, png_bytes: bytes) -> None:
    files = [
        ("files", ("good.png", png_bytes, "image/png")),
        ("files", ("bad.png", b"nope", "image/png")),
    ]
    response = client.post("/v1/predict/batch", files=files)
    assert response.status_code == 400
    assert "file[1]" in response.json()["detail"]
