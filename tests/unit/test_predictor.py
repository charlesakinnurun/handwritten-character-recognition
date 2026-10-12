"""Unit tests for the predictor abstraction and response formatting."""

from __future__ import annotations

from pathlib import Path

import pytest

from handwriting_intelligence.exceptions import ArtifactError
from handwriting_intelligence.inference.predictor import load_predictor


def test_predict_single(tiny_artifact: Path, png_bytes: bytes) -> None:
    predictor = load_predictor(tiny_artifact, device="cpu")
    prediction = predictor.predict(png_bytes, top_k=5)
    assert 0 <= prediction.class_index < predictor.class_set.num_classes
    assert prediction.label == predictor.class_set.label_of(prediction.class_index)
    assert 0.0 <= prediction.confidence <= 1.0
    assert len(prediction.top_k) == 5
    probs = [alt.probability for alt in prediction.top_k]
    assert probs == sorted(probs, reverse=True)
    assert prediction.model_name == "cnn_baseline"


def test_predict_batch(tiny_artifact: Path, png_bytes: bytes) -> None:
    predictor = load_predictor(tiny_artifact, device="cpu")
    predictions = predictor.predict_batch([png_bytes, png_bytes], top_k=3)
    assert len(predictions) == 2
    assert all(len(p.top_k) == 3 for p in predictions)


def test_predict_batch_empty_raises(tiny_artifact: Path) -> None:
    predictor = load_predictor(tiny_artifact, device="cpu")
    with pytest.raises(ValueError, match="no images"):
        predictor.predict_batch([])


def test_prediction_to_dict(tiny_artifact: Path, png_bytes: bytes) -> None:
    import json

    predictor = load_predictor(tiny_artifact, device="cpu")
    payload = predictor.predict(png_bytes).to_dict()
    json.dumps(payload)
    assert set(payload) >= {"class_index", "label", "confidence", "top_k", "model_name"}


def test_info_reports_no_word_recognition(tiny_artifact: Path) -> None:
    info = load_predictor(tiny_artifact, device="cpu").info()
    assert info["supports_word_recognition"] is False
    assert info["num_classes"] == 10
    assert len(info["class_names"]) == 10


def test_load_missing_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        load_predictor(tmp_path / "nope.pt")
