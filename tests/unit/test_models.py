"""Unit tests for model construction, shapes and the CRNN decoder."""

from __future__ import annotations

import pytest
import torch

from handwriting_intelligence.config import ModelConfig
from handwriting_intelligence.exceptions import ModelError
from handwriting_intelligence.models import (
    CNNBaseline,
    CNNImproved,
    available_models,
    build_model,
    count_parameters,
    describe_model,
)
from handwriting_intelligence.models.crnn import CRNN, build_crnn


def test_registry_contains_expected_models() -> None:
    assert set(available_models()) == {"cnn_baseline", "cnn_improved"}


def test_baseline_output_shape() -> None:
    model = CNNBaseline(10, ModelConfig(name="cnn_baseline", channels=[8, 16], hidden_dim=16))
    out = model(torch.randn(4, 1, 28, 28))
    assert tuple(out.shape) == (4, 10)
    assert model.num_classes == 10


def test_improved_output_shape() -> None:
    model = CNNImproved(47, ModelConfig(name="cnn_improved", channels=[8, 16, 32], hidden_dim=16))
    out = model(torch.randn(2, 1, 28, 28))
    assert tuple(out.shape) == (2, 47)


def test_models_support_non_square_input() -> None:
    model = CNNImproved(3, ModelConfig(name="cnn_improved", channels=[8], hidden_dim=8))
    assert tuple(model(torch.randn(1, 1, 28, 32)).shape) == (1, 3)


def test_build_model_requires_num_classes() -> None:
    with pytest.raises(ModelError, match="num_classes is required"):
        build_model(ModelConfig(name="cnn_baseline", channels=[8]))


def test_build_model_unknown_name() -> None:
    with pytest.raises(ModelError, match="unknown model"):
        build_model(_stub_unknown(), 3)


def _stub_unknown() -> ModelConfig:
    # ``model_construct`` bypasses validation so we can exercise the unknown-name path.
    return ModelConfig.model_construct(name="does_not_exist", num_classes=3, channels=[8])


def test_describe_and_count() -> None:
    model = CNNBaseline(10, ModelConfig(name="cnn_baseline", channels=[8], hidden_dim=8))
    info = describe_model(model)
    assert info["architecture"] == "CNNBaseline"
    assert info["parameters"] == count_parameters(model)
    assert info["size_bytes"] > 0


def test_predict_proba_sums_to_one() -> None:
    model = CNNBaseline(10, ModelConfig(name="cnn_baseline", channels=[8], hidden_dim=8))
    proba = model.predict_proba(torch.randn(3, 1, 28, 28))
    assert torch.allclose(proba.sum(dim=1), torch.ones(3), atol=1e-5)


def test_crnn_output_shape_and_blank() -> None:
    cfg = ModelConfig(name="crnn", channels=[8, 16], rnn_hidden=16, rnn_layers=2, dropout=0.1)
    model = build_crnn(num_classes=5, cfg=cfg, input_height=32)
    out = model(torch.randn(3, 1, 32, 64))
    assert out.shape[1] == 3  # batch
    assert out.shape[2] == 6  # 5 characters + blank
    # log-softmax over classes sums to exp(0)=1
    assert torch.allclose(out.exp().sum(dim=2), torch.ones(out.shape[0], out.shape[1]), atol=1e-4)


def test_crnn_greedy_decode_collapses_repeats_and_blanks() -> None:
    # frames: 1,1,0(blank),2,2,3  ->  [1,2,3]
    tokens = [1, 1, 0, 2, 2, 3]
    log_probs = torch.full((len(tokens), 1, 4), -10.0)
    for t, token in enumerate(tokens):
        log_probs[t, 0, token] = 0.0
    decoded = CRNN.decode_greedy(log_probs)
    assert decoded == [[1, 2, 3]]
