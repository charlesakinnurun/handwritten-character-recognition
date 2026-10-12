"""Unit tests for inference-time input validation."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from handwriting_intelligence.config import PreprocessConfig
from handwriting_intelligence.exceptions import ImageValidationError
from handwriting_intelligence.inference.preprocessing import (
    ImageMetadata,
    preprocess_display_array,
    preprocess_tensor,
    validate_image_bytes,
)


def _png(size: tuple[int, int] = (28, 28), mode: str = "L") -> bytes:
    array = (np.random.default_rng(0).random((size[1], size[0])) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def test_valid_png_returns_metadata() -> None:
    meta = validate_image_bytes(_png())
    assert isinstance(meta, ImageMetadata)
    assert meta.image_format == "PNG"
    assert meta.width == 28 and meta.height == 28


def test_empty_bytes_rejected() -> None:
    with pytest.raises(ImageValidationError, match="empty request body"):
        validate_image_bytes(b"")


def test_oversized_bytes_rejected() -> None:
    with pytest.raises(ImageValidationError, match="too large"):
        validate_image_bytes(_png(), max_bytes=10)


def test_too_small_dimensions_rejected() -> None:
    with pytest.raises(ImageValidationError, match="too small"):
        validate_image_bytes(_png((2, 2)))


def test_too_large_dimensions_rejected() -> None:
    with pytest.raises(ImageValidationError, match="too large"):
        validate_image_bytes(_png((50, 50)), max_side=32)


def test_non_image_bytes_rejected() -> None:
    with pytest.raises(ImageValidationError, match="could not decode"):
        validate_image_bytes(b"definitely not an image")


def test_unsupported_format_rejected() -> None:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((28, 28), dtype=np.uint8)).save(buffer, format="PPM")
    with pytest.raises(ImageValidationError, match="unsupported image format"):
        validate_image_bytes(buffer.getvalue())


def test_preprocess_tensor_shape() -> None:
    tensor = preprocess_tensor(_png(), PreprocessConfig())
    assert tuple(tensor.shape) == (1, 28, 28)


def test_preprocess_display_array_shape() -> None:
    array = preprocess_display_array(_png((100, 40)), PreprocessConfig(image_size=28))
    assert array.shape == (28, 28)
    assert array.dtype == np.uint8
