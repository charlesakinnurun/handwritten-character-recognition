"""Unit tests for the preprocessing transforms."""

from __future__ import annotations

import numpy as np
import pytest

from handwriting_intelligence.config import PreprocessConfig, TrainingConfig
from handwriting_intelligence.data.transforms import (
    array_to_tensor,
    build_dataset_transform,
    build_inference_transform,
    center_on_canvas,
    crop_to_content,
    load_image,
    normalize_polarity,
    preprocess_to_array,
    threshold_array,
)
from handwriting_intelligence.exceptions import ImageValidationError


def _cfg(**kwargs) -> PreprocessConfig:
    return PreprocessConfig(**kwargs)


def test_normalize_polarity_auto_inverts_light_background() -> None:
    light_bg = np.full((28, 28), 255, dtype=np.uint8)
    light_bg[10:18, 10:18] = 0
    result = normalize_polarity(light_bg, "auto")
    assert float(np.median(result)) < 128  # background is now dark


def test_normalize_polarity_auto_keeps_dark_background() -> None:
    dark_bg = np.zeros((28, 28), dtype=np.uint8)
    dark_bg[10:18, 10:18] = 255
    result = normalize_polarity(dark_bg, "auto")
    assert float(np.median(result)) == 0.0


def test_normalize_polarity_policies() -> None:
    arr = np.full((4, 4), 10, dtype=np.uint8)
    assert (normalize_polarity(arr, "never") == arr).all()
    assert (normalize_polarity(arr, "always") == 255 - arr).all()
    with pytest.raises(ImageValidationError):
        normalize_polarity(arr, "sideways")


def test_crop_to_content_bbox() -> None:
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[5:10, 7:12] = 200
    cropped = crop_to_content(arr)
    assert cropped.shape == (5, 5)
    assert (cropped == 200).all()


def test_crop_empty_returns_input() -> None:
    arr = np.zeros((28, 28), dtype=np.uint8)
    assert crop_to_content(arr).shape == (28, 28)


def test_threshold_array() -> None:
    arr = np.array([[10, 200], [128, 127]], dtype=np.uint8)
    out = threshold_array(arr, 128)
    assert out.tolist() == [[0, 255], [255, 0]]


def test_center_on_canvas_shape_and_margins() -> None:
    from PIL import Image

    glyph = Image.fromarray(np.full((28, 28), 255, dtype=np.uint8))
    canvas = center_on_canvas(glyph, canvas=28, content=20)
    assert canvas.shape == (28, 28)
    # content box is 20 -> there must be some empty padding rows/cols
    assert canvas[0].max() == 0 or canvas[:, 0].max() == 0


def test_center_on_canvas_rejects_bad_content() -> None:
    from PIL import Image

    glyph = Image.fromarray(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ImageValidationError):
        center_on_canvas(glyph, canvas=8, content=20)


@pytest.mark.parametrize("size", [(28, 28), (64, 200), (100, 40)])
def test_preprocess_to_array_always_returns_canvas(size: tuple[int, int]) -> None:
    rng = np.random.default_rng(0)
    image = (rng.random(size) * 255).astype(np.uint8)
    out = preprocess_to_array(image, _cfg(image_size=28))
    assert out.shape == (28, 28)
    assert out.dtype == np.uint8


def test_array_to_tensor_normalization() -> None:
    arr = np.zeros((28, 28), dtype=np.uint8)
    tensor = array_to_tensor(arr, normalize=True, mean=0.1307, std=0.3081)
    assert tensor.shape == (1, 28, 28)
    assert tensor.dtype.is_floating_point
    # A zero pixel normalizes to -mean/std
    assert tensor.flatten()[0].item() == pytest.approx(-0.1307 / 0.3081, rel=1e-4)


def test_build_dataset_transform_shapes() -> None:
    transform = build_dataset_transform(
        _cfg(), TrainingConfig(augment=False, device="cpu"), train=False
    )
    arr = np.zeros((28, 28), dtype=np.uint8)
    out = transform(arr)
    assert tuple(out.shape) == (1, 28, 28)


def test_build_dataset_transform_augmentation_runs() -> None:
    transform = build_dataset_transform(
        _cfg(), TrainingConfig(augment=True, device="cpu"), train=True
    )
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[10:18, 10:18] = 255
    out = transform(arr)
    assert tuple(out.shape) == (1, 28, 28)


def test_build_inference_transform_from_bytes(png_bytes: bytes) -> None:
    transform = build_inference_transform(_cfg())
    tensor = transform(png_bytes)
    assert tuple(tensor.shape) == (1, 28, 28)


def test_load_image_from_array_and_bytes(png_bytes: bytes) -> None:
    assert load_image(png_bytes).size == (28, 28)
    assert load_image(np.zeros((10, 10), dtype=np.uint8)).mode == "L"


def test_load_image_invalid_raises() -> None:
    with pytest.raises(ImageValidationError):
        load_image(b"this is not an image")


def test_load_image_wrong_array_shape_raises() -> None:
    with pytest.raises(ImageValidationError):
        load_image(np.zeros((3, 3, 3, 3), dtype=np.uint8))
