"""Image transforms and preprocessing.

There are deliberately **two** transform pipelines, and they share the same
normalization constants:

1. :func:`build_dataset_transform` - used for MNIST/EMNIST samples that are *already*
   centred 28x28 grayscale characters. It only converts to a tensor, optionally
   augments, and normalizes. It must not crop or resize (that would destroy the
   dataset's canonical framing).
2. :func:`build_inference_transform` - used for arbitrary user uploads, which have
   unknown size, polarity, framing and stroke weight. It normalizes polarity,
   removes noise, crops to the stroke bounding box, resizes with a fixed
   aspect-ratio-preserving margin and centres the glyph - mirroring the canonical
   MNIST preprocessing so that train and serve distributions match.

Every step is a small pure function so each can be unit-tested in isolation.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

from handwriting_intelligence.exceptions import ImageValidationError

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from handwriting_intelligence.config import PreprocessConfig, TrainingConfig

#: Any of the image sources accepted by :func:`load_image`.
ImageSource = str | Path | bytes | np.ndarray | Image.Image


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------
def load_image(source: ImageSource) -> Image.Image:
    """Load ``source`` as a single-channel (``"L"``) PIL image.

    Accepts a path, raw bytes, a NumPy array or a PIL image. Bytes and paths are
    fully decoded, so a corrupt upload raises here rather than producing garbage.

    Raises:
        ImageValidationError: if the source cannot be decoded as an image.
    """
    if isinstance(source, Image.Image):
        return source.convert("L")
    if isinstance(source, np.ndarray):
        return _array_to_pil(source)
    try:
        image = Image.open(io.BytesIO(source)) if isinstance(source, bytes) else Image.open(source)
        image.load()  # force full decode; truncated files fail here
        return image.convert("L")
    except ImageValidationError:
        raise
    except Exception as exc:  # PIL raises a zoo of exception types
        raise ImageValidationError(f"could not decode image: {exc}") from exc


def _array_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a NumPy array to a grayscale PIL image."""
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = arr[..., :3].mean(axis=2)
    if arr.ndim != 2:
        raise ImageValidationError(f"expected a 2D or HxWxC array, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


# ---------------------------------------------------------------------------
# Preprocessing steps (pure functions on uint8 arrays)
# ---------------------------------------------------------------------------
def normalize_polarity(array: np.ndarray, policy: str = "auto") -> np.ndarray:
    """Ensure a light glyph on a dark background (the MNIST convention).

    ``policy="auto"`` estimates the background via the image median: a light
    background (median > 127) is inverted. This is a heuristic and is documented
    as such; for images with a huge black foreground it can mis-fire.
    """
    if policy == "never":
        return array
    if policy == "always":
        return 255 - array
    if policy != "auto":
        raise ImageValidationError(f"unknown invert policy: {policy!r}")
    return 255 - array if float(np.median(array)) > 127.0 else array


def denoise_array(array: np.ndarray) -> np.ndarray:
    """Apply a light median filter to remove salt-and-pepper speckle."""
    return np.asarray(Image.fromarray(array).filter(ImageFilter.MedianFilter(size=3)))


def threshold_array(array: np.ndarray, threshold: int) -> np.ndarray:
    """Binarize: pixels above ``threshold`` become 255, the rest 0."""
    return np.where(array >= threshold, 255, 0).astype(np.uint8)


def crop_to_content(array: np.ndarray, pad: int = 0) -> np.ndarray:
    """Crop to the bounding box of non-background pixels, then pad.

    Returns the input unchanged when the image is empty (all background), so a
    blank upload degrades gracefully instead of raising.
    """
    mask = array > 0
    if not mask.any():
        return array
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    cropped = array[r0:r1, c0:c1]
    if pad > 0:
        cropped = np.pad(cropped, pad, mode="constant", constant_values=0)
    return cropped


def _resize_preserving_aspect(pil: Image.Image, max_side: int) -> Image.Image:
    """Resize so the longest side equals ``max_side``, preserving aspect ratio."""
    width, height = pil.size
    if width == 0 or height == 0:
        raise ImageValidationError("image has zero width or height")
    scale = max_side / float(max(width, height))
    new_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return pil.resize(new_size, resample=Image.Resampling.LANCZOS)


def center_on_canvas(pil: Image.Image, canvas: int, content: int) -> np.ndarray:
    """Place ``pil`` centred on a ``canvas`` x ``canvas`` black canvas.

    The glyph is first fitted into a ``content`` x ``content`` box (aspect ratio
    preserved), then pasted so its bounding box is centred on the canvas. This
    mirrors the standard MNIST preprocessing (20x20 glyph centred in 28x28).
    """
    if content > canvas:
        raise ImageValidationError("content box cannot exceed canvas")
    fitted = np.asarray(_resize_preserving_aspect(pil, max_side=content))
    canvas_array = np.zeros((canvas, canvas), dtype=np.uint8)
    height, width = fitted.shape
    top = (canvas - height) // 2
    left = (canvas - width) // 2
    canvas_array[top : top + height, left : left + width] = fitted
    return canvas_array


def preprocess_to_array(source: ImageSource, cfg: PreprocessConfig) -> np.ndarray:
    """Run the full inference preprocessing chain and return a uint8 canvas.

    Steps: decode -> polarity -> optional denoise -> optional threshold ->
    crop-to-content (when enabled or required by a non-square/oversized input) ->
    aspect-preserving resize -> centre on a ``cfg.image_size`` canvas.

    Returns:
        A ``(image_size, image_size)`` ``uint8`` array. The result is also what the
        demo displays, so users can see exactly what the model sees.
    """
    array = np.asarray(load_image(source))
    array = normalize_polarity(array, cfg.invert)
    if cfg.denoise:
        array = denoise_array(array)
    if cfg.threshold is not None:
        array = threshold_array(array, cfg.threshold)

    # Always crop to content for arbitrary uploads: the raw framing is unknown and
    # padding/border artefacts otherwise dominate the 28x28 downsample.
    array = crop_to_content(array)

    canvas = cfg.image_size
    content = max(1, round(canvas * 5 / 7))  # 20 for a 28 canvas
    canvas_array = center_on_canvas(_array_to_pil(array), canvas=canvas, content=content)
    return canvas_array


def array_to_tensor(
    array: np.ndarray,
    *,
    normalize: bool = True,
    mean: float = 0.1307,
    std: float = 0.3081,
    add_channel: bool = True,
) -> torch.Tensor:
    """Convert a uint8 HxW array to a float tensor, optionally normalized."""
    import torch

    tensor = torch.tensor(np.ascontiguousarray(array), dtype=torch.float32).div_(255.0)
    if add_channel:
        tensor = tensor.unsqueeze(0)
    if normalize:
        tensor = (tensor - mean) / std
    return tensor


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------
def _augmentation_transform(cfg: PreprocessConfig, training_cfg: TrainingConfig) -> Callable:
    """Build the (optional) augmentation pipeline applied to training tensors."""
    import torchvision.transforms.v2 as T

    if not training_cfg.augment:
        return T.Identity()
    translate = (training_cfg.augment_translate, training_cfg.augment_translate)
    scale = (1.0 - training_cfg.augment_scale, 1.0 + training_cfg.augment_scale)
    return T.Compose(
        [
            T.RandomAffine(
                degrees=training_cfg.augment_rotation,
                translate=translate,
                scale=scale,
                fill=0.0,
            ),
            # Mild elastic-free smoothing via small blur keeps strokes plausible.
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        ]
    )


def build_dataset_transform(cfg: PreprocessConfig, training_cfg, *, train: bool) -> Callable:
    """Transform for canonical 28x28 dataset samples: tensor + optional aug + normalize.

    The returned callable maps a ``uint8`` HxW array to a normalized
    ``float32`` tensor of shape ``(1, image_size, image_size)``.
    """
    augment = _augmentation_transform(cfg, training_cfg) if train else None

    def transform(array: np.ndarray) -> torch.Tensor:
        tensor = array_to_tensor(
            array,
            normalize=False,
            add_channel=True,
        )
        if augment is not None:
            tensor = augment(tensor)
        if cfg.normalize:
            tensor = (tensor - cfg.mean) / cfg.std
        return tensor

    return transform


def build_inference_transform(cfg: PreprocessConfig) -> Callable[[ImageSource], torch.Tensor]:
    """Transform for arbitrary uploads: full preprocessing then normalize."""

    def transform(source: ImageSource) -> torch.Tensor:
        array = preprocess_to_array(source, cfg)
        return array_to_tensor(
            array,
            normalize=cfg.normalize,
            mean=cfg.mean,
            std=cfg.std,
        )

    return transform
