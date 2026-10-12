"""Inference-time preprocessing and input validation.

The numerical preprocessing is defined once in
:mod:`handwriting_intelligence.data.transforms` so training and serving cannot
drift apart. This module adds the *boundary* concerns that only matter when data
comes from an untrusted client: hard limits on byte size, decoded dimensions and
accepted formats.

Security note: file names and ``Content-Type`` headers are **never** trusted.
Validation is performed on the fully decoded image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from handwriting_intelligence.config import PreprocessConfig
from handwriting_intelligence.data.transforms import (
    ImageSource,
    array_to_tensor,
    load_image,
    preprocess_to_array,
)
from handwriting_intelligence.exceptions import ImageValidationError

#: Formats we will decode. Chosen to cover the common upload cases without
#: pulling in exotic decoders.
ALLOWED_FORMATS = frozenset({"PNG", "JPEG", "JPG", "BMP", "WEBP", "TIFF", "GIF"})

#: Hard limits applied to every upload before any model sees it.
DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MiB
DEFAULT_MIN_SIDE = 4
DEFAULT_MAX_SIDE = 4096


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Validated properties of a decoded image."""

    width: int
    height: int
    image_format: str
    mode: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "format": self.image_format,
            "mode": self.mode,
            "size_bytes": self.size_bytes,
        }


def validate_image_bytes(
    data: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    min_side: int = DEFAULT_MIN_SIDE,
    max_side: int = DEFAULT_MAX_SIDE,
) -> ImageMetadata:
    """Validate raw upload bytes and return decoded metadata.

    Checks, in order: byte-size limit, decodability, format allow-list, and
    dimension bounds. Raises :class:`ImageValidationError` with a specific reason.

    Note:
        ``Image.MAX_IMAGE_PIXELS`` guard is intentionally left at PIL's default,
        which already rejects decompression-bomb-scale images by raising.
    """
    if not data:
        raise ImageValidationError("empty request body: no image bytes provided")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"image is too large: {len(data)} bytes exceeds the {max_bytes}-byte limit"
        )

    try:
        with _open_bytes(data) as image:
            width, height = image.size
            image_format = (image.format or "UNKNOWN").upper()
            mode = image.mode
    except Image.DecompressionBombError as exc:  # pragma: no cover - depends on PIL version
        raise ImageValidationError(f"image rejected as a decompression bomb: {exc}") from exc
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(f"could not decode image: {exc}") from exc

    if image_format not in ALLOWED_FORMATS:
        raise ImageValidationError(
            f"unsupported image format '{image_format}'; allowed: {sorted(ALLOWED_FORMATS)}"
        )
    if width < min_side or height < min_side:
        raise ImageValidationError(
            f"image is too small: {width}x{height}, minimum side is {min_side}px"
        )
    if width > max_side or height > max_side:
        raise ImageValidationError(
            f"image is too large: {width}x{height}, maximum side is {max_side}px"
        )

    return ImageMetadata(
        width=width,
        height=height,
        image_format=image_format,
        mode=mode,
        size_bytes=len(data),
    )


def _open_bytes(data: bytes):
    """Open PIL from bytes while forcing a full decode via a transient buffer."""
    import io

    buffer = io.BytesIO(data)
    image = Image.open(buffer)
    image.load()
    return image


def preprocess_tensor(source: ImageSource, cfg: PreprocessConfig):
    """Full preprocessing for inference: arbitrary image -> normalized tensor ``(1,H,W)``."""
    array = preprocess_to_array(source, cfg)
    return array_to_tensor(array, normalize=cfg.normalize, mean=cfg.mean, std=cfg.std)


def preprocess_display_array(source: ImageSource, cfg: PreprocessConfig) -> np.ndarray:
    """Return the exact ``uint8`` canvas the model sees (for the demo to display)."""
    return preprocess_to_array(source, cfg)


__all__ = [
    "ALLOWED_FORMATS",
    "DEFAULT_MAX_SIDE",
    "DEFAULT_MAX_UPLOAD_BYTES",
    "DEFAULT_MIN_SIDE",
    "ImageMetadata",
    "load_image",
    "preprocess_display_array",
    "preprocess_tensor",
    "validate_image_bytes",
]
