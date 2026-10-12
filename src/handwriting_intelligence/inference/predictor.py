"""The predictor: the single object that turns an image into a prediction.

Everything the API and the demo do flows through :class:`Predictor`. Keeping the
model-loading, preprocessing and softmax logic here (rather than in the endpoints)
means the serving layer is thin and can be tested with a fake predictor.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from handwriting_intelligence.classes import ClassSet
from handwriting_intelligence.config import PreprocessConfig
from handwriting_intelligence.data.transforms import ImageSource
from handwriting_intelligence.exceptions import ArtifactError
from handwriting_intelligence.inference.preprocessing import preprocess_tensor
from handwriting_intelligence.training.checkpoints import (
    artifact_to_checkpoint_payload,
    load_artifact,
)
from handwriting_intelligence.training.checkpoints import (
    model_from_checkpoint as _model_from_checkpoint,
)
from handwriting_intelligence.utils.device import resolve_device
from handwriting_intelligence.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Alternative:
    """One entry of a top-k prediction."""

    class_index: int
    label: str
    probability: float


@dataclass(slots=True)
class Prediction:
    """A single classification result."""

    class_index: int
    label: str
    confidence: float
    top_k: list[Alternative] = field(default_factory=list)
    model_name: str = ""
    model_version: str = ""
    inference_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (top_k expanded recursively)."""
        return {
            "class_index": self.class_index,
            "label": self.label,
            "confidence": self.confidence,
            "top_k": [asdict(a) for a in self.top_k],
            "model_name": self.model_name,
            "model_version": self.model_version,
            "inference_ms": self.inference_ms,
        }


class Predictor:
    """Wraps a trained model with its class set and preprocessing pipeline.

    Args:
        model: An ``nn.Module`` returning logits of shape ``(B, num_classes)``.
        class_set: The label mapping the model was trained with.
        preprocessing: Preprocessing configuration loaded from the artifact.
        model_name: Architecture name (for metadata).
        model_version: Human-readable version string.
        model_info: Optional parameter-count / size summary.
        device: Device to run inference on.
        metadata: Extra provenance (dataset, training summary).
    """

    def __init__(
        self,
        model: nn.Module,
        class_set: ClassSet,
        preprocessing: PreprocessConfig,
        *,
        model_name: str = "unknown",
        model_version: str = "unknown",
        model_info: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device if device is not None else resolve_device("auto")
        self.model = model.to(self.device).eval()
        self.class_set = class_set
        self.preprocessing = preprocessing
        self.model_name = model_name
        self.model_version = model_version
        self.model_info = model_info or {}
        self.metadata = metadata or {}
        self._lock = threading.Lock()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
        map_location: str | torch.device = "cpu",
    ) -> Predictor:
        """Load a predictor from a serialized inference artifact.

        Raises:
            ArtifactError: if the artifact is missing or malformed.
        """
        payload = load_artifact(path, map_location=map_location)
        model = _model_from_checkpoint(artifact_to_checkpoint_payload(payload), device="cpu")
        preprocessing = PreprocessConfig.model_validate(payload["preprocessing"])
        class_set = ClassSet.from_metadata(payload["class_set"])
        resolved_device = device if device is not None else resolve_device("auto")
        model = model.to(resolved_device)
        predictor = cls(
            model,
            class_set,
            preprocessing,
            model_name=str(payload.get("model_name", "unknown")),
            model_version=str(payload.get("version", "unknown")),
            model_info=payload.get("model_info") or {},
            metadata={
                "data": payload.get("data"),
                "training_summary": payload.get("training_summary"),
                "seed": payload.get("seed"),
                "saved_at": payload.get("saved_at"),
            },
            device=torch.device(resolved_device),
        )

        # Sanity: the model output width must match the class set, or predictions
        # would be silently mislabelled.
        if int(getattr(model, "num_classes", -1)) != class_set.num_classes:
            raise ArtifactError(
                f"artifact model outputs {getattr(model, 'num_classes', '?')} classes but its "
                f"class set defines {class_set.num_classes}"
            )
        logger.info(
            "predictor ready",
            extra={"model": predictor.model_name, "classes": class_set.num_classes},
        )
        return predictor

    # -- inference ----------------------------------------------------------
    def _probabilities(self, tensor: torch.Tensor) -> np.ndarray:
        with self._lock, torch.inference_mode():
            logits = self.model(tensor.to(self.device))
            probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def predict(self, source: ImageSource, *, top_k: int = 5) -> Prediction:
        """Classify a single image source (path, bytes, array or PIL image)."""
        return self.predict_batch([source], top_k=top_k)[0]

    def predict_batch(self, sources: Sequence[ImageSource], *, top_k: int = 5) -> list[Prediction]:
        """Classify several images in one forward pass.

        Raises:
            ValueError: if ``sources`` is empty.
        """
        if not sources:
            raise ValueError("predict_batch received no images")
        started = time.perf_counter()
        batch = torch.stack([preprocess_tensor(src, self.preprocessing) for src in sources])
        probs = self._probabilities(batch)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_image_ms = elapsed_ms / len(sources)
        return [self._to_prediction(row, top_k=top_k, latency=per_image_ms) for row in probs]

    def _to_prediction(
        self, probabilities: np.ndarray, *, top_k: int, latency: float
    ) -> Prediction:
        k = max(1, min(top_k, len(probabilities)))
        order = np.argsort(-probabilities)[:k]
        best = int(order[0])
        alternatives = [
            Alternative(
                class_index=int(idx),
                label=self.class_set.label_of(int(idx)),
                probability=float(probabilities[int(idx)]),
            )
            for idx in order
        ]
        return Prediction(
            class_index=best,
            label=self.class_set.label_of(best),
            confidence=float(probabilities[best]),
            top_k=alternatives,
            model_name=self.model_name,
            model_version=self.model_version,
            inference_ms=round(latency, 3),
        )

    # -- metadata -----------------------------------------------------------
    def info(self) -> dict[str, Any]:
        """Return model metadata suitable for ``GET /v1/model/info``."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "architecture": type(self.model).__name__,
            "num_classes": self.class_set.num_classes,
            "class_names": list(self.class_set.labels),
            "class_kind": self.class_set.kind,
            "class_description": self.class_set.description,
            "supports_word_recognition": False,
            "model_info": self.model_info,
            "dataset": self.class_set.name,
            "metadata": self.metadata,
        }


def load_predictor(path: str | Path, *, device: torch.device | str | None = None) -> Predictor:
    """Convenience wrapper around :meth:`Predictor.from_artifact`."""
    return Predictor.from_artifact(path, device=device)
