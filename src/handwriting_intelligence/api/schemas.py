"""Pydantic request/response schemas for the inference API.

These models define the *public contract*. They are deliberately explicit (no bare
dicts) so the OpenAPI schema is accurate and clients get typed responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for ``GET /health`` (liveness)."""

    status: str = Field(description="Always 'ok' when the process is serving.")
    version: str = Field(description="Package version.")
    uptime_seconds: float = Field(description="Seconds since the process started.")


class ReadyResponse(BaseModel):
    """Response for ``GET /ready`` (readiness)."""

    ready: bool = Field(description="True when a model artifact is loaded and usable.")
    detail: str = Field(description="Human-readable explanation.")
    model_name: str | None = Field(default=None, description="Loaded model architecture.")
    num_classes: int | None = Field(default=None, description="Number of output classes.")


class ClassProbability(BaseModel):
    """A single (label, probability) pair."""

    class_index: int
    label: str
    probability: float


class PredictionResponse(BaseModel):
    """Response for ``POST /v1/predict``."""

    request_id: str
    predicted_class: str = Field(description="Human-readable predicted label.")
    class_index: int
    confidence: float = Field(description="Softmax probability of the predicted class.")
    top_k: list[ClassProbability]
    model_name: str
    model_version: str
    inference_ms: float


class BatchPredictionResponse(BaseModel):
    """Response for ``POST /v1/predict/batch``."""

    request_id: str
    count: int
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    """Response for ``GET /v1/model/info``."""

    model_name: str
    model_version: str
    architecture: str
    num_classes: int
    class_names: list[str]
    class_kind: str
    class_description: str
    supports_word_recognition: bool = Field(
        description="Always False for character models; see docs for the CRNN extension."
    )
    dataset: str
    model_info: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    request_id: str
    error: str = Field(description="Short machine-readable error code.")
    detail: str = Field(description="Human-readable explanation.")
