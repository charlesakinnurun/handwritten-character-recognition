"""Application settings and dependency providers.

Configuration comes from environment variables prefixed ``HWI_`` (see
``.env.example``) or a ``.env`` file. The predictor is provided through a FastAPI
dependency so tests can override it with a stub and never need a trained model.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from fastapi import Request
from pydantic_settings import BaseSettings, SettingsConfigDict

from handwriting_intelligence.inference.predictor import Predictor, load_predictor
from handwriting_intelligence.utils.logging import get_logger

logger = get_logger(__name__)

_PROCESS_START = time.monotonic()


class Settings(BaseSettings):
    """Runtime settings for the API service."""

    model_config = SettingsConfigDict(
        env_prefix="HWI_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    artifact_path: Path = Path("models/mnist_cnn/artifact.pt")

    log_level: str = "INFO"
    log_format: str = "json"

    host: str = "0.0.0.0"
    port: int = 8000

    max_upload_bytes: int = 5 * 1024 * 1024
    max_batch_size: int = 32

    @property
    def uptime_seconds(self) -> float:
        """Seconds since the process imported this module."""
        return round(time.monotonic() - _PROCESS_START, 3)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def get_predictor(request: Request) -> Predictor:
    """Return the predictor loaded during application startup.

    Raises:
        ArtifactError: if no artifact was loaded (checked by ``/ready`` and the
            prediction endpoints, which translate it into a 503).
    """
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        from handwriting_intelligence.exceptions import ArtifactError

        reason = getattr(request.app.state, "predictor_error", "no model artifact loaded")
        raise ArtifactError(reason)
    return predictor


def try_load_predictor(settings: Settings) -> tuple[Predictor | None, str]:
    """Attempt to load the predictor, returning ``(predictor, error_message)``.

    A failure here must not prevent the process from starting: the service still
    answers ``/health`` and reports *why* it is not ready via ``/ready``.
    """
    try:
        predictor = load_predictor(settings.artifact_path)
        return predictor, ""
    except Exception as exc:  # surfaced verbatim in /ready, not swallowed
        message = f"{type(exc).__name__}: {exc}"
        logger.warning("predictor not loaded", extra={"reason": message})
        return None, message
