"""FastAPI application factory and endpoints.

Design:
    * ``create_app`` builds a fully configured app. Tests call it with an injected
      ``predictor`` so they never need a trained artifact.
    * The model is loaded **once** during startup (lifespan) and stored on
      ``app.state``. If loading fails the service still starts and reports the
      reason through ``/ready`` - a container should be able to become healthy,
      then report not-ready, rather than crash-looping invisibly.
    * A single middleware assigns a request id, times the request and emits a
      structured log line. Image contents are never logged.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from handwriting_intelligence import __version__
from handwriting_intelligence.api.dependencies import (
    Settings,
    get_predictor,
    get_settings,
    try_load_predictor,
)
from handwriting_intelligence.api.schemas import (
    BatchPredictionResponse,
    ClassProbability,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
)
from handwriting_intelligence.exceptions import ArtifactError, ImageValidationError
from handwriting_intelligence.inference.predictor import Predictor
from handwriting_intelligence.inference.preprocessing import validate_image_bytes
from handwriting_intelligence.utils.logging import (
    configure_logging,
    get_logger,
    get_request_id,
    set_request_id,
)

logger = get_logger(__name__)


def _to_prediction_response(request_id: str, prediction) -> PredictionResponse:
    return PredictionResponse(
        request_id=request_id,
        predicted_class=prediction.label,
        class_index=prediction.class_index,
        confidence=prediction.confidence,
        top_k=[
            ClassProbability(
                class_index=alt.class_index, label=alt.label, probability=alt.probability
            )
            for alt in prediction.top_k
        ],
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        inference_ms=prediction.inference_ms,
    )


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload with a hard byte cap (never buffers an unbounded body)."""
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ImageValidationError(f"uploaded file exceeds the {max_bytes}-byte limit")
    return data


def create_app(
    settings: Settings | None = None,
    *,
    predictor: Predictor | None = None,
) -> FastAPI:
    """Build and return the FastAPI application."""
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved_settings.log_level, resolved_settings.log_format)
        if predictor is not None:
            app.state.predictor = predictor
            app.state.predictor_error = ""
        else:
            loaded, error = try_load_predictor(resolved_settings)
            app.state.predictor = loaded
            app.state.predictor_error = error
        logger.info(
            "application startup",
            extra={
                "version": __version__,
                "artifact": str(resolved_settings.artifact_path),
                "model_loaded": app.state.predictor is not None,
            },
        )
        yield
        logger.info("application shutdown")

    app = FastAPI(
        title="Handwriting Intelligence API",
        description=(
            "Character recognition for handwritten digits and letters. "
            "This model classifies **isolated characters**; it does not perform "
            "word or sentence recognition."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings

    # -- middleware: request id, timing, size guard, structured logging -----
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        set_request_id(request_id)
        started = time.perf_counter()

        # Cheap first line of defence against oversized bodies before parsing.
        content_length = request.headers.get("content-length")
        body_cap = (
            resolved_settings.max_upload_bytes * max(1, resolved_settings.max_batch_size)
            + 1_048_576
        )
        if (
            content_length is not None
            and content_length.isdigit()
            and int(content_length) > body_cap
        ):
            return JSONResponse(
                status_code=413,
                content=ErrorResponse(
                    request_id=request_id,
                    error="payload_too_large",
                    detail=f"request body exceeds the {body_cap}-byte limit",
                ).model_dump(),
            )

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "unhandled error", extra={"method": request.method, "path": request.url.path}
            )
            set_request_id("")
            raise

        duration_ms = (time.perf_counter() - started) * 1000.0
        response.headers["x-request-id"] = request_id
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        set_request_id("")
        return response

    # -- error handlers -----------------------------------------------------
    @app.exception_handler(ImageValidationError)
    async def _image_error_handler(request: Request, exc: ImageValidationError):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                request_id=get_request_id(),
                error="invalid_image",
                detail=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(ArtifactError)
    async def _artifact_error_handler(request: Request, exc: ArtifactError):
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                request_id=get_request_id(),
                error="model_unavailable",
                detail=str(exc),
            ).model_dump(),
        )

    # -- endpoints ----------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        """Liveness: the process is up and serving."""
        return HealthResponse(
            status="ok", version=__version__, uptime_seconds=resolved_settings.uptime_seconds
        )

    @app.get("/ready", response_model=ReadyResponse, tags=["system"])
    async def ready(request: Request):
        """Readiness: a usable model artifact is loaded."""
        predictor_obj = getattr(request.app.state, "predictor", None)
        if predictor_obj is None:
            return JSONResponse(
                status_code=503,
                content=ReadyResponse(
                    ready=False,
                    detail=getattr(request.app.state, "predictor_error", "no model loaded"),
                ).model_dump(),
            )
        return ReadyResponse(
            ready=True,
            detail="model artifact loaded",
            model_name=predictor_obj.model_name,
            num_classes=predictor_obj.class_set.num_classes,
        )

    @app.get("/v1/model/info", response_model=ModelInfoResponse, tags=["model"])
    async def model_info(
        predictor_obj: Annotated[Predictor, Depends(get_predictor)],
    ) -> ModelInfoResponse:
        """Metadata about the loaded model and its supported class set."""
        return ModelInfoResponse(**predictor_obj.info())

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"])
    async def predict(
        file: Annotated[UploadFile, File(description="A single character image.")],
        predictor_obj: Annotated[Predictor, Depends(get_predictor)],
    ) -> PredictionResponse:
        """Classify a single uploaded character image."""
        data = await _read_upload(file, resolved_settings.max_upload_bytes)
        validate_image_bytes(data, max_bytes=resolved_settings.max_upload_bytes)
        prediction = predictor_obj.predict(data)
        return _to_prediction_response(get_request_id(), prediction)

    @app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
    async def predict_batch(
        files: Annotated[list[UploadFile], File(description="One or more character images.")],
        predictor_obj: Annotated[Predictor, Depends(get_predictor)],
    ) -> BatchPredictionResponse:
        """Classify several images (bounded by ``HWI_MAX_BATCH_SIZE``)."""
        if not files:
            raise ImageValidationError("no files provided")
        if len(files) > resolved_settings.max_batch_size:
            raise ImageValidationError(
                f"batch size {len(files)} exceeds the limit of {resolved_settings.max_batch_size}"
            )
        payloads: list[bytes] = []
        for index, file in enumerate(files):
            data = await _read_upload(file, resolved_settings.max_upload_bytes)
            try:
                validate_image_bytes(data, max_bytes=resolved_settings.max_upload_bytes)
            except ImageValidationError as exc:
                raise ImageValidationError(f"file[{index}]: {exc}") from exc
            payloads.append(data)
        predictions = predictor_obj.predict_batch(payloads)
        request_id = get_request_id()
        return BatchPredictionResponse(
            request_id=request_id,
            count=len(predictions),
            predictions=[_to_prediction_response(request_id, p) for p in predictions],
        )

    return app


def _iter_route_paths(app: FastAPI) -> Iterable[str]:  # pragma: no cover - diagnostics
    return (route.path for route in app.routes if hasattr(route, "path"))


#: Uvicorn entry point: ``uvicorn handwriting_intelligence.api.main:app``.
app = create_app()
