# ---------------------------------------------------------------------------
# Handwriting Intelligence - inference API image.
#
# Build:
#   docker build -t handwriting-intelligence:latest .
# Run (mount a trained artifact read-only):
#   docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" handwriting-intelligence:latest
#
# The image installs CPU-only PyTorch wheels to keep it as small as possible and
# runs the API as an unprivileged user.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp1 is required by PyTorch; curl is used by the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller than the default CUDA wheels).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Copy only what the package build needs.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# The demo is optional but lets the same image serve Streamlit.
COPY app ./app

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app
USER appuser

ENV HWI_ARTIFACT_PATH=/app/models/mnist_cnn/artifact.pt \
    HWI_LOG_FORMAT=json \
    HWI_LOG_LEVEL=INFO \
    HWI_HOST=0.0.0.0 \
    HWI_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "handwriting_intelligence.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
