# Deployment guide

This guide covers the local development workflow and a production-style deployment
of the inference API. It describes **how** to deploy; it does not claim any
particular environment is live.

## 1. Local development workflow

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 1. acquire data (see docs/data.md)
python -m handwriting_intelligence.cli.download --config configs/mnist_cnn.yaml

# 2. train a model
python -m handwriting_intelligence.cli.train --config configs/mnist_cnn.yaml

# 3. evaluate it
python -m handwriting_intelligence.cli.evaluate \
    --artifact models/mnist_cnn/artifact.pt --config configs/mnist_cnn.yaml

# 4. run the API and the demo
python -m handwriting_intelligence.cli.serve            # http://localhost:8000/docs
python -m streamlit run app/streamlit_app.py            # http://localhost:8501
```

Quality gates (also run in CI):

```bash
ruff check src tests scripts app
ruff format --check src tests scripts app
mypy src
pytest
```

## 2. Configuration

All runtime configuration is environment-driven (prefix `HWI_`) and validated by
`pydantic-settings`. See [`.env.example`](../.env.example).

| Variable | Default | Purpose |
| --- | --- | --- |
| `HWI_ARTIFACT_PATH` | `models/mnist_cnn/artifact.pt` | Artifact the API serves |
| `HWI_DATA_DIR` | `data` | Dataset/cache root |
| `HWI_MODEL_DIR` | `models` | Artifact root |
| `HWI_LOG_LEVEL` | `INFO` | Log verbosity |
| `HWI_LOG_FORMAT` | `json` | `json` (prod) or `text` (dev) |
| `HWI_HOST` / `HWI_PORT` | `0.0.0.0` / `8000` | Bind address |
| `HWI_MAX_UPLOAD_BYTES` | `5242880` | Per-file upload cap |
| `HWI_MAX_BATCH_SIZE` | `32` | Max images per batch request |

**Secrets.** Never commit `.env`. In production, inject configuration through your
platform's secret manager or environment injection mechanism. No credentials are
needed to serve a model; the only sensitive configuration is the artifact path and
any future external-service tokens (e.g. an error-tracking DSN), which must come
from the environment.

## 3. Container deployment

```bash
# Build (CPU-only torch, non-root user, healthcheck baked in)
docker build -t handwriting-intelligence:0.1.0 .

# Run with the trained artifact mounted read-only
docker run --rm -p 8000:8000 \
    -v "$PWD/models:/app/models:ro" \
    -e HWI_ARTIFACT_PATH=/app/models/mnist_cnn/artifact.pt \
    -e HWI_LOG_FORMAT=json \
    handwriting-intelligence:0.1.0

# Or via compose
docker compose up --build
```

The image:

* installs **CPU-only** PyTorch wheels (smaller; no CUDA),
* runs as the unprivileged user `appuser` (uid 10001),
* exposes a `HEALTHCHECK` against `/health`,
* serves with `uvicorn`.

### Health and readiness probes

| Endpoint | Meaning | Suggested use |
| --- | --- | --- |
| `GET /health` | process is up | liveness probe |
| `GET /ready` | model artifact loaded (200) / not loaded (503) | readiness probe |

Wire `/health` to the liveness probe and `/ready` to the readiness probe so that a
pod with a missing artifact is taken out of rotation instead of serving errors.

## 4. Resource expectations (qualitative)

* The model is small (a few hundred thousand parameters; see the `artifact.pt`
  size in the run summary). A single CPU core is enough for low request rates.
* Memory footprint is dominated by the PyTorch runtime, not the model. A container
  limit in the hundreds of MB is comfortable for the character models.
* Startup time is dominated by importing PyTorch and loading the artifact
  (order of a second on CPU).
* Scaling is horizontal (stateless replicas behind a load balancer). Vertical
  scaling (more CPU) reduces latency for batches.

No cloud cost figures are given here because they depend entirely on the provider
and region.

## 5. Operational metrics to monitor

See `docs/architecture.md` §9. In short: request latency percentiles, 4xx vs 5xx
rates, `/ready` flapping, input-size drift, and the fraction of low-confidence
predictions. Structured JSON logs carry a `request_id` for correlation; add an
error-tracking hook (e.g. a Sentry DSN from the environment) if desired.

## 6. Graceful degradation

If the artifact is missing or corrupt, the service **still starts**:

* `/health` returns `200` (the process is alive),
* `/ready` returns `503` with the exact error,
* `/v1/predict*` and `/v1/model/info` return `503` with a structured error body.

This avoids crash-loops and makes the failure obvious to operators.

## 7. Release checklist

1. `ruff`, `mypy` and `pytest` are green.
2. `docker build` succeeds and the container's healthcheck passes.
3. `/ready` returns `200` with the expected artifact.
4. A smoke `curl` against `/v1/predict` returns a sensible label.
5. The model card and `docs/experiments.md` reflect the artifact being shipped.
