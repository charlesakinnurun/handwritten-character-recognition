# Handwriting Intelligence — Character Recognition System

[![CI](https://github.com/charlesakinnurun/handwritten-character-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/charlesakinnurun/handwritten-character-recognition/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.2-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-shaped handwritten **character recognition** system: reusable
training/evaluation pipelines, reproducible experiments, a FastAPI inference
service, an interactive demo, automated tests and CI — built with PyTorch.

It recognises **isolated** handwritten digits and characters (MNIST and the
EMNIST splits). A separate CRNN extension targets word/line recognition and is
documented but **not claimed to work** until trained on a sequence dataset.

> **Honesty note.** Every number in this repository comes from a run you can
> reproduce with the documented commands. Nothing here is estimated or invented.

---

## 1. Why this project

Most character-recognition repos are notebooks. This one is a **product**: it
separates data, models, training, evaluation and serving into layers that can be
tested independently; it records the provenance of every run; it validates
untrusted input at the system boundary; and it fails loudly instead of returning
a plausible-looking wrong answer. The goal is to demonstrate the engineering
judgement expected of a machine-learning engineer, not just model accuracy.

## 2. Features

* **Two architectures** — a compact CNN baseline and a deeper residual CNN, plus a
  CRNN sequence model (experimental).
* **Two + datasets** — MNIST and EMNIST (`balanced`, `letters`, `digits`,
  `byclass`, `bymerge`) behind one interface, with correct label conventions.
* **Deterministic experiments** — seeded splits, provenance capture, resumable
  checkpoints, early stopping, LR schedules, guarded AMP.
* **Real evaluation** — accuracy, macro/weighted P/R/F1, per-class metrics,
  confusion matrices, top-k, confidence analysis, majority-class baseline,
  latency and model size.
* **Inference API** — versioned FastAPI endpoints with request IDs, size limits,
  input validation, structured logging, and a clean predictor abstraction.
* **Interactive demo** — Streamlit app to upload or draw a character.
* **Engineering hygiene** — typed, linted, `mypy`-checked, tested, containerised,
  and wired into GitHub Actions.

## 3. Architecture

```
config (YAML) ─► data ─► models ─► training ─► checkpoints ─► artifact
                                                                  │
                                             ┌────────────────────┴───────────────┐
                                             ▼                                    ▼
                                    FastAPI /v1/predict                Streamlit demo
```

Full data flow, model boundaries, and lifecycle diagrams are in
[`docs/architecture.md`](docs/architecture.md).

**Documentation index:** [Architecture](docs/architecture.md) ·
[Data & licensing](docs/data.md) · [Experiments](docs/experiments.md) ·
[Model card](docs/model_card.md) · [Deployment](docs/deployment.md).

## 4. Installation

Requires **Python 3.11+**.

```bash
git clone https://github.com/charlesakinnurun/handwritten-character-recognition.git
cd handwriting-intelligence

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -e ".[dev]"     # runtime + test/lint/type tools
# optional: pip install -e ".[demo]"    # drawing canvas for the demo
```

Verify:

```bash
python -m pytest -q
```

## 5. Datasets

See [`docs/data.md`](docs/data.md) for full details, licensing and layouts.

* **MNIST** — discovered locally or downloaded from the official mirror
  (MD5-verified).
* **EMNIST balanced** — read from an extracted JPEG corpus, OR use torchvision for
  the other splits.

Download/verify what a config needs:

```bash
python -m handwriting_intelligence.cli.download --config configs/mnist_cnn.yaml
```

## 6. Models

| Name | Description | Where |
| --- | --- | --- |
| `cnn_baseline` | Conv→BN→ReLU→pool ×2, MLP head | `models/cnn.py` |
| `cnn_improved` | Residual blocks + global average pooling + wider head | `models/cnn.py` |
| `crnn` | CNN + BiLSTM + CTC (sequence / word recognition) | `models/crnn.py` |

The number of classes is always derived from the dataset's class set, never hard
coded.

## 7. Training

```bash
# MNIST baseline
python -m handwriting_intelligence.cli.train --config configs/mnist_cnn.yaml

# MNIST improved (augmentation + cosine schedule + residuals)
python -m handwriting_intelligence.cli.train --config configs/mnist_cnn_improved.yaml

# EMNIST balanced (CPU quick pass)
python -m handwriting_intelligence.cli.train --config configs/emnist_cnn.yaml \
    --max-train-samples 30000 --max-val-samples 3000 --epochs 3

# Resume an interrupted run
python -m handwriting_intelligence.cli.train --config configs/mnist_cnn.yaml --resume auto
```

Each run writes to `runs/<experiment>/`: `config.yaml`, `environment.json`,
`metrics.jsonl`, `best.pt`, `last.pt`, `summary.json`. The serving artifact is
written to `models/<experiment>/artifact.pt`.

## 8. Evaluation

```bash
python -m handwriting_intelligence.cli.evaluate \
    --artifact models/mnist_cnn/artifact.pt \
    --config configs/mnist_cnn.yaml \
    --split test
```

This prints a metrics table and writes `reports/<experiment>/{metrics_test.json,
confusion_matrix_test.png, examples_correct_test.png, examples_incorrect_test.png}`.

### Results (measured)

Measured on the official MNIST test set and the EMNIST Balanced held-out split
(CPU-only, 2 PyTorch threads). Full methodology, averaging choices and error
analysis: [`docs/experiments.md`](docs/experiments.md).

| Dataset | Model | Test accuracy | F1 (macro) | Top-5 | Majority baseline | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | `cnn_baseline` | 0.9907 | 0.9907 | 1.0000 | 0.1135 | 0.93 ms |
| MNIST | `cnn_improved` | **0.9953** | **0.9953** | 1.0000 | 0.1135 | 1.79 ms |
| EMNIST Balanced (47 classes) † | `cnn_improved` | 0.8126 | 0.8084 | 0.9821 | 0.0166 | 3.16 ms |

† The EMNIST Balanced run is a **quick CPU verification pass** (3 epochs on 30,000
of 112,800 training images, seeded held-out split, `official_test: false`) — it is
*not* a tuned result; more data and epochs would improve it. The **improved CNN
beats the baseline by +0.46 pp on MNIST with ~19% fewer parameters** and
better-calibrated confidences (wrong-prediction confidence 0.56 vs 0.73).

Full methodology, averaging choices and error analysis are in
[`docs/experiments.md`](docs/experiments.md).

## 9. Inference API

```bash
python -m handwriting_intelligence.cli.serve            # or: uvicorn handwriting_intelligence.api.main:app
```

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/ready` | Model/artifact readiness |
| GET | `/v1/model/info` | Model metadata + supported classes |
| POST | `/v1/predict` | Classify a single image |
| POST | `/v1/predict/batch` | Classify multiple images (bounded) |

Example:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/predict \
     -F "file=@examples/sample_digit_7.png"
```

```json
{
  "request_id": "<per-request uuid4 hex>",
  "predicted_class": "7",
  "class_index": 7,
  "confidence": 0.9932,
  "top_k": [
    {"class_index": 7, "label": "7", "probability": 0.9932},
    {"class_index": 2, "label": "2", "probability": 0.0067},
    {"class_index": 1, "label": "1", "probability": 0.0001}
  ],
  "model_name": "cnn_baseline",
  "model_version": "0.1.0",
  "inference_ms": 5.69
}
```

(Values above are from an actual local run against
`models/mnist_cnn/artifact.pt`; `inference_ms` varies with hardware.)

Invalid uploads return a structured `400`; a missing model returns `503`. The
service still starts and answers `/health`, reporting the reason via `/ready`.

## 10. Interactive demo

```bash
python -m streamlit run app/streamlit_app.py
# or: make demo
```

Upload an image or draw one; you'll see the exact 28×28 canvas the model sees, the
prediction, confidence and top-k. The demo states that it classifies isolated
characters only.

## 11. Docker

```bash
docker build -t handwriting-intelligence:latest .
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" handwriting-intelligence:latest
# or
docker compose up --build
```

The image installs CPU-only PyTorch, runs as a non-root user, and defines a
`/health` healthcheck.

## 12. Testing

```bash
python -m pytest                 # unit + integration + smoke
python -m pytest tests/unit      # fast unit tests only
python -m pytest --cov           # with coverage
```

Tests use synthetic fixtures and never download a full dataset or train a full
model. Run the linters and type checker:

```bash
ruff check src tests scripts app
ruff format --check src tests scripts app
mypy src
```

## 13. Limitations and known failure cases

* **Isolated characters only.** A multi-character image is not segmented; the
  model returns one label.
* **No calibration.** Confidence is a softmax score, not a calibrated probability.
* **Domain shift.** Camera photos, coloured backgrounds, unusual stroke weight,
  and non-Latin scripts are out of distribution.
* **EMNIST balanced local runs** use a held-out split carved from training because
  the shipped corpus contains only training data (`official_test: false`).
* **Confusable characters.** Visually similar classes (e.g. `0/O`, `1/l/I`, `S/5`)
  are the dominant confusions — see the per-class metrics and confusion matrix.
* **CPU-only in this repo's verification.** GPU paths exist but are guarded and
  were not exercised here.

## 14. Roadmap toward word and sentence recognition

1. ✅ Isolated character classification (this repository).
2. 🔜 CRNN + CTC trained on a word/line dataset (IAM), reported with **CER/WER**.
3. 🔜 Beam-search decoding and a language model for real words.
4. 🔜 Line-level segmentation and full-page transcription.

The CRNN is implemented in `models/crnn.py`; it is **not trained** in this
repository and no word-level accuracy is claimed. See `docs/architecture.md`.

## 15. Model card & data licensing

* [`docs/model_card.md`](docs/model_card.md)
* [`docs/data.md`](docs/data.md) — MNIST/EMNIST/IAM terms and citation.

## 16. Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). In short: keep changes typed, tested and
documented; run `ruff`, `mypy` and `pytest` before opening a PR; never commit
datasets, weights or secrets.

## 17. License & citation

Code released under the [MIT License](LICENSE).

If you use this project, please also cite the datasets:

```bibtex
@article{lecun1998mnist, title={The MNIST database of handwritten digits}, author={LeCun, Yann and Cortes, Corinna and Burges, Christopher}, year={1998}}
@article{cohen2017emnist, title={EMNIST: Extending MNIST to handwritten letters}, author={Cohen, Gregory and Afshar, Saeed and Tapsoba, Jonathan and Van Schaik, Andr{\'e}}, journal={IJCNN}, year={2017}}
```
