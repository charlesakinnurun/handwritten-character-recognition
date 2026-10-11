# Architecture

This document explains how Handwriting Intelligence is put together, why the
boundaries are drawn where they are, and the trade-offs taken along the way.

## 1. System context

```
                        ┌───────────────────────────────────────────────┐
                        │                  Config (YAML)                 │
                        └───────────────┬───────────────────────────────┘
                                        │ validated by pydantic
             ┌──────────────────────────┼───────────────────────────────┐
             ▼                          ▼                               ▼
   ┌──────────────────┐      ┌────────────────────┐          ┌──────────────────┐
   │   data package    │      │   models package    │          │ training package │
   │ IDX / corpus      │      │ registry + CNN/CRNN │          │ Trainer / ckpt / │
   │ deterministic     │      │ pure nn.Modules     │          │ metrics          │
   │ splits            │      └─────────┬──────────┘          └────────┬─────────┘
   └────────┬─────────┘                │                               │
            │  uint8 images            │  logits (B, C)                │ checkpoint /*.pt
            ▼                          ▼                               ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                            inference package                               │
   │  preprocessing (train/serve parity)  →  Predictor (class_set + model)       │
   └───────────────┬───────────────────────────────────────────────┬───────────┘
                   │ artifact.pt                                    │
                   ▼                                                ▼
        ┌────────────────────┐                          ┌────────────────────┐
        │  FastAPI service   │                          │  Streamlit demo    │
        │  /v1/predict[...]  │                          │  upload / draw     │
        └────────────────────┘                          └────────────────────┘
```

## 2. Package boundaries

| Package | Responsibility | Depends on |
| --- | --- | --- |
| `config` | Parse/validate YAML into typed models | — |
| `classes` | Label conventions & class sets | — |
| `data` | Dataset discovery, loading, splits, transforms | `config`, `classes` |
| `models` | Architecture registry and `nn.Module`s | `config` |
| `training` | Training loop, checkpoints, metrics | `data`, `models`, `config` |
| `inference` | Preprocessing validation + `Predictor` | `data.transforms`, `training.checkpoints` |
| `api` | HTTP contract | `inference` |
| `cli` | Thin orchestration for scripts | everything above |

The dependency arrows only ever point **downward**. `models` knows nothing about
`data`; `inference` knows nothing about `api`. This is what makes each layer
independently testable: model tests need no data, API tests need no dataset.

## 3. Data flow

1. `Config.load` reads YAML and validates it (unknown keys, ranges, and dataset
   consistency are all rejected at load time).
2. `build_datasets` resolves the requested dataset:
   * MNIST from IDX files (discovered locally, or downloaded from the official
     mirror and MD5-verified), or
   * EMNIST `balanced`/`bymerge` from the local JPEG corpus (validated against
     `emnist.csv`, transposed to the upright orientation, cached to `.npz`), or
   * other EMNIST splits via torchvision.
3. A **seeded permutation** produces disjoint train/val/test indices. The
   validation split always comes from *training* data. The official test set is
   only ever used for final evaluation — never for model selection.
4. Loaders return `(uint8 HxW image, int label)`. A transform stack converts to a
   normalized `float32` tensor and (optionally) augments.

## 4. Model boundaries

* Every model returns **raw logits** `(B, C)`. Softmax is applied only at the
  point of interpretation, so `CrossEntropyLoss` is correct and callers can
  compute top-k, entropy, etc.
* The **number of classes is always derived from the dataset's class set**, never
  hard-coded. A config that disagrees with the dataset is rejected up front.
* `CNNBaseline` and `CNNImproved` live in the character registry
  (`available_models()`).
* `CRNN` is deliberately **not** in that registry: its output is a *sequence*
  (`T, B, C`), not a single class, so it must not be selected by the
  character-training CLI.

### Why a single-character classifier cannot do words

A character classifier receives one framed glyph and emits one of `C` labels.
For an image of a *word* there is no way to know how many characters are present,
where one ends and the next begins, or to keep the output consistent across the
sequence. Word recognition therefore requires a model that maps a variable-width
image to a *variable-length* label sequence. The CRNN formalises this:

```
image (1 x H x W) → CNN features → per-column features → BiLSTM → per-frame logits
                                                         → CTC (alignment-free) → decode
```

CTC introduces a blank symbol (index `0`) so the model can emit "no character"
between glyphs; greedy decoding collapses repeats and blanks, and beam search can
refine the result. This extension is implemented in `models/crnn.py` and is **not
claimed to work** until trained and evaluated on a word/line dataset (e.g. IAM)
with character/word error rate. See `docs/experiments.md`.

## 5. Training lifecycle

```
seed_everything(config.seed)
  → build_dataloaders        (train / val / test)
  → build_model              (num_classes from class_set)
  → Trainer(...)             (optimizer, scheduler, AMP, loss)
  → [resume from checkpoint] (model + optimizer + epoch + history)
  → fit(train, val)
       for each epoch:
         train_epoch  → train_loss, train_accuracy
         validate     → val_loss, val_accuracy
         scheduler.step()
         write metrics.jsonl
         save last.pt; save best.pt when the monitored metric improves
         early-stop if no improvement for `patience` epochs
  → select best checkpoint → save_artifact(models/<name>/artifact.pt)
```

* **Reproducibility**: seed, git commit and library versions are recorded before
  training starts (`environment.json`), so an interrupted run is still described.
* **AMP** is enabled only when CUDA is available *and* the config allows it.
* **Interrupts**: `Ctrl-C` writes `interrupt.pt` before the exception propagates.
* **Resume** restores optimizer/epoch/history and never rewrites the run's config.

## 6. Checkpoint vs artifact lifecycle

| | Checkpoint (`runs/<name>/*.pt`) | Artifact (`models/<name>/artifact.pt`) |
| --- | --- | --- |
| Purpose | resume training | serve predictions |
| Contains | model + optimizer state, epoch, history, full config | model state, class set, preprocessing, provenance |
| Size | larger (optimizer moments) | smaller |
| Committed | never | never |

Both carry a `format_version`. Loading a checkpoint as an artifact (or vice
versa) fails with a precise message, and a newer format is rejected rather than
mis-read. Writes are atomic (temp file + `os.replace`).

## 7. Inference lifecycle

```
create_app(settings) ── lifespan ──► try_load_predictor(artifact_path)
        │                                  │
        │                             success? ──► app.state.predictor
        │                                  └─ fail ──► app.state.predictor_error
        ▼
  request ─► middleware: request-id + timing + size guard
           ─► validate bytes (size, decode, format allow-list, dimensions)
           ─► Predictor.predict (preprocess → forward → softmax → top-k)
           ─► typed JSON response
```

* The model is loaded **once** at startup, never per request.
* A missing artifact does not crash the service: `/health` stays 200 and `/ready`
  returns 503 with the reason. This is what lets a container become healthy and
  *then* report not-ready instead of crash-looping.
* The same preprocessing code path is used for training-time evaluation and
  serving, eliminating train/serve skew.

## 8. Key engineering decisions

| Decision | Rationale |
| --- | --- |
| Custom numpy IDX reader instead of relying on library paths | The reference MNIST mirror and local layouts differ; explicit validation gives clear errors and catches truncation. |
| Class set serialized into every artifact | Prevents predicting with the wrong label mapping; the API can report exactly which classes a model supports. |
| Deterministic seeded split shared across experiments | Baseline vs improved comparisons must use the *same* validation set to be meaningful. |
| `predictor` injected through a FastAPI dependency | API tests run without a trained model. |
| CRNN kept out of the classifier registry | Its sequence output would silently corrupt a classification pipeline. |
| No message queue / DB / Kubernetes | There is no demonstrated requirement; a single process serving a small model is simpler and observable. |

## 9. Operational metrics to monitor

* **Latency**: p50/p95/p99 of `/v1/predict` (from structured logs).
* **Error rates**: 4xx (client input) vs 5xx (server/model) per endpoint.
* **Readiness**: `/ready` flapping indicates artifact/volume problems.
* **Input drift**: distribution of decoded image sizes and mean pixel values.
* **Confidence drift**: fraction of predictions below a confidence threshold.
* **Throughput & saturation**: requests/sec, CPU utilisation, queue wait.

These are intentionally *not* wired to a vendor here; the structured JSON logs and
the `request_id` provide the correlation keys any collector (e.g. an APM agent)
needs.
