"""Checkpoint and inference-artifact serialization.

Two related but distinct formats are produced:

* **Checkpoints** (``*.pt`` under a run directory) are for *resuming training*:
  they carry model + optimizer state, epoch, scheduler-relevant metrics and the
  full training configuration. They are large and never committed.
* **Artifacts** (``artifact.pt``) are for *serving*: a self-contained bundle with
  the model weights, the class set, the preprocessing config and provenance. The
  API and demo load only artifacts, never checkpoints.

Both are written atomically (temp file + ``os.replace``) so an interrupted write
cannot leave a half-written file that later fails to load.

Every checkpoint records ``format_version``; a future incompatible change can then
fail with a precise message instead of a confusing ``KeyError``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from handwriting_intelligence.classes import ClassSet
from handwriting_intelligence.config import Config, ModelConfig
from handwriting_intelligence.exceptions import ArtifactError, CheckpointError
from handwriting_intelligence.models.base import build_model, describe_model

CHECKPOINT_FORMAT_VERSION = 1
ARTIFACT_FORMAT_VERSION = 1


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    """Write ``payload`` to ``path`` atomically and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    return out


# ---------------------------------------------------------------------------
# Checkpoints (training)
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    best_metric: float,
    monitor: str,
    config: Config,
    class_set: ClassSet,
    history: list[dict[str, float]],
    environment: dict[str, Any],
    model_info: dict[str, Any] | None = None,
) -> Path:
    """Serialize a training checkpoint atomically."""
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "checkpoint",
        "saved_at": _now(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "monitor": monitor,
        "model_name": config.model.name,
        "model_config": config.model.model_dump(mode="json"),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": config.to_dict(),
        "class_set": class_set.to_metadata(),
        "history": history,
        "environment": environment,
        "model_info": model_info or describe_model(model),
    }
    return _atomic_torch_save(payload, path)


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load and validate a checkpoint.

    Raises:
        CheckpointError: if the file is missing, not a checkpoint, or a newer
            format than this code understands.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise CheckpointError(f"checkpoint not found: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"failed to load checkpoint {ckpt_path}: {exc}") from exc
    _validate_payload(
        payload, ckpt_path, expected_kind="checkpoint", version=CHECKPOINT_FORMAT_VERSION
    )
    return payload


def model_from_checkpoint(
    checkpoint: dict[str, Any], *, device: torch.device | str = "cpu"
) -> nn.Module:
    """Rebuild a model from a checkpoint's stored architecture + weights."""
    model_config = ModelConfig.model_validate(checkpoint["model_config"])
    class_set = ClassSet.from_metadata(checkpoint["class_set"])
    model = build_model(model_config, num_classes=class_set.num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Artifacts (serving)
# ---------------------------------------------------------------------------
def save_artifact(
    path: str | Path,
    *,
    model: nn.Module,
    class_set: ClassSet,
    config: Config,
    environment: dict[str, Any],
    training_summary: dict[str, Any] | None = None,
) -> Path:
    """Serialize a self-contained inference artifact atomically."""
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "kind": "artifact",
        "artifact_kind": "character_classifier",
        "saved_at": _now(),
        "model_name": config.model.name,
        "model_config": config.model.model_dump(mode="json"),
        "model_state": model.state_dict(),
        "preprocessing": config.preprocessing.model_dump(mode="json"),
        "data": config.data.model_dump(mode="json"),
        "experiment": config.experiment.model_dump(mode="json"),
        "seed": config.seed,
        "class_set": class_set.to_metadata(),
        "model_info": describe_model(model),
        "training_summary": training_summary or {},
        "environment": environment,
        "version": _package_version(),
    }
    return _atomic_torch_save(payload, path)


def load_artifact(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate an inference artifact.

    Raises:
        ArtifactError: if the file is missing, not an artifact, or malformed.
    """
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ArtifactError(
            f"model artifact not found: {artifact_path}\n"
            "Train a model first, e.g. `python -m handwriting_intelligence.cli.train "
            "--config configs/mnist_cnn.yaml`."
        )
    try:
        payload = torch.load(artifact_path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ArtifactError(f"failed to load artifact {artifact_path}: {exc}") from exc
    _validate_payload(
        payload, artifact_path, expected_kind="artifact", version=ARTIFACT_FORMAT_VERSION
    )
    for required in ("model_config", "model_state", "class_set"):
        if required not in payload:
            raise ArtifactError(f"artifact {artifact_path} is missing required key '{required}'")
    return payload


def artifact_to_checkpoint_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    """Adapt an artifact payload to the shape expected by :func:`model_from_checkpoint`.

    This lets the predictor reuse the model-rebuild logic without caring whether the
    source was a checkpoint or an artifact.
    """
    return {
        "model_config": artifact["model_config"],
        "model_state": artifact["model_state"],
        "class_set": artifact["class_set"],
    }


def _package_version() -> str:
    from handwriting_intelligence import __version__

    return __version__


def _validate_payload(payload: Any, path: Path, *, expected_kind: str, version: int) -> None:
    """Validate the common envelope fields shared by checkpoints and artifacts."""
    error_cls = CheckpointError if expected_kind == "checkpoint" else ArtifactError
    if not isinstance(payload, dict):
        raise error_cls(f"{path} is not a {expected_kind} mapping")
    found_version = payload.get("format_version")
    if found_version is None:
        raise error_cls(f"{path} has no format_version; it is not a valid {expected_kind}")
    if int(found_version) > version:
        raise error_cls(
            f"{path} uses format_version {found_version}, newer than this build supports "
            f"({version}). Upgrade the package or retrain."
        )
    kind = payload.get("kind")
    if kind is not None and kind != expected_kind:
        raise error_cls(f"{path} is a '{kind}', not a '{expected_kind}'")
