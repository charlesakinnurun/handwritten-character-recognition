"""The training loop.

A single :class:`Trainer` owns optimization, validation, early stopping, learning
rate scheduling, checkpointing and metric logging. Nothing about *what* data or
*which* architecture is hard-coded here: the trainer receives a model, a config and
loaders, which is what makes it reusable across datasets and independently
testable with synthetic fixtures.

Notable behaviours:

* **Device-agnostic.** The device is resolved via :func:`resolve_device`; CUDA is
  never assumed.
* **Guarded AMP.** Mixed precision is only enabled on CUDA, and ``mixed_precision``
  can be forced on/off explicitly.
* **Interrupt-safe.** ``Ctrl-C`` saves an ``interrupt.pt`` checkpoint before
  re-raising, so long CPU runs are never silently lost.
* **Resumable without clobbering.** :meth:`Trainer.resume` restores optimizer,
  epoch, history and best metric; the run directory and config are preserved.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from handwriting_intelligence.classes import ClassSet
from handwriting_intelligence.config import Config, TrainingConfig
from handwriting_intelligence.exceptions import CheckpointError
from handwriting_intelligence.training.checkpoints import load_checkpoint, save_checkpoint
from handwriting_intelligence.utils.device import amp_supported, resolve_device
from handwriting_intelligence.utils.logging import get_logger
from handwriting_intelligence.utils.reproducibility import environment_info

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingResult:
    """Outcome of a training run."""

    history: list[dict[str, float]]
    best_metric: float
    best_epoch: int
    monitor: str
    best_checkpoint: Path
    last_checkpoint: Path
    interrupted: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Construct the configured optimizer for ``model``."""
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unsupported optimizer: {cfg.optimizer}")  # pragma: no cover


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainingConfig, *, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Construct the configured learning-rate scheduler (or ``None``)."""
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.scheduler == "plateau":
        mode: Literal["min", "max"] = "min" if cfg.monitor == "val_loss" else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=cfg.patience, factor=cfg.gamma
        )
    raise ValueError(f"unsupported scheduler: {cfg.scheduler}")  # pragma: no cover


def resolve_amp(preference: bool | str, device: torch.device) -> bool:
    """Resolve the ``mixed_precision`` preference into a concrete boolean."""
    if preference is True:
        if not amp_supported(device):
            raise ValueError("mixed_precision=True was requested but AMP requires CUDA")
        return True
    if preference is False:
        return False
    return amp_supported(device)


class Trainer:
    """Owns the full training lifecycle for a character classifier."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        class_set: ClassSet,
        *,
        device: torch.device | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.class_set = class_set
        self.device = device if device is not None else resolve_device(config.training.device)
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, config.training)
        self.scheduler = build_scheduler(
            self.optimizer, config.training, epochs=config.training.epochs
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
        self.amp = resolve_amp(config.training.mixed_precision, self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.output_dir = Path(output_dir) if output_dir is not None else config.run_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

        self.history: list[dict[str, float]] = []
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = math.inf if config.training.monitor == "val_loss" else -math.inf
        self.best_epoch = -1
        self._epochs_without_improvement = 0
        self._interrupted = False

        #: When True, a plateau scheduler is stepped with the monitored metric.
        self._plateau = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    # -- metric bookkeeping -------------------------------------------------
    def _is_improvement(self, value: float) -> bool:
        if self.config.training.monitor == "val_loss":
            return value < self.best_metric - 1e-9
        return value > self.best_metric + 1e-9

    @staticmethod
    def _metric_from(metrics: dict[str, float], monitor: str) -> float:
        key = "val_loss" if monitor == "val_loss" else "val_accuracy"
        return float(metrics[key])

    # -- epoch loops --------------------------------------------------------
    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """Run one training epoch and return average loss / accuracy."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

            if self.amp:
                self.scaler.scale(loss).backward()
                if self.config.training.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.training.grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip
                    )
                self.optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += batch_size
            self.global_step += 1

            if self.config.training.log_every and step % self.config.training.log_every == 0:
                logger.info(
                    "train step",
                    extra={
                        "epoch": epoch,
                        "step": step,
                        "loss": round(float(loss.item()), 4),
                    },
                )
        return {
            "train_loss": total_loss / max(1, seen),
            "train_accuracy": correct / max(1, seen),
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Run the validation loop and return average loss / accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        seen = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += batch_size
        return {
            "val_loss": total_loss / max(1, seen),
            "val_accuracy": correct / max(1, seen),
        }

    # -- orchestration ------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingResult:
        """Train for the configured number of epochs with early stopping."""
        cfg = self.config.training
        epochs = cfg.epochs
        try:
            for epoch in range(self.start_epoch, epochs):
                started = time.perf_counter()
                train_metrics = self.train_epoch(train_loader, epoch)
                val_metrics = self.validate(val_loader)
                elapsed = time.perf_counter() - started

                lr = float(self.optimizer.param_groups[0]["lr"])
                record: dict[str, float] = {
                    "epoch": float(epoch),
                    **train_metrics,
                    **val_metrics,
                    "lr": lr,
                    "epoch_seconds": round(elapsed, 3),
                }
                self.history.append(record)
                self._append_metrics(record)

                current = self._metric_from(record, cfg.monitor)
                improved = self._is_improvement(current)
                if improved:
                    self.best_metric = current
                    self.best_epoch = epoch
                    self._epochs_without_improvement = 0
                else:
                    self._epochs_without_improvement += 1

                self._advance_scheduler(val_metrics)

                logger.info(
                    "epoch complete", extra={"epoch": epoch, **train_metrics, **val_metrics}
                )
                self._save_checkpoint(epoch, tag="last")

                if improved:
                    self._save_checkpoint(epoch, tag="best")

                if cfg.early_stopping and self._epochs_without_improvement >= cfg.patience:
                    logger.info(
                        "early stopping",
                        extra={"epoch": epoch, "patience": cfg.patience},
                    )
                    break
        except KeyboardInterrupt:
            logger.warning("training interrupted - saving interrupt checkpoint")
            self._save_checkpoint(max(self.start_epoch, self._current_epoch()), tag="interrupt")
            self._interrupted = True

        return TrainingResult(
            history=self.history,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            monitor=self.config.training.monitor,
            best_checkpoint=self.output_dir / "best.pt",
            last_checkpoint=self.output_dir / "last.pt",
            interrupted=self._interrupted,
        )

    def _current_epoch(self) -> int:
        return int(self.history[-1]["epoch"]) if self.history else self.start_epoch

    def _advance_scheduler(self, val_metrics: dict[str, float]) -> None:
        if self.scheduler is None:
            return
        if self._plateau:
            metric = self._metric_from(val_metrics, self.config.training.monitor)
            self.scheduler.step(metric)  # type: ignore[arg-type]
        else:
            self.scheduler.step()  # type: ignore[call-arg]

    def _save_checkpoint(self, epoch: int, *, tag: str) -> Path:
        return save_checkpoint(
            self.output_dir / f"{tag}.pt",
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            global_step=self.global_step,
            best_metric=self.best_metric,
            monitor=self.config.training.monitor,
            config=self.config,
            class_set=self.class_set,
            history=self.history,
            environment=environment_info(),
        )

    def _append_metrics(self, record: dict[str, float]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def resume(self, checkpoint_path: str | Path) -> None:
        """Restore model, optimizer, scheduler, history and best metric from a checkpoint.

        The run directory and configuration are left untouched, so resuming never
        overwrites experiment metadata.
        """
        ckpt_path = Path(checkpoint_path)
        payload = load_checkpoint(ckpt_path, map_location=self.device)
        if payload.get("model_name") != self.config.model.name:
            raise CheckpointError(
                f"checkpoint model '{payload.get('model_name')}' does not match config "
                f"model '{self.config.model.name}'"
            )
        self.model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        self.start_epoch = int(payload["epoch"]) + 1
        self.global_step = int(payload.get("global_step", 0))
        self.best_metric = float(payload["best_metric"])
        self.history = list(payload.get("history", []))
        self.best_epoch = int(payload.get("best_metric_epoch", payload["epoch"]))
        logger.info("resumed checkpoint", extra={"path": str(ckpt_path), "epoch": self.start_epoch})
