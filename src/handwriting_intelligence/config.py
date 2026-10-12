"""Experiment configuration: schema, validation and (de)serialization.

Configuration is *data*, not code. Every hyperparameter lives in a YAML file that
is parsed into the strongly-typed :class:`Config` model below. Invalid
configuration fails at load time with a precise error rather than halfway through
a training run.

Example:
    >>> from handwriting_intelligence.config import load_config
    >>> cfg = load_config("configs/mnist_cnn.yaml")  # doctest: +SKIP
    >>> cfg.data.name
    'mnist'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from handwriting_intelligence.classes import SUPPORTED_DATASETS, build_class_set
from handwriting_intelligence.exceptions import ConfigurationError

ModelName = Literal["cnn_baseline", "cnn_improved", "crnn"]
OptimizerName = Literal["sgd", "adam", "adamw"]
SchedulerName = Literal["none", "cosine", "step", "plateau"]
DevicePreference = Literal["auto", "cpu", "cuda", "mps"]
MixPrecision = bool | Literal["auto"]
InvertPolicy = Literal["auto", "always", "never"]


class _Base(BaseModel):
    """Base model that forbids unknown keys so typos are caught immediately."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExperimentConfig(_Base):
    """Identity and provenance of a single run."""

    name: str = Field(default="experiment", min_length=1)
    output_dir: Path = Path("runs")
    notes: str = ""


class DataConfig(_Base):
    """Dataset selection and loading behaviour."""

    name: str = "mnist"
    root: Path = Path("data")
    download: bool = True
    val_split: float = Field(default=0.1, ge=0.0, lt=1.0)
    batch_size: int = Field(default=128, ge=1)
    eval_batch_size: int = Field(default=512, ge=1)
    num_workers: int = Field(default=0, ge=0)
    max_train_samples: int | None = Field(default=None, ge=1)
    max_val_samples: int | None = Field(default=None, ge=1)
    max_test_samples: int | None = Field(default=None, ge=1)
    cache: bool = True

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if value not in SUPPORTED_DATASETS:
            supported = ", ".join(SUPPORTED_DATASETS)
            raise ValueError(f"unknown dataset '{value}'; supported: {supported}")
        return value


class PreprocessConfig(_Base):
    """Image preprocessing shared by training and inference.

    ``invert="auto"`` detects polarity (dark-on-light vs light-on-dark) and
    normalizes to a light stroke on a dark background, matching MNIST/EMNIST.
    """

    image_size: int = Field(default=28, ge=8, le=256)
    invert: InvertPolicy = "auto"
    normalize: bool = True
    mean: float = 0.1307
    std: float = Field(default=0.3081, gt=0.0)
    crop_to_content: bool = False
    threshold: int | None = Field(default=None, ge=0, le=255)
    denoise: bool = False


class ModelConfig(_Base):
    """Architecture selection and shape parameters."""

    name: ModelName = "cnn_baseline"
    num_classes: int | None = Field(default=None, ge=2)
    dropout: float = Field(default=0.25, ge=0.0, lt=1.0)
    channels: list[int] = Field(default_factory=lambda: [32, 64])
    hidden_dim: int = Field(default=128, ge=1)
    rnn_hidden: int = Field(default=256, ge=1)
    rnn_layers: int = Field(default=2, ge=1)

    @field_validator("channels")
    @classmethod
    def _validate_channels(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("`channels` must contain at least one block width")
        if any(c <= 0 for c in value):
            raise ValueError("all channel widths must be positive")
        return value


class TrainingConfig(_Base):
    """Optimization, scheduling and reproducibility knobs."""

    epochs: int = Field(default=5, ge=1)
    optimizer: OptimizerName = "adam"
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    scheduler: SchedulerName = "none"
    step_size: int = Field(default=3, ge=1)
    gamma: float = Field(default=0.5, gt=0.0)
    patience: int = Field(default=3, ge=1)
    early_stopping: bool = True
    grad_clip: float | None = Field(default=None, gt=0.0)
    mixed_precision: MixPrecision = "auto"
    device: DevicePreference = "auto"
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    monitor: Literal["val_loss", "val_accuracy"] = "val_loss"
    augment: bool = False
    augment_rotation: float = Field(default=10.0, ge=0.0)
    augment_translate: float = Field(default=0.1, ge=0.0, lt=1.0)
    augment_scale: float = Field(default=0.1, ge=0.0, lt=1.0)
    log_every: int = Field(default=50, ge=0, description="Log every N steps; 0 disables.")
    resume: Path | None = None


class Config(_Base):
    """Top-level experiment configuration."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessConfig = Field(default_factory=PreprocessConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # -- validation ---------------------------------------------------------
    @model_validator(mode="after")
    def _validate_consistency(self) -> Config:
        # The number of outputs is always dictated by the dataset's label convention
        # (including intentional placeholder indices such as EMNIST letters index 0).
        inferred = build_class_set(self.data.name).num_classes
        if self.model.num_classes is not None and self.model.num_classes != inferred:
            raise ConfigurationError(
                f"model.num_classes={self.model.num_classes} does not match dataset "
                f"'{self.data.name}' which has {inferred} classes"
            )
        return self

    # -- derived helpers ----------------------------------------------------
    @property
    def num_classes(self) -> int:
        """Number of target classes (always derived from the dataset)."""
        return build_class_set(self.data.name).num_classes

    @property
    def run_dir(self) -> Path:
        """Directory that holds the artifacts for this run."""
        return Path(self.experiment.output_dir) / self.experiment.name

    def resolved(self) -> Config:
        """Return a copy with ``model.num_classes`` filled in from the dataset."""
        data = self.model_dump()
        data["model"]["num_classes"] = self.num_classes
        return Config.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dump (paths become strings)."""
        return self.model_dump(mode="json")

    # -- IO ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Validate a plain mapping into a :class:`Config`."""
        try:
            return cls.model_validate(data)
        except ConfigurationError:
            raise
        except Exception as exc:  # pydantic ValidationError
            raise ConfigurationError(f"invalid configuration:\n{exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load and validate a YAML configuration file."""
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise ConfigurationError(f"configuration file not found: {cfg_path}")
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"could not parse YAML in {cfg_path}: {exc}") from exc
        if raw is None:
            raise ConfigurationError(f"configuration file is empty: {cfg_path}")
        if not isinstance(raw, dict):
            raise ConfigurationError(f"configuration root must be a mapping: {cfg_path}")
        return cls.from_dict(raw)

    def dump(self, path: str | Path) -> Path:
        """Write the configuration to ``path`` as YAML (creating parent dirs)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        return out


def load_config(path: str | Path) -> Config:
    """Module-level convenience wrapper around :meth:`Config.load`."""
    return Config.load(path)
