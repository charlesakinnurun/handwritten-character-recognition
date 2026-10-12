"""``hi-train`` - train a character classifier from a YAML config.

The command is deliberately thin: it wires configuration -> data -> model ->
trainer -> artifact, and leaves all real logic in the reusable modules. Every run
writes, into ``runs/<experiment>/``: ``config.yaml``, ``environment.json``,
``metrics.jsonl``, ``best.pt``, ``last.pt`` and (by default) a serving artifact at
``models/<experiment>/artifact.pt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from handwriting_intelligence import __version__
from handwriting_intelligence.cli._common import (
    add_common_options,
    apply_overrides,
    resolve_resume_path,
)
from handwriting_intelligence.config import Config
from handwriting_intelligence.data import build_dataloaders
from handwriting_intelligence.exceptions import CheckpointError
from handwriting_intelligence.models import build_model, describe_model
from handwriting_intelligence.training.checkpoints import (
    load_checkpoint,
    model_from_checkpoint,
    save_artifact,
)
from handwriting_intelligence.training.trainer import Trainer
from handwriting_intelligence.utils.device import resolve_device
from handwriting_intelligence.utils.logging import configure_logging, get_logger
from handwriting_intelligence.utils.reproducibility import environment_info, seed_everything

logger = get_logger("hi.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hi-train", description="Train a handwritten character recognition model."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--output-dir", default=None, help="Override experiment.output_dir.")
    parser.add_argument(
        "--artifact-dir",
        default="models",
        help="Where to write the serving artifact (default: models/<experiment>/).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a checkpoint path, or 'auto' to use runs/<name>/last.pt.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument(
        "--device", default=None, choices=["auto", "cpu", "cuda", "mps"], help="Override device."
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--no-download", action="store_true", help="Never download; fail if data is missing."
    )
    add_common_options(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_format)

    cfg = apply_overrides(Config.load(args.config), args)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.training.device)
    logger.info(
        "starting run",
        extra={
            "experiment": cfg.experiment.name,
            "dataset": cfg.data.name,
            "model": cfg.model.name,
            "device": str(device),
            "version": __version__,
        },
    )

    loaders = build_dataloaders(cfg, pin_memory=(device.type == "cuda"))
    model = build_model(cfg.model, loaders.class_set.num_classes)
    logger.info("model built", extra=describe_model(model))

    trainer = Trainer(model, cfg, loaders.class_set, device=device)
    resume_path = resolve_resume_path(args.resume, trainer.output_dir)
    if resume_path is not None:
        trainer.resume(resume_path)

    # Persist reproducibility metadata *before* training so an interrupted run is
    # still fully described.
    cfg.dump(trainer.output_dir / "config.yaml")
    (trainer.output_dir / "environment.json").write_text(
        json.dumps(environment_info(), indent=2), encoding="utf-8"
    )

    result = trainer.fit(loaders.train, loaders.val)

    # Select the best model (fall back to last if best was never recorded).
    checkpoint_path = result.best_checkpoint
    if not checkpoint_path.is_file():
        checkpoint_path = result.last_checkpoint
    if not checkpoint_path.is_file():
        raise CheckpointError("training finished without producing a checkpoint")
    best_model = model_from_checkpoint(load_checkpoint(checkpoint_path), device="cpu")

    artifact_dir = Path(args.artifact_dir) / cfg.experiment.name
    artifact_path = save_artifact(
        artifact_dir / "artifact.pt",
        model=best_model,
        class_set=loaders.class_set,
        config=cfg,
        environment=environment_info(),
        training_summary={
            "best_metric": result.best_metric,
            "best_epoch": result.best_epoch,
            "monitor": result.monitor,
            "epochs_run": len(result.history),
            "interrupted": result.interrupted,
            "dataset": loaders.meta,
        },
    )

    summary = {
        "experiment": cfg.experiment.name,
        "dataset": cfg.data.name,
        "model": cfg.model.name,
        "num_classes": loaders.class_set.num_classes,
        "best_metric": result.best_metric,
        "best_epoch": result.best_epoch,
        "monitor": result.monitor,
        "epochs_run": len(result.history),
        "interrupted": result.interrupted,
        "history": result.history,
        "artifact": str(artifact_path),
        "checkpoint": str(checkpoint_path),
        "class_set": loaders.class_set.to_metadata(),
        "environment": environment_info(),
    }
    (trainer.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTraining complete")
    print(f"  experiment   : {cfg.experiment.name}")
    print(f"  best {result.monitor:<12}: {result.best_metric:.4f} (epoch {result.best_epoch})")
    print(f"  epochs run   : {len(result.history)}")
    print(f"  checkpoint   : {checkpoint_path}")
    print(f"  artifact     : {artifact_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
