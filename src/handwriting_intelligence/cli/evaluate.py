"""``hi-evaluate`` - evaluate a trained artifact on a held-out split.

Produces a metrics JSON, a confusion-matrix PNG and example grids of correct and
misclassified samples. The preprocessing is taken from the artifact so evaluation
uses exactly the same pipeline as serving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from handwriting_intelligence.cli._common import add_common_options
from handwriting_intelligence.config import Config
from handwriting_intelligence.data import build_dataloaders
from handwriting_intelligence.evaluation import (
    baseline_report,
    confidence_analysis,
    evaluate_model,
    format_metrics_table,
    save_confusion_matrix_png,
    save_examples_grid,
)
from handwriting_intelligence.exceptions import ConfigurationError
from handwriting_intelligence.training.checkpoints import (
    artifact_to_checkpoint_payload,
    load_artifact,
    model_from_checkpoint,
)
from handwriting_intelligence.utils.device import resolve_device
from handwriting_intelligence.utils.logging import configure_logging, get_logger

logger = get_logger("hi.evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hi-evaluate", description="Evaluate a trained inference artifact."
    )
    parser.add_argument(
        "--artifact", default="models/mnist_cnn/artifact.pt", help="Path to artifact.pt."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config describing the dataset/splits (must match the artifact's dataset).",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--report-dir", default=None, help="Where to write reports.")
    parser.add_argument("--examples", type=int, default=16, help="Examples per grid (0 disables).")
    add_common_options(parser)
    return parser


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_format)

    artifact = load_artifact(args.artifact)
    cfg = Config.load(args.config)
    if cfg.data.name != artifact["class_set"]["name"]:
        raise ConfigurationError(
            f"config dataset '{cfg.data.name}' does not match artifact dataset "
            f"'{artifact['class_set']['name']}'"
        )
    # Use the artifact's preprocessing so evaluation matches serving exactly.
    data = cfg.to_dict()
    data["preprocessing"] = artifact["preprocessing"]
    cfg = Config.from_dict(data).resolved()

    device = resolve_device(cfg.training.device)
    loaders = build_dataloaders(cfg, pin_memory=(device.type == "cuda"))
    model = model_from_checkpoint(artifact_to_checkpoint_payload(artifact), device=device)
    loader = loaders.test if args.split == "test" else loaders.val

    from handwriting_intelligence.models.base import model_size_bytes

    result = evaluate_model(
        model,
        loader,
        labels=list(loaders.class_set.labels),
        device=device,
        model_size_bytes=model_size_bytes(model),
    )

    report_dir = Path(args.report_dir) if args.report_dir else Path("reports") / cfg.experiment.name
    report_dir.mkdir(parents=True, exist_ok=True)

    probs = _softmax(result.logits)
    conf = confidence_analysis(result.y_true, result.y_pred, probs)

    # Majority-class baseline uses the *training* split distribution.
    train_labels = getattr(loaders.train.dataset, "labels", np.array([]))
    baseline = baseline_report(train_labels, result.y_true)

    payload = {
        "artifact": str(args.artifact),
        "dataset": cfg.data.name,
        "split": args.split,
        "official_test": loaders.meta.get("official_test"),
        "model": artifact.get("model_name"),
        "metrics": result.to_dict(),
        "confidence_analysis": conf,
        "baseline_majority": baseline,
    }
    metrics_path = report_dir / f"metrics_{args.split}.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    save_confusion_matrix_png(
        result.metrics,
        report_dir / f"confusion_matrix_{args.split}.png",
        title=f"{cfg.experiment.name} - {args.split}",
    )

    if args.examples > 0:
        images = np.stack(
            [loader.dataset.raw_image(i) for i in range(len(loader.dataset))]  # type: ignore[arg-type, attr-defined]
        )
        save_examples_grid(
            images,
            result.y_true,
            result.y_pred,
            probs[np.arange(len(result.y_pred)), result.y_pred],
            report_dir / f"examples_correct_{args.split}.png",
            correct=True,
            count=args.examples,
            title=f"correct ({cfg.data.name})",
        )
        save_examples_grid(
            images,
            result.y_true,
            result.y_pred,
            probs[np.arange(len(result.y_pred)), result.y_pred],
            report_dir / f"examples_incorrect_{args.split}.png",
            correct=False,
            count=args.examples,
            title=f"misclassified ({cfg.data.name})",
        )

    print(f"\nEvaluation - {cfg.experiment.name} [{args.split}]")
    print(format_metrics_table(result.metrics, top_n_classes=10))
    print(f"\nmajority-class baseline accuracy : {baseline['accuracy']:.4f}")
    print(f"model size                      : {result.model_size_bytes / 1024:.1f} KiB")
    print(f"inference (single-image est.)   : {result.inference_ms_per_image:.3f} ms/image")
    print(
        f"mean confidence (correct/wrong) : {conf['mean_confidence_correct']:.4f} / {conf['mean_confidence_incorrect']:.4f}"
    )
    print(f"\nreports written to {report_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
