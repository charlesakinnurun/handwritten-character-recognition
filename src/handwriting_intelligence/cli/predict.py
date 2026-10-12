"""``hi-predict`` - classify a single image with a trained artifact."""

from __future__ import annotations

import argparse
import json

from handwriting_intelligence.cli._common import add_common_options
from handwriting_intelligence.inference.predictor import load_predictor
from handwriting_intelligence.utils.device import resolve_device
from handwriting_intelligence.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hi-predict", description="Classify a handwritten character image."
    )
    parser.add_argument("--artifact", default="models/mnist_cnn/artifact.pt")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    add_common_options(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_format)
    predictor = load_predictor(args.artifact, device=resolve_device(args.device))
    prediction = predictor.predict(args.image, top_k=args.top_k)
    print(json.dumps(prediction.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
