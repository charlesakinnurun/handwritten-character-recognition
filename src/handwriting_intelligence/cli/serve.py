"""``hi-serve`` - run the FastAPI inference service with uvicorn."""

from __future__ import annotations

import argparse

from handwriting_intelligence.api.dependencies import get_settings
from handwriting_intelligence.cli._common import add_common_options
from handwriting_intelligence.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hi-serve", description="Serve the inference API.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (development).")
    add_common_options(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_format)
    settings = get_settings()
    import uvicorn

    uvicorn.run(
        "handwriting_intelligence.api.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
