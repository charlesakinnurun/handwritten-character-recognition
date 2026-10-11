"""Thin wrapper for ``python scripts/predict.py``."""

from __future__ import annotations

from handwriting_intelligence.cli.predict import main

if __name__ == "__main__":
    raise SystemExit(main())
