"""Thin wrapper for ``python scripts/evaluate.py``."""

from __future__ import annotations

from handwriting_intelligence.cli.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
