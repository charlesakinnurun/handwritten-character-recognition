"""Thin wrapper so ``python scripts/train.py`` works without console scripts.

All logic lives in :mod:`handwriting_intelligence.cli.train`.
"""

from __future__ import annotations

from handwriting_intelligence.cli.train import main

if __name__ == "__main__":
    raise SystemExit(main())
