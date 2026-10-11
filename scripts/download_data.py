"""Thin wrapper for ``python scripts/download_data.py``."""

from __future__ import annotations

from handwriting_intelligence.cli.download import main

if __name__ == "__main__":
    raise SystemExit(main())
