"""Command-line entry points for Handwriting Intelligence.

Each module exposes ``main(argv=None) -> int`` and is also registered as a
console script (``hi-train``, ``hi-evaluate``, ...). The thin wrappers in
``scripts/`` simply call these functions.
"""

from __future__ import annotations

__all__ = ["download", "evaluate", "predict", "serve", "train"]
