"""Filesystem path helpers.

All paths in the codebase are :class:`pathlib.Path` objects. This module centralises
the few places where we need to locate the project root or create directories.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root.

    The package lives at ``<root>/src/handwriting_intelligence``; three parents up
    from this file is the repository root.
    """
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a :class:`Path`."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
