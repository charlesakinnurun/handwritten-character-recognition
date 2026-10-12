"""Structured logging.

The project emits **structured** logs. In ``json`` mode each record is a single
JSON object with a stable set of fields, which is what a log aggregator expects in
production. In ``text`` mode a compact human-readable line is emitted for local
development.

A ``request_id`` is carried in a :class:`contextvars.ContextVar` so that every log
line emitted while handling an HTTP request is automatically correlated, without
threading the identifier through every function signature.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

#: Correlation identifier for the current request/task (empty outside a request).
_request_id: ContextVar[str] = ContextVar("request_id", default="")

#: Attributes present on every LogRecord; anything else is treated as structured
#: "extra" data and included verbatim in the JSON payload.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


def set_request_id(value: str) -> None:
    """Bind a request identifier to the current context."""
    _request_id.set(value)


def get_request_id() -> str:
    """Return the request identifier bound to the current context (or "")."""
    return _request_id.get()


class _JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if rid := _request_id.get():
            payload["request_id"] = rid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """Compact single-line format for local development."""

    def format(self, record: logging.LogRecord) -> str:
        rid = _request_id.get()
        prefix = f"[{rid}] " if rid else ""
        base = (
            f"{datetime.fromtimestamp(record.created, tz=UTC):%H:%M:%S} "
            f"{record.levelname:<7} {record.name}: {prefix}{record.getMessage()}"
        )
        extras = {
            k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
        }
        if extras:
            base += " " + " ".join(f"{k}={v!r}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str | int = "INFO", fmt: str = "json") -> None:
    """Configure the root logger exactly once.

    Args:
        level: Logging level name or numeric value.
        fmt: ``"json"`` for structured output, anything else for text.
    """
    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_JsonFormatter() if fmt == "json" else _TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Third-party libraries are noisy at INFO; keep them at WARNING.
    for noisy in ("uvicorn.access", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
