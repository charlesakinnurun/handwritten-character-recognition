"""Typed exceptions used across the project.

The project fails loudly and specifically: every error raised at a boundary has a
dedicated exception type so callers (CLI, API, tests) can react precisely instead
of catching broad ``Exception`` objects.
"""

from __future__ import annotations


class HandwritingIntelligenceError(Exception):
    """Base class for all project-specific errors."""


class ConfigurationError(HandwritingIntelligenceError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


class DatasetError(HandwritingIntelligenceError):
    """Raised when a dataset cannot be located, downloaded, read or validated."""


class DatasetNotFoundError(DatasetError):
    """Raised when the requested dataset is unavailable locally and downloads are off."""


class ModelError(HandwritingIntelligenceError):
    """Raised for invalid model construction or unsupported architectures."""


class CheckpointError(HandwritingIntelligenceError):
    """Raised when a checkpoint is missing, corrupt or incompatible."""


class ArtifactError(HandwritingIntelligenceError):
    """Raised when a serialized inference artifact cannot be loaded or is incomplete."""


class ImageValidationError(HandwritingIntelligenceError):
    """Raised when an uploaded image fails validation (type, size, dimensions)."""


class SchemaMismatchError(HandwritingIntelligenceError):
    """Raised when an artifact's class set does not match the requested dataset."""
