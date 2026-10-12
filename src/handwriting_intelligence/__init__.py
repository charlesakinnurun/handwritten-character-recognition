"""Handwriting Intelligence - production-grade handwritten character recognition.

The package is organised into independent, individually testable layers:

* :mod:`handwriting_intelligence.config` - validated experiment configuration.
* :mod:`handwriting_intelligence.data` - dataset loaders, class maps, transforms.
* :mod:`handwriting_intelligence.models` - model registry and architectures.
* :mod:`handwriting_intelligence.training` - training loop, checkpoints, metrics.
* :mod:`handwriting_intelligence.inference` - preprocessing + predictor abstraction.
* :mod:`handwriting_intelligence.api` - FastAPI serving application.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
