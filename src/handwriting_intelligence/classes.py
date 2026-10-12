"""Class-set definitions for the supported datasets.

A :class:`ClassSet` is the single source of truth that maps a model's integer
output index to a human-readable label. It is serialized into every checkpoint and
inference artifact so that a prediction can *never* be silently reinterpreted
against the wrong label convention.

Why this module exists
----------------------
MNIST and the different EMNIST splits do **not** share label conventions:

* **MNIST / EMNIST digits** - labels ``0..9`` are exactly the characters ``'0'..'9'``.
* **EMNIST letters** - the raw labels are ``1..26`` for ``'a'..'z'`` and index
  ``0`` is *unused* (torchvision exposes ``["N/A", "a", ..., "z"]``). A naive
  ``num_classes=26`` silently shifts every prediction by one.
* **EMNIST balanced / bymerge** - 47 classes. Uppercase ``A-Z`` and digits are
  kept, but 15 lowercase letters that are visually confusable with their uppercase
  counterparts (``c i j k l m o p s u v w x y z``) are *merged* into uppercase.
* **EMNIST byclass** - all 62 alphanumeric classes.

The exact EMNIST merge rule below mirrors the reference implementation used by
``torchvision.datasets.EMNIST`` (``_merged_classes``), so indices agree with the
official IDX files distributed by NIST.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Literal

from handwriting_intelligence.exceptions import ConfigurationError

#: Lowercase letters that the EMNIST ``balanced`` / ``bymerge`` splits merge with
#: their uppercase form. Mirrors ``torchvision.datasets.EMNIST._merged_classes``.
EMNIST_MERGED_CLASSES: frozenset[str] = frozenset(
    {"c", "i", "j", "k", "l", "m", "o", "p", "s", "u", "v", "w", "x", "y", "z"}
)

_ALL_ALPHANUMERIC: set[str] = set(string.digits + string.ascii_letters)

ClassKind = Literal["digits", "letters", "alphanumeric"]


@dataclass(frozen=True, slots=True)
class ClassSet:
    """An ordered mapping from model output index to display label.

    Attributes:
        name: Dataset identifier the class set belongs to (e.g. ``"emnist_balanced"``).
        labels: Ordered labels; ``labels[i]`` is the character for class index ``i``.
        kind: Semantic family of the character set.
        description: Human-readable explanation, safe to surface in an API / model card.
        has_placeholder: True when index ``0`` is an unused placeholder (EMNIST letters).
    """

    name: str
    labels: tuple[str, ...]
    kind: ClassKind
    description: str
    has_placeholder: bool = False

    @property
    def num_classes(self) -> int:
        """Number of output classes, including any placeholder index."""
        return len(self.labels)

    def label_of(self, index: int) -> str:
        """Return the display label for a class index.

        Raises:
            IndexError: if ``index`` is outside the valid range.
        """
        if not 0 <= index < len(self.labels):
            raise IndexError(f"class index {index} out of range [0, {len(self.labels)})")
        return self.labels[index]

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-serializable description persisted into artifacts."""
        return {
            "name": self.name,
            "labels": list(self.labels),
            "kind": self.kind,
            "description": self.description,
            "has_placeholder": self.has_placeholder,
        }

    @classmethod
    def from_metadata(cls, data: dict[str, object]) -> ClassSet:
        """Reconstruct a :class:`ClassSet` from :meth:`to_metadata` output."""
        try:
            raw_labels = data["labels"]
            if not isinstance(raw_labels, (list, tuple)):
                raise TypeError("'labels' must be a list or tuple")
            return cls(
                name=str(data["name"]),
                labels=tuple(str(x) for x in raw_labels),
                kind=data["kind"],  # type: ignore[arg-type]
                description=str(data.get("description", "")),
                has_placeholder=bool(data.get("has_placeholder", False)),
            )
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(f"invalid class-set metadata: {exc}") from exc


def emnist_balanced_labels() -> tuple[str, ...]:
    """Return the 47 labels of the EMNIST ``balanced`` split in index order.

    Ordering follows ``sorted(digits + ascii_letters - merged)``: digits ``0-9``,
    then uppercase ``A-Z``, then the 11 surviving lowercase letters
    (``a b d e f g h n q r t``).
    """
    return tuple(sorted(_ALL_ALPHANUMERIC - EMNIST_MERGED_CLASSES))


def emnist_bymerge_labels() -> tuple[str, ...]:
    """Return the 47 labels of the EMNIST ``bymerge`` split.

    Identical class set to ``balanced``; the two splits differ only in how many
    samples per class they contain, not in their label space.
    """
    return emnist_balanced_labels()


def emnist_byclass_labels() -> tuple[str, ...]:
    """Return the 62 labels (digits + upper + lower) of the EMNIST ``byclass`` split."""
    return tuple(sorted(_ALL_ALPHANUMERIC))


def emnist_letters_labels() -> tuple[str, ...]:
    """Return the ``letters`` labels: index ``0`` is a placeholder, ``1..26`` are ``a..z``."""
    return ("N/A", *string.ascii_lowercase)


#: Registry of dataset name -> factory constructing its :class:`ClassSet`.
_CLASS_SET_FACTORIES: dict[str, tuple[ClassKind, str, bool]] = {
    "mnist": ("digits", "MNIST handwritten digits 0-9.", False),
    "emnist_digits": ("digits", "EMNIST digits split, 0-9 (10 classes).", False),
    "emnist_letters": (
        "letters",
        "EMNIST letters split. NOTE: index 0 is an unused placeholder; a-z map to 1-26.",
        True,
    ),
    "emnist_balanced": (
        "alphanumeric",
        "EMNIST balanced split: digits, uppercase A-Z, and 11 non-merged lowercase letters.",
        False,
    ),
    "emnist_bymerge": (
        "alphanumeric",
        "EMNIST bymerge split: same 47-class space as balanced.",
        False,
    ),
    "emnist_byclass": (
        "alphanumeric",
        "EMNIST byclass split: all 62 alphanumeric classes (0-9, A-Z, a-z).",
        False,
    ),
}


def build_class_set(dataset_name: str) -> ClassSet:
    """Build the :class:`ClassSet` for a dataset name.

    Args:
        dataset_name: One of the keys in :data:`SUPPORTED_DATASETS`.

    Raises:
        ConfigurationError: if the name is unknown.
    """
    if dataset_name not in _CLASS_SET_FACTORIES:
        supported = ", ".join(sorted(_CLASS_SET_FACTORIES))
        raise ConfigurationError(f"unknown dataset '{dataset_name}'; supported: {supported}")

    kind, description, has_placeholder = _CLASS_SET_FACTORIES[dataset_name]
    if dataset_name in {"mnist", "emnist_digits"}:
        labels: tuple[str, ...] = tuple(string.digits)
    elif dataset_name == "emnist_letters":
        labels = emnist_letters_labels()
    elif dataset_name in {"emnist_balanced", "emnist_bymerge"}:
        labels = emnist_balanced_labels()
    elif dataset_name == "emnist_byclass":
        labels = emnist_byclass_labels()
    else:  # pragma: no cover - unreachable, guarded above
        raise ConfigurationError(f"no label builder for '{dataset_name}'")

    return ClassSet(
        name=dataset_name,
        labels=labels,
        kind=kind,
        description=description,
        has_placeholder=has_placeholder,
    )


#: Dataset names recognised by the loaders and the class-set registry.
SUPPORTED_DATASETS: tuple[str, ...] = tuple(sorted(_CLASS_SET_FACTORIES))
