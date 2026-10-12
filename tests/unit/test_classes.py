"""Unit tests for dataset class-set mappings."""

from __future__ import annotations

import pytest

from handwriting_intelligence.classes import (
    ClassSet,
    build_class_set,
    emnist_balanced_labels,
    emnist_byclass_labels,
    emnist_letters_labels,
)
from handwriting_intelligence.exceptions import ConfigurationError


def test_mnist_class_set() -> None:
    cs = build_class_set("mnist")
    assert cs.num_classes == 10
    assert cs.labels == tuple("0123456789")
    assert cs.kind == "digits"
    assert cs.label_of(3) == "3"


def test_emnist_balanced_class_set() -> None:
    cs = build_class_set("emnist_balanced")
    assert cs.num_classes == 47
    assert cs.labels[0] == "0"
    assert cs.labels[9] == "9"
    assert cs.labels[10] == "A"
    assert cs.labels[35] == "Z"
    # Merged lowercase letters are absent; their uppercase form is present.
    assert "C" in cs.labels
    assert "c" not in cs.labels
    assert cs.kind == "alphanumeric"


def test_emnist_balanced_matches_reference_rule() -> None:
    assert emnist_balanced_labels() == tuple(sorted(emnist_balanced_labels()))
    assert len(emnist_balanced_labels()) == 47
    assert set(emnist_balanced_labels()) - set(__import__("string").digits) == set(
        emnist_byclass_labels()
    ) - set(__import__("string").digits) - set("cijklmopsuvwxyz")


def test_emnist_letters_placeholder() -> None:
    cs = build_class_set("emnist_letters")
    assert cs.num_classes == 27
    assert cs.has_placeholder is True
    assert cs.labels[0] == "N/A"
    assert cs.labels[1] == "a"
    assert cs.labels[26] == "z"
    assert emnist_letters_labels()[0] == "N/A"


def test_emnist_byclass_62_classes() -> None:
    assert build_class_set("emnist_byclass").num_classes == 62


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ConfigurationError, match="unknown dataset"):
        build_class_set("cifar10")


def test_label_of_out_of_range() -> None:
    cs = build_class_set("mnist")
    with pytest.raises(IndexError):
        cs.label_of(10)
    with pytest.raises(IndexError):
        cs.label_of(-1)


def test_metadata_roundtrip() -> None:
    cs = build_class_set("emnist_balanced")
    restored = ClassSet.from_metadata(cs.to_metadata())
    assert restored == cs


def test_metadata_invalid_raises() -> None:
    with pytest.raises(ConfigurationError):
        ClassSet.from_metadata({"name": "x"})
