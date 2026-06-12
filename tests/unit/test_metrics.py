"""Unit tests for metrics helpers."""

from __future__ import annotations

import pytest

from src.metrics import classification_metrics


def test_classification_metrics_basic():
    scores = classification_metrics(
        true_positives=80,
        false_positives=20,
        true_negatives=900,
        false_negatives=20,
    )
    assert scores["precision"] == 0.8
    assert scores["recall"] == 0.8
    assert scores["f1"] == pytest.approx(0.8)


def test_classification_metrics_zero_division():
    scores = classification_metrics(
        true_positives=0,
        false_positives=0,
        true_negatives=10,
        false_negatives=0,
    )
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0
    assert scores["f1"] == 0.0
