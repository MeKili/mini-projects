"""Tests for classification metrics."""

import math

import pytest

from classification_metrics.core import (
    ConfusionMatrix,
    accuracy,
    confusion_matrix,
    f1,
    precision,
    recall,
    roc_auc,
)


def test_confusion_matrix_all_correct() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    cm = confusion_matrix(y_true, y_pred)
    assert cm == ConfusionMatrix(
        true_negatives=2, false_positives=0, false_negatives=0, true_positives=2
    )


def test_confusion_matrix_with_errors() -> None:
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 0]
    cm = confusion_matrix(y_true, y_pred)
    assert cm.true_negatives == 2
    assert cm.false_positives == 1
    assert cm.false_negatives == 1
    assert cm.true_positives == 2


def test_confusion_matrix_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        confusion_matrix([0, 1], [0])


def test_confusion_matrix_invalid_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        confusion_matrix([0, 2], [0, 1])


def test_accuracy_perfect() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_half() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    assert accuracy(y_true, y_pred) == 0.0


def test_accuracy_75_percent() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    assert accuracy(y_true, y_pred) == 0.75


def test_accuracy_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        accuracy([], [])


def test_precision_perfect() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    assert precision(y_true, y_pred) == 1.0


def test_precision_all_false_positives() -> None:
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    assert precision(y_true, y_pred) == 0.0


def test_precision_no_positive_predictions() -> None:
    y_true = [0, 1, 1]
    y_pred = [0, 0, 0]
    assert precision(y_true, y_pred) == 0.0


def test_precision_partial() -> None:
    y_true = [0, 1, 1, 1, 0]
    y_pred = [1, 1, 1, 0, 1]
    assert precision(y_true, y_pred) == 0.5


def test_recall_perfect() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    assert recall(y_true, y_pred) == 1.0


def test_recall_all_false_negatives() -> None:
    y_true = [1, 1, 1]
    y_pred = [0, 0, 0]
    assert recall(y_true, y_pred) == 0.0


def test_recall_no_positive_truth() -> None:
    y_true = [0, 0, 0]
    y_pred = [1, 0, 1]
    assert recall(y_true, y_pred) == 0.0


def test_recall_partial() -> None:
    y_true = [0, 1, 1, 1, 0]
    y_pred = [1, 1, 1, 0, 1]
    assert recall(y_true, y_pred) == (2 / 3)


def test_f1_perfect() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    assert f1(y_true, y_pred) == 1.0


def test_f1_all_wrong() -> None:
    y_true = [0, 1]
    y_pred = [1, 0]
    assert f1(y_true, y_pred) == 0.0


def test_f1_partial() -> None:
    y_true = [1, 1, 1, 0]
    y_pred = [1, 1, 0, 0]
    p = 1.0
    r = 2 / 3
    expected = 2 * (p * r) / (p + r)
    assert math.isclose(f1(y_true, y_pred), expected)


def test_roc_auc_perfect() -> None:
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(y_true, y_scores) == 1.0


def test_roc_auc_worst() -> None:
    y_true = [0, 0, 1, 1]
    y_scores = [0.9, 0.8, 0.2, 0.1]
    assert roc_auc(y_true, y_scores) == 0.0


def test_roc_auc_random() -> None:
    y_true = [0, 1, 0, 1]
    y_scores = [0.3, 0.7, 0.4, 0.6]
    result = roc_auc(y_true, y_scores)
    assert 0.0 <= result <= 1.0


def test_roc_auc_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        roc_auc([0, 1], [0.5])


def test_roc_auc_invalid_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        roc_auc([0, 2], [0.5, 0.5])


def test_roc_auc_no_positives() -> None:
    with pytest.raises(ValueError, match="both positive and negative"):
        roc_auc([0, 0], [0.5, 0.5])


def test_roc_auc_no_negatives() -> None:
    with pytest.raises(ValueError, match="both positive and negative"):
        roc_auc([1, 1], [0.5, 0.5])


def test_roc_auc_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        roc_auc([], [])
