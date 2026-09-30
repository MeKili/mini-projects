"""Classification metrics: precision, recall, F1, ROC-AUC, confusion matrix."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfusionMatrix:
    """2x2 confusion matrix for binary classification."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionMatrix:
    """Return a 2x2 confusion matrix from binary labels (0 or 1).

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        ConfusionMatrix with tn, fp, fn, tp counts.

    Raises:
        ValueError: If lengths don't match or labels are not binary (0 or 1).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tn = fp = fn = tp = 0
    for true, pred in zip(y_true, y_pred, strict=True):
        if true not in (0, 1) or pred not in (0, 1):
            raise ValueError("labels must be binary (0 or 1)")
        if true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        else:
            tp += 1

    return ConfusionMatrix(
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
    )


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return accuracy: (TP + TN) / (TP + TN + FP + FN)."""
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    cm = confusion_matrix(y_true, y_pred)
    total = cm.true_positives + cm.true_negatives + cm.false_positives + cm.false_negatives
    if total == 0:
        return 0.0
    return (cm.true_positives + cm.true_negatives) / total


def precision(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return precision: TP / (TP + FP).

    Returns 0.0 if no positive predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    denom = cm.true_positives + cm.false_positives
    if denom == 0:
        return 0.0
    return cm.true_positives / denom


def recall(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return recall (sensitivity): TP / (TP + FN).

    Returns 0.0 if no positive ground truth.
    """
    cm = confusion_matrix(y_true, y_pred)
    denom = cm.true_positives + cm.false_negatives
    if denom == 0:
        return 0.0
    return cm.true_positives / denom


def f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return F1 score: 2 * (precision * recall) / (precision + recall).

    Returns 0.0 if both precision and recall are 0.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    if denom == 0:
        return 0.0
    return 2 * (p * r) / denom


def roc_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    """Return ROC-AUC score by counting concordant/discordant pairs.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_scores: Predicted probability scores (not binary predictions).

    Returns:
        ROC-AUC score in [0, 1].

    Raises:
        ValueError: If lengths don't match, no positive or negative samples.
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")

    if len(y_true) == 0:
        raise ValueError("y_true and y_scores must be non-empty")

    # Separate positives and negatives
    pos_scores = []
    neg_scores = []
    for true, score in zip(y_true, y_scores, strict=True):
        if true not in (0, 1):
            raise ValueError("y_true must be binary (0 or 1)")
        if true == 1:
            pos_scores.append(score)
        else:
            neg_scores.append(score)

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError("y_true must have both positive and negative samples")

    # Count concordant pairs
    concordant = 0
    total = 0
    for pos in pos_scores:
        for neg in neg_scores:
            total += 1
            if pos > neg:
                concordant += 1

    return concordant / total if total > 0 else 0.0
