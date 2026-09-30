"""Classification metrics for binary and multiclass evaluation."""

from __future__ import annotations

from classification_metrics.core import (
    accuracy,
    confusion_matrix,
    f1,
    precision,
    recall,
    roc_auc,
)

__all__ = [
    "confusion_matrix",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]
