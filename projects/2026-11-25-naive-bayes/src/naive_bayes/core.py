"""Gaussian Naive Bayes classifier for binary and multiclass classification."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class _Gaussian:
    mean: float
    variance: float


@dataclass
class _ClassStats:
    count: int
    feature_stats: dict[int, _Gaussian]


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier with fit and predict methods."""

    def __init__(self) -> None:
        self.class_stats: dict[int, _ClassStats] = {}
        self.classes: list[int] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> GaussianNaiveBayes:
        """Fit the classifier on training data."""
        if len(X) == 0:
            raise ValueError("training data cannot be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X[0]) == 0:
            raise ValueError("features cannot be empty")

        feature_count = len(X[0])
        self.class_stats = {}

        for class_label, features in zip(y, X, strict=True):
            if len(features) != feature_count:
                raise ValueError("all samples must have the same number of features")

            if class_label not in self.class_stats:
                self.class_stats[class_label] = _ClassStats(count=0, feature_stats={})

            self.class_stats[class_label].count += 1
            for feature_idx, _value in enumerate(features):
                self.class_stats[class_label].feature_stats[feature_idx] = None  # type: ignore

        self.classes = sorted(self.class_stats.keys())

        for class_label in self.classes:
            class_samples = [X[i] for i in range(len(y)) if y[i] == class_label]
            for feature_idx in range(feature_count):
                values = [sample[feature_idx] for sample in class_samples]
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                variance = max(variance, 1e-9)
                self.class_stats[class_label].feature_stats[feature_idx] = _Gaussian(
                    mean=mean, variance=variance
                )

        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Predict class labels for samples in X."""
        return [self.classes[i] for i in [self._predict_single(x) for x in X]]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[dict[int, float]]:
        """Return prediction probabilities for each class."""
        result = []
        for sample in X:
            probs = self._predict_proba_single(sample)
            result.append(probs)
        return result

    def _predict_single(self, sample: Sequence[float]) -> int:
        probs = self._predict_proba_single(sample)
        return max(self.classes, key=lambda c: probs[c])

    def _predict_proba_single(self, sample: Sequence[float]) -> dict[int, float]:
        probs: dict[int, float] = {}
        total_samples = sum(stats.count for stats in self.class_stats.values())

        for class_label in self.classes:
            prior = self.class_stats[class_label].count / total_samples
            likelihood = 1.0

            for feature_idx, value in enumerate(sample):
                gaussian = self.class_stats[class_label].feature_stats[feature_idx]
                numerator = math.exp(-((value - gaussian.mean) ** 2) / (2 * gaussian.variance))
                denominator = math.sqrt(2 * math.pi * gaussian.variance)
                likelihood *= numerator / denominator

            probs[class_label] = prior * likelihood

        total = sum(probs.values())
        if total > 0:
            for class_label in probs:
                probs[class_label] /= total
        else:
            for class_label in self.classes:
                probs[class_label] = 1.0 / len(self.classes)

        return probs
