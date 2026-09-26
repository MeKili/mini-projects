"""K-nearest neighbors classifier using Euclidean distance."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


class KNNClassifier:
    """K-nearest neighbors classifier that stores training data and predicts by majority vote."""

    def __init__(self, k: int) -> None:
        """Initialize with the number of neighbors to consider.

        Args:
            k: Number of nearest neighbors to use for prediction (must be >= 1).

        Raises:
            ValueError: If k < 1.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.X_train: list[list[float]] = []
        self.y_train: list[int] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> KNNClassifier:
        """Store training data.

        Args:
            X: Training feature vectors (each should be a sequence of floats).
            y: Training class labels (each should be an integer).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X and y have different lengths or if X is empty.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X) == 0:
            raise ValueError("X must not be empty")

        self.X_train = [list(x) for x in X]
        self.y_train = list(y)
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Predict class labels for test samples.

        For each test sample, find the k nearest training samples by Euclidean
        distance and return the majority class label among them.

        Args:
            X: Test feature vectors.

        Returns:
            List of predicted class labels.

        Raises:
            ValueError: If classifier has not been fitted yet.
        """
        if not self.X_train:
            raise ValueError("Classifier must be fitted before prediction")

        predictions = []
        for test_sample in X:
            test_list = list(test_sample)
            distances = [
                (self._euclidean_distance(test_list, train), label)
                for train, label in zip(self.X_train, self.y_train, strict=True)
            ]
            distances.sort(key=lambda x: x[0])
            k_nearest_labels = [label for _, label in distances[: self.k]]
            counter = Counter(k_nearest_labels)
            most_common_label = counter.most_common(1)[0][0]
            predictions.append(most_common_label)

        return predictions

    @staticmethod
    def _euclidean_distance(x: list[float], y: list[float]) -> float:
        """Calculate Euclidean distance between two vectors.

        Args:
            x: First vector.
            y: Second vector.

        Returns:
            The Euclidean distance.

        Raises:
            ValueError: If vectors have different lengths.
        """
        if len(x) != len(y):
            raise ValueError("vectors must have the same length")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y, strict=True)))
