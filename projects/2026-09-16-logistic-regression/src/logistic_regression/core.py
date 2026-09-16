"""Binary logistic regression classifier with gradient descent."""

from __future__ import annotations

import math
from collections.abc import Sequence


class LogisticRegression:
    """Binary logistic regression model trained with gradient descent.

    Uses the logistic sigmoid function for binary classification (0/1).
    """

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000) -> None:
        """Initialize the model with hyperparameters.

        Args:
            learning_rate: Step size for gradient descent updates.
            max_iterations: Maximum number of training iterations.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights: list[float] = []
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Sigmoid activation function: 1 / (1 + exp(-z))."""
        if z > 100:
            return 1.0
        if z < -100:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        """Train the model using gradient descent.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,) with values 0 or 1.
        """
        X_list = [list(row) for row in X]
        y_list = list(y)

        if not X_list:
            raise ValueError("X must not be empty")
        if len(X_list) != len(y_list):
            raise ValueError("X and y must have the same length")

        n_features = len(X_list[0])
        if any(len(row) != n_features for row in X_list):
            raise ValueError("All samples must have the same number of features")

        if any(label not in (0, 1) for label in y_list):
            raise ValueError("All labels must be 0 or 1")

        self.weights = [0.0] * n_features
        self.bias = 0.0
        n_samples = len(X_list)

        for _ in range(self.max_iterations):
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                z = self.bias + sum(w * x for w, x in zip(self.weights, X_list[i], strict=True))
                pred = self._sigmoid(z)
                error = pred - y_list[i]

                db += error
                for j in range(n_features):
                    dw[j] += error * X_list[i][j]

            self.bias -= (self.learning_rate / n_samples) * db
            for j in range(n_features):
                self.weights[j] -= (self.learning_rate / n_samples) * dw[j]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Predict class probabilities for each sample.

        Returns probabilities of the positive class (1).

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Probabilities in [0, 1].
        """
        if not self.weights:
            raise ValueError("Model must be fitted before prediction")

        probas = []
        for row in X:
            z = self.bias + sum(w * x for w, x in zip(self.weights, row, strict=True))
            probas.append(self._sigmoid(z))
        return probas

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Predict class labels for each sample.

        Uses a threshold of 0.5 on predicted probabilities.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Predicted labels (0 or 1).
        """
        probas = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probas]

    def accuracy(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> float:
        """Compute accuracy on given data.

        Args:
            X: Input features.
            y: True labels (0 or 1).

        Returns:
            Fraction of correct predictions.
        """
        predictions = self.predict(X)
        y_list = list(y)
        if not predictions:
            return 0.0
        correct = sum(1 for pred, true in zip(predictions, y_list, strict=True) if pred == true)
        return correct / len(predictions)
