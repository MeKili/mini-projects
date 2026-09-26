"""knn-classifier — a typed, from-scratch k-nearest neighbors classifier.

Fits on training data and predicts class labels for test samples by finding the k
nearest neighbors in the training set and returning the majority class.
Supports Euclidean distance only. Pure Python, no dependencies.
"""

from knn_classifier.core import KNNClassifier

__all__ = ["KNNClassifier"]
__version__ = "0.1.0"
