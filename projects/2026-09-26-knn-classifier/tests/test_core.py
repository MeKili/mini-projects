"""Tests for k-nearest neighbors classifier."""

import pytest

from knn_classifier.core import KNNClassifier


class TestKNNClassifierInit:
    """Tests for KNNClassifier initialization."""

    def test_init_valid_k(self) -> None:
        """Valid k values should not raise."""
        classifier = KNNClassifier(k=1)
        assert classifier.k == 1

        classifier = KNNClassifier(k=5)
        assert classifier.k == 5

    def test_init_invalid_k(self) -> None:
        """k < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="k must be at least 1"):
            KNNClassifier(k=0)

        with pytest.raises(ValueError, match="k must be at least 1"):
            KNNClassifier(k=-1)


class TestKNNClassifierFit:
    """Tests for the fit method."""

    def test_fit_basic(self) -> None:
        """Fitting should store training data."""
        classifier = KNNClassifier(k=3)
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [0, 1, 0]
        classifier.fit(X, y)

        assert len(classifier.X_train) == 3
        assert classifier.y_train == [0, 1, 0]

    def test_fit_returns_self(self) -> None:
        """fit() should return self for method chaining."""
        classifier = KNNClassifier(k=3)
        result = classifier.fit([[1.0, 2.0]], [0])
        assert result is classifier

    def test_fit_empty_x(self) -> None:
        """Fitting with empty X should raise ValueError."""
        classifier = KNNClassifier(k=3)
        with pytest.raises(ValueError, match="X must not be empty"):
            classifier.fit([], [])

    def test_fit_mismatched_lengths(self) -> None:
        """Fitting with mismatched X and y lengths should raise."""
        classifier = KNNClassifier(k=3)
        with pytest.raises(ValueError, match="X and y must have the same length"):
            classifier.fit([[1.0, 2.0], [3.0, 4.0]], [0])

    def test_fit_stores_copy(self) -> None:
        """fit() should store a copy of data, not references."""
        classifier = KNNClassifier(k=3)
        X = [[1.0, 2.0]]
        y = [0]
        classifier.fit(X, y)
        X[0][0] = 999.0
        y[0] = 999

        assert classifier.X_train[0][0] == 1.0
        assert classifier.y_train[0] == 0


class TestKNNClassifierPredict:
    """Tests for the predict method."""

    def test_predict_unfitted(self) -> None:
        """predict() on unfitted classifier should raise."""
        classifier = KNNClassifier(k=3)
        with pytest.raises(ValueError, match="Classifier must be fitted"):
            classifier.predict([[1.0, 2.0]])

    def test_predict_k1(self) -> None:
        """With k=1, should predict the nearest neighbor's label."""
        classifier = KNNClassifier(k=1)
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        y = [0, 1, 2]
        classifier.fit(X, y)

        predictions = classifier.predict([[0.1, 0.1]])
        assert predictions == [0]

    def test_predict_majority_vote(self) -> None:
        """Should use majority vote among k nearest neighbors."""
        classifier = KNNClassifier(k=3)
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]
        y = [0, 0, 1, 2]
        classifier.fit(X, y)

        predictions = classifier.predict([[0.5, 0.5]])
        assert predictions == [0]

    def test_predict_multiple_samples(self) -> None:
        """Should predict multiple test samples."""
        classifier = KNNClassifier(k=2)
        X = [[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]]
        y = [0, 0, 1, 1]
        classifier.fit(X, y)

        predictions = classifier.predict([[0.3, 0.3], [10.5, 10.5]])
        assert len(predictions) == 2
        assert predictions[0] == 0
        assert predictions[1] == 1

    def test_predict_exact_match(self) -> None:
        """Should correctly predict when test sample is identical to training sample."""
        classifier = KNNClassifier(k=1)
        X = [[1.0, 2.0], [3.0, 4.0]]
        y = [5, 6]
        classifier.fit(X, y)

        predictions = classifier.predict([[1.0, 2.0]])
        assert predictions == [5]

    def test_predict_k_equals_n(self) -> None:
        """With k=n (number of training samples), should use all samples."""
        classifier = KNNClassifier(k=4)
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]
        y = [0, 0, 1, 2]
        classifier.fit(X, y)

        predictions = classifier.predict([[5.0, 5.0]])
        assert len(predictions) == 1


class TestEuclideanDistance:
    """Tests for Euclidean distance calculation."""

    def test_euclidean_distance_identical_vectors(self) -> None:
        """Distance between identical vectors should be 0."""
        distance = KNNClassifier._euclidean_distance([1.0, 2.0], [1.0, 2.0])
        assert distance == 0.0

    def test_euclidean_distance_axis_aligned(self) -> None:
        """Distance along single axis should be the difference."""
        distance = KNNClassifier._euclidean_distance([0.0, 0.0], [3.0, 0.0])
        assert distance == 3.0

    def test_euclidean_distance_diagonal(self) -> None:
        """3-4-5 triangle test."""
        distance = KNNClassifier._euclidean_distance([0.0, 0.0], [3.0, 4.0])
        assert distance == 5.0

    def test_euclidean_distance_higher_dimensions(self) -> None:
        """Should work with arbitrary dimensions."""
        distance = KNNClassifier._euclidean_distance([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        assert abs(distance - (3**0.5)) < 1e-9

    def test_euclidean_distance_mismatched_length(self) -> None:
        """Different vector lengths should raise ValueError."""
        with pytest.raises(ValueError, match="vectors must have the same length"):
            KNNClassifier._euclidean_distance([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_euclidean_distance_negative_values(self) -> None:
        """Should handle negative values correctly."""
        distance = KNNClassifier._euclidean_distance([-1.0, -1.0], [1.0, 1.0])
        assert distance == (8**0.5)
