"""Tests for Gaussian Naive Bayes classifier."""

import math

import pytest

from naive_bayes.core import GaussianNaiveBayes


def test_fit_basic() -> None:
    clf = GaussianNaiveBayes()
    X = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]]
    y = [0, 0, 1, 1]
    clf.fit(X, y)
    assert clf.classes == [0, 1]
    assert len(clf.class_stats) == 2


def test_fit_empty_data() -> None:
    clf = GaussianNaiveBayes()
    with pytest.raises(ValueError, match="training data cannot be empty"):
        clf.fit([], [])


def test_fit_mismatched_lengths() -> None:
    clf = GaussianNaiveBayes()
    with pytest.raises(ValueError, match="X and y must have the same length"):
        clf.fit([[1.0]], [0, 1])


def test_fit_mismatched_feature_count() -> None:
    clf = GaussianNaiveBayes()
    with pytest.raises(ValueError, match="all samples must have the same number"):
        clf.fit([[1.0], [1.0, 2.0]], [0, 1])


def test_fit_empty_features() -> None:
    clf = GaussianNaiveBayes()
    with pytest.raises(ValueError, match="features cannot be empty"):
        clf.fit([[]], [0])


def test_predict_binary_classification() -> None:
    clf = GaussianNaiveBayes()
    X = [[1.0, 1.0], [1.5, 1.5], [5.0, 5.0], [6.0, 6.0]]
    y = [0, 0, 1, 1]
    clf.fit(X, y)

    predictions = clf.predict([[1.0, 1.0], [5.0, 5.0]])
    assert predictions == [0, 1]


def test_predict_multiclass() -> None:
    clf = GaussianNaiveBayes()
    X = [
        [1.0, 1.0],
        [1.5, 1.5],
        [5.0, 5.0],
        [5.5, 5.5],
        [9.0, 9.0],
        [9.5, 9.5],
    ]
    y = [0, 0, 1, 1, 2, 2]
    clf.fit(X, y)

    predictions = clf.predict([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])
    assert predictions == [0, 1, 2]


def test_predict_proba() -> None:
    clf = GaussianNaiveBayes()
    X = [[1.0, 1.0], [1.5, 1.5], [5.0, 5.0], [6.0, 6.0]]
    y = [0, 0, 1, 1]
    clf.fit(X, y)

    probs = clf.predict_proba([[1.0, 1.0]])
    assert len(probs) == 1
    assert len(probs[0]) == 2
    assert 0 in probs[0] and 1 in probs[0]
    assert math.isclose(sum(probs[0].values()), 1.0)


def test_predict_proba_multiclass() -> None:
    clf = GaussianNaiveBayes()
    X = [
        [1.0, 1.0],
        [1.5, 1.5],
        [5.0, 5.0],
        [5.5, 5.5],
        [9.0, 9.0],
        [9.5, 9.5],
    ]
    y = [0, 0, 1, 1, 2, 2]
    clf.fit(X, y)

    probs = clf.predict_proba([[1.0, 1.0]])
    assert len(probs[0]) == 3
    assert math.isclose(sum(probs[0].values()), 1.0)
    assert 0 in probs[0] and 1 in probs[0] and 2 in probs[0]


def test_predict_imbalanced_classes() -> None:
    clf = GaussianNaiveBayes()
    X = [[1.0], [1.1], [1.2], [5.0]]
    y = [0, 0, 0, 1]
    clf.fit(X, y)

    predictions = clf.predict([[1.0]])
    assert predictions[0] == 0


def test_predict_single_feature() -> None:
    clf = GaussianNaiveBayes()
    X = [[1.0], [2.0], [3.0], [5.0], [6.0], [7.0]]
    y = [0, 0, 0, 1, 1, 1]
    clf.fit(X, y)

    predictions = clf.predict([[2.0], [6.0]])
    assert predictions == [0, 1]


def test_predict_negative_values() -> None:
    clf = GaussianNaiveBayes()
    X = [[-5.0, -5.0], [-4.0, -4.0], [5.0, 5.0], [6.0, 6.0]]
    y = [0, 0, 1, 1]
    clf.fit(X, y)

    predictions = clf.predict([[-5.0, -5.0], [5.0, 5.0]])
    assert predictions == [0, 1]


def test_predict_zero_variance_handled() -> None:
    clf = GaussianNaiveBayes()
    X = [[1.0, 1.0], [1.0, 1.0], [5.0, 5.0], [5.0, 5.0]]
    y = [0, 0, 1, 1]
    clf.fit(X, y)

    predictions = clf.predict([[1.0, 1.0], [5.0, 5.0]])
    assert predictions == [0, 1]


def test_fit_returns_self() -> None:
    clf = GaussianNaiveBayes()
    result = clf.fit([[1.0]], [0])
    assert result is clf


def test_predict_multiple_samples() -> None:
    clf = GaussianNaiveBayes()
    X = [[1.0, 1.0], [1.5, 1.5], [5.0, 5.0], [6.0, 6.0]]
    y = [0, 0, 1, 1]
    clf.fit(X, y)

    predictions = clf.predict([[1.0, 1.0], [1.5, 1.5], [5.0, 5.0], [6.0, 6.0]])
    assert predictions == [0, 0, 1, 1]


def test_classes_sorted() -> None:
    clf = GaussianNaiveBayes()
    X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    y = [2, 2, 1, 1, 0, 0]
    clf.fit(X, y)
    assert clf.classes == [0, 1, 2]


def test_predict_proba_sum_to_one() -> None:
    clf = GaussianNaiveBayes()
    X = [[1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [6.0, 6.0]]
    y = [0, 0, 1, 1]
    clf.fit(X, y)

    probs_list = clf.predict_proba([[2.0, 2.0], [3.0, 3.0], [5.5, 5.5]])
    for probs in probs_list:
        assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
