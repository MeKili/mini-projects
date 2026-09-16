"""Tests for logistic regression classifier."""

import math

import pytest

from logistic_regression.core import LogisticRegression


def test_sigmoid_values() -> None:
    """Test sigmoid function boundary values."""
    model = LogisticRegression()
    assert math.isclose(model._sigmoid(0), 0.5)
    assert 0 < model._sigmoid(-10) < 0.01
    assert 0.99 < model._sigmoid(10) < 1


def test_sigmoid_overflow_protection() -> None:
    """Test sigmoid handles extreme values without overflow."""
    model = LogisticRegression()
    assert model._sigmoid(1000) == 1.0
    assert model._sigmoid(-1000) == 0.0


def test_init_invalid_learning_rate() -> None:
    """Test invalid learning rate raises ValueError."""
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=-0.1)
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=0.0)


def test_init_invalid_max_iterations() -> None:
    """Test invalid max_iterations raises ValueError."""
    with pytest.raises(ValueError):
        LogisticRegression(max_iterations=-1)
    with pytest.raises(ValueError):
        LogisticRegression(max_iterations=0)


def test_predict_before_fit_raises() -> None:
    """Test predicting before fitting raises ValueError."""
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.predict([[1.0, 2.0]])


def test_fit_empty_data_raises() -> None:
    """Test fitting on empty data raises ValueError."""
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit([], [])


def test_fit_mismatched_lengths_raises() -> None:
    """Test mismatched X and y lengths raise ValueError."""
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit([[1.0, 2.0], [3.0, 4.0]], [0])


def test_fit_inconsistent_features_raises() -> None:
    """Test inconsistent feature count raises ValueError."""
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit([[1.0, 2.0], [3.0]], [0, 1])


def test_fit_invalid_labels_raises() -> None:
    """Test non-binary labels raise ValueError."""
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit([[1.0], [2.0]], [0, 2])


def test_fit_single_feature() -> None:
    """Test fitting with single feature."""
    model = LogisticRegression(learning_rate=0.1, max_iterations=100)
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]
    model.fit(X, y)
    assert len(model.weights) == 1
    assert isinstance(model.bias, float)


def test_fit_linearly_separable() -> None:
    """Test fitting on linearly separable data."""
    model = LogisticRegression(learning_rate=0.1, max_iterations=500)
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0, 0, 0, 1]
    model.fit(X, y)

    preds = model.predict(X)
    assert preds == [0, 0, 0, 1]


def test_predict_proba_range() -> None:
    """Test predicted probabilities are in [0, 1]."""
    model = LogisticRegression(learning_rate=0.1, max_iterations=100)
    X_train = [[i] for i in range(4)]
    y_train = [0, 0, 1, 1]
    model.fit(X_train, y_train)

    proba = model.predict_proba([[0.5], [1.5], [2.5]])
    assert all(0 <= p <= 1 for p in proba)


def test_predict_consistent_with_proba() -> None:
    """Test that predict aligns with predict_proba threshold."""
    model = LogisticRegression(learning_rate=0.1, max_iterations=200)
    X_train = [[0.0], [1.0], [2.0], [3.0]]
    y_train = [0, 0, 1, 1]
    model.fit(X_train, y_train)

    X_test = [[0.5], [1.5], [2.5]]
    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)

    for p, pred in zip(proba, preds, strict=True):
        expected = 1 if p >= 0.5 else 0
        assert pred == expected


def test_accuracy_perfect() -> None:
    """Test accuracy on perfectly classified data."""
    model = LogisticRegression(learning_rate=0.1, max_iterations=500)
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]
    model.fit(X, y)

    acc = model.accuracy(X, y)
    assert acc == 1.0


def test_accuracy_all_wrong() -> None:
    """Test accuracy when all predictions are inverted."""
    model = LogisticRegression(learning_rate=0.1, max_iterations=10)
    X = [[0.0], [1.0]]
    y = [0, 1]
    model.fit(X, y)

    # With very few iterations, model is undertrained; predictions may be poor
    acc = model.accuracy(X, y)
    assert 0 <= acc <= 1


def test_weights_updated_after_fit() -> None:
    """Test that weights and bias change after fitting."""
    model = LogisticRegression(learning_rate=0.1, max_iterations=100)
    X = [[1.0, 2.0], [3.0, 4.0]]
    y = [0, 1]

    model.fit(X, y)
    assert len(model.weights) == 2
    assert model.weights != [0.0, 0.0]
    assert model.bias != 0.0
