"""Tests for linear regression implementation."""

import pytest

from linear_regression import LinearRegression, fit, rmse


class TestFit:
    """Test the fit function."""

    def test_perfect_line(self) -> None:
        """Fit a perfect line y = 2x + 1."""
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 3.0, 5.0, 7.0]
        model = fit(xs, ys)
        assert model.slope == pytest.approx(2.0)
        assert model.intercept == pytest.approx(1.0)

    def test_identity_line(self) -> None:
        """Fit y = x."""
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 2.0, 3.0, 4.0]
        model = fit(xs, ys)
        assert model.slope == pytest.approx(1.0)
        assert model.intercept == pytest.approx(0.0)

    def test_horizontal_line(self) -> None:
        """Fit y = 5 (constant)."""
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [5.0, 5.0, 5.0, 5.0]
        model = fit(xs, ys)
        assert model.slope == pytest.approx(0.0)
        assert model.intercept == pytest.approx(5.0)

    def test_negative_slope(self) -> None:
        """Fit a line with negative slope."""
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [10.0, 8.0, 6.0, 4.0]
        model = fit(xs, ys)
        assert model.slope == pytest.approx(-2.0)
        assert model.intercept == pytest.approx(10.0)

    def test_single_point(self) -> None:
        """Fit with a single point: should give intercept = y, slope = 0."""
        xs = [5.0]
        ys = [10.0]
        model = fit(xs, ys)
        assert model.slope == pytest.approx(0.0)
        assert model.intercept == pytest.approx(10.0)

    def test_vertical_x_values(self) -> None:
        """Fit when all x values are the same (zero denominator)."""
        xs = [2.0, 2.0, 2.0]
        ys = [1.0, 2.0, 3.0]
        model = fit(xs, ys)
        assert model.slope == pytest.approx(0.0)
        assert model.intercept == pytest.approx(2.0)  # mean of ys

    def test_mismatched_lengths(self) -> None:
        """Should raise ValueError for mismatched xs and ys."""
        xs = [1.0, 2.0]
        ys = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="same length"):
            fit(xs, ys)

    def test_empty_data(self) -> None:
        """Should raise ValueError for empty data."""
        with pytest.raises(ValueError, match="empty data"):
            fit([], [])


class TestPredict:
    """Test the predict method."""

    def test_predict_on_fitted_line(self) -> None:
        """Predict should work on points close to fitted line."""
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 3.0, 5.0, 7.0]
        model = fit(xs, ys)
        assert model.predict(0.0) == pytest.approx(1.0)
        assert model.predict(1.0) == pytest.approx(3.0)
        assert model.predict(2.0) == pytest.approx(5.0)
        assert model.predict(3.0) == pytest.approx(7.0)

    def test_predict_outside_range(self) -> None:
        """Predict should work for x values outside training range."""
        xs = [0.0, 1.0, 2.0]
        ys = [0.0, 2.0, 4.0]
        model = fit(xs, ys)
        assert model.predict(10.0) == pytest.approx(20.0)
        assert model.predict(-5.0) == pytest.approx(-10.0)

    def test_predict_with_model_constructor(self) -> None:
        """Create model directly and test predict."""
        model = LinearRegression(slope=3.0, intercept=2.0)
        assert model.predict(0.0) == pytest.approx(2.0)
        assert model.predict(1.0) == pytest.approx(5.0)
        assert model.predict(5.0) == pytest.approx(17.0)


class TestRSquared:
    """Test the r_squared method."""

    def test_perfect_fit(self) -> None:
        """R² should be 1.0 for perfect fit."""
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 3.0, 5.0, 7.0]
        model = fit(xs, ys)
        assert model.r_squared(xs, ys) == pytest.approx(1.0)

    def test_poor_fit(self) -> None:
        """R² should be low for poor fit."""
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [10.0, 5.0, 8.0, 1.0]  # random-looking
        model = fit(xs, ys)
        r2 = model.r_squared(xs, ys)
        assert 0.0 <= r2 < 1.0

    def test_constant_y(self) -> None:
        """R² edge case: all y values are the same."""
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [5.0, 5.0, 5.0, 5.0]
        model = fit(xs, ys)
        r2 = model.r_squared(xs, ys)
        assert r2 == pytest.approx(1.0)  # perfect fit to horizontal line

    def test_r_squared_negative(self) -> None:
        """R² can be negative if model is worse than mean."""
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 2.0, 3.0, 4.0]
        model = LinearRegression(slope=10.0, intercept=0.0)  # terrible model
        r2 = model.r_squared(xs, ys)
        assert r2 < 0.0

    def test_r_squared_mismatched_lengths(self) -> None:
        """Should raise ValueError for mismatched lengths."""
        model = LinearRegression(slope=1.0, intercept=0.0)
        with pytest.raises(ValueError, match="same length"):
            model.r_squared([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_r_squared_empty_data(self) -> None:
        """Should raise ValueError for empty data."""
        model = LinearRegression(slope=1.0, intercept=0.0)
        with pytest.raises(ValueError, match="empty data"):
            model.r_squared([], [])


class TestRMSE:
    """Test the rmse function."""

    def test_rmse_perfect_fit(self) -> None:
        """RMSE should be 0 for perfect fit."""
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 3.0, 5.0, 7.0]
        model = fit(xs, ys)
        assert rmse(model, xs, ys) == pytest.approx(0.0)

    def test_rmse_constant_error(self) -> None:
        """RMSE with constant error."""
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 1.0, 2.0, 3.0]
        model = LinearRegression(slope=1.0, intercept=1.0)
        # Each point is 1 unit too high, so RMSE = 1.0
        assert rmse(model, xs, ys) == pytest.approx(1.0)

    def test_rmse_mismatched_lengths(self) -> None:
        """Should raise ValueError for mismatched lengths."""
        model = LinearRegression(slope=1.0, intercept=0.0)
        with pytest.raises(ValueError, match="same length"):
            rmse(model, [1.0, 2.0], [1.0, 2.0, 3.0])

    def test_rmse_empty_data(self) -> None:
        """Should raise ValueError for empty data."""
        model = LinearRegression(slope=1.0, intercept=0.0)
        with pytest.raises(ValueError, match="empty data"):
            rmse(model, [], [])

    def test_rmse_single_point(self) -> None:
        """RMSE with a single point."""
        model = LinearRegression(slope=2.0, intercept=1.0)
        xs = [1.0]
        ys = [4.0]  # model predicts 2*1+1=3, error is 1
        assert rmse(model, xs, ys) == pytest.approx(1.0)


class TestIntegration:
    """Integration tests with realistic data."""

    def test_noisy_line(self) -> None:
        """Fit a noisy line and check model properties."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.1, 4.0, 5.9, 8.1, 10.0]  # roughly y = 2x
        model = fit(xs, ys)
        assert model.slope == pytest.approx(2.0, abs=0.1)
        assert model.intercept == pytest.approx(0.0, abs=0.1)
        r2 = model.r_squared(xs, ys)
        assert 0.95 < r2 <= 1.0  # strong fit
        err = rmse(model, xs, ys)
        assert err < 0.2

    def test_negative_coordinates(self) -> None:
        """Test with negative coordinates."""
        xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
        ys = [-3.0, -1.0, 1.0, 3.0, 5.0]
        model = fit(xs, ys)
        assert model.slope == pytest.approx(2.0)
        assert model.intercept == pytest.approx(1.0)
        assert model.r_squared(xs, ys) == pytest.approx(1.0)
