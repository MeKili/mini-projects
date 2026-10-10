"""Tests for streaming quantile estimator."""

import math
import random

import pytest

from quantile_estimator import QuantileEstimator


class TestQuantileEstimator:
    """Test suite for P-squared quantile estimator."""

    def test_init_invalid_p(self) -> None:
        """Reject p outside (0, 1)."""
        with pytest.raises(ValueError, match="p must be in"):
            QuantileEstimator(p=0.0)
        with pytest.raises(ValueError, match="p must be in"):
            QuantileEstimator(p=1.0)
        with pytest.raises(ValueError, match="p must be in"):
            QuantileEstimator(p=-0.5)
        with pytest.raises(ValueError, match="p must be in"):
            QuantileEstimator(p=1.5)

    def test_estimate_before_update(self) -> None:
        """Estimate fails if no data added."""
        est = QuantileEstimator()
        with pytest.raises(ValueError, match="No data added"):
            est.estimate()

    def test_single_value(self) -> None:
        """Estimate with one value."""
        est = QuantileEstimator(p=0.5)
        est.update(42.0)
        assert est.estimate() == 42.0

    def test_two_values(self) -> None:
        """Estimate with two values."""
        est = QuantileEstimator(p=0.5)
        est.update(10.0)
        est.update(20.0)
        estimate = est.estimate()
        assert 10.0 <= estimate <= 20.0

    def test_five_values_median(self) -> None:
        """Estimate median of five values."""
        est = QuantileEstimator(p=0.5)
        values = [3.0, 1.0, 4.0, 2.0, 5.0]
        for v in values:
            est.update(v)
        estimate = est.estimate()
        assert estimate == 3.0

    def test_sorted_stream(self) -> None:
        """Estimate median from sorted stream."""
        est = QuantileEstimator(p=0.5)
        for i in range(1, 101):
            est.update(float(i))
        estimate = est.estimate()
        assert 40.0 <= estimate <= 60.0

    def test_reverse_sorted_stream(self) -> None:
        """Estimate median from reverse sorted stream."""
        est = QuantileEstimator(p=0.5)
        for i in range(100, 0, -1):
            est.update(float(i))
        estimate = est.estimate()
        assert 45.0 <= estimate <= 55.0

    def test_random_stream_median(self) -> None:
        """Estimate median of random values."""
        random.seed(42)
        values = [random.gauss(50.0, 15.0) for _ in range(1000)]
        est = QuantileEstimator(p=0.5)
        for v in values:
            est.update(v)

        estimate = est.estimate()
        sorted_values = sorted(values)
        true_median = sorted_values[500]
        assert abs(estimate - true_median) < 5.0

    def test_quartiles(self) -> None:
        """Estimate Q1, median, Q3."""
        random.seed(42)
        values = [random.gauss(50.0, 15.0) for _ in range(1000)]
        sorted_values = sorted(values)

        q1_true = sorted_values[250]
        q2_true = sorted_values[500]
        q3_true = sorted_values[750]

        est_q1 = QuantileEstimator(p=0.25)
        est_q2 = QuantileEstimator(p=0.5)
        est_q3 = QuantileEstimator(p=0.75)

        for v in values:
            est_q1.update(v)
            est_q2.update(v)
            est_q3.update(v)

        q1_est = est_q1.estimate()
        q2_est = est_q2.estimate()
        q3_est = est_q3.estimate()

        assert abs(q1_est - q1_true) < 12.0
        assert abs(q2_est - q2_true) < 12.0
        assert abs(q3_est - q3_true) < 12.0

    def test_percentile_95(self) -> None:
        """Estimate 95th percentile."""
        random.seed(42)
        values = [random.gauss(50.0, 15.0) for _ in range(1000)]
        sorted_values = sorted(values)
        p95_true = sorted_values[950]

        est = QuantileEstimator(p=0.95)
        for v in values:
            est.update(v)

        p95_est = est.estimate()
        assert abs(p95_est - p95_true) < 15.0

    def test_uniform_distribution(self) -> None:
        """Estimate median of uniform distribution."""
        random.seed(42)
        values = [random.uniform(0.0, 100.0) for _ in range(1000)]
        sorted_values = sorted(values)
        true_median = sorted_values[500]

        est = QuantileEstimator(p=0.5)
        for v in values:
            est.update(v)

        estimate = est.estimate()
        assert abs(estimate - true_median) < 8.0

    def test_extreme_values(self) -> None:
        """Handle extreme values correctly."""
        est = QuantileEstimator(p=0.5)
        est.update(-1e6)
        est.update(-1e3)
        est.update(0.0)
        est.update(1e3)
        est.update(1e6)
        estimate = est.estimate()
        assert estimate == 0.0

    def test_negative_values(self) -> None:
        """Handle negative values."""
        est = QuantileEstimator(p=0.5)
        for i in range(-50, 51):
            est.update(float(i))
        estimate = est.estimate()
        assert abs(estimate - 0.0) < 5.0

    def test_duplicate_values(self) -> None:
        """Handle duplicate values."""
        est = QuantileEstimator(p=0.5)
        for _ in range(100):
            est.update(42.0)
        assert est.estimate() == 42.0

    def test_small_fractional_values(self) -> None:
        """Handle small fractional values."""
        est = QuantileEstimator(p=0.5)
        for i in range(100):
            est.update(float(i) * 0.01)
        estimate = est.estimate()
        assert 0.3 <= estimate <= 0.7

    def test_estimated_range_valid(self) -> None:
        """Estimate stays within data range."""
        est = QuantileEstimator(p=0.5)
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        for v in values:
            est.update(v)

        estimate = est.estimate()
        assert 10.0 <= estimate <= 90.0

    def test_incremental_estimates_sorted(self) -> None:
        """Check that estimates are in reasonable range for different quantiles."""
        random.seed(42)
        values = [random.gauss(50.0, 15.0) for _ in range(500)]
        sorted_vals = sorted(values)

        est_p25 = QuantileEstimator(p=0.25)
        est_p50 = QuantileEstimator(p=0.5)
        est_p75 = QuantileEstimator(p=0.75)

        for v in values:
            est_p25.update(v)
            est_p50.update(v)
            est_p75.update(v)

        e25 = est_p25.estimate()
        e50 = est_p50.estimate()
        e75 = est_p75.estimate()

        # Check that estimates are reasonable (close to true quantiles)
        true_p25 = sorted_vals[125]
        true_p50 = sorted_vals[250]
        true_p75 = sorted_vals[375]

        assert abs(e25 - true_p25) < 15.0
        assert abs(e50 - true_p50) < 15.0
        assert abs(e75 - true_p75) < 15.0

    def test_large_stream(self) -> None:
        """Estimate median on large stream."""
        random.seed(42)
        est = QuantileEstimator(p=0.5)
        values: list[float] = []

        for _ in range(10000):
            v = random.gauss(100.0, 20.0)
            est.update(v)
            values.append(v)

        estimate = est.estimate()
        true_median = sorted(values)[5000]
        assert abs(estimate - true_median) < 10.0

    def test_exponential_distribution(self) -> None:
        """Estimate median of exponential distribution."""
        random.seed(42)
        values = [-math.log(1.0 - random.random()) for _ in range(1000)]
        sorted_values = sorted(values)
        true_median = sorted_values[500]

        est = QuantileEstimator(p=0.5)
        for v in values:
            est.update(v)

        estimate = est.estimate()
        assert abs(estimate - true_median) < 0.5
