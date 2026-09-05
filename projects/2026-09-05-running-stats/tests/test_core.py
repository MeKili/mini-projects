"""Tests for running mean and variance tracking."""

import math

from running_stats.core import RunningStat


def test_empty_stat() -> None:
    """Empty RunningStat has zero count and zero mean."""
    stat = RunningStat()
    assert stat.get_count() == 0
    assert stat.get_mean() == 0.0
    assert stat.get_variance() == 0.0


def test_single_value() -> None:
    """Single value has that value as mean, zero variance."""
    stat = RunningStat()
    stat.update(5.0)
    assert stat.get_count() == 1
    assert stat.get_mean() == 5.0
    assert stat.get_variance() == 0.0
    assert stat.get_population_variance() == 0.0


def test_two_values() -> None:
    """Two equal values have zero variance."""
    stat = RunningStat()
    stat.update(3.0)
    stat.update(3.0)
    assert stat.get_count() == 2
    assert stat.get_mean() == 3.0
    assert stat.get_variance() == 0.0


def test_simple_mean() -> None:
    """Mean of [1, 2, 3] is 2."""
    stat = RunningStat()
    stat.update(1.0)
    stat.update(2.0)
    stat.update(3.0)
    assert stat.get_count() == 3
    assert math.isclose(stat.get_mean(), 2.0)


def test_simple_variance() -> None:
    """Sample variance of [1, 2, 3] is 1.0 (N-1 = 2 denominator)."""
    stat = RunningStat()
    stat.update(1.0)
    stat.update(2.0)
    stat.update(3.0)
    assert math.isclose(stat.get_variance(), 1.0)


def test_population_variance() -> None:
    """Population variance of [1, 2, 3] is 2/3 (N = 3 denominator)."""
    stat = RunningStat()
    stat.update(1.0)
    stat.update(2.0)
    stat.update(3.0)
    assert math.isclose(stat.get_population_variance(), 2.0 / 3.0)


def test_std_dev() -> None:
    """Standard deviation is sqrt(variance)."""
    stat = RunningStat()
    for val in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
        stat.update(val)
    variance = stat.get_variance()
    std = stat.get_std()
    assert math.isclose(std, math.sqrt(variance))


def test_add_values() -> None:
    """add_values processes multiple values correctly."""
    stat1 = RunningStat()
    stat2 = RunningStat()

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        stat1.update(v)
    stat2.add_values(values)

    assert stat1.get_mean() == stat2.get_mean()
    assert math.isclose(stat1.get_variance(), stat2.get_variance())


def test_negative_values() -> None:
    """Correctly handles negative values."""
    stat = RunningStat()
    stat.add_values([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert stat.get_mean() == 0.0
    assert stat.get_count() == 5


def test_large_values() -> None:
    """Welford's algorithm is stable with large values."""
    stat = RunningStat()
    stat.add_values([1e6, 1e6 + 1.0, 1e6 + 2.0])
    expected_mean = 1e6 + 1.0
    assert math.isclose(stat.get_mean(), expected_mean, rel_tol=1e-9)


def test_reset() -> None:
    """reset() returns to initial state."""
    stat = RunningStat()
    stat.add_values([1.0, 2.0, 3.0])
    stat.reset()
    assert stat.get_count() == 0
    assert stat.get_mean() == 0.0
    assert stat.get_variance() == 0.0


def test_incremental_updates() -> None:
    """Incremental updates match batch updates."""
    stat_batch = RunningStat()
    stat_batch.add_values([1.0, 2.0, 3.0, 4.0, 5.0])

    stat_incremental = RunningStat()
    stat_incremental.update(1.0)
    stat_incremental.update(2.0)
    stat_incremental.update(3.0)
    stat_incremental.update(4.0)
    stat_incremental.update(5.0)

    assert stat_batch.get_mean() == stat_incremental.get_mean()
    assert math.isclose(stat_batch.get_variance(), stat_incremental.get_variance())


def test_float_precision() -> None:
    """Correctly computes variance for floats with different precisions."""
    stat = RunningStat()
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    stat.add_values(values)
    assert stat.get_count() == 5
    assert stat.get_variance() > 0.0


def test_zero_values() -> None:
    """Handles all-zero values correctly."""
    stat = RunningStat()
    stat.add_values([0.0, 0.0, 0.0])
    assert stat.get_mean() == 0.0
    assert stat.get_variance() == 0.0
