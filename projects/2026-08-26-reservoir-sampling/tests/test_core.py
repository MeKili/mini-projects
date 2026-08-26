"""Tests for reservoir sampling."""

import pytest

from reservoir_sampling import ReservoirSampler, sample, sample_iter


def test_sample_exact_size() -> None:
    """Sample k items from a stream of size k."""
    result = sample(range(5), k=5, seed=42)
    assert len(result) == 5
    assert set(result) == set(range(5))


def test_sample_smaller_than_k() -> None:
    """Sample k items from a stream smaller than k."""
    result = sample(range(3), k=5, seed=42)
    assert len(result) == 3
    assert set(result) == set(range(3))


def test_sample_larger_than_k() -> None:
    """Sample k items from a stream larger than k."""
    result = sample(range(100), k=10, seed=42)
    assert len(result) == 10
    assert all(0 <= x < 100 for x in result)
    assert len(set(result)) == 10


def test_sample_single_item() -> None:
    """Sample single item."""
    result = sample(range(100), k=1, seed=42)
    assert len(result) == 1
    assert 0 <= result[0] < 100


def test_sample_empty_stream() -> None:
    """Sample from empty stream."""
    result = sample([], k=5, seed=42)
    assert len(result) == 0


def test_sample_reproducibility() -> None:
    """Seed ensures reproducible results."""
    result1 = sample(range(1000), k=10, seed=12345)
    result2 = sample(range(1000), k=10, seed=12345)
    assert result1 == result2


def test_sample_different_seeds() -> None:
    """Different seeds give different results (with high probability)."""
    result1 = sample(range(1000), k=10, seed=1)
    result2 = sample(range(1000), k=10, seed=2)
    assert result1 != result2


def test_sampler_class_single_stream() -> None:
    """ReservoirSampler works with single stream."""
    sampler = ReservoirSampler(5, seed=42)
    result = sampler.sample(range(20))
    assert len(result) == 5
    assert all(0 <= x < 20 for x in result)


def test_sampler_class_incremental() -> None:
    """ReservoirSampler can add items one by one."""
    sampler = ReservoirSampler(3, seed=42)
    for i in range(10):
        sampler.add(i)
    result = sampler.reservoir.copy()
    assert len(result) == 3
    assert all(0 <= x < 10 for x in result)


def test_sampler_count() -> None:
    """ReservoirSampler tracks count correctly."""
    sampler = ReservoirSampler(5, seed=42)
    sampler.sample(range(20))
    assert sampler.count == 20


def test_sampler_iter() -> None:
    """sample_iter yields k items."""
    result = list(sample_iter(range(100), k=5, seed=42))
    assert len(result) == 5
    assert all(0 <= x < 100 for x in result)


def test_sample_iter_empty() -> None:
    """sample_iter with empty stream."""
    result = list(sample_iter([], k=5, seed=42))
    assert len(result) == 0


def test_invalid_k_zero() -> None:
    """Reject k=0."""
    with pytest.raises(ValueError, match="k must be at least 1"):
        ReservoirSampler(0)


def test_invalid_k_negative() -> None:
    """Reject negative k."""
    with pytest.raises(ValueError, match="k must be at least 1"):
        ReservoirSampler(-1)


def test_sample_function_invalid_k() -> None:
    """sample function rejects invalid k."""
    with pytest.raises(ValueError, match="k must be at least 1"):
        sample(range(10), k=0)


def test_distribution_uniformity() -> None:
    """Verify that samples cover multiple distinct items."""
    samples = [
        sample(range(1000), k=50, seed=i) for i in range(10)
    ]
    all_items = set()
    for sample_list in samples:
        all_items.update(sample_list)

    assert len(all_items) > 100


def test_large_stream() -> None:
    """Sample from very large stream."""

    def large_stream() -> range:
        return range(1_000_000)

    result = sample(large_stream(), k=100, seed=42)
    assert len(result) == 100
    assert all(0 <= x < 1_000_000 for x in result)
    assert len(set(result)) == 100


def test_generator_stream() -> None:
    """Sample from generator stream."""

    def gen() -> range:
        yield from range(50)

    result = sample(gen(), k=10, seed=42)
    assert len(result) == 10
    assert all(0 <= x < 50 for x in result)
