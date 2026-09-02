"""Tests for Bloom filter implementation."""


import pytest

from bloom_filter.core import BloomFilter


def test_add_and_contains() -> None:
    """Test basic add and contains operations."""
    bf = BloomFilter(100, 0.01)
    bf.add("hello")
    assert "hello" in bf


def test_no_false_negatives() -> None:
    """Never return False for an item that was added (no false negatives)."""
    bf = BloomFilter(1000, 0.01)
    items = ["apple", "banana", "cherry", "date", "elderberry"]
    for item in items:
        bf.add(item)
    for item in items:
        assert item in bf


def test_absent_item_likely_not_present() -> None:
    """Items never added should usually not be present (allows false positives)."""
    bf = BloomFilter(100, 0.01)
    for i in range(50):
        bf.add(f"item_{i}")

    absent = [f"absent_{i}" for i in range(50)]
    present_count = sum(1 for item in absent if item in bf)

    assert present_count < len(absent) * 0.05


def test_duplicate_adds() -> None:
    """Adding the same item multiple times is idempotent."""
    bf = BloomFilter(100, 0.01)
    bf.add("hello")
    bf.add("hello")
    bf.add("hello")
    assert "hello" in bf


def test_empty_filter() -> None:
    """Empty filter does not contain anything."""
    bf = BloomFilter(100, 0.01)
    assert "anything" not in bf


def test_single_item() -> None:
    """Single item can be added and retrieved."""
    bf = BloomFilter(1, 0.01)
    bf.add("single")
    assert "single" in bf


def test_invalid_expected_elements() -> None:
    """Reject non-positive expected_elements."""
    with pytest.raises(ValueError, match="expected_elements must be positive"):
        BloomFilter(0, 0.01)

    with pytest.raises(ValueError, match="expected_elements must be positive"):
        BloomFilter(-5, 0.01)


def test_invalid_false_positive_rate_zero() -> None:
    """Reject FPR of 0."""
    with pytest.raises(ValueError, match="false_positive_rate must be in"):
        BloomFilter(100, 0.0)


def test_invalid_false_positive_rate_one() -> None:
    """Reject FPR of 1 or greater."""
    with pytest.raises(ValueError, match="false_positive_rate must be in"):
        BloomFilter(100, 1.0)

    with pytest.raises(ValueError, match="false_positive_rate must be in"):
        BloomFilter(100, 1.5)


def test_optimal_size_calculation() -> None:
    """Optimal size formula produces reasonable bit array dimensions."""
    size = BloomFilter._optimal_size(1000, 0.01)
    assert size > 0
    assert size < 1000 * 100


def test_optimal_hash_count() -> None:
    """Optimal hash function count is positive and reasonable."""
    size = BloomFilter._optimal_size(100, 0.01)
    count = BloomFilter._optimal_hash_count(size, 100)
    assert count > 0
    assert count < 100


def test_different_fpr_targets() -> None:
    """Stricter FPR targets use more space."""
    size_01 = BloomFilter._optimal_size(1000, 0.1)
    size_001 = BloomFilter._optimal_size(1000, 0.01)
    size_0001 = BloomFilter._optimal_size(1000, 0.001)

    assert size_01 < size_001 < size_0001


def test_len_tracks_additions() -> None:
    """__len__ reflects the number of items added (counting duplicates)."""
    bf = BloomFilter(100, 0.01)
    assert len(bf) == 0
    bf.add("first")
    assert len(bf) == 1
    bf.add("second")
    assert len(bf) == 2
    bf.add("first")
    assert len(bf) == 3


def test_len_caps_at_expected() -> None:
    """__len__ does not exceed expected_elements."""
    bf = BloomFilter(5, 0.01)
    for i in range(10):
        bf.add(f"item_{i}")
    assert len(bf) <= 5


def test_contains_operator() -> None:
    """__contains__ allows 'in' operator syntax."""
    bf = BloomFilter(100, 0.01)
    bf.add("test")
    assert "test" in bf
    assert "missing" not in bf


def test_large_dataset() -> None:
    """Works correctly on larger datasets."""
    bf = BloomFilter(10000, 0.01)
    items = [f"item_{i}" for i in range(5000)]
    for item in items:
        bf.add(item)

    for item in items:
        assert item in bf

    absent_count = 0
    for i in range(5000, 10000):
        if f"item_{i}" in bf:
            absent_count += 1

    assert absent_count < 100
