"""Tests for LRU cache behavior and edge cases."""

import pytest

from lru_cache.core import LRUCache


def test_put_and_get() -> None:
    cache: LRUCache[str, int] = LRUCache(2)  # type: ignore[assignment]
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2


def test_get_nonexistent_returns_none() -> None:
    cache: LRUCache[str, int] = LRUCache(1)  # type: ignore[assignment]
    assert cache.get("missing") is None


def test_capacity_one() -> None:
    cache: LRUCache[str, str] = LRUCache(1)  # type: ignore[assignment]
    cache.put("a", "x")
    assert cache.get("a") == "x"
    cache.put("b", "y")
    assert cache.get("a") is None
    assert cache.get("b") == "y"


def test_evicts_least_recently_used() -> None:
    cache: LRUCache[str, int] = LRUCache(2)  # type: ignore[assignment]
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_update_existing_key() -> None:
    cache: LRUCache[str, int] = LRUCache(2)  # type: ignore[assignment]
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    assert cache.get("a") == 10
    assert len(cache.cache) == 2


def test_update_moves_to_end() -> None:
    cache: LRUCache[str, int] = LRUCache(2)  # type: ignore[assignment]
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 10


def test_capacity_zero_raises() -> None:
    with pytest.raises(ValueError):
        LRUCache(0)


def test_capacity_negative_raises() -> None:
    with pytest.raises(ValueError):
        LRUCache(-5)


def test_large_capacity() -> None:
    cache: LRUCache[int, int] = LRUCache(1000)  # type: ignore[assignment]
    for i in range(1000):
        cache.put(i, i * 2)
    for i in range(1000):
        assert cache.get(i) == i * 2


def test_overflow_evicts() -> None:
    cache: LRUCache[int, int] = LRUCache(10)  # type: ignore[assignment]
    for i in range(20):
        cache.put(i, i)
    assert len(cache.cache) == 10
    for i in range(10):
        assert cache.get(i) is None
    for i in range(10, 20):
        assert cache.get(i) == i


def test_mixed_access_pattern() -> None:
    cache: LRUCache[str, str] = LRUCache(3)  # type: ignore[assignment]
    cache.put("a", "1")
    cache.put("b", "2")
    cache.put("c", "3")
    assert cache.get("a") == "1"
    cache.put("d", "4")
    assert cache.get("b") is None
    assert cache.get("a") == "1"
    assert cache.get("c") == "3"
    assert cache.get("d") == "4"
