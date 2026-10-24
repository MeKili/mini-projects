"""Tests for skip list operations."""

import random

from skip_list.core import SkipList


def test_search_not_found() -> None:
    sl: SkipList[int, str] = SkipList()
    sl.insert(10, "a")
    assert sl.search(5) is None
    assert sl.search(20) is None


def test_search_found() -> None:
    sl: SkipList[int, str] = SkipList()
    sl.insert(10, "a")
    sl.insert(20, "b")
    assert sl.search(10) == "a"
    assert sl.search(20) == "b"


def test_insert_overwrite() -> None:
    sl: SkipList[int, str] = SkipList()
    sl.insert(10, "first")
    assert sl.search(10) == "first"
    sl.insert(10, "second")
    assert sl.search(10) == "second"


def test_insert_maintains_order() -> None:
    sl: SkipList[int, str] = SkipList()
    for i in [50, 30, 70, 10, 90]:
        sl.insert(i, f"v{i}")
    x = sl.header.forward[0]
    values = []
    while x is not None:
        values.append(x.key)
        x = x.forward[0]
    assert values == [10, 30, 50, 70, 90]


def test_delete_not_found() -> None:
    sl: SkipList[int, str] = SkipList()
    sl.insert(10, "a")
    assert sl.delete(20) is False
    assert sl.search(10) == "a"


def test_delete_found() -> None:
    sl: SkipList[int, str] = SkipList()
    sl.insert(10, "a")
    assert sl.delete(10) is True
    assert sl.search(10) is None


def test_delete_multiple() -> None:
    sl: SkipList[int, str] = SkipList()
    for i in [10, 20, 30]:
        sl.insert(i, f"v{i}")
    assert sl.delete(20) is True
    assert sl.search(10) == "v10"
    assert sl.search(20) is None
    assert sl.search(30) == "v30"


def test_range_search_empty() -> None:
    sl: SkipList[int, str] = SkipList()
    sl.insert(50, "a")
    assert sl.range_search(10, 40) == []


def test_range_search_inclusive() -> None:
    sl: SkipList[int, str] = SkipList()
    for i in [10, 20, 30, 40, 50]:
        sl.insert(i, f"v{i}")
    result = sl.range_search(20, 40)
    assert result == [(20, "v20"), (30, "v30"), (40, "v40")]


def test_range_search_full() -> None:
    sl: SkipList[int, str] = SkipList()
    for i in [10, 20, 30]:
        sl.insert(i, f"v{i}")
    result = sl.range_search(0, 100)
    assert result == [(10, "v10"), (20, "v20"), (30, "v30")]


def test_len() -> None:
    sl: SkipList[int, str] = SkipList()
    assert len(sl) == 0
    for i in range(5):
        sl.insert(i, f"v{i}")
    assert len(sl) == 5
    sl.delete(2)
    assert len(sl) == 4


def test_contains() -> None:
    sl: SkipList[int, str] = SkipList()
    sl.insert(10, "a")
    assert 10 in sl
    assert 20 not in sl


def test_string_keys() -> None:
    sl: SkipList[str, int] = SkipList()
    sl.insert("apple", 1)
    sl.insert("banana", 2)
    sl.insert("cherry", 3)
    assert sl.search("banana") == 2
    result = sl.range_search("apple", "banana")
    assert result == [("apple", 1), ("banana", 2)]


def test_float_keys() -> None:
    sl: SkipList[float, str] = SkipList()
    for val in [1.5, 2.7, 0.3]:
        sl.insert(val, f"v{val}")
    assert sl.search(2.7) == "v2.7"


def test_stress_random_operations() -> None:
    random.seed(42)
    sl: SkipList[int, int] = SkipList()
    inserted = set()

    for _ in range(100):
        key = random.randint(0, 50)
        sl.insert(key, key * 2)
        inserted.add(key)

    for key in inserted:
        assert sl.search(key) == key * 2

    for _ in range(30):
        key = random.choice(list(inserted))
        if sl.delete(key):
            inserted.discard(key)

    for key in inserted:
        assert sl.search(key) == key * 2
        assert key in sl

    for key in range(0, 51):
        if key not in inserted:
            assert sl.search(key) is None
