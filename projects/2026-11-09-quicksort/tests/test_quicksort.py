"""Tests for quicksort implementation."""

from quicksort import quicksort


def test_empty_list() -> None:
    """Empty list stays empty."""
    arr: list[int] = []
    quicksort(arr)
    assert arr == []


def test_single_element() -> None:
    """Single element is sorted."""
    arr = [42]
    quicksort(arr)
    assert arr == [42]


def test_two_elements_sorted() -> None:
    """Two elements already sorted."""
    arr = [1, 2]
    quicksort(arr)
    assert arr == [1, 2]


def test_two_elements_reverse() -> None:
    """Two elements in reverse order."""
    arr = [2, 1]
    quicksort(arr)
    assert arr == [1, 2]


def test_random_integers() -> None:
    """Random integers are sorted correctly."""
    arr = [64, 34, 25, 12, 22, 11, 90]
    quicksort(arr)
    assert arr == [11, 12, 22, 25, 34, 64, 90]


def test_already_sorted() -> None:
    """Already sorted array is handled correctly (tests pivot selection)."""
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    quicksort(arr)
    assert arr == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_reverse_sorted() -> None:
    """Reverse sorted array is handled correctly."""
    arr = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    quicksort(arr)
    assert arr == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_all_same() -> None:
    """All elements the same."""
    arr = [5, 5, 5, 5, 5]
    quicksort(arr)
    assert arr == [5, 5, 5, 5, 5]


def test_duplicates() -> None:
    """Array with many duplicates."""
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    quicksort(arr)
    assert arr == [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]


def test_negative_numbers() -> None:
    """Array with negative numbers."""
    arr = [3, -1, 4, -5, 0, 2, -3]
    quicksort(arr)
    assert arr == [-5, -3, -1, 0, 2, 3, 4]


def test_floats() -> None:
    """Array of floats."""
    arr = [3.14, 2.71, 1.41, 1.73, 2.0]
    quicksort(arr)
    assert arr == [1.41, 1.73, 2.0, 2.71, 3.14]


def test_strings() -> None:
    """Array of strings (lexicographic order)."""
    arr = ["zebra", "apple", "mango", "banana"]
    quicksort(arr)
    assert arr == ["apple", "banana", "mango", "zebra"]


def test_large_array() -> None:
    """Large array to verify O(n log n) performance."""
    arr = list(range(1000, 0, -1))
    quicksort(arr)
    assert arr == list(range(1, 1001))


def test_inplace_modification() -> None:
    """Verify that sorting modifies the original list in-place."""
    arr = [3, 1, 2]
    original_id = id(arr)
    quicksort(arr)
    assert id(arr) == original_id
    assert arr == [1, 2, 3]


def test_stability_not_required() -> None:
    """Quicksort is not stable, but elements are still sorted correctly."""

    class Item:
        def __init__(self, value: int, index: int):
            self.value = value
            self.index = index

        def __lt__(self, other: object) -> bool:
            if not isinstance(other, Item):
                return NotImplemented
            return self.value < other.value

        def __le__(self, other: object) -> bool:
            if not isinstance(other, Item):
                return NotImplemented
            return self.value <= other.value

        def __gt__(self, other: object) -> bool:
            if not isinstance(other, Item):
                return NotImplemented
            return self.value > other.value

    items = [Item(1, 0), Item(1, 1), Item(1, 2)]
    quicksort(items)
    assert all(item.value == 1 for item in items)
