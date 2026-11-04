"""Tests for segment tree implementation."""

import pytest

from segment_tree import SegmentTree


class TestSegmentTreeMin:
    """Test segment tree with minimum aggregate."""

    def test_build_single_element(self) -> None:
        """Test building tree with single element."""
        tree = SegmentTree([5], min)
        assert tree.query(0, 0) == 5

    def test_build_multiple_elements(self) -> None:
        """Test building tree with multiple elements."""
        arr = [3, 2, 6, 9, 1, 5]
        tree = SegmentTree(arr, min)
        assert tree.query(0, 5) == 1

    def test_query_full_range(self) -> None:
        """Test query on full range."""
        arr = [3, 2, 6, 9, 1, 5]
        tree = SegmentTree(arr, min)
        assert tree.query(0, len(arr) - 1) == 1

    def test_query_partial_range(self) -> None:
        """Test query on partial ranges."""
        arr = [3, 2, 6, 9, 1, 5]
        tree = SegmentTree(arr, min)
        assert tree.query(0, 2) == 2  # min(3, 2, 6)
        assert tree.query(2, 4) == 1  # min(6, 9, 1)
        assert tree.query(4, 5) == 1  # min(1, 5)

    def test_query_single_element(self) -> None:
        """Test query on single element ranges."""
        arr = [3, 2, 6, 9, 1, 5]
        tree = SegmentTree(arr, min)
        for i, val in enumerate(arr):
            assert tree.query(i, i) == val

    def test_update_and_query(self) -> None:
        """Test updating and querying."""
        arr = [3, 2, 6, 9, 1, 5]
        tree = SegmentTree(arr, min)

        # Update index 4 from 1 to 10
        tree.update(4, 10)
        assert tree.query(2, 4) == 6  # min(6, 9, 10)
        assert tree.query(0, 5) == 2  # min of all updated

        # Update index 1 from 2 to 0
        tree.update(1, 0)
        assert tree.query(0, 5) == 0

    def test_multiple_updates(self) -> None:
        """Test multiple sequential updates."""
        arr = [5, 3, 7, 2, 8]
        tree = SegmentTree(arr, min)

        tree.update(0, 1)
        assert tree.query(0, 4) == 1

        tree.update(2, 0)
        assert tree.query(0, 4) == 0

        tree.update(3, 10)
        assert tree.query(0, 4) == 0

    def test_query_invalid_range(self) -> None:
        """Test query with invalid ranges."""
        tree = SegmentTree([1, 2, 3], min)

        with pytest.raises(IndexError):
            tree.query(-1, 2)

        with pytest.raises(IndexError):
            tree.query(0, 3)

        with pytest.raises(IndexError):
            tree.query(2, 0)

    def test_update_invalid_index(self) -> None:
        """Test update with invalid indices."""
        tree = SegmentTree([1, 2, 3], min)

        with pytest.raises(IndexError):
            tree.update(-1, 5)

        with pytest.raises(IndexError):
            tree.update(3, 5)

    def test_empty_array(self) -> None:
        """Test that empty array raises error."""
        with pytest.raises(ValueError):
            SegmentTree([], min)


class TestSegmentTreeMax:
    """Test segment tree with maximum aggregate."""

    def test_max_queries(self) -> None:
        """Test max aggregate queries."""
        arr = [3, 2, 6, 9, 1, 5]
        tree = SegmentTree(arr, max)

        assert tree.query(0, 5) == 9
        assert tree.query(0, 2) == 6
        assert tree.query(2, 4) == 9
        assert tree.query(4, 5) == 5

    def test_max_update(self) -> None:
        """Test max aggregate with updates."""
        arr = [3, 2, 6]
        tree = SegmentTree(arr, max)

        assert tree.query(0, 2) == 6
        tree.update(0, 10)
        assert tree.query(0, 2) == 10


class TestSegmentTreeSum:
    """Test segment tree with sum aggregate."""

    def test_sum_queries(self) -> None:
        """Test sum aggregate queries."""
        arr = [1, 2, 3, 4, 5]
        tree = SegmentTree(arr, lambda a, b: a + b)

        assert tree.query(0, 4) == 15
        assert tree.query(0, 2) == 6  # 1 + 2 + 3
        assert tree.query(2, 4) == 12  # 3 + 4 + 5

    def test_sum_update(self) -> None:
        """Test sum aggregate with updates."""
        arr = [1, 2, 3, 4, 5]
        tree = SegmentTree(arr, lambda a, b: a + b)

        tree.update(2, 10)  # Change 3 to 10
        assert tree.query(0, 4) == 22  # 1 + 2 + 10 + 4 + 5


class TestSegmentTreeFloats:
    """Test segment tree with floating point values."""

    def test_float_min(self) -> None:
        """Test min with floats."""
        arr = [3.5, 2.1, 6.7, 1.2]
        tree = SegmentTree(arr, min)
        assert tree.query(0, 3) == 1.2

    def test_float_max(self) -> None:
        """Test max with floats."""
        arr = [3.5, 2.1, 6.7, 1.2]
        tree = SegmentTree(arr, max)
        assert tree.query(0, 3) == 6.7

    def test_float_update(self) -> None:
        """Test update with floats."""
        arr = [1.5, 2.5, 3.5]
        tree = SegmentTree(arr, min)

        tree.update(1, 0.5)
        assert tree.query(0, 2) == 0.5


class TestSegmentTreeStrings:
    """Test segment tree with strings (lexicographic ordering)."""

    def test_string_min(self) -> None:
        """Test min with strings."""
        arr = ["apple", "banana", "cherry"]
        tree = SegmentTree(arr, min)
        assert tree.query(0, 2) == "apple"

    def test_string_update(self) -> None:
        """Test update with strings."""
        arr = ["dog", "cat", "bird"]
        tree = SegmentTree(arr, min)

        tree.update(0, "aardvark")
        assert tree.query(0, 2) == "aardvark"
