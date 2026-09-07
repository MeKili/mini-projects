"""Tests for binary min-heap data structure."""

import pytest

from min_heap import MinHeap


class TestMinHeapBasics:
    """Test basic initialization and operations."""

    def test_init_empty_heap(self) -> None:
        """Test that initialization creates an empty heap."""
        heap: MinHeap[int] = MinHeap()
        assert len(heap) == 0
        assert not heap

    def test_push_single_element(self) -> None:
        """Test pushing a single element."""
        heap: MinHeap[int] = MinHeap()
        heap.push(5)
        assert len(heap) == 1
        assert heap.peek() == 5

    def test_push_maintains_heap_property(self) -> None:
        """Test that push maintains min-heap property."""
        heap: MinHeap[int] = MinHeap()
        values = [10, 5, 20, 2, 8, 15]
        for v in values:
            heap.push(v)
        assert heap.peek() == 2

    def test_pop_single_element(self) -> None:
        """Test popping the only element."""
        heap: MinHeap[int] = MinHeap()
        heap.push(42)
        assert heap.pop() == 42
        assert len(heap) == 0

    def test_pop_returns_minimum_sorted(self) -> None:
        """Test that popping returns elements in sorted order."""
        heap: MinHeap[int] = MinHeap()
        values = [10, 5, 20, 2, 8, 15]
        for v in values:
            heap.push(v)
        result = [heap.pop() for _ in range(len(values))]
        assert result == sorted(values)

    def test_peek_doesnt_remove(self) -> None:
        """Test that peek doesn't modify the heap."""
        heap: MinHeap[int] = MinHeap()
        heap.push(10)
        assert heap.peek() == 10
        assert len(heap) == 1
        assert heap.peek() == 10

    def test_bool_empty_heap(self) -> None:
        """Test bool conversion for empty heap."""
        heap: MinHeap[int] = MinHeap()
        assert not heap
        heap.push(1)
        assert heap
        heap.pop()
        assert not heap


class TestMinHeapPushPop:
    """Test push and pop operations."""

    def test_push_pop_alternating(self) -> None:
        """Test alternating push and pop operations."""
        heap: MinHeap[int] = MinHeap()
        heap.push(5)
        heap.push(3)
        assert heap.pop() == 3
        heap.push(7)
        heap.push(1)
        assert heap.pop() == 1
        assert heap.pop() == 5
        assert heap.pop() == 7

    def test_push_pop_duplicates(self) -> None:
        """Test heap with duplicate elements."""
        heap: MinHeap[int] = MinHeap()
        for v in [5, 2, 5, 2, 8]:
            heap.push(v)
        assert heap.pop() == 2
        assert heap.pop() == 2
        assert heap.pop() == 5
        assert heap.pop() == 5
        assert heap.pop() == 8

    def test_push_large_batch(self) -> None:
        """Test pushing a large batch of elements."""
        heap: MinHeap[int] = MinHeap()
        values = list(range(100, 0, -1))
        for v in values:
            heap.push(v)
        assert len(heap) == 100
        assert heap.peek() == 1

    def test_pop_all_returns_sorted(self) -> None:
        """Test that popping all elements returns sorted sequence."""
        heap: MinHeap[int] = MinHeap()
        import random

        values = [random.randint(1, 1000) for _ in range(50)]
        for v in values:
            heap.push(v)
        result = [heap.pop() for _ in range(len(values))]
        assert result == sorted(values)


class TestMinHeapErrors:
    """Test error cases."""

    def test_pop_empty_raises(self) -> None:
        """Test that pop on empty heap raises IndexError."""
        heap: MinHeap[int] = MinHeap()
        with pytest.raises(IndexError, match="pop from empty heap"):
            heap.pop()

    def test_peek_empty_raises(self) -> None:
        """Test that peek on empty heap raises IndexError."""
        heap: MinHeap[int] = MinHeap()
        with pytest.raises(IndexError, match="peek at empty heap"):
            heap.peek()

    def test_pop_until_empty(self) -> None:
        """Test popping all elements until empty then error."""
        heap: MinHeap[int] = MinHeap()
        heap.push(1)
        heap.pop()
        with pytest.raises(IndexError):
            heap.pop()


class TestMinHeapHeapify:
    """Test heapify class method."""

    def test_heapify_empty(self) -> None:
        """Test heapify with empty iterable."""
        heap = MinHeap.heapify([])
        assert len(heap) == 0

    def test_heapify_single_element(self) -> None:
        """Test heapify with single element."""
        heap = MinHeap.heapify([42])
        assert len(heap) == 1
        assert heap.peek() == 42

    def test_heapify_unordered(self) -> None:
        """Test heapify with unordered values."""
        values = [10, 5, 20, 2, 8, 15]
        heap = MinHeap.heapify(values)
        assert heap.peek() == 2
        result = [heap.pop() for _ in range(len(values))]
        assert result == sorted(values)

    def test_heapify_generator(self) -> None:
        """Test heapify with generator expression."""
        values = list(range(20, 0, -1))
        heap = MinHeap.heapify(v for v in values)
        result = [heap.pop() for _ in range(len(values))]
        assert result == sorted(values)

    def test_heapify_large_list(self) -> None:
        """Test heapify with large list."""
        import random

        values = [random.randint(1, 10000) for _ in range(1000)]
        heap = MinHeap.heapify(values)
        result = [heap.pop() for _ in range(len(values))]
        assert result == sorted(values)


class TestMinHeapTypes:
    """Test heap with different types."""

    def test_heap_strings(self) -> None:
        """Test heap with string elements."""
        heap: MinHeap[str] = MinHeap()
        words = ["dog", "apple", "cat", "banana"]
        for w in words:
            heap.push(w)
        result = [heap.pop() for _ in range(len(words))]
        assert result == sorted(words)

    def test_heap_floats(self) -> None:
        """Test heap with float elements."""
        heap: MinHeap[float] = MinHeap()
        values = [3.14, 2.71, 1.41, 1.73]
        for v in values:
            heap.push(v)
        result = [heap.pop() for _ in range(len(values))]
        assert result == sorted(values)

    def test_heap_mixed_numeric(self) -> None:
        """Test heap with mixed int/float (comparable) elements."""
        heap: MinHeap[float] = MinHeap()
        values: list[float] = [5, 2.5, 10, 1.5]
        for v in values:
            heap.push(v)
        result = [heap.pop() for _ in range(len(values))]
        assert result == sorted(values)


class TestMinHeapHeapProperty:
    """Test that the heap property is maintained."""

    def test_heap_property_after_pushes(self) -> None:
        """Test that heap property holds after multiple pushes."""
        heap: MinHeap[int] = MinHeap()
        for v in [5, 3, 7, 1, 9, 2, 8]:
            heap.push(v)
            self._verify_heap_property(heap)

    def test_heap_property_after_pops(self) -> None:
        """Test that heap property holds after pops."""
        heap: MinHeap[int] = MinHeap()
        values = [5, 3, 7, 1, 9, 2, 8]
        for v in values:
            heap.push(v)
        for _ in range(len(values)):
            heap.pop()
            if len(heap) > 0:
                self._verify_heap_property(heap)

    def test_heap_property_after_heapify(self) -> None:
        """Test that heap property holds after heapify."""
        values = [10, 5, 20, 2, 8, 15, 3, 25]
        heap = MinHeap.heapify(values)
        self._verify_heap_property(heap)

    @staticmethod
    def _verify_heap_property(heap: MinHeap[int]) -> None:
        """Verify that the heap satisfies the min-heap property."""
        h = heap._heap
        for i in range(len(h)):
            left_idx = 2 * i + 1
            right_idx = 2 * i + 2
            if left_idx < len(h):
                assert h[i] <= h[left_idx], f"Parent {h[i]} > left child {h[left_idx]}"
            if right_idx < len(h):
                assert h[i] <= h[right_idx], f"Parent {h[i]} > right child {h[right_idx]}"


class TestMinHeapEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_element_all_ops(self) -> None:
        """Test all operations on heap with single element."""
        heap: MinHeap[int] = MinHeap()
        heap.push(100)
        assert len(heap) == 1
        assert bool(heap)
        assert heap.peek() == 100
        assert heap.pop() == 100
        assert len(heap) == 0
        assert not bool(heap)

    def test_many_identical_elements(self) -> None:
        """Test heap with many identical elements."""
        heap: MinHeap[int] = MinHeap()
        for _ in range(100):
            heap.push(42)
        for _ in range(100):
            assert heap.pop() == 42

    def test_negative_numbers(self) -> None:
        """Test heap with negative numbers."""
        heap: MinHeap[int] = MinHeap()
        values = [5, -3, 0, -10, 8]
        for v in values:
            heap.push(v)
        result = [heap.pop() for _ in range(len(values))]
        assert result == sorted(values)

    def test_heap_with_none_comparison(self) -> None:
        """Test heap with tuple elements (custom comparable type)."""
        heap: MinHeap[tuple[int, str]] = MinHeap()
        items = [(3, "c"), (1, "a"), (2, "b")]
        for item in items:
            heap.push(item)
        result = [heap.pop() for _ in range(len(items))]
        assert result == sorted(items)
