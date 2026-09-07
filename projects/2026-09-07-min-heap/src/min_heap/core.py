"""Binary min-heap implementation with O(log n) operations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


class MinHeap[T]:
    """A binary min-heap (priority queue) for efficiently retrieving minimum elements.

    Supports O(log n) push and pop operations, and O(n) heap construction via heapify.
    Elements are ordered by their natural comparison order.
    """

    def __init__(self) -> None:
        """Initialize an empty min-heap."""
        self._heap: list[T] = []

    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return len(self._heap)

    def __bool__(self) -> bool:
        """Return True if the heap is non-empty."""
        return bool(self._heap)

    def push(self, value: T) -> None:
        """Add a value to the heap and maintain the heap property in O(log n)."""
        self._heap.append(value)
        self._sift_up(len(self._heap) - 1)

    def pop(self) -> T:
        """Remove and return the minimum element in O(log n). Raises IndexError if empty."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        if len(self._heap) == 1:
            return self._heap.pop()
        min_val = self._heap[0]
        self._heap[0] = self._heap.pop()
        self._sift_down(0)
        return min_val

    def peek(self) -> T:
        """Return the minimum element without removing it. Raises IndexError if empty."""
        if not self._heap:
            raise IndexError("peek at empty heap")
        return self._heap[0]

    @classmethod
    def heapify(cls, iterable: Iterable[T]) -> MinHeap[T]:
        """Build a min-heap from an iterable in O(n) time."""
        heap: MinHeap[T] = cls()
        heap._heap = list(iterable)
        for i in range(len(heap._heap) // 2 - 1, -1, -1):
            heap._sift_down(i)
        return heap

    def _sift_up(self, idx: int) -> None:
        """Move element at idx up the heap until heap property is satisfied."""
        while idx > 0:
            parent_idx = (idx - 1) // 2
            if self._heap[idx] < self._heap[parent_idx]:  # type: ignore[operator]
                self._heap[idx], self._heap[parent_idx] = (
                    self._heap[parent_idx],
                    self._heap[idx],
                )
                idx = parent_idx
            else:
                break

    def _sift_down(self, idx: int) -> None:
        """Move element at idx down the heap until heap property is satisfied."""
        while True:
            smallest = idx
            left_idx = 2 * idx + 1
            right_idx = 2 * idx + 2
            if (
                left_idx < len(self._heap) and self._heap[left_idx] < self._heap[smallest]  # type: ignore[operator]
            ):
                smallest = left_idx
            if (
                right_idx < len(self._heap) and self._heap[right_idx] < self._heap[smallest]  # type: ignore[operator]
            ):
                smallest = right_idx
            if smallest != idx:
                self._heap[idx], self._heap[smallest] = self._heap[smallest], self._heap[idx]
                idx = smallest
            else:
                break
