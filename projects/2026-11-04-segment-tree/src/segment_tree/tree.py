"""Segment tree implementation for range minimum queries."""

from collections.abc import Callable


class SegmentTree[T]:
    """A segment tree for efficient range minimum queries and point updates.

    Supports O(log n) range minimum queries and point updates on an array.
    """

    def __init__(self, arr: list[T], key: Callable[[T, T], T]) -> None:
        """Build a segment tree from an array.

        Args:
            arr: The array to build the tree from.
            key: A binary function that computes the aggregate (e.g., min, max, sum).
        """
        self.key = key
        self.n = len(arr)
        if self.n == 0:
            raise ValueError("Array cannot be empty")

        # Use 1-indexed tree: index 0 is unused
        self.tree: list[T] = [arr[0]] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr: list[T], node: int, start: int, end: int) -> None:
        """Build the segment tree recursively."""
        if start == end:
            self.tree[node] = arr[start]
            return

        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        self._build(arr, left_child, start, mid)
        self._build(arr, right_child, mid + 1, end)

        self.tree[node] = self.key(self.tree[left_child], self.tree[right_child])

    def query(self, left: int, right: int) -> T:
        """Query the aggregate in range [left, right] (inclusive).

        Args:
            left: Left boundary (inclusive).
            right: Right boundary (inclusive).

        Returns:
            The aggregate value in the given range.

        Raises:
            IndexError: If the range is invalid.
        """
        if left < 0 or right >= self.n or left > right:
            raise IndexError(f"Invalid range [{left}, {right}] for array of size {self.n}")

        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node: int, start: int, end: int, left: int, right: int) -> T:
        """Query recursively."""
        if left <= start and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        left_has_overlap = left <= mid
        right_has_overlap = right > mid

        if left_has_overlap and right_has_overlap:
            left_result = self._query(left_child, start, mid, left, right)
            right_result = self._query(right_child, mid + 1, end, left, right)
            return self.key(left_result, right_result)
        elif left_has_overlap:
            return self._query(left_child, start, mid, left, right)
        else:
            return self._query(right_child, mid + 1, end, left, right)

    def update(self, index: int, value: T) -> None:
        """Update the element at the given index to a new value.

        Args:
            index: The index to update.
            value: The new value.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if index < 0 or index >= self.n:
            raise IndexError(f"Index {index} out of bounds for array of size {self.n}")

        self._update(1, 0, self.n - 1, index, value)

    def _update(self, node: int, start: int, end: int, index: int, value: T) -> None:
        """Update recursively."""
        if start == end:
            self.tree[node] = value
            return

        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        if index <= mid:
            self._update(left_child, start, mid, index, value)
        else:
            self._update(right_child, mid + 1, end, index, value)

        self.tree[node] = self.key(self.tree[left_child], self.tree[right_child])
