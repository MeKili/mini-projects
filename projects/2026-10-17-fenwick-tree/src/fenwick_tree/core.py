"""Fenwick tree (binary indexed tree) for efficient range sum queries and updates."""

from __future__ import annotations


class FenwickTree:
    """Fenwick tree supporting O(log n) point updates and range sum queries on integers."""

    def __init__(self, n: int) -> None:
        """Initialize a Fenwick tree with n positions (0-indexed, 0 to n-1)."""
        if n < 0:
            raise ValueError("size must be non-negative")
        self.n = n
        self.tree: list[int] = [0] * (n + 1)

    def update(self, i: int, delta: int) -> None:
        """Add delta to position i (0-indexed)."""
        if not (0 <= i < self.n):
            raise IndexError(f"index {i} out of range [0, {self.n})")
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def range_sum(self, left: int, right: int) -> int:
        """Sum of elements in [left, right) (0-indexed, half-open interval)."""
        if not (0 <= left <= right <= self.n):
            raise ValueError(f"invalid range [{left}, {right}) for size {self.n}")
        return self._prefix_sum(right) - self._prefix_sum(left)

    def _prefix_sum(self, i: int) -> int:
        """Sum of elements in [0, i) (0-indexed, half-open interval)."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def __repr__(self) -> str:
        return f"FenwickTree(n={self.n})"
