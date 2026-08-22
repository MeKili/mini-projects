"""Union-Find (Disjoint Set Union) with path compression and union by rank."""

from __future__ import annotations


class UnionFind:
    """Fast, typed union-find data structure for connectivity queries.

    Supports O(α(n)) amortized time for both find and union operations,
    where α is the inverse Ackermann function (effectively constant).
    """

    def __init__(self, n: int) -> None:
        """Initialize union-find with n disjoint sets (elements 0 to n-1)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self.parent: list[int] = list(range(n))
        self.rank: list[int] = [0] * n
        self._components = n

    def find(self, x: int) -> int:
        """Find the root representative of the set containing x (with path compression)."""
        if not (0 <= x < len(self.parent)):
            raise IndexError(f"x={x} out of range [0, {len(self.parent)})")
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union the sets containing x and y. Return True if they were in different sets."""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        self._components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Return True if x and y are in the same connected component."""
        return self.find(x) == self.find(y)

    def components(self) -> int:
        """Return the number of connected components."""
        return self._components
