"""Skip list implementation: probabilistic balanced search structure."""

from __future__ import annotations

import random
from typing import TypeVar

K = TypeVar("K", bound=int | float | str)
V = TypeVar("V")


class Node[K: int | float | str, V]:
    """A node in the skip list with key, value, and forward pointers."""

    def __init__(self, key: K, value: V, level: int) -> None:
        self.key = key
        self.value = value
        self.forward: list[Node[K, V] | None] = [None] * (level + 1)


class SkipList[K: int | float | str, V]:
    """Probabilistic skip list with O(log n) search, insert, and delete.

    Each node has a random level (height); probability of reaching level i is 0.5^i.
    Search goes right within a level, then drops down; insert/delete follow similarly.
    """

    def __init__(self, max_level: int = 16, p: float = 0.5) -> None:
        """Initialize skip list with max level and probability parameter p.

        Args:
            max_level: Maximum level (height) any node can have.
            p: Probability of a node promoting to the next level (default 0.5).
        """
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.header: Node[K, V] = Node(key=None, value=None, level=max_level)  # type: ignore

    def _random_level(self) -> int:
        """Return a random level 0..max_level weighted by probability p."""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def search(self, key: K) -> V | None:
        """Search for key; return value if found, None otherwise."""
        x: Node[K, V] | None = self.header
        for i in range(self.level, -1, -1):
            while x is not None and x.forward[i] is not None and x.forward[i].key < key:  # type: ignore
                x = x.forward[i]
            # x.forward[i].key >= key or x.forward[i] is None
        if x is not None:
            x = x.forward[0]
        if x is not None and x.key == key:
            return x.value
        return None

    def insert(self, key: K, value: V) -> None:
        """Insert (key, value), overwriting if key exists."""
        new_level = self._random_level()
        if new_level > self.level:
            self.level = new_level

        update: list[Node[K, V]] = [self.header] * (new_level + 1)
        x: Node[K, V] | None = self.header
        for i in range(self.level, -1, -1):
            while x is not None and x.forward[i] is not None and x.forward[i].key < key:  # type: ignore
                x = x.forward[i]
            if i <= new_level and x is not None:
                update[i] = x

        if x is not None:
            x = x.forward[0]
        if x is not None and x.key == key:
            x.value = value
        else:
            new_node = Node(key, value, new_level)
            for i in range(new_level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node

    def delete(self, key: K) -> bool:
        """Delete the node with key; return True if found and deleted, False otherwise."""
        update: list[Node[K, V]] = [self.header] * (self.level + 1)
        x: Node[K, V] | None = self.header
        for i in range(self.level, -1, -1):
            while x is not None and x.forward[i] is not None and x.forward[i].key < key:  # type: ignore
                x = x.forward[i]
            if x is not None:
                update[i] = x

        if x is not None:
            x = x.forward[0]
        if x is None or x.key != key:
            return False

        for i in range(self.level + 1):
            if update[i].forward[i] != x:
                break
            update[i].forward[i] = x.forward[i]

        while self.level > 0 and self.header.forward[self.level] is None:
            self.level -= 1

        return True

    def range_search(self, start: K, end: K) -> list[tuple[K, V]]:
        """Return all (key, value) pairs in [start, end] (inclusive) in sorted order."""
        result: list[tuple[K, V]] = []
        x: Node[K, V] | None = self.header
        for i in range(self.level, -1, -1):
            while x is not None and x.forward[i] is not None and x.forward[i].key < start:  # type: ignore
                x = x.forward[i]
        if x is not None:
            x = x.forward[0]
        while x is not None and x.key <= end:  # type: ignore
            result.append((x.key, x.value))
            x = x.forward[0]
        return result

    def __len__(self) -> int:
        """Return number of nodes in the skip list."""
        count = 0
        x = self.header.forward[0]
        while x is not None:
            count += 1
            x = x.forward[0]
        return count

    def __contains__(self, key: K) -> bool:
        """Return whether key is in the skip list."""
        return self.search(key) is not None
