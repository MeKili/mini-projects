"""LRU cache with O(1) get and put operations."""

from __future__ import annotations


class _Node[K, V]:
    """A doubly-linked node storing a key-value pair."""

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.prev: _Node[K, V] | None = None
        self.next: _Node[K, V] | None = None


class LRUCache[K, V]:
    """A Least Recently Used cache that evicts the oldest access when full.

    Both get() and put() are O(1) operations. Maintains insertion/access order
    using a doubly-linked list, indexed by a dict for O(1) lookups.
    """

    def __init__(self, capacity: int) -> None:
        """Create a cache with a maximum capacity.

        Args:
            capacity: Maximum number of items the cache can hold. Must be > 0.

        Raises:
            ValueError: If capacity <= 0.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.cache: dict[K, _Node[K, V]] = {}
        self.head: _Node[K, V] | None = None
        self.tail: _Node[K, V] | None = None

    def get(self, key: K) -> V | None:
        """Retrieve a value by key, marking it as recently used.

        Returns None if the key is not in the cache.
        """
        if key not in self.cache:
            return None
        node = self.cache[key]
        self._move_to_end(node)
        return node.value

    def put(self, key: K, value: V) -> None:
        """Store a key-value pair, evicting the LRU item if at capacity.

        If the key already exists, update its value and mark it as recently used.
        """
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_end(node)
            return

        node = _Node(key, value)
        self.cache[key] = node

        if self.head is None:
            self.head = self.tail = node
        else:
            assert self.tail is not None
            node.prev = self.tail
            self.tail.next = node
            self.tail = node

        if len(self.cache) > self.capacity:
            assert self.head is not None
            evicted = self.head
            del self.cache[evicted.key]
            self.head = evicted.next
            if self.head:
                self.head.prev = None
            else:
                self.tail = None

    def _move_to_end(self, node: _Node[K, V]) -> None:
        """Move a node to the end of the list (most recently used)."""
        if node is self.tail:
            return

        if node is self.head:
            self.head = node.next
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

        node.prev = self.tail
        node.next = None
        if self.tail:
            self.tail.next = node
        self.tail = node
