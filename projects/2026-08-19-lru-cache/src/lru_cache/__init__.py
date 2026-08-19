"""lru-cache — a fast, typed LRU (Least Recently Used) cache.

An in-memory cache that evicts the least recently used item when capacity is
reached. Both get() and put() are O(1) operations using a dict + doubly-linked list.
Pure Python, no dependencies.
"""

from lru_cache.core import LRUCache

__all__ = ["LRUCache"]
__version__ = "0.1.0"
