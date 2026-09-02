"""Bloom filter — a space-efficient probabilistic set data structure."""

from __future__ import annotations

import math


class BloomFilter:
    """A Bloom filter for fast set membership testing with false positive rate.

    Guaranteed never to return false negatives (missed membership), but may return
    false positives with probability bounded by the false_positive_rate parameter.
    """

    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01) -> None:
        """Initialize a Bloom filter.

        Args:
            expected_elements: Expected number of distinct elements to add.
            false_positive_rate: Target upper bound on false positive probability.

        Raises:
            ValueError: If parameters are invalid.
        """
        if expected_elements <= 0:
            raise ValueError("expected_elements must be positive")
        if not (0 < false_positive_rate < 1):
            raise ValueError("false_positive_rate must be in (0, 1)")

        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate

        self.bit_array_size = self._optimal_size(expected_elements, false_positive_rate)
        self.num_hash_functions = self._optimal_hash_count(self.bit_array_size, expected_elements)
        self._bit_array: bytearray = bytearray((self.bit_array_size + 7) // 8)
        self._count = 0

    @staticmethod
    def _optimal_size(elements: int, fpr: float) -> int:
        """Compute optimal bit array size given element count and target FPR."""
        return max(1, int(-elements * math.log(fpr) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_hash_count(size: int, elements: int) -> int:
        """Compute optimal number of hash functions."""
        return max(1, int((size / elements) * math.log(2)))

    def _hash(self, item: str, seed: int) -> int:
        """Compute a hash value for item using the given seed."""
        value = 0
        for char in item:
            value = ((value << 5) + value + ord(char) + seed) & 0xFFFFFFFF
        return value % self.bit_array_size

    def add(self, item: str) -> None:
        """Add an item to the filter. Safe to call multiple times with same item."""
        for seed in range(self.num_hash_functions):
            pos = self._hash(item, seed)
            byte_idx = pos // 8
            bit_idx = pos % 8
            self._bit_array[byte_idx] |= 1 << bit_idx
        if self._count < self.expected_elements:
            self._count += 1

    def contains(self, item: str) -> bool:
        """Check if item might be in the filter.

        Returns True if item was definitely added, or False if it might have been added
        (false positive). Never returns False for an item that was definitely added.
        """
        for seed in range(self.num_hash_functions):
            pos = self._hash(item, seed)
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self._bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def __contains__(self, item: str) -> bool:
        """Support 'item in bloom_filter' syntax."""
        return self.contains(item)

    def __len__(self) -> int:
        """Return the number of items added (approximate, counts duplicates)."""
        return self._count
