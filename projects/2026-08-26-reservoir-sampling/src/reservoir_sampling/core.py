"""Reservoir sampling: select k random items from a stream in a single pass.

Algorithm R (Vitter, 1985): maintains a reservoir of k items and replaces them with
decreasing probability as the stream grows. Guarantees uniform random selection with O(k)
space and single-pass O(n) time.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator


class ReservoirSampler[T]:
    """Select k random items from an infinite stream in a single pass."""

    def __init__(self, k: int, seed: int | None = None) -> None:
        """Initialize a reservoir sampler.

        Args:
            k: number of items to sample
            seed: optional random seed for reproducibility

        Raises:
            ValueError: if k < 1
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.reservoir: list[T] = []
        self.count = 0
        if seed is not None:
            random.seed(seed)

    def add(self, item: T) -> None:
        """Add an item to the stream."""
        self.count += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            j = random.randint(0, self.count - 1)
            if j < self.k:
                self.reservoir[j] = item

    def sample(self, stream: Iterable[T]) -> list[T]:
        """Consume a stream and return k random samples.

        Args:
            stream: an iterable of items

        Returns:
            list of up to k random items from the stream
        """
        for item in stream:
            self.add(item)
        return self.reservoir.copy()


def sample[T](stream: Iterable[T], k: int, seed: int | None = None) -> list[T]:
    """Return k random samples from a stream.

    Args:
        stream: an iterable of items
        k: number of samples to draw
        seed: optional random seed for reproducibility

    Returns:
        list of up to k random items from the stream

    Raises:
        ValueError: if k < 1
    """
    sampler: ReservoirSampler[T] = ReservoirSampler(k, seed)
    return sampler.sample(stream)


def sample_iter[T](stream: Iterable[T], k: int, seed: int | None = None) -> Iterator[T]:
    """Yield k random samples from a stream one by one.

    Args:
        stream: an iterable of items
        k: number of samples to draw
        seed: optional random seed for reproducibility

    Yields:
        random items from the stream

    Raises:
        ValueError: if k < 1
    """
    sampler: ReservoirSampler[T] = ReservoirSampler(k, seed)
    for item in stream:
        sampler.add(item)
    yield from sampler.reservoir
