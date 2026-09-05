"""Streaming mean and variance using Welford's algorithm for numerical stability."""

from __future__ import annotations

import math


class RunningStat:
    """Numerically stable running mean and variance tracker.

    Uses Welford's online algorithm to compute mean and variance in a single pass
    without storing all values. Handles edge cases (empty, single value, etc.).
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        """Add a single value to the running statistics."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def add_values(self, values: list[float] | tuple[float, ...]) -> None:
        """Add multiple values to the running statistics."""
        for value in values:
            self.update(value)

    def get_mean(self) -> float:
        """Return the running mean."""
        return self.mean

    def get_variance(self) -> float:
        """Return the running sample variance (N-1 denominator).

        Returns 0.0 if fewer than 2 values have been added.
        """
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    def get_population_variance(self) -> float:
        """Return the running population variance (N denominator).

        Returns 0.0 if no values have been added.
        """
        if self.count == 0:
            return 0.0
        return self.m2 / self.count

    def get_std(self) -> float:
        """Return the running sample standard deviation (N-1 denominator)."""
        return math.sqrt(self.get_variance())

    def get_population_std(self) -> float:
        """Return the running population standard deviation (N denominator)."""
        return math.sqrt(self.get_population_variance())

    def get_count(self) -> int:
        """Return the number of values added so far."""
        return self.count

    def reset(self) -> None:
        """Reset all statistics to initial state."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
