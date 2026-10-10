"""Streaming quantile estimator using reservoir sampling.

Maintains a reservoir of k recent values and estimates quantiles
from their sorted order, enabling O(k log k) time per update.
"""


class QuantileEstimator:
    """Streaming quantile estimator using sorted reservoir.

    Maintains a fixed-size reservoir of recent values and estimates
    target quantiles from the sorted reservoir. Useful for streaming
    data where recent values are more representative.
    """

    def __init__(self, p: float = 0.5, capacity: int = 256) -> None:
        """Initialize estimator for quantile p (0 < p < 1).

        Args:
            p: Target quantile (0.5 for median, 0.95 for 95th percentile).
            capacity: Size of reservoir to maintain (default 256).

        Raises:
            ValueError: If p is not in (0, 1).
        """
        if not 0 < p < 1:
            raise ValueError(f"p must be in (0, 1), got {p}")

        self.p = p
        self.capacity = capacity
        self.n = 0
        self.reservoir: list[float] = []

    def update(self, value: float) -> None:
        """Process a new value and update quantile estimate.

        Args:
            value: Next value in the stream.
        """
        self.n += 1

        # Add to reservoir if not at capacity, otherwise replace randomly
        if len(self.reservoir) < self.capacity:
            self.reservoir.append(value)
        else:
            # With probability 1/n, keep this value
            import random

            if random.randint(1, self.n) <= self.capacity:
                idx = random.randint(0, self.capacity - 1)
                self.reservoir[idx] = value

    def estimate(self) -> float:
        """Return current estimate of the target quantile.

        Returns:
            Estimated value at quantile p.
        """
        if self.n == 0:
            raise ValueError("No data added yet")

        if len(self.reservoir) == 0:
            raise ValueError("Reservoir is empty")

        # Sort reservoir and find quantile
        sorted_vals = sorted(self.reservoir)
        idx = int(self.p * (len(sorted_vals) - 1))
        idx = min(max(idx, 0), len(sorted_vals) - 1)

        return sorted_vals[idx]
