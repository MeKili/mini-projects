"""Core knapsack solver implementation."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """An item with weight and value."""

    name: str
    weight: int
    value: int

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")
        if self.value < 0:
            raise ValueError(f"value must be non-negative, got {self.value}")


@dataclass(frozen=True)
class Solution:
    """Result of knapsack solver."""

    max_value: int
    selected_items: tuple[Item, ...]
    total_weight: int


def knapsack(items: Sequence[Item], capacity: int) -> Solution:
    """Solve 0/1 knapsack problem using dynamic programming.

    Args:
        items: Sequence of Item objects with weight and value.
        capacity: Maximum weight the knapsack can hold.

    Returns:
        Solution with maximum value, selected items, and total weight.

    Raises:
        ValueError: If capacity is negative.
    """
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative, got {capacity}")

    n = len(items)
    if n == 0 or capacity == 0:
        return Solution(max_value=0, selected_items=(), total_weight=0)

    # dp[i][w] = max value using items 0..i-1 with capacity w
    dp: list[list[int]] = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        item = items[i - 1]
        for w in range(capacity + 1):
            # Option 1: don't take item i-1
            dp[i][w] = dp[i - 1][w]
            # Option 2: take item i-1 if it fits
            if w >= item.weight:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - item.weight] + item.value)

    # Backtrack to find selected items
    selected: list[Item] = []
    w = capacity
    for i in range(n, 0, -1):
        # If value changed from previous row, this item was selected
        if dp[i][w] != dp[i - 1][w]:
            selected.append(items[i - 1])
            w -= items[i - 1].weight

    selected.reverse()
    total_weight = sum(item.weight for item in selected)

    return Solution(
        max_value=dp[n][capacity],
        selected_items=tuple(selected),
        total_weight=total_weight,
    )
