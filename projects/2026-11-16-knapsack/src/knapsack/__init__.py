"""0/1 knapsack solver with dynamic programming and item backtracking.

Solves the 0/1 knapsack problem: given items with weights and values,
find the subset that maximizes total value within capacity constraint.
Returns both the maximum value and the selected items via backtracking.
"""

from knapsack.solver import Item, knapsack

__all__ = ["knapsack", "Item"]
