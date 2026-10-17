"""fenwick-tree — a compact data structure for efficient range sum queries.

A Fenwick tree (binary indexed tree) stores prefix sums in a way that enables O(log n)
range sum queries and point updates on an array of numbers. Pure Python, no dependencies.
"""

from fenwick_tree.core import FenwickTree

__all__ = ["FenwickTree"]
__version__ = "0.1.0"
