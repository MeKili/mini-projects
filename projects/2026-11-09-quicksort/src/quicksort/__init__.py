"""quicksort — in-place quicksort with Hoare partitioning and median-of-three pivot selection.

Quicksort is a divide-and-conquer sorting algorithm that achieves O(n log n) average-case
time and O(1) space (in-place). Uses Hoare's partitioning scheme for better cache locality
and median-of-three pivot selection to avoid pathological cases on already-sorted data.
"""

from quicksort.core import quicksort

__all__ = ["quicksort"]
__version__ = "0.1.0"
