"""Core quicksort implementation with Hoare partitioning and median-of-three pivot."""

from typing import Protocol, TypeVar


class Comparable(Protocol):
    """Protocol for types that support comparison operators."""

    def __lt__(self, other: object) -> bool: ...

    def __le__(self, other: object) -> bool: ...

    def __gt__(self, other: object) -> bool: ...

    def __ge__(self, other: object) -> bool: ...


T = TypeVar("T", bound=Comparable)


def quicksort(arr: list[T]) -> None:
    """Sort array in-place using quicksort with Hoare partitioning.

    Uses median-of-three pivot selection to avoid O(n²) on already-sorted data.
    Time: O(n log n) average, O(n²) worst case.
    Space: O(log n) for recursion stack.
    """
    if len(arr) <= 1:
        return
    _quicksort_impl(arr, 0, len(arr) - 1)


def _quicksort_impl(arr: list[T], lo: int, hi: int) -> None:
    """Recursively sort arr[lo:hi+1] in-place."""
    if lo < hi:
        p = _partition(arr, lo, hi)
        _quicksort_impl(arr, lo, p - 1)
        _quicksort_impl(arr, p + 1, hi)


def _partition(arr: list[T], lo: int, hi: int) -> int:
    """Partition using Lomuto scheme with median-of-three pivot.

    Partitions arr[lo:hi+1] around a pivot so that arr[lo:p] <= pivot and arr[p+1:hi+1] > pivot.
    Returns position p where the pivot ends up.
    """
    pivot_idx = _median_of_three(arr, lo, hi)
    arr[hi], arr[pivot_idx] = arr[pivot_idx], arr[hi]
    pivot = arr[hi]

    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[hi] = arr[hi], arr[i]
    return i


def _median_of_three(arr: list[T], lo: int, hi: int) -> int:
    """Return index of median of arr[lo], arr[mid], arr[hi]."""
    mid = (lo + hi) // 2
    if arr[lo] > arr[mid]:
        lo, mid = mid, lo
    if arr[mid] > arr[hi]:
        mid = hi
        if arr[lo] > arr[mid]:
            mid = lo
    return mid
