"""Longest increasing subsequence (LIS) solver using O(n log n) dynamic programming."""

import bisect
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def lis_length(seq: Sequence[int]) -> int:
    """Return the length of the longest increasing subsequence.

    Uses dynamic programming with binary search for O(n log n) time.
    Space complexity: O(n).
    """
    if not seq:
        return 0

    tails: list[int] = []
    for num in seq:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)


def lis_subsequence(seq: Sequence[int]) -> list[int]:
    """Return the longest increasing subsequence.

    Uses dynamic programming with binary search and reconstruction.
    Returns the actual subsequence in order of appearance.
    """
    if not seq:
        return []

    tails: list[int] = []
    indices: list[int] = []
    parents: list[int] = [-1] * len(seq)

    for i, num in enumerate(seq):
        pos = bisect.bisect_left(tails, num)
        if pos > 0:
            parents[i] = indices[pos - 1]
        if pos == len(tails):
            tails.append(num)
            indices.append(i)
        else:
            tails[pos] = num
            indices[pos] = i

    result: list[int] = []
    idx = indices[-1] if indices else -1
    while idx >= 0:
        result.append(seq[idx])
        idx = parents[idx]
    result.reverse()
    return result


def lis_with_key[T](seq: Sequence[T], key: Callable[[T], int]) -> list[T]:
    """Return the longest increasing subsequence using a key function.

    The key function maps each element to an integer for comparison.
    """
    if not seq:
        return []

    keys = [key(x) for x in seq]
    tails: list[int] = []
    indices: list[int] = []
    parents: list[int] = [-1] * len(seq)

    for i, k in enumerate(keys):
        pos = bisect.bisect_left(tails, k)
        if pos > 0:
            parents[i] = indices[pos - 1]
        if pos == len(tails):
            tails.append(k)
            indices.append(i)
        else:
            tails[pos] = k
            indices[pos] = i

    result: list[T] = []
    idx = indices[-1] if indices else -1
    while idx >= 0:
        result.append(seq[idx])
        idx = parents[idx]
    result.reverse()
    return result
