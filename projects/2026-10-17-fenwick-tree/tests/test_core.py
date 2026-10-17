"""Tests for Fenwick tree (binary indexed tree)."""

import pytest

from fenwick_tree.core import FenwickTree


def test_fenwick_init_empty() -> None:
    ft = FenwickTree(0)
    assert ft.n == 0


def test_fenwick_init_positive() -> None:
    ft = FenwickTree(5)
    assert ft.n == 5


def test_fenwick_init_negative_raises() -> None:
    with pytest.raises(ValueError):
        FenwickTree(-1)


def test_fenwick_single_update_and_range_sum() -> None:
    ft = FenwickTree(5)
    ft.update(2, 10)
    assert ft.range_sum(0, 5) == 10
    assert ft.range_sum(2, 3) == 10
    assert ft.range_sum(0, 2) == 0


def test_fenwick_multiple_updates() -> None:
    ft = FenwickTree(5)
    ft.update(0, 1)
    ft.update(1, 2)
    ft.update(2, 3)
    ft.update(3, 4)
    ft.update(4, 5)
    assert ft.range_sum(0, 5) == 15


def test_fenwick_partial_ranges() -> None:
    ft = FenwickTree(10)
    for i in range(10):
        ft.update(i, i + 1)
    assert ft.range_sum(0, 10) == 55
    assert ft.range_sum(0, 5) == 15
    assert ft.range_sum(5, 10) == 40
    assert ft.range_sum(2, 7) == 25


def test_fenwick_negative_delta() -> None:
    ft = FenwickTree(5)
    ft.update(2, 10)
    ft.update(2, -3)
    assert ft.range_sum(0, 5) == 7


def test_fenwick_zero_delta() -> None:
    ft = FenwickTree(3)
    ft.update(1, 5)
    ft.update(1, 0)
    assert ft.range_sum(0, 3) == 5


def test_fenwick_single_element_range() -> None:
    ft = FenwickTree(3)
    ft.update(0, 10)
    ft.update(1, 20)
    ft.update(2, 30)
    assert ft.range_sum(0, 1) == 10
    assert ft.range_sum(1, 2) == 20
    assert ft.range_sum(2, 3) == 30


def test_fenwick_empty_range() -> None:
    ft = FenwickTree(5)
    ft.update(2, 100)
    assert ft.range_sum(2, 2) == 0
    assert ft.range_sum(0, 0) == 0
    assert ft.range_sum(5, 5) == 0


def test_fenwick_update_out_of_bounds_negative() -> None:
    ft = FenwickTree(5)
    with pytest.raises(IndexError):
        ft.update(-1, 10)


def test_fenwick_update_out_of_bounds_too_large() -> None:
    ft = FenwickTree(5)
    with pytest.raises(IndexError):
        ft.update(5, 10)


def test_fenwick_range_sum_invalid_left() -> None:
    ft = FenwickTree(5)
    with pytest.raises(ValueError):
        ft.range_sum(-1, 3)


def test_fenwick_range_sum_invalid_right() -> None:
    ft = FenwickTree(5)
    with pytest.raises(ValueError):
        ft.range_sum(0, 6)


def test_fenwick_range_sum_inverted_range() -> None:
    ft = FenwickTree(5)
    with pytest.raises(ValueError):
        ft.range_sum(4, 2)


def test_fenwick_repr() -> None:
    ft = FenwickTree(10)
    assert repr(ft) == "FenwickTree(n=10)"


def test_fenwick_stress_large_array() -> None:
    n = 1000
    ft = FenwickTree(n)
    for i in range(n):
        ft.update(i, i % 10)
    expected = sum((i % 10) for i in range(n))
    assert ft.range_sum(0, n) == expected


def test_fenwick_cumulative_operations() -> None:
    ft = FenwickTree(4)
    ft.update(0, 5)
    assert ft.range_sum(0, 4) == 5
    ft.update(1, 3)
    assert ft.range_sum(0, 4) == 8
    ft.update(0, -2)
    assert ft.range_sum(0, 4) == 6
    assert ft.range_sum(0, 2) == 6
    assert ft.range_sum(2, 4) == 0
