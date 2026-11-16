"""Tests for knapsack solver."""

import pytest

from knapsack import Item, knapsack


def test_empty_knapsack() -> None:
    """Empty knapsack with zero capacity."""
    items = [Item("a", 5, 10)]
    result = knapsack(items, 0)
    assert result.max_value == 0
    assert result.selected_items == ()
    assert result.total_weight == 0


def test_empty_items() -> None:
    """No items available."""
    result = knapsack([], 10)
    assert result.max_value == 0
    assert result.selected_items == ()
    assert result.total_weight == 0


def test_single_item_fits() -> None:
    """Single item that fits."""
    items = [Item("gold", 5, 100)]
    result = knapsack(items, 10)
    assert result.max_value == 100
    assert result.selected_items == (Item("gold", 5, 100),)
    assert result.total_weight == 5


def test_single_item_too_heavy() -> None:
    """Single item that doesn't fit."""
    items = [Item("gold", 15, 100)]
    result = knapsack(items, 10)
    assert result.max_value == 0
    assert result.selected_items == ()
    assert result.total_weight == 0


def test_classic_example() -> None:
    """Classic textbook example: capacity 10, 4 items."""
    items = [
        Item("a", 5, 60),
        Item("b", 3, 50),
        Item("c", 4, 70),
        Item("d", 2, 30),
    ]
    result = knapsack(items, 10)
    # Optimal: b(3,50) + c(4,70) + d(2,30) = weight 9, value 150
    assert result.max_value == 150
    assert result.total_weight == 9
    assert len(result.selected_items) == 3
    assert Item("c", 4, 70) in result.selected_items
    assert Item("b", 3, 50) in result.selected_items


def test_multiple_items_exact_capacity() -> None:
    """Items that exactly fill capacity."""
    items = [
        Item("x", 3, 10),
        Item("y", 2, 8),
        Item("z", 5, 20),
    ]
    result = knapsack(items, 10)
    assert result.max_value == 38
    assert result.total_weight == 10
    assert set(result.selected_items) == {
        Item("x", 3, 10),
        Item("y", 2, 8),
        Item("z", 5, 20),
    }


def test_all_items_too_heavy() -> None:
    """All items exceed capacity."""
    items = [
        Item("a", 20, 100),
        Item("b", 15, 80),
    ]
    result = knapsack(items, 10)
    assert result.max_value == 0
    assert result.selected_items == ()


def test_duplicate_items() -> None:
    """Multiple items with same weight/value."""
    items = [
        Item("gem1", 2, 10),
        Item("gem2", 2, 10),
        Item("gem3", 2, 10),
    ]
    result = knapsack(items, 6)
    assert result.max_value == 30
    assert result.total_weight == 6
    assert len(result.selected_items) == 3


def test_zero_value_items() -> None:
    """Items with zero value should not be selected."""
    items = [
        Item("worthless", 5, 0),
        Item("gold", 3, 50),
    ]
    result = knapsack(items, 8)
    assert result.max_value == 50
    assert result.selected_items == (Item("gold", 3, 50),)


def test_high_value_density() -> None:
    """High value-to-weight ratio items."""
    items = [
        Item("diamond", 1, 100),
        Item("ruby", 2, 150),
        Item("stone", 10, 50),
    ]
    result = knapsack(items, 3)
    # Optimal: diamond(1,100) + ruby(2,150) = weight 3, value 250
    assert result.max_value == 250
    assert result.total_weight == 3
    assert len(result.selected_items) == 2


def test_large_capacity() -> None:
    """Capacity larger than sum of all weights."""
    items = [
        Item("a", 2, 10),
        Item("b", 3, 20),
    ]
    result = knapsack(items, 100)
    assert result.max_value == 30
    assert result.total_weight == 5
    assert len(result.selected_items) == 2


def test_item_validation_negative_weight() -> None:
    """Item with negative weight raises error."""
    with pytest.raises(ValueError, match="weight must be non-negative"):
        Item("bad", -5, 10)


def test_item_validation_negative_value() -> None:
    """Item with negative value raises error."""
    with pytest.raises(ValueError, match="value must be non-negative"):
        Item("bad", 5, -10)


def test_knapsack_validation_negative_capacity() -> None:
    """Negative capacity raises error."""
    items = [Item("a", 5, 10)]
    with pytest.raises(ValueError, match="capacity must be non-negative"):
        knapsack(items, -5)


def test_complex_scenario() -> None:
    """Complex realistic scenario with many items."""
    items = [
        Item("laptop", 5, 1200),
        Item("book", 1, 15),
        Item("phone", 2, 800),
        Item("watch", 1, 200),
        Item("shoes", 3, 80),
        Item("jacket", 4, 150),
        Item("backpack", 2, 50),
    ]
    result = knapsack(items, 10)
    assert result.max_value > 0
    assert result.total_weight <= 10
    # Should include high-value items
    assert any(i.name in ["laptop", "phone"] for i in result.selected_items)


def test_order_independence() -> None:
    """Result should be same regardless of item order."""
    items = [
        Item("a", 3, 20),
        Item("b", 4, 30),
        Item("c", 2, 15),
    ]
    result1 = knapsack(items, 7)

    # Reverse order
    result2 = knapsack(list(reversed(items)), 7)

    assert result1.max_value == result2.max_value
    assert result1.total_weight == result2.total_weight
    assert set(result1.selected_items) == set(result2.selected_items)
