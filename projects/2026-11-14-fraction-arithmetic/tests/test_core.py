"""Tests for Fraction arithmetic."""

import math

import pytest

from fraction_arithmetic import Fraction


def test_fraction_simplification() -> None:
    """Test automatic simplification."""
    assert Fraction(2, 4).numerator == 1
    assert Fraction(2, 4).denominator == 2
    assert Fraction(6, 9) == Fraction(2, 3)


def test_fraction_default_denominator() -> None:
    """Test default denominator of 1."""
    f = Fraction(5)
    assert f.numerator == 5
    assert f.denominator == 1


def test_fraction_sign_normalization() -> None:
    """Test that sign is always in numerator."""
    assert Fraction(-1, 2) == Fraction(1, -2)
    assert Fraction(-1, 2).numerator == -1
    assert Fraction(-1, 2).denominator == 2
    assert Fraction(1, -2).numerator == -1
    assert Fraction(1, -2).denominator == 2


def test_fraction_zero_denominator() -> None:
    """Test that zero denominator raises ValueError."""
    with pytest.raises(ValueError, match="denominator cannot be zero"):
        Fraction(1, 0)


def test_fraction_equality() -> None:
    """Test equality comparison."""
    assert Fraction(1, 2) == Fraction(2, 4)
    assert Fraction(1, 2) == Fraction(1, 2)
    assert Fraction(1, 2) != Fraction(1, 3)
    assert Fraction(0, 5) == Fraction(0, 1)


def test_fraction_ordering() -> None:
    """Test ordering comparisons."""
    assert Fraction(1, 3) < Fraction(1, 2)
    assert Fraction(1, 2) < Fraction(2, 3)
    assert Fraction(1, 2) <= Fraction(1, 2)
    assert Fraction(1, 2) <= Fraction(2, 3)
    assert Fraction(1, 2) > Fraction(1, 3)
    assert Fraction(2, 3) > Fraction(1, 2)
    assert Fraction(1, 2) >= Fraction(1, 2)
    assert Fraction(2, 3) >= Fraction(1, 2)


def test_fraction_negative_ordering() -> None:
    """Test ordering with negative fractions."""
    assert Fraction(-1, 2) < Fraction(1, 2)
    assert Fraction(-1, 2) < Fraction(0, 1)
    assert Fraction(-2, 3) < Fraction(-1, 3)


def test_fraction_addition() -> None:
    """Test fraction addition."""
    assert Fraction(1, 2) + Fraction(1, 3) == Fraction(5, 6)
    assert Fraction(1, 2) + Fraction(1, 2) == Fraction(1, 1)
    assert Fraction(1, 3) + Fraction(2, 3) == Fraction(1, 1)
    assert Fraction(1, 2) + Fraction(0, 1) == Fraction(1, 2)


def test_fraction_subtraction() -> None:
    """Test fraction subtraction."""
    assert Fraction(1, 2) - Fraction(1, 3) == Fraction(1, 6)
    assert Fraction(1, 2) - Fraction(1, 2) == Fraction(0, 1)
    assert Fraction(1, 3) - Fraction(2, 3) == Fraction(-1, 3)


def test_fraction_multiplication() -> None:
    """Test fraction multiplication."""
    assert Fraction(1, 2) * Fraction(1, 3) == Fraction(1, 6)
    assert Fraction(2, 3) * Fraction(3, 4) == Fraction(1, 2)
    assert Fraction(5, 1) * Fraction(1, 5) == Fraction(1, 1)
    assert Fraction(0, 1) * Fraction(10, 1) == Fraction(0, 1)


def test_fraction_division() -> None:
    """Test fraction division."""
    assert Fraction(1, 2) / Fraction(1, 3) == Fraction(3, 2)
    assert Fraction(2, 3) / Fraction(3, 4) == Fraction(8, 9)
    assert Fraction(1, 1) / Fraction(2, 1) == Fraction(1, 2)


def test_fraction_division_by_zero() -> None:
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="cannot divide by zero"):
        Fraction(1, 2) / Fraction(0, 1)


def test_fraction_negation() -> None:
    """Test fraction negation."""
    assert -Fraction(1, 2) == Fraction(-1, 2)
    assert -Fraction(-1, 2) == Fraction(1, 2)
    assert -Fraction(0, 1) == Fraction(0, 1)


def test_fraction_float_conversion() -> None:
    """Test conversion to float."""
    assert math.isclose(float(Fraction(1, 2)), 0.5)
    assert math.isclose(float(Fraction(1, 3)), 1 / 3)
    assert math.isclose(float(Fraction(-1, 2)), -0.5)


def test_fraction_string_representation() -> None:
    """Test string representation."""
    assert str(Fraction(1, 2)) == "1/2"
    assert str(Fraction(5, 1)) == "5"
    assert str(Fraction(-1, 2)) == "-1/2"
    assert str(Fraction(0, 1)) == "0"


def test_fraction_repr() -> None:
    """Test repr representation."""
    assert repr(Fraction(1, 2)) == "Fraction(1, 2)"
    assert repr(Fraction(5, 1)) == "Fraction(5, 1)"


def test_fraction_hash() -> None:
    """Test that equal fractions have equal hash."""
    f1 = Fraction(1, 2)
    f2 = Fraction(2, 4)
    assert hash(f1) == hash(f2)
    assert {f1, f2} == {f1}  # Set should deduplicate


def test_fraction_immutable() -> None:
    """Test that Fraction is immutable."""
    f = Fraction(1, 2)
    with pytest.raises(AttributeError):
        f.numerator = 3  # type: ignore


def test_fraction_arithmetic_chain() -> None:
    """Test chaining arithmetic operations."""
    result = (Fraction(1, 2) + Fraction(1, 3) - Fraction(1, 6)) * Fraction(2, 1) / Fraction(1, 2)
    assert result == Fraction(8, 3)


def test_fraction_large_numbers() -> None:
    """Test with large numbers."""
    f = Fraction(10**9, 2 * 10**9)
    assert f == Fraction(1, 2)
    assert float(f) == 0.5
