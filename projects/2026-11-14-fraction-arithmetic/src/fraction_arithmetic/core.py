"""Immutable rational number type with automatic simplification."""

from __future__ import annotations


def _gcd(a: int, b: int) -> int:
    """Compute greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return abs(a)


class Fraction:
    """Immutable rational number with automatic simplification."""

    __slots__ = ("_numerator", "_denominator")

    def __init__(self, numerator: int, denominator: int = 1) -> None:
        """Initialize a fraction with automatic simplification and sign normalization.

        Args:
            numerator: The numerator of the fraction.
            denominator: The denominator of the fraction (default 1).

        Raises:
            ValueError: If denominator is zero.
        """
        if denominator == 0:
            raise ValueError("denominator cannot be zero")

        gcd = _gcd(numerator, denominator)
        self._numerator = numerator // gcd
        self._denominator = denominator // gcd

        # Normalize sign: negative sign in numerator only
        if self._denominator < 0:
            self._numerator = -self._numerator
            self._denominator = -self._denominator

    @property
    def numerator(self) -> int:
        """The numerator of the simplified fraction."""
        return self._numerator

    @property
    def denominator(self) -> int:
        """The denominator of the simplified fraction."""
        return self._denominator

    def __float__(self) -> float:
        """Convert to float."""
        return self._numerator / self._denominator

    def __str__(self) -> str:
        """String representation as 'numerator/denominator' or 'numerator' if denominator is 1."""
        if self._denominator == 1:
            return str(self._numerator)
        return f"{self._numerator}/{self._denominator}"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Fraction({self._numerator}, {self._denominator})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another Fraction."""
        if not isinstance(other, Fraction):
            return NotImplemented
        return self._numerator == other._numerator and self._denominator == other._denominator

    def __lt__(self, other: object) -> bool:
        """Check if less than another Fraction."""
        if not isinstance(other, Fraction):
            return NotImplemented
        # Compare a/b < c/d by cross-multiplying: a*d < b*c
        return self._numerator * other._denominator < self._denominator * other._numerator

    def __le__(self, other: object) -> bool:
        """Check if less than or equal to another Fraction."""
        if not isinstance(other, Fraction):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        """Check if greater than another Fraction."""
        if not isinstance(other, Fraction):
            return NotImplemented
        return other < self

    def __ge__(self, other: object) -> bool:
        """Check if greater than or equal to another Fraction."""
        if not isinstance(other, Fraction):
            return NotImplemented
        return self == other or self > other

    def __add__(self, other: object) -> Fraction:
        """Add two fractions."""
        if not isinstance(other, Fraction):
            return NotImplemented
        num = self._numerator * other._denominator + other._numerator * self._denominator
        denom = self._denominator * other._denominator
        return Fraction(num, denom)

    def __sub__(self, other: object) -> Fraction:
        """Subtract two fractions."""
        if not isinstance(other, Fraction):
            return NotImplemented
        num = self._numerator * other._denominator - other._numerator * self._denominator
        denom = self._denominator * other._denominator
        return Fraction(num, denom)

    def __mul__(self, other: object) -> Fraction:
        """Multiply two fractions."""
        if not isinstance(other, Fraction):
            return NotImplemented
        return Fraction(
            self._numerator * other._numerator,
            self._denominator * other._denominator,
        )

    def __truediv__(self, other: object) -> Fraction:
        """Divide two fractions."""
        if not isinstance(other, Fraction):
            return NotImplemented
        if other._numerator == 0:
            raise ValueError("cannot divide by zero fraction")
        return Fraction(
            self._numerator * other._denominator,
            self._denominator * other._numerator,
        )

    def __neg__(self) -> Fraction:
        """Negate a fraction."""
        return Fraction(-self._numerator, self._denominator)

    def __hash__(self) -> int:
        """Hash a fraction (safe because Fraction is immutable and simplified)."""
        return hash((self._numerator, self._denominator))
