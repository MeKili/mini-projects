"""Tests for expression evaluator."""

import pytest

from expr_eval import evaluate


class TestBasicArithmetic:
    """Test basic arithmetic operations."""

    def test_addition(self) -> None:
        assert evaluate("2 + 3") == 5.0
        assert evaluate("10 + 20 + 30") == 60.0

    def test_subtraction(self) -> None:
        assert evaluate("5 - 2") == 3.0
        assert evaluate("10 - 3 - 2") == 5.0

    def test_multiplication(self) -> None:
        assert evaluate("2 * 3") == 6.0
        assert evaluate("2 * 3 * 4") == 24.0

    def test_division(self) -> None:
        assert evaluate("6 / 2") == 3.0
        assert evaluate("100 / 10 / 2") == 5.0

    def test_single_number(self) -> None:
        assert evaluate("42") == 42.0
        assert evaluate("0") == 0.0


class TestOperatorPrecedence:
    """Test that * and / have higher precedence than + and -."""

    def test_mult_before_add(self) -> None:
        assert evaluate("2 + 3 * 4") == 14.0
        assert evaluate("3 * 4 + 2") == 14.0

    def test_div_before_sub(self) -> None:
        assert evaluate("10 - 6 / 2") == 7.0
        assert evaluate("6 / 2 - 1") == 2.0

    def test_complex_precedence(self) -> None:
        assert evaluate("2 + 3 * 4 - 5 / 2") == 11.5
        assert evaluate("1 + 2 * 3 + 4 * 5") == 27.0


class TestParentheses:
    """Test parentheses for grouping."""

    def test_simple_parens(self) -> None:
        assert evaluate("(2 + 3) * 4") == 20.0
        assert evaluate("2 * (3 + 4)") == 14.0

    def test_nested_parens(self) -> None:
        assert evaluate("((2 + 3) * 4)") == 20.0
        assert evaluate("((10 - 2) / 2)") == 4.0

    def test_parens_override_precedence(self) -> None:
        assert evaluate("(2 + 3) * 4") == 20.0  # 5 * 4
        assert evaluate("2 + (3 * 4)") == 14.0  # 2 + 12
        assert evaluate("2 * (3 + 4)") == 14.0  # 2 * 7

    def test_multiple_parens(self) -> None:
        assert evaluate("(2 + 3) * (4 - 1)") == 15.0


class TestDecimals:
    """Test decimal numbers."""

    def test_simple_decimal(self) -> None:
        assert evaluate("1.5 + 2.5") == 4.0
        assert evaluate("3.5 * 2") == 7.0

    def test_zero_decimal(self) -> None:
        assert evaluate("0.5 + 0.5") == 1.0

    def test_decimal_division(self) -> None:
        assert evaluate("5.0 / 2.0") == 2.5

    def test_leading_zero(self) -> None:
        assert evaluate("0.25 * 4") == 1.0


class TestWhitespace:
    """Test handling of whitespace."""

    def test_spaces_around_ops(self) -> None:
        assert evaluate("2 + 3") == 5.0
        assert evaluate("2+3") == 5.0

    def test_excess_whitespace(self) -> None:
        assert evaluate("  2  +  3  ") == 5.0

    def test_tabs_and_newlines(self) -> None:
        assert evaluate("2\t+\n3") == 5.0


class TestErrors:
    """Test error handling."""

    def test_division_by_zero(self) -> None:
        with pytest.raises(ValueError, match="Division by zero"):
            evaluate("1 / 0")

    def test_missing_closing_paren(self) -> None:
        with pytest.raises(ValueError, match="Expected closing parenthesis"):
            evaluate("(2 + 3")

    def test_extra_closing_paren(self) -> None:
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluate("(2 + 3))")

    def test_invalid_character(self) -> None:
        with pytest.raises(ValueError, match="Unexpected character"):
            evaluate("2 & 3")

    def test_empty_parens(self) -> None:
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluate("()")

    def test_missing_operand(self) -> None:
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluate("2 +")

    def test_double_operator(self) -> None:
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluate("2 + + 3")

    def test_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluate("")


class TestComplexExpressions:
    """Test complex real-world expressions."""

    def test_quadratic(self) -> None:
        # x^2 + 2*x + 1 at x=3: 9 + 6 + 1 = 16
        assert evaluate("3 * 3 + 2 * 3 + 1") == 16.0

    def test_average(self) -> None:
        # (a + b + c) / 3 for a=10, b=20, c=30
        assert evaluate("(10 + 20 + 30) / 3") == 20.0

    def test_nested_computation(self) -> None:
        assert evaluate("((2 + 3) * (4 + 5) - 1) / 2") == 22.0

    def test_chain_operations(self) -> None:
        assert evaluate("10 - 2 - 3 - 1") == 4.0
        assert evaluate("100 / 2 / 5") == 10.0
