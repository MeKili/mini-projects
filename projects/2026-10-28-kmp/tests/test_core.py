"""Tests for KMP string matching."""

from kmp import build_failure_function, find_pattern


class TestFailureFunction:
    """Test the failure function construction."""

    def test_single_char(self) -> None:
        """Single character has zero failure value."""
        assert build_failure_function("a") == [0]

    def test_no_prefix_suffix(self) -> None:
        """Pattern with no repeating prefix/suffix."""
        assert build_failure_function("abc") == [0, 0, 0]

    def test_repeating_prefix(self) -> None:
        """Pattern with repeating prefix."""
        assert build_failure_function("aaaa") == [0, 1, 2, 3]

    def test_partial_repeat(self) -> None:
        """Pattern with partial repeating prefix/suffix."""
        assert build_failure_function("ababab") == [0, 0, 1, 2, 3, 4]

    def test_abab_pattern(self) -> None:
        """Test common pattern."""
        assert build_failure_function("abab") == [0, 0, 1, 2]

    def test_aaab_pattern(self) -> None:
        """Test pattern with prefix repeat."""
        assert build_failure_function("aaab") == [0, 1, 2, 0]

    def test_abcabc_pattern(self) -> None:
        """Test pattern with full repeat."""
        assert build_failure_function("abcabc") == [0, 0, 0, 1, 2, 3]


class TestFindPattern:
    """Test pattern finding."""

    def test_single_match(self) -> None:
        """Find pattern in simple text."""
        assert find_pattern("hello world", "world") == [6]

    def test_multiple_matches(self) -> None:
        """Find multiple occurrences."""
        assert find_pattern("aaaa", "aa") == [0, 1, 2]

    def test_overlapping_matches(self) -> None:
        """Handle overlapping pattern matches."""
        assert find_pattern("ababa", "aba") == [0, 2]

    def test_no_match(self) -> None:
        """Return empty when pattern not found."""
        assert find_pattern("hello", "xyz") == []

    def test_pattern_at_start(self) -> None:
        """Pattern at beginning of text."""
        assert find_pattern("hello world", "hello") == [0]

    def test_pattern_at_end(self) -> None:
        """Pattern at end of text."""
        assert find_pattern("hello world", "ld") == [9]

    def test_pattern_equals_text(self) -> None:
        """Pattern equals entire text."""
        assert find_pattern("test", "test") == [0]

    def test_empty_pattern(self) -> None:
        """Empty pattern returns empty."""
        assert find_pattern("hello", "") == []

    def test_empty_text(self) -> None:
        """Empty text returns empty."""
        assert find_pattern("", "a") == []

    def test_pattern_longer_than_text(self) -> None:
        """Pattern longer than text."""
        assert find_pattern("hi", "hello") == []

    def test_single_char_pattern(self) -> None:
        """Single character pattern."""
        assert find_pattern("mississippi", "i") == [1, 4, 7, 10]

    def test_repeating_pattern(self) -> None:
        """Repeating pattern in text."""
        assert find_pattern("aaaaaa", "aaa") == [0, 1, 2, 3]

    def test_complex_pattern(self) -> None:
        """Complex overlapping pattern."""
        assert find_pattern("ababababab", "abab") == [0, 2, 4, 6]

    def test_case_sensitive(self) -> None:
        """Matching is case-sensitive."""
        assert find_pattern("Hello", "hello") == []
        assert find_pattern("Hello", "Hello") == [0]

    def test_special_chars(self) -> None:
        """Pattern with special characters."""
        assert find_pattern("test.file.txt", ".") == [4, 9]

    def test_long_pattern(self) -> None:
        """Long pattern search."""
        text = "The quick brown fox jumps over the lazy dog"
        assert find_pattern(text, "quick brown") == [4]

    def test_repeated_within_pattern(self) -> None:
        """Pattern search when pattern has internal repetition."""
        assert find_pattern("aabaabaac", "aabaac") == [3]

    def test_all_same_chars(self) -> None:
        """Text and pattern with all same characters."""
        assert find_pattern("aaaaaaa", "aa") == [0, 1, 2, 3, 4, 5]

    def test_whitespace_pattern(self) -> None:
        """Pattern with whitespace."""
        assert find_pattern("a b a b a", " a ") == [3]
