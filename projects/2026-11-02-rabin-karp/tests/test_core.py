"""Tests for Rabin-Karp rolling hash pattern matching."""

import pytest

from rabin_karp import count_occurrences, find_all_matches, find_first_match


class TestFindAllMatches:
    def test_single_match(self) -> None:
        assert find_all_matches("hello world", "world") == [6]

    def test_multiple_matches(self) -> None:
        assert find_all_matches("ababab", "ab") == [0, 2, 4]

    def test_overlapping_matches(self) -> None:
        assert find_all_matches("aaaa", "aa") == [0, 1, 2]

    def test_no_match(self) -> None:
        assert find_all_matches("hello", "xyz") == []

    def test_pattern_at_start(self) -> None:
        assert find_all_matches("pattern test", "pattern") == [0]

    def test_pattern_at_end(self) -> None:
        assert find_all_matches("test pattern", "pattern") == [5]

    def test_entire_text_is_pattern(self) -> None:
        assert find_all_matches("hello", "hello") == [0]

    def test_single_character_pattern(self) -> None:
        assert find_all_matches("aaa", "a") == [0, 1, 2]

    def test_single_character_match(self) -> None:
        result = find_all_matches("programming", "g")
        assert result == [3, 10]

    def test_repeated_pattern(self) -> None:
        assert find_all_matches("aaaaaa", "aaa") == [0, 1, 2, 3]

    def test_case_sensitive(self) -> None:
        assert find_all_matches("Hello hello", "hello") == [6]

    def test_special_characters(self) -> None:
        assert find_all_matches("hello@world@test", "@") == [5, 11]

    def test_whitespace_in_pattern(self) -> None:
        assert find_all_matches("hello world test", "o w") == [4]

    def test_unicode_characters(self) -> None:
        assert find_all_matches("café café", "café") == [0, 5]

    def test_pattern_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="pattern cannot be empty"):
            find_all_matches("hello", "")

    def test_pattern_longer_than_text_raises(self) -> None:
        with pytest.raises(ValueError, match="pattern cannot be longer than text"):
            find_all_matches("hi", "hello")


class TestFindFirstMatch:
    def test_finds_first_occurrence(self) -> None:
        assert find_first_match("abab", "ab") == 0

    def test_finds_first_of_multiple(self) -> None:
        assert find_first_match("abcabc", "bc") == 1

    def test_returns_none_for_no_match(self) -> None:
        assert find_first_match("hello", "xyz") is None

    def test_single_character(self) -> None:
        assert find_first_match("hello", "l") == 2

    def test_entire_text(self) -> None:
        assert find_first_match("hello", "hello") == 0

    def test_at_end(self) -> None:
        assert find_first_match("hello", "lo") == 3

    def test_pattern_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="pattern cannot be empty"):
            find_first_match("hello", "")


class TestCountOccurrences:
    def test_no_overlapping_occurrences(self) -> None:
        assert count_occurrences("abcabc", "ab") == 2

    def test_overlapping_matches_are_non_overlapping(self) -> None:
        assert count_occurrences("aaaa", "aa") == 2

    def test_no_match(self) -> None:
        assert count_occurrences("hello", "xyz") == 0

    def test_single_match(self) -> None:
        assert count_occurrences("hello world", "world") == 1

    def test_repeated_non_overlapping(self) -> None:
        assert count_occurrences("ababab", "ab") == 3

    def test_pattern_equals_text(self) -> None:
        assert count_occurrences("hello", "hello") == 1

    def test_single_character(self) -> None:
        assert count_occurrences("aaa", "a") == 3

    def test_pattern_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="pattern cannot be empty"):
            count_occurrences("hello", "")

    def test_overlapping_vs_non_overlapping(self) -> None:
        # "aaa" has overlapping matches of "aa" at [0,1]
        # non-overlapping count should be 1
        assert count_occurrences("aaa", "aa") == 1

    def test_complex_overlapping(self) -> None:
        # "abababa" with pattern "aba"
        # matches at [0, 2, 4]
        # non-overlapping: [0] (covers 0-2), then [3, 4] invalid since 4 is before 3
        # Actually [0] covers 0-2, [4] is valid → 2 matches
        assert count_occurrences("abababa", "aba") == 2


class TestCustomPrime:
    def test_custom_prime_parameter(self) -> None:
        # Should work with different prime
        assert find_all_matches("hello world", "world", prime=1009) == [6]

    def test_custom_prime_first_match(self) -> None:
        assert find_first_match("hello world", "world", prime=1009) == 6

    def test_custom_prime_count(self) -> None:
        assert count_occurrences("ababab", "ab", prime=1009) == 3


class TestEdgeCases:
    def test_two_character_text_two_character_pattern(self) -> None:
        assert find_all_matches("ab", "ab") == [0]

    def test_long_text_short_pattern(self) -> None:
        text = "a" * 1000 + "b" + "a" * 1000
        assert find_all_matches(text, "b") == [1000]

    def test_repeated_characters(self) -> None:
        assert find_all_matches("aaaaaaa", "aa") == [0, 1, 2, 3, 4, 5]

    def test_pattern_all_same_character(self) -> None:
        assert find_all_matches("bbbbb", "bbb") == [0, 1, 2]

    def test_numeric_strings(self) -> None:
        assert find_all_matches("123123123", "23") == [1, 4, 7]

    def test_mixed_whitespace(self) -> None:
        assert find_all_matches("a b\tc\nd", " b\tc") == [1]
