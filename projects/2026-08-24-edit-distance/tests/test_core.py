"""Tests for Levenshtein distance and fuzzy matching."""

import pytest

from edit_distance import closest_match, distance, similarity_ratio


class TestDistance:
    """Tests for the distance function."""

    def test_identical_strings(self) -> None:
        assert distance("hello", "hello") == 0
        assert distance("", "") == 0
        assert distance("a", "a") == 0

    def test_one_empty_string(self) -> None:
        assert distance("", "abc") == 3
        assert distance("abc", "") == 3
        assert distance("", "x") == 1

    def test_single_character_edits(self) -> None:
        assert distance("cat", "bat") == 1  # substitution
        assert distance("cat", "ca") == 1  # deletion
        assert distance("ca", "cat") == 1  # insertion

    def test_multiple_edits(self) -> None:
        assert distance("kitten", "sitting") == 3
        assert distance("saturday", "sunday") == 3
        assert distance("abc", "def") == 3

    def test_case_sensitivity(self) -> None:
        assert distance("Hello", "hello") == 1
        assert distance("HELLO", "hello") == 5

    def test_longer_strings(self) -> None:
        assert distance("algorithm", "altruistic") == 6
        assert distance("programming", "programmer") == 3

    def test_unicode_strings(self) -> None:
        assert distance("café", "cafe") == 1
        assert distance("🎉", "🎉") == 0
        assert distance("🎉", "🎊") == 1


class TestSimilarityRatio:
    """Tests for the similarity_ratio function."""

    def test_identical_strings(self) -> None:
        assert similarity_ratio("hello", "hello") == 1.0
        assert similarity_ratio("", "") == 1.0

    def test_completely_different(self) -> None:
        assert similarity_ratio("abc", "def") == 0.0

    def test_partial_match(self) -> None:
        ratio = similarity_ratio("cat", "bat")
        assert 0.0 < ratio < 1.0

    def test_empty_string(self) -> None:
        assert similarity_ratio("hello", "") == 0.0
        assert similarity_ratio("", "hello") == 0.0

    def test_one_char_difference(self) -> None:
        ratio = similarity_ratio("kitten", "sitting")
        assert ratio == pytest.approx((7 - 3) / 7)

    def test_symmetry(self) -> None:
        ratio1 = similarity_ratio("hello", "world")
        ratio2 = similarity_ratio("world", "hello")
        assert ratio1 == ratio2

    def test_single_character(self) -> None:
        assert similarity_ratio("a", "a") == 1.0
        assert similarity_ratio("a", "b") == 0.0


class TestClosestMatch:
    """Tests for the closest_match function."""

    def test_empty_candidates(self) -> None:
        assert closest_match("hello", []) is None

    def test_exact_match(self) -> None:
        candidates = ["hello", "world", "help"]
        result = closest_match("hello", candidates)
        assert result == ("hello", 1.0)

    def test_closest_match(self) -> None:
        candidates = ["cat", "dog", "bat"]
        result = closest_match("car", candidates)
        assert result is not None
        assert result[0] in ["cat", "bat"]

    def test_threshold_filtering(self) -> None:
        candidates = ["hello", "world"]
        result = closest_match("hello", candidates, threshold=0.9)
        assert result == ("hello", 1.0)

        result = closest_match("goodbye", candidates, threshold=0.9)
        assert result is None

    def test_multiple_same_distance(self) -> None:
        candidates = ["cat", "bat", "hat"]
        result = closest_match("mat", candidates)
        assert result is not None
        assert result[0] in ["cat", "bat", "hat"]
        assert result[1] == pytest.approx(2.0 / 3.0)

    def test_single_candidate(self) -> None:
        result = closest_match("hello", ["hello"])
        assert result == ("hello", 1.0)

        result = closest_match("hello", ["world"])
        assert result is not None
        assert result[0] == "world"

    def test_threshold_exactly_met(self) -> None:
        candidates = ["cat"]
        ratio = similarity_ratio("car", "cat")
        result = closest_match("car", candidates, threshold=ratio)
        assert result is not None
        assert result[0] == "cat"
        assert result[1] == pytest.approx(ratio)

    def test_threshold_just_missed(self) -> None:
        candidates = ["cat"]
        ratio = similarity_ratio("car", "cat")
        result = closest_match("car", candidates, threshold=ratio + 0.01)
        assert result is None

    def test_practical_spell_checking(self) -> None:
        words = ["python", "java", "javascript", "ruby", "go"]
        result = closest_match("javaScript", words)
        assert result is not None
        assert result[0] == "javascript"
