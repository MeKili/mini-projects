"""Tests for longest common subsequence."""

from lcs.core import lcs_length, lcs_sequence


def test_lcs_identical() -> None:
    assert lcs_length("abc", "abc") == 3
    assert lcs_sequence("abc", "abc") == "abc"


def test_lcs_no_overlap() -> None:
    assert lcs_length("abc", "def") == 0
    assert lcs_sequence("abc", "def") == ""


def test_lcs_empty_string() -> None:
    assert lcs_length("", "abc") == 0
    assert lcs_length("abc", "") == 0
    assert lcs_sequence("", "abc") == ""
    assert lcs_sequence("abc", "") == ""


def test_lcs_both_empty() -> None:
    assert lcs_length("", "") == 0
    assert lcs_sequence("", "") == ""


def test_lcs_example_1() -> None:
    a, b = "AGGTAB", "GXTXAYB"
    assert lcs_length(a, b) == 4
    assert lcs_sequence(a, b) == "GTAB"


def test_lcs_example_2() -> None:
    a, b = "ABCDGH", "AEDFHR"
    assert lcs_length(a, b) == 3
    assert lcs_sequence(a, b) == "ADH"


def test_lcs_one_char() -> None:
    assert lcs_length("a", "a") == 1
    assert lcs_sequence("a", "a") == "a"
    assert lcs_length("a", "b") == 0
    assert lcs_sequence("a", "b") == ""


def test_lcs_prefix() -> None:
    a, b = "abc", "abcdef"
    assert lcs_length(a, b) == 3
    assert lcs_sequence(a, b) == "abc"


def test_lcs_interspersed() -> None:
    a, b = "aXbYc", "abc"
    assert lcs_length(a, b) == 3
    assert lcs_sequence(a, b) == "abc"


def test_lcs_repeated_chars() -> None:
    a, b = "aaa", "aa"
    assert lcs_length(a, b) == 2
    assert lcs_sequence(a, b) == "aa"


def test_lcs_longer_sequences() -> None:
    a = "ACCGGTCGAGTGCGCGGAAGCCGGCCGAA"
    b = "GTCGTTCGGAATGCCCCGC"
    result_length = lcs_length(a, b)
    result_seq = lcs_sequence(a, b)
    assert result_length == len(result_seq)
    assert result_length >= 10
