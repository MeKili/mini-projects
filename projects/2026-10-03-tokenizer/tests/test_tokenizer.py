"""Tests for tokenizer module."""

import pytest

from tokenizer import (
    CamelCaseTokenizer,
    PunctuationTokenizer,
    RegexTokenizer,
    WhitespaceTokenizer,
    tokenize,
)


class TestWhitespaceTokenizer:
    """Test whitespace-based tokenization."""

    def test_simple_split(self) -> None:
        tok = WhitespaceTokenizer()
        assert tok.tokenize("hello world") == ["hello", "world"]

    def test_multiple_spaces(self) -> None:
        tok = WhitespaceTokenizer()
        assert tok.tokenize("hello   world") == ["hello", "world"]

    def test_tabs_and_newlines(self) -> None:
        tok = WhitespaceTokenizer()
        assert tok.tokenize("hello\tworld\ntest") == ["hello", "world", "test"]

    def test_empty_string(self) -> None:
        tok = WhitespaceTokenizer()
        assert tok.tokenize("") == []

    def test_only_whitespace(self) -> None:
        tok = WhitespaceTokenizer()
        assert tok.tokenize("   \t\n  ") == []

    def test_single_word(self) -> None:
        tok = WhitespaceTokenizer()
        assert tok.tokenize("hello") == ["hello"]


class TestPunctuationTokenizer:
    """Test punctuation-aware tokenization."""

    def test_simple_text(self) -> None:
        tok = PunctuationTokenizer()
        result = tok.tokenize("Hello, world!")
        assert result == ["Hello", ",", "world", "!"]

    def test_mixed_punctuation(self) -> None:
        tok = PunctuationTokenizer()
        result = tok.tokenize("It's great.")
        assert result == ["It", "'", "s", "great", "."]

    def test_no_punctuation(self) -> None:
        tok = PunctuationTokenizer()
        assert tok.tokenize("hello world") == ["hello", "world"]

    def test_only_punctuation(self) -> None:
        tok = PunctuationTokenizer()
        assert tok.tokenize("!@#$%") == ["!", "@", "#", "$", "%"]

    def test_empty_string(self) -> None:
        tok = PunctuationTokenizer()
        assert tok.tokenize("") == []

    def test_hyphenated_word(self) -> None:
        tok = PunctuationTokenizer()
        result = tok.tokenize("well-known")
        assert result == ["well", "-", "known"]


class TestRegexTokenizer:
    """Test regex-based tokenization."""

    def test_default_word_pattern(self) -> None:
        tok = RegexTokenizer()
        result = tok.tokenize("hello123 world!")
        assert result == ["hello123", "world"]

    def test_custom_pattern(self) -> None:
        tok = RegexTokenizer(r"\d+")
        result = tok.tokenize("abc123def456ghi")
        assert result == ["123", "456"]

    def test_letter_only_pattern(self) -> None:
        tok = RegexTokenizer(r"[a-z]+")
        result = tok.tokenize("Hello123World456test")
        assert result == ["ello", "orld", "test"]

    def test_empty_result(self) -> None:
        tok = RegexTokenizer(r"\d+")
        assert tok.tokenize("abc def ghi") == []

    def test_camelcase_pattern(self) -> None:
        tok = RegexTokenizer(r"[A-Z]?[a-z]+")
        result = tok.tokenize("HelloWorld")
        assert result == ["Hello", "World"]

    def test_empty_string(self) -> None:
        tok = RegexTokenizer()
        assert tok.tokenize("") == []


class TestCamelCaseTokenizer:
    """Test camelCase/PascalCase tokenization."""

    def test_camelcase(self) -> None:
        tok = CamelCaseTokenizer()
        result = tok.tokenize("helloWorld")
        assert result == ["hello", "World"]

    def test_pascalcase(self) -> None:
        tok = CamelCaseTokenizer()
        result = tok.tokenize("HelloWorld")
        assert result == ["Hello", "World"]

    def test_consecutive_capitals(self) -> None:
        tok = CamelCaseTokenizer()
        result = tok.tokenize("HTTPServer")
        assert result == ["HTTP", "Server"]

    def test_snake_case(self) -> None:
        tok = CamelCaseTokenizer()
        result = tok.tokenize("hello_world")
        assert result == ["hello_world"]

    def test_mixed_separators(self) -> None:
        tok = CamelCaseTokenizer()
        result = tok.tokenize("myVar_name.property")
        assert result == ["my", "Var_name", "property"]

    def test_single_word(self) -> None:
        tok = CamelCaseTokenizer()
        assert tok.tokenize("hello") == ["hello"]

    def test_empty_string(self) -> None:
        tok = CamelCaseTokenizer()
        assert tok.tokenize("") == []

    def test_numbers_in_camelcase(self) -> None:
        tok = CamelCaseTokenizer()
        result = tok.tokenize("var123Name456")
        assert result == ["var123", "Name456"]


class TestTokenizeFunction:
    """Test the convenience tokenize function."""

    def test_whitespace_strategy(self) -> None:
        result = tokenize("hello world", strategy="whitespace")
        assert result == ["hello", "world"]

    def test_punctuation_strategy(self) -> None:
        result = tokenize("Hello, world!", strategy="punctuation")
        assert result == ["Hello", ",", "world", "!"]

    def test_regex_strategy(self) -> None:
        result = tokenize("hello world", strategy="regex")
        assert result == ["hello", "world"]

    def test_camelcase_strategy(self) -> None:
        result = tokenize("helloWorld", strategy="camelcase")
        assert result == ["hello", "World"]

    def test_default_strategy(self) -> None:
        result = tokenize("hello world")
        assert result == ["hello", "world"]

    def test_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            tokenize("hello", strategy="unknown")


class TestEdgeCases:
    """Test edge cases and special inputs."""

    def test_unicode_text(self) -> None:
        tok = WhitespaceTokenizer()
        result = tok.tokenize("café naïve résumé")
        assert result == ["café", "naïve", "résumé"]

    def test_very_long_text(self) -> None:
        tok = WhitespaceTokenizer()
        long_text = " ".join(["word"] * 10000)
        result = tok.tokenize(long_text)
        assert len(result) == 10000
        assert all(w == "word" for w in result)

    def test_punctuation_with_numbers(self) -> None:
        tok = PunctuationTokenizer()
        result = tok.tokenize("123-456")
        assert result == ["123", "-", "456"]
