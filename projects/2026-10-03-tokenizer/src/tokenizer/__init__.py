"""
Text tokenizer with multiple strategies: whitespace, punctuation-aware, and regex-based.
Useful for splitting text into tokens for NLP tasks.
"""

import re
import string
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Base class for text tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Split text into tokens."""


class WhitespaceTokenizer(Tokenizer):
    """Simple whitespace-based tokenizer."""

    def tokenize(self, text: str) -> list[str]:
        """Split on whitespace and filter empty strings."""
        return [t for t in text.split() if t]


class PunctuationTokenizer(Tokenizer):
    """Tokenizer that separates punctuation from words."""

    def tokenize(self, text: str) -> list[str]:
        """Split text, separating punctuation and words."""
        tokens: list[str] = []
        current = ""

        for char in text:
            if char in string.punctuation:
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(char)
            elif char.isspace():
                if current:
                    tokens.append(current)
                    current = ""
            else:
                current += char

        if current:
            tokens.append(current)

        return tokens


class RegexTokenizer(Tokenizer):
    """Tokenizer using a regex pattern."""

    def __init__(self, pattern: str = r"\w+"):
        """
        Initialize with a regex pattern.

        Args:
            pattern: Regex pattern to match tokens (default: word characters)
        """
        self.pattern = pattern

    def tokenize(self, text: str) -> list[str]:
        """Extract tokens matching the regex pattern."""
        return re.findall(self.pattern, text)


class CamelCaseTokenizer(Tokenizer):
    """Tokenizer that splits camelCase and PascalCase identifiers."""

    def tokenize(self, text: str) -> list[str]:
        """Split camelCase/PascalCase, whitespace, and punctuation."""
        s1 = re.sub("([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
        s2 = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", s1)
        s3 = re.sub(r"[^\w\s]", " ", s2)
        return [t for t in s3.split() if t]


def tokenize(text: str, strategy: str = "whitespace") -> list[str]:
    """
    Tokenize text using the specified strategy.

    Args:
        text: Text to tokenize
        strategy: One of 'whitespace', 'punctuation', 'regex', or 'camelcase'

    Returns:
        List of tokens

    Raises:
        ValueError: If strategy is unknown
    """
    tokenizers: dict[str, Tokenizer] = {
        "whitespace": WhitespaceTokenizer(),
        "punctuation": PunctuationTokenizer(),
        "regex": RegexTokenizer(),
        "camelcase": CamelCaseTokenizer(),
    }

    if strategy not in tokenizers:
        raise ValueError(f"Unknown strategy: {strategy}")

    return tokenizers[strategy].tokenize(text)


__all__ = [
    "Tokenizer",
    "WhitespaceTokenizer",
    "PunctuationTokenizer",
    "RegexTokenizer",
    "CamelCaseTokenizer",
    "tokenize",
]
