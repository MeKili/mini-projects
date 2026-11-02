"""Rabin-Karp rolling hash algorithm for efficient pattern matching."""

from __future__ import annotations


def find_all_matches(text: str, pattern: str, prime: int = 101) -> list[int]:
    """Find all starting indices where pattern occurs in text using rolling hash.

    Args:
        text: The text to search in.
        pattern: The pattern to find.
        prime: A prime number for the rolling hash base (default 101).

    Returns:
        List of starting indices where pattern is found, sorted in ascending order.

    Raises:
        ValueError: If pattern is empty or longer than text.
    """
    if not pattern:
        raise ValueError("pattern cannot be empty")
    if len(pattern) > len(text):
        raise ValueError("pattern cannot be longer than text")

    # Use a large prime for the base of the hash
    base = 256
    matches: list[int] = []

    if len(pattern) == 0:
        return matches

    # Precompute base^(pattern_len - 1) % prime for removal in rolling hash
    power = pow(base, len(pattern) - 1, prime)

    # Compute initial hash for pattern and first window of text
    pattern_hash = 0
    window_hash = 0

    for i in range(len(pattern)):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % prime
        window_hash = (window_hash * base + ord(text[i])) % prime

    # Slide the window through the text
    for i in range(len(text) - len(pattern) + 1):
        # If hashes match, verify the actual strings (to avoid false positives)
        if pattern_hash == window_hash and text[i : i + len(pattern)] == pattern:
            matches.append(i)

        # Compute hash for next window
        if i < len(text) - len(pattern):
            # Remove leading character from window_hash
            window_hash = (
                (window_hash - ord(text[i]) * power) * base + ord(text[i + len(pattern)])
            ) % prime
            # Ensure positive hash
            window_hash = (window_hash + prime) % prime

    return matches


def find_first_match(text: str, pattern: str, prime: int = 101) -> int | None:
    """Find the first occurrence of pattern in text.

    Args:
        text: The text to search in.
        pattern: The pattern to find.
        prime: A prime number for the rolling hash base (default 101).

    Returns:
        The index of the first match, or None if not found.

    Raises:
        ValueError: If pattern is empty or longer than text.
    """
    matches = find_all_matches(text, pattern, prime)
    return matches[0] if matches else None


def count_occurrences(text: str, pattern: str, prime: int = 101) -> int:
    """Count the number of non-overlapping occurrences of pattern in text.

    Args:
        text: The text to search in.
        pattern: The pattern to find.
        prime: A prime number for the rolling hash base (default 101).

    Returns:
        The count of non-overlapping matches.

    Raises:
        ValueError: If pattern is empty or longer than text.
    """
    matches = find_all_matches(text, pattern, prime)
    if not matches:
        return 0

    # Greedily consume non-overlapping matches
    count = 0
    last_match_end = -1
    for match_start in matches:
        if match_start >= last_match_end:
            count += 1
            last_match_end = match_start + len(pattern)

    return count
