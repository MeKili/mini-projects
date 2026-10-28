"""KMP string matching implementation."""


def build_failure_function(pattern: str) -> list[int]:
    """
    Build the KMP failure function (prefix function) for a pattern.

    The failure function at position i tells us the length of the longest
    proper prefix of pattern[0:i+1] that is also a suffix of pattern[0:i+1].

    Time: O(m) where m is the pattern length.
    """
    m = len(pattern)
    failure = [0] * m

    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        failure[i] = j

    return failure


def find_pattern(text: str, pattern: str) -> list[int]:
    """
    Find all occurrences of pattern in text using KMP algorithm.

    Returns a list of starting indices where the pattern is found.

    Time: O(n + m) where n is text length and m is pattern length.
    Space: O(m) for the failure function.
    """
    if not pattern or not text:
        return []

    if len(pattern) > len(text):
        return []

    failure = build_failure_function(pattern)
    matches: list[int] = []

    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = failure[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            matches.append(i - len(pattern) + 1)
            j = failure[j - 1]

    return matches
