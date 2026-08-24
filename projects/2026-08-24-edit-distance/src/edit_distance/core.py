"""Levenshtein distance and fuzzy string matching."""

from __future__ import annotations


def distance(s1: str, s2: str) -> int:
    """Return the Levenshtein distance between two strings.

    The distance is the minimum number of single-character edits (insertions,
    deletions, substitutions) required to transform one string into another.
    """
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr_row = [i] + [0] * len(s2)
        for j, c2 in enumerate(s2, 1):
            insert = prev_row[j] + 1
            delete = curr_row[j - 1] + 1
            replace = prev_row[j - 1] + (0 if c1 == c2 else 1)
            curr_row[j] = min(insert, delete, replace)
        prev_row = curr_row
    return prev_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """Return a similarity score between 0.0 and 1.0.

    1.0 means the strings are identical; 0.0 means they are completely different.
    Computed as (max_len - distance) / max_len, where max_len is the longer string.
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance(s1, s2) / max_len)


def closest_match(
    query: str, candidates: list[str], threshold: float = 0.0
) -> tuple[str, float] | None:
    """Find the closest match in candidates to the query string.

    Returns (matched_string, score) for the best match with similarity >= threshold,
    or None if no match meets the threshold or candidates is empty.
    """
    if not candidates:
        return None
    best = max(((c, similarity_ratio(query, c)) for c in candidates), key=lambda x: x[1])
    return best if best[1] >= threshold else None
