"""Longest common subsequence using dynamic programming.

The longest common subsequence (LCS) of two sequences is the longest sequence that appears
in both input sequences in the same order (but not necessarily consecutive). For example,
LCS("AGGTAB", "GXTXAYB") is "GTAB".

Uses O(m*n) time and space via standard DP table construction.
"""

from __future__ import annotations


def lcs_length(a: str, b: str) -> int:
    """Return the length of the longest common subsequence of two strings."""
    m, n = len(a), len(b)
    dp: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def lcs_sequence(a: str, b: str) -> str:
    """Return the longest common subsequence of two strings.

    If multiple LCS exist, returns the one found by left-preferring reconstruction.
    """
    m, n = len(a), len(b)
    dp: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    result: list[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return "".join(reversed(result))
