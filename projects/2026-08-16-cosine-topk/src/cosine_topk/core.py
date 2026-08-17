"""Cosine similarity and top-k retrieval over dense float vectors."""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = Sequence[float]


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Return the cosine similarity of two equal-length vectors (0.0 if either is zero)."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_k(query: Vector, vectors: Sequence[Vector], k: int) -> list[tuple[int, float]]:
    """Return the (index, score) of the ``k`` vectors most similar to ``query``.

    Sorted by descending similarity, ties broken by ascending index.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scored = [(index, cosine_similarity(query, vector)) for index, vector in enumerate(vectors)]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]
