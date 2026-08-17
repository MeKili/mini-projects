"""Tests for cosine similarity and top-k retrieval."""

import math

from cosine_topk.core import cosine_similarity, top_k


def test_cosine_identical() -> None:
    assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)


def test_cosine_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_length_mismatch() -> None:
    import pytest

    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


def test_top_k_orders_by_similarity() -> None:
    query = [1.0, 0.0]
    vectors = [[0.0, 1.0], [1.0, 0.0], [0.9, 0.1]]
    result = top_k(query, vectors, 2)
    assert [index for index, _ in result] == [1, 2]


def test_top_k_zero_returns_empty() -> None:
    assert top_k([1.0], [[1.0]], 0) == []
