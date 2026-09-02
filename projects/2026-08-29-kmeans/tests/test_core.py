"""Tests for k-means clustering."""

import math

import pytest

from kmeans.core import KMeans


def test_kmeans_init_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        KMeans(k=0)

    with pytest.raises(ValueError, match="k must be positive"):
        KMeans(k=-1)


def test_kmeans_init_invalid_iterations() -> None:
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        KMeans(k=2, max_iterations=0)


def test_kmeans_fit_empty_points() -> None:
    kmeans = KMeans(k=2)
    with pytest.raises(ValueError, match="points cannot be empty"):
        kmeans.fit([])


def test_kmeans_fit_dimension_mismatch() -> None:
    kmeans = KMeans(k=2)
    points = [[1.0, 2.0], [3.0, 4.0, 5.0]]
    with pytest.raises(ValueError, match="same dimension"):
        kmeans.fit(points)


def test_kmeans_single_cluster() -> None:
    kmeans = KMeans(k=1, random_seed=42)
    points = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    result = kmeans.fit(points)

    assert len(result.centroids) == 1
    assert len(result.assignments) == 3
    assert all(a == 0 for a in result.assignments)
    centroid = list(result.centroids[0])
    assert math.isclose(centroid[0], 2.0)
    assert math.isclose(centroid[1], 2.0)


def test_kmeans_two_clusters_well_separated() -> None:
    kmeans = KMeans(k=2, random_seed=42)
    points = [
        [0.0, 0.0],
        [0.5, 0.5],
        [10.0, 10.0],
        [10.5, 10.5],
    ]
    result = kmeans.fit(points)

    assert len(result.centroids) == 2
    assert len(result.assignments) == 4

    close_points = [i for i, a in enumerate(result.assignments) if a == result.assignments[0]]
    far_points = [i for i, a in enumerate(result.assignments) if a != result.assignments[0]]

    assert len(close_points) == 2
    assert len(far_points) == 2
    assert set(close_points) == {0, 1} or set(close_points) == {2, 3}


def test_kmeans_more_clusters_than_points() -> None:
    kmeans = KMeans(k=5, random_seed=42)
    points = [[1.0, 1.0], [2.0, 2.0]]
    result = kmeans.fit(points)

    assert len(result.centroids) == 5
    assert len(result.assignments) == 2
    assert all(0 <= a < 5 for a in result.assignments)


def test_kmeans_1d_points() -> None:
    kmeans = KMeans(k=2, random_seed=42)
    points = [[0.0], [1.0], [10.0], [11.0]]
    result = kmeans.fit(points)

    assert len(result.centroids) == 2
    assert len(result.assignments) == 4

    low_cluster = [i for i, a in enumerate(result.assignments) if a == result.assignments[0]]
    high_cluster = [i for i, a in enumerate(result.assignments) if a != result.assignments[0]]

    assert len(low_cluster) == 2
    assert len(high_cluster) == 2


def test_kmeans_deterministic_with_seed() -> None:
    points = [[float(i), float(i)] for i in range(10)]

    result1 = KMeans(k=3, random_seed=123).fit(points)
    result2 = KMeans(k=3, random_seed=123).fit(points)

    assert result1.assignments == result2.assignments
    for c1, c2 in zip(result1.centroids, result2.centroids, strict=True):
        assert all(math.isclose(x, y) for x, y in zip(c1, c2, strict=True))


def test_kmeans_convergence() -> None:
    kmeans = KMeans(k=2, max_iterations=50, random_seed=42)
    points = [
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0],
        [10.1, 10.1],
    ]
    result = kmeans.fit(points)

    assert len(result.centroids) == 2
    assert len(result.assignments) == 4
