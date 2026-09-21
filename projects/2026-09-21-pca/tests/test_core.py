"""Tests for PCA (principal component analysis)."""

import math

import pytest

from pca.core import PCA


def test_pca_fit_transform_2d_to_1d() -> None:
    """PCA should reduce 2D data to 1D."""
    X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    pca = PCA(n_components=1)
    result = pca.fit_transform(X)
    assert len(result) == 3
    assert len(result[0]) == 1


def test_pca_mean_computation() -> None:
    """PCA should compute mean correctly."""
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    pca = PCA(n_components=1)
    pca.fit(X)
    assert pca.mean is not None
    assert math.isclose(pca.mean[0], 3.0)
    assert math.isclose(pca.mean[1], 4.0)


def test_pca_preserves_variance_direction() -> None:
    """PC1 should align with direction of maximum variance."""
    X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    pca = PCA(n_components=1)
    pca.fit(X)
    assert pca.components is not None
    pc = pca.components[0]
    # First component should have larger absolute value (variance in x-direction)
    assert abs(pc[0]) > abs(pc[1])


def test_pca_explained_variance() -> None:
    """Explained variance should decrease for successive components."""
    X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    pca = PCA(n_components=2)
    pca.fit(X)
    assert pca.explained_variance is not None
    assert len(pca.explained_variance) == 2
    assert pca.explained_variance[0] >= pca.explained_variance[1]


def test_pca_fit_then_transform() -> None:
    """PCA.fit() followed by transform() should match fit_transform()."""
    X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    pca1 = PCA(n_components=1)
    result1 = pca1.fit_transform(X)

    pca2 = PCA(n_components=1)
    pca2.fit(X)
    result2 = pca2.transform(X)

    assert len(result1) == len(result2)
    for r1, r2 in zip(result1, result2, strict=True):
        for v1, v2 in zip(r1, r2, strict=True):
            assert math.isclose(v1, v2, abs_tol=1e-9)


def test_pca_n_components_exceeds_features() -> None:
    """PCA should raise error if n_components > n_features."""
    X = [[1.0, 2.0], [3.0, 4.0]]
    pca = PCA(n_components=3)
    with pytest.raises(ValueError, match="n_components"):
        pca.fit(X)


def test_pca_n_components_invalid() -> None:
    """PCA should reject non-positive n_components."""
    with pytest.raises(ValueError, match="n_components must be positive"):
        PCA(n_components=0)
    with pytest.raises(ValueError, match="n_components must be positive"):
        PCA(n_components=-1)


def test_pca_empty_data() -> None:
    """PCA should reject empty data."""
    pca = PCA(n_components=1)
    with pytest.raises(ValueError, match="non-empty"):
        pca.fit([])


def test_pca_transform_without_fit() -> None:
    """PCA.transform() should fail if fit() wasn't called."""
    pca = PCA(n_components=1)
    with pytest.raises(ValueError, match="fit"):
        pca.transform([[1.0, 2.0]])


def test_pca_multi_sample_reduction() -> None:
    """PCA should reduce 5D data to 2D with 10 samples."""
    X = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [4.0, 5.0, 6.0, 7.0, 8.0],
        [5.0, 6.0, 7.0, 8.0, 9.0],
    ]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    assert len(result) == 5
    assert len(result[0]) == 2
    # Verify reduced data is not all zeros
    assert any(abs(x) > 1e-6 for row in result for x in row)
