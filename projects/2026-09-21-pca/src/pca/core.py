"""Principal component analysis (PCA) for dimensionality reduction."""

from __future__ import annotations

import math
from collections.abc import Sequence


class PCA:
    """Principal component analysis via eigendecomposition of centered data.

    Fit on training data to learn mean and principal components, then transform
    new data into the reduced subspace. All computations are pure Python.
    """

    def __init__(self, n_components: int) -> None:
        """Initialize PCA with the number of components to retain."""
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.mean: list[float] | None = None
        self.components: list[list[float]] | None = None
        self.explained_variance: list[float] | None = None

    def fit(self, X: Sequence[Sequence[float]]) -> PCA:
        """Fit PCA on training data X (n_samples x n_features)."""
        if not X or not X[0]:
            raise ValueError("X must be non-empty")
        n_samples = len(X)
        n_features = len(X[0])

        if self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed n_features ({n_features})"
            )

        # Compute mean and center the data
        self.mean = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        X_centered = [[X[i][j] - self.mean[j] for j in range(n_features)] for i in range(n_samples)]

        # Compute covariance matrix: (X_centered^T @ X_centered) / (n_samples - 1)
        cov = self._covariance_matrix(X_centered, n_samples, n_features)

        # Eigendecomposition of covariance matrix
        eigenvalues, eigenvectors = self._eigen_decomposition(cov, n_features)

        # Sort by eigenvalues descending and take top n_components
        pairs = sorted(
            zip(eigenvalues, eigenvectors, strict=True),
            key=lambda p: p[0],
            reverse=True,
        )
        self.explained_variance = [p[0] for p in pairs[: self.n_components]]
        self.components = [list(p[1]) for p in pairs[: self.n_components]]

        return self

    def transform(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        """Project X onto the principal components."""
        if self.mean is None or self.components is None:
            raise ValueError("fit() must be called before transform()")
        n_features = len(self.mean)
        # Center and project
        return [
            [
                sum((X[i][j] - self.mean[j]) * self.components[k][j] for j in range(n_features))
                for k in range(len(self.components))
            ]
            for i in range(len(X))
        ]

    def fit_transform(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        """Fit PCA and transform X in one step."""
        return self.fit(X).transform(X)

    def _covariance_matrix(
        self, X_centered: list[list[float]], n_samples: int, n_features: int
    ) -> list[list[float]]:
        """Compute covariance matrix from centered data."""
        cov = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                cov[i][j] = (
                    sum(X_centered[s][i] * X_centered[s][j] for s in range(n_samples))
                    / (n_samples - 1)
                    if n_samples > 1
                    else 0.0
                )
        return cov

    def _eigen_decomposition(
        self, matrix: list[list[float]], n: int
    ) -> tuple[list[float], list[list[float]]]:
        """Approximate eigendecomposition via power iteration."""
        eigenvalues: list[float] = []
        eigenvectors: list[list[float]] = []

        # Make a working copy
        A = [row[:] for row in matrix]

        for _ in range(self.n_components):
            # Power iteration to find dominant eigenvector
            v = [1.0 / math.sqrt(n)] * n
            for _ in range(50):  # 50 iterations usually converges
                Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
                norm = math.sqrt(sum(x * x for x in Av))
                if norm < 1e-10:
                    break
                v = [x / norm for x in Av]

            # Eigenvalue is v^T @ A @ v
            Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
            eigenvalue = sum(v[i] * Av[i] for i in range(n))
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)

            # Deflate: A = A - eigenvalue * v * v^T
            for i in range(n):
                for j in range(n):
                    A[i][j] -= eigenvalue * v[i] * v[j]

        return eigenvalues, eigenvectors
