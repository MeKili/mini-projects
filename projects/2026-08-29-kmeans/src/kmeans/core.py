"""K-means clustering algorithm implementation."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

Point = Sequence[float]


@dataclass
class KMeans:
    """K-means clusterer: partition points into k clusters by iterative refinement."""

    k: int
    max_iterations: int = 100
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

    def fit(self, points: Sequence[Point]) -> KMeansResult:
        """Fit clusters to points and return centroids and assignments.

        Uses k-means++ initialization for better starting centroids.
        """
        if not points:
            raise ValueError("points cannot be empty")

        dim = len(points[0])
        if any(len(p) != dim for p in points):
            raise ValueError("all points must have the same dimension")

        rng = random.Random(self.random_seed)
        centroids = self._init_centroids(points, rng)

        for _ in range(self.max_iterations):
            assignments = [self._nearest_centroid(point, centroids) for point in points]

            new_centroids = self._update_centroids(points, assignments, dim)

            if all(
                self._distance(old, new) < 1e-6
                for old, new in zip(centroids, new_centroids, strict=True)
            ):
                break

            centroids = new_centroids

        assignments = [self._nearest_centroid(point, centroids) for point in points]

        return KMeansResult(
            centroids=centroids,
            assignments=assignments,
        )

    def _init_centroids(self, points: Sequence[Point], rng: random.Random) -> list[list[float]]:
        """Initialize centroids using k-means++ seeding."""
        centroids: list[list[float]] = []

        centroids.append(list(rng.choice(list(points))))

        for _ in range(1, self.k):
            distances = [
                min(self._distance(point, centroid) ** 2 for centroid in centroids)
                for point in points
            ]
            total_dist = sum(distances)
            if total_dist == 0:
                centroids.append(list(rng.choice(list(points))))
            else:
                probabilities = [d / total_dist for d in distances]
                idx = rng.choices(range(len(points)), weights=probabilities, k=1)[0]
                centroids.append(list(points[idx]))

        return centroids

    def _nearest_centroid(self, point: Point, centroids: Sequence[Point]) -> int:
        """Return index of nearest centroid to point."""
        return min(
            range(len(centroids)),
            key=lambda i: self._distance(point, centroids[i]),
        )

    def _update_centroids(
        self,
        points: Sequence[Point],
        assignments: Sequence[int],
        dim: int,
    ) -> list[list[float]]:
        """Compute new centroids as mean of assigned points."""
        new_centroids: list[list[float]] = [[0.0] * dim for _ in range(self.k)]
        counts = [0] * self.k

        for point, cluster in zip(points, assignments, strict=True):
            for j, val in enumerate(point):
                new_centroids[cluster][j] += val
            counts[cluster] += 1

        for i in range(self.k):
            if counts[i] > 0:
                for j in range(dim):
                    new_centroids[i][j] /= counts[i]
            else:
                new_centroids[i] = [0.0] * dim

        return new_centroids

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        """Euclidean distance between two points."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


@dataclass
class KMeansResult:
    """Result of k-means fitting."""

    centroids: Sequence[Point]
    assignments: Sequence[int]
