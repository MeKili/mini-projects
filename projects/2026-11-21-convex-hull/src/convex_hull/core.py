"""Graham scan algorithm for computing the convex hull of a set of 2D points."""

from __future__ import annotations

import math
from typing import NamedTuple


class Point(NamedTuple):
    """A 2D point with integer coordinates."""

    x: int
    y: int


def _cross_product(o: Point, a: Point, b: Point) -> int:
    """Return the cross product of vectors OA and OB.

    Positive: counter-clockwise turn, Negative: clockwise turn, Zero: collinear.
    """
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def _distance_squared(p1: Point, p2: Point) -> int:
    """Return the squared distance between two points."""
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2


def convex_hull(points: list[Point]) -> list[Point]:
    """Return the convex hull of a set of points using Graham scan.

    Points are returned in counter-clockwise order starting from the
    bottom-most point (lowest y, leftmost x if tied).

    Args:
        points: A list of 2D points.

    Returns:
        A list of points forming the convex hull in counter-clockwise order.

    Raises:
        ValueError: If fewer than 3 points are provided.
    """
    if len(points) < 3:
        raise ValueError("at least 3 points are required")

    # Find the starting point: lowest y, leftmost x if tied
    start = min(points, key=lambda p: (p.y, p.x))

    # Sort all other points by polar angle, then by distance
    def polar_angle_and_distance(p: Point) -> tuple[float, int]:
        """Return (polar angle in radians, distance squared) for sorting."""
        if p == start:
            return (-math.pi, 0)
        angle = math.atan2(p.y - start.y, p.x - start.x)
        dist = _distance_squared(start, p)
        return (angle, dist)

    sorted_points = sorted(points, key=polar_angle_and_distance)

    # Build the hull using a stack
    hull: list[Point] = []
    for p in sorted_points:
        # Remove points that make a clockwise turn
        while len(hull) >= 2 and _cross_product(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull
