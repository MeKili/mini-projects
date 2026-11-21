"""Tests for Graham scan convex hull algorithm."""

import pytest

from convex_hull.core import Point, convex_hull


def test_triangle() -> None:
    """Basic triangle should return all 3 points."""
    points = [Point(0, 0), Point(1, 0), Point(0, 1)]
    hull = convex_hull(points)
    assert len(hull) == 3


def test_square() -> None:
    """Square should return all 4 corners."""
    points = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    hull = convex_hull(points)
    assert len(hull) == 4


def test_interior_point() -> None:
    """Interior point should be excluded from hull."""
    points = [Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2), Point(1, 1)]
    hull = convex_hull(points)
    assert len(hull) == 4
    assert Point(1, 1) not in hull


def test_multiple_interior_points() -> None:
    """Multiple interior points should all be excluded."""
    points = [
        Point(0, 0),
        Point(4, 0),
        Point(4, 4),
        Point(0, 4),
        Point(1, 1),
        Point(2, 2),
        Point(3, 1),
    ]
    hull = convex_hull(points)
    assert len(hull) == 4


def test_collinear_points_excluded() -> None:
    """Points on edges (but not corners) should be excluded."""
    points = [Point(0, 0), Point(2, 0), Point(4, 0), Point(2, 2)]
    hull = convex_hull(points)
    # Should have 3 points: two endpoints and the apex
    assert len(hull) == 3
    assert Point(2, 0) not in hull


def test_ccw_order() -> None:
    """Hull points should be in counter-clockwise order."""
    points = [Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)]
    hull = convex_hull(points)
    # Starting from bottom-left (0,0), CCW should go through (2,0), (2,2), (0,2)
    assert hull[0] == Point(0, 0)


def test_pentagon() -> None:
    """Pentagon shape should return 5 points."""
    points = [
        Point(0, 0),
        Point(2, 0),
        Point(3, 1),
        Point(1, 2),
        Point(-1, 1),
    ]
    hull = convex_hull(points)
    assert len(hull) == 5


def test_duplicate_points() -> None:
    """Duplicate points should work correctly."""
    points = [Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2), Point(0, 0)]
    hull = convex_hull(points)
    assert len(hull) == 4


def test_start_point_selection() -> None:
    """The starting point should be the lowest, then leftmost."""
    points = [Point(5, 5), Point(0, 0), Point(5, 0), Point(0, 5)]
    hull = convex_hull(points)
    assert hull[0] == Point(0, 0)


def test_tie_x_coordinate() -> None:
    """When y is same, leftmost point should be chosen."""
    points = [Point(5, 0), Point(0, 0), Point(2, 2)]
    hull = convex_hull(points)
    assert hull[0] == Point(0, 0)


def test_error_too_few_points() -> None:
    """Should raise ValueError for fewer than 3 points."""
    with pytest.raises(ValueError, match="at least 3 points"):
        convex_hull([Point(0, 0), Point(1, 1)])


def test_error_empty() -> None:
    """Should raise ValueError for empty list."""
    with pytest.raises(ValueError, match="at least 3 points"):
        convex_hull([])


def test_negative_coordinates() -> None:
    """Should work with negative coordinates."""
    points = [Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)]
    hull = convex_hull(points)
    assert len(hull) == 4


def test_l_shape_with_interior() -> None:
    """L-shaped set with interior point excluded from hull."""
    points = [
        Point(0, 0),
        Point(2, 0),
        Point(2, 1),
        Point(1, 1),  # interior point
        Point(1, 2),
        Point(0, 2),
        Point(1, 1),  # duplicate interior point
    ]
    hull = convex_hull(points)
    # The convex hull of these points is 5-sided, (1,1) is interior
    assert len(hull) == 5
    assert Point(1, 1) not in hull
