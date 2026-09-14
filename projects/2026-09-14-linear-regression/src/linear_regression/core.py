"""Simple linear regression: fit a line to 2D points, predict, and compute R²."""

import math
from typing import NamedTuple


class LinearRegression(NamedTuple):
    """Fitted linear regression model: y = slope * x + intercept."""

    slope: float
    intercept: float

    def predict(self, x: float) -> float:
        """Predict y value for a given x using the fitted line."""
        return self.slope * x + self.intercept

    def r_squared(self, xs: list[float], ys: list[float]) -> float:
        """Compute R² (coefficient of determination) on the given data."""
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length")
        if len(xs) == 0:
            raise ValueError("Cannot compute R² on empty data")

        mean_y = sum(ys) / len(ys)
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        ss_res = sum((y - self.predict(x)) ** 2 for x, y in zip(xs, ys, strict=True))

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - (ss_res / ss_tot)


def fit(xs: list[float], ys: list[float]) -> LinearRegression:
    """Fit a line to 2D points using least squares.

    Args:
        xs: x-coordinates of points
        ys: y-coordinates of points

    Returns:
        LinearRegression model with slope and intercept

    Raises:
        ValueError: if xs and ys have different lengths or are empty
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) == 0:
        raise ValueError("Cannot fit on empty data")

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator = sum((x - mean_x) ** 2 for x in xs)

    slope = 0.0 if denominator == 0 else numerator / denominator

    intercept = mean_y - slope * mean_x

    return LinearRegression(slope=slope, intercept=intercept)


def rmse(model: LinearRegression, xs: list[float], ys: list[float]) -> float:
    """Compute root mean squared error on the given data."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) == 0:
        raise ValueError("Cannot compute RMSE on empty data")

    ss_res = sum((y - model.predict(x)) ** 2 for x, y in zip(xs, ys, strict=True))
    return math.sqrt(ss_res / len(xs))
