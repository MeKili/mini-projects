"""linear-regression — simple two-variable linear regression with prediction and evaluation.

Fit a line to 2D points using least squares, predict y-values, and compute R² and RMSE metrics.
Pure Python, no dependencies.
"""

from linear_regression.core import LinearRegression, fit, rmse

__all__ = ["LinearRegression", "fit", "rmse"]
__version__ = "0.1.0"
