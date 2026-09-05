"""running-stats — streaming mean and variance using Welford's algorithm.

Numerically stable computation of mean and variance from a data stream, without storing
all values in memory. Useful for normalizing features or detecting anomalies.
"""

from running_stats.core import RunningStat

__all__ = ["RunningStat"]
__version__ = "0.1.0"
