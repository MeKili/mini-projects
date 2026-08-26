"""reservoir-sampling — select k random items from a stream in a single pass.

Algorithm R (Vitter, 1985) maintains a reservoir of k items with uniform random selection
from unknown-size streams using O(k) space and single-pass O(n) time.
"""

from reservoir_sampling.core import ReservoirSampler, sample, sample_iter

__all__ = ["ReservoirSampler", "sample", "sample_iter"]
__version__ = "0.1.0"
