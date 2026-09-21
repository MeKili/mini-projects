"""pca — Principal component analysis for dimensionality reduction.

Fit PCA on training data to learn the principal components, then transform any
data matrix into the reduced subspace. Pure Python, no dependencies.
"""

from pca.core import PCA

__all__ = ["PCA"]
__version__ = "0.1.0"
