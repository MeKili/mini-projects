"""kmeans — K-means clustering from scratch.

Given a dataset and k (number of clusters), partition points into k clusters by
repeatedly assigning each point to the nearest centroid and updating centroids
until convergence.
"""

from kmeans.core import KMeans

__all__ = ["KMeans"]
__version__ = "0.1.0"
