"""cosine-topk — a tiny in-memory top-k similarity search over dense vectors.

Given a query vector and a list of candidate vectors, return the indices of the k
most similar by cosine similarity. Pure Python, no dependencies.
"""

from cosine_topk.core import cosine_similarity, top_k

__all__ = ["cosine_similarity", "top_k"]
__version__ = "0.1.0"
