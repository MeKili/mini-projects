"""tfidf — TF-IDF (term frequency–inverse document frequency) vectorizer.

Compute TF-IDF vectors from a corpus of text documents. Pure Python, no dependencies.
Includes cosine similarity for comparing document vectors.
"""

from tfidf.core import TfIdfVectorizer, cosine_similarity

__all__ = ["TfIdfVectorizer", "cosine_similarity"]
__version__ = "0.1.0"
