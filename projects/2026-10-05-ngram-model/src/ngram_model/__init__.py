"""N-gram language model for text analysis and prediction.

Build and query n-gram models to analyze text patterns, predict next tokens,
and measure text similarity. Supports any n from 1 (unigrams) to sequence length.
"""

from ngram_model.model import NGramModel

__all__ = ["NGramModel"]
