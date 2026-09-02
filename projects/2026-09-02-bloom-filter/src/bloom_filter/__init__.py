"""bloom-filter — a space-efficient probabilistic data structure for fast membership testing.

Given a false positive rate target, automatically sizes a bit array and chooses hash functions
to minimize space while bounding the probability of false positives. Never produces false negatives.
"""

from bloom_filter.core import BloomFilter

__all__ = ["BloomFilter"]
__version__ = "0.1.0"
