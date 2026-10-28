"""Knuth-Morris-Pratt string pattern matching algorithm."""

from .core import build_failure_function, find_pattern

__all__ = ["find_pattern", "build_failure_function"]
