"""Markov chain text generator with variable-order models."""

from __future__ import annotations

import random
from collections import defaultdict


class MarkovChain:
    """First-order Markov chain for text generation.

    Builds a probabilistic model from training text where each word
    transitions to the next with learned probabilities.
    """

    def __init__(self, order: int = 1, seed: int | None = None) -> None:
        """Initialize a Markov chain with given order.

        Args:
            order: Number of preceding words to condition on (typically 1 or 2).
            seed: Random seed for reproducible generation.
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.transitions: dict[tuple[str, ...], list[str]] = defaultdict(list)
        self.seed = seed

    def train(self, text: str) -> None:
        """Learn transition probabilities from text.

        Args:
            text: Training text, whitespace-tokenized.
        """
        tokens = text.split()
        if len(tokens) <= self.order:
            return

        for i in range(len(tokens) - self.order):
            context = tuple(tokens[i : i + self.order])
            next_token = tokens[i + self.order]
            self.transitions[context].append(next_token)

    def generate(self, length: int, start: tuple[str, ...] | None = None) -> str:
        """Generate text of given length starting from context.

        Args:
            length: Number of tokens to generate.
            start: Initial context tuple; if None, use a random one.

        Returns:
            Generated text as whitespace-separated tokens.
        """
        if length < 1:
            raise ValueError("length must be >= 1")
        if not self.transitions:
            raise ValueError("chain has not been trained")

        if self.seed is not None:
            random.seed(self.seed)

        if start is None:
            start = random.choice(list(self.transitions.keys()))
        elif start not in self.transitions:
            raise ValueError(f"start context {start} not in trained transitions")

        result: list[str] = list(start)
        context = start

        for _ in range(length - self.order):
            next_options = self.transitions[context]
            if not next_options:
                break
            next_token = random.choice(next_options)
            result.append(next_token)
            context = tuple(result[-(self.order) :])

        return " ".join(result[:length])

    def most_common_continuation(self, context: tuple[str, ...]) -> str | None:
        """Return the most common next token for a given context.

        Args:
            context: Tuple of preceding words.

        Returns:
            Most frequent next token, or None if context unseen.
        """
        if context not in self.transitions:
            return None
        options = self.transitions[context]
        return max(set(options), key=options.count)
