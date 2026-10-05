"""Core N-gram model implementation."""

from collections import Counter, defaultdict
from collections.abc import Sequence


class NGramModel:
    """Build and query n-gram language models."""

    def __init__(self, n: int = 2) -> None:
        """Initialize model for n-grams of size n.

        Args:
            n: The n-gram size (1=unigrams, 2=bigrams, etc.). Must be >= 1.

        Raises:
            ValueError: If n < 1.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.ngrams: Counter[tuple[str, ...]] = Counter()
        self.contexts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.total_count = 0

    def build(self, tokens: Sequence[str]) -> None:
        """Build model from a sequence of tokens.

        Args:
            tokens: List of string tokens to build n-grams from.

        Raises:
            ValueError: If tokens has fewer than n elements.
        """
        if len(tokens) < self.n:
            raise ValueError(
                f"Token sequence must have at least {self.n} elements, got {len(tokens)}"
            )

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            self.ngrams[ngram] += 1
            self.total_count += 1

            if self.n > 1:
                context = ngram[:-1]
                target = ngram[-1]
                self.contexts[context][target] += 1

    def count(self, ngram: Sequence[str]) -> int:
        """Get count of an n-gram.

        Args:
            ngram: Sequence of tokens to look up.

        Returns:
            Count of this n-gram (0 if not found).

        Raises:
            ValueError: If ngram has wrong length.
        """
        if len(ngram) != self.n:
            raise ValueError(f"ngram must have {self.n} tokens, got {len(ngram)}")
        return self.ngrams[tuple(ngram)]

    def probability(self, ngram: Sequence[str]) -> float:
        """Get probability of an n-gram (MLE).

        Args:
            ngram: Sequence of tokens.

        Returns:
            MLE probability of this n-gram (0.0 if unseen).

        Raises:
            ValueError: If ngram has wrong length or model is empty.
        """
        if self.total_count == 0:
            raise ValueError("Model is empty; call build() first")
        if len(ngram) != self.n:
            raise ValueError(f"ngram must have {self.n} tokens, got {len(ngram)}")
        count = self.count(ngram)
        return count / self.total_count

    def predict_next(self, context: Sequence[str], top_k: int = 5) -> list[tuple[str, float]]:
        """Predict next token given context.

        Uses conditional probability P(token | context) estimated from n-gram counts.
        Ranks by probability descending.

        Args:
            context: Sequence of preceding tokens (must be n-1 tokens for n-grams).
            top_k: Return top-k predictions.

        Returns:
            List of (token, probability) tuples, sorted by probability descending.

        Raises:
            ValueError: If context length != n-1.
        """
        if self.n == 1:
            raise ValueError("Cannot predict with unigram model; need n >= 2")
        if len(context) != self.n - 1:
            raise ValueError(f"context must have {self.n - 1} tokens, got {len(context)}")

        context_key = tuple(context)
        if context_key not in self.contexts:
            return []

        context_count = sum(self.contexts[context_key].values())
        predictions = [
            (token, count / context_count)
            for token, count in self.contexts[context_key].most_common(top_k)
        ]
        return predictions

    def most_common(self, k: int = 10) -> list[tuple[tuple[str, ...], int]]:
        """Get k most frequent n-grams.

        Args:
            k: Number of n-grams to return.

        Returns:
            List of (ngram, count) tuples sorted by count descending.
        """
        return self.ngrams.most_common(k)

    def vocab_size(self) -> int:
        """Get number of unique n-grams seen."""
        return len(self.ngrams)

    def total_ngrams(self) -> int:
        """Get total count of all n-grams."""
        return self.total_count
