"""TF-IDF (term frequency–inverse document frequency) vectorizer for text data."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


class TfIdfVectorizer:
    """Compute TF-IDF vectors from a corpus of documents."""

    def __init__(self) -> None:
        """Initialize an empty vectorizer (must call ``fit`` before transform)."""
        self.vocab: dict[str, int] = {}
        self.idf_values: dict[str, float] = {}
        self.num_docs: int = 0

    def fit(self, documents: Sequence[Sequence[str]]) -> None:
        """Learn vocabulary and IDF values from documents (list of tokenized documents).

        Each document is a sequence of lowercased tokens. Overwrites any previous fit.
        """
        self.vocab = {}
        self.idf_values = {}
        self.num_docs = len(documents)

        if self.num_docs == 0:
            return

        doc_freq: Counter[str] = Counter()
        for doc in documents:
            unique_terms = set(doc)
            doc_freq.update(unique_terms)

        for term, freq in sorted(doc_freq.items()):
            self.vocab[term] = len(self.vocab)
            self.idf_values[term] = math.log((self.num_docs + 1) / (freq + 1))

    def transform(self, document: Sequence[str]) -> dict[str, float]:
        """Convert a tokenized document to TF-IDF scores (term -> score).

        Returns a dict mapping terms in the vocabulary to their TF-IDF scores.
        Terms not in the vocabulary are ignored.
        """
        if not self.vocab:
            return {}

        term_freq = Counter(document)
        doc_len = len(document)
        tfidf: dict[str, float] = {}

        for term, freq in term_freq.items():
            if term in self.vocab:
                tf = freq / doc_len if doc_len > 0 else 0.0
                idf = self.idf_values[term]
                tfidf[term] = tf * idf

        return tfidf

    def fit_transform(self, documents: Sequence[Sequence[str]]) -> list[dict[str, float]]:
        """Fit the vectorizer and transform all documents in one call."""
        self.fit(documents)
        return [self.transform(doc) for doc in documents]


def cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors (dicts)."""
    dot_product = sum(vec1.get(term, 0.0) * vec2[term] for term in vec2)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)
