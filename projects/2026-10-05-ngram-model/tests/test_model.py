"""Tests for N-gram model."""

import pytest

from ngram_model import NGramModel


class TestNGramModelInit:
    """Test model initialization."""

    def test_init_valid_n(self) -> None:
        """Valid n values should initialize."""
        model = NGramModel(n=1)
        assert model.n == 1

        model = NGramModel(n=3)
        assert model.n == 3

    def test_init_invalid_n(self) -> None:
        """n < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            NGramModel(n=0)

        with pytest.raises(ValueError, match="n must be >= 1"):
            NGramModel(n=-1)

    def test_init_default_n(self) -> None:
        """Default n should be 2."""
        model = NGramModel()
        assert model.n == 2


class TestBuild:
    """Test model building from tokens."""

    def test_build_unigrams(self) -> None:
        """Unigrams should be built correctly."""
        model = NGramModel(n=1)
        tokens = ["a", "b", "a", "c"]
        model.build(tokens)

        assert model.total_ngrams() == 4
        assert model.vocab_size() == 3
        assert model.count(["a"]) == 2
        assert model.count(["b"]) == 1
        assert model.count(["c"]) == 1

    def test_build_bigrams(self) -> None:
        """Bigrams should be built correctly."""
        model = NGramModel(n=2)
        tokens = ["the", "cat", "in", "the", "hat"]
        model.build(tokens)

        assert model.total_ngrams() == 4
        assert model.vocab_size() == 4
        assert model.count(["the", "cat"]) == 1
        assert model.count(["the", "hat"]) == 1
        assert model.count(["in", "the"]) == 1
        assert model.count(["cat", "in"]) == 1

    def test_build_trigrams(self) -> None:
        """Trigrams should be built correctly."""
        model = NGramModel(n=3)
        tokens = ["a", "b", "c", "a", "b", "d"]
        model.build(tokens)

        assert model.total_ngrams() == 4
        assert model.count(["a", "b", "c"]) == 1
        assert model.count(["a", "b", "d"]) == 1

    def test_build_insufficient_tokens(self) -> None:
        """Sequence shorter than n should raise ValueError."""
        model = NGramModel(n=3)
        with pytest.raises(ValueError, match="Token sequence must have at least 3"):
            model.build(["a", "b"])

    def test_build_exact_length(self) -> None:
        """Sequence with exactly n tokens should work."""
        model = NGramModel(n=2)
        model.build(["hello", "world"])
        assert model.total_ngrams() == 1
        assert model.count(["hello", "world"]) == 1


class TestCount:
    """Test counting n-grams."""

    def test_count_found(self) -> None:
        """Should return correct count for found n-grams."""
        model = NGramModel(n=2)
        model.build(["a", "b", "a", "b"])

        assert model.count(["a", "b"]) == 2

    def test_count_not_found(self) -> None:
        """Should return 0 for unseen n-grams."""
        model = NGramModel(n=2)
        model.build(["a", "b", "c"])

        assert model.count(["b", "c"]) == 1
        assert model.count(["x", "y"]) == 0

    def test_count_wrong_length(self) -> None:
        """Wrong-length query should raise ValueError."""
        model = NGramModel(n=2)
        model.build(["a", "b"])

        with pytest.raises(ValueError, match="ngram must have 2 tokens"):
            model.count(["a"])

        with pytest.raises(ValueError, match="ngram must have 2 tokens"):
            model.count(["a", "b", "c"])


class TestProbability:
    """Test probability estimation."""

    def test_probability_basic(self) -> None:
        """Probabilities should sum to 1."""
        model = NGramModel(n=1)
        model.build(["a", "a", "b"])

        prob_a = model.probability(["a"])
        prob_b = model.probability(["b"])
        assert abs((prob_a + prob_b) - 1.0) < 1e-9

    def test_probability_values(self) -> None:
        """Probabilities should match frequencies."""
        model = NGramModel(n=1)
        model.build(["a", "a", "b"])

        # a appears 2 times out of 3
        assert abs(model.probability(["a"]) - 2 / 3) < 1e-9
        # b appears 1 time out of 3
        assert abs(model.probability(["b"]) - 1 / 3) < 1e-9

    def test_probability_unseen(self) -> None:
        """Unseen n-grams should have probability 0."""
        model = NGramModel(n=2)
        model.build(["a", "b"])

        assert model.probability(["x", "y"]) == 0.0

    def test_probability_empty_model(self) -> None:
        """Querying empty model should raise ValueError."""
        model = NGramModel(n=1)
        with pytest.raises(ValueError, match="Model is empty"):
            model.probability(["a"])

    def test_probability_wrong_length(self) -> None:
        """Wrong-length query should raise ValueError."""
        model = NGramModel(n=2)
        model.build(["a", "b"])

        with pytest.raises(ValueError, match="ngram must have 2 tokens"):
            model.probability(["a"])


class TestPredictNext:
    """Test next-token prediction."""

    def test_predict_next_bigram(self) -> None:
        """Bigram model should predict next token correctly."""
        model = NGramModel(n=2)
        model.build(["the", "cat", "the", "dog", "the", "hat"])

        predictions = model.predict_next(["the"])
        assert len(predictions) == 3
        assert predictions[0][0] in ["cat", "dog", "hat"]
        # All should appear with equal probability (each appears once after "the")
        for _token, prob in predictions:
            assert abs(prob - 1 / 3) < 1e-9

    def test_predict_next_biased(self) -> None:
        """Should rank by frequency."""
        model = NGramModel(n=2)
        model.build(["a", "b", "a", "b", "a", "c"])

        predictions = model.predict_next(["a"])
        # "b" appears 2 times after "a", "c" appears 1 time
        assert predictions[0][0] == "b"
        assert abs(predictions[0][1] - 2 / 3) < 1e-9
        assert predictions[1][0] == "c"
        assert abs(predictions[1][1] - 1 / 3) < 1e-9

    def test_predict_next_top_k(self) -> None:
        """Should respect top_k parameter."""
        model = NGramModel(n=2)
        model.build(["a", "b", "a", "c", "a", "d", "a", "e"])

        predictions = model.predict_next(["a"], top_k=2)
        assert len(predictions) == 2

    def test_predict_next_unknown_context(self) -> None:
        """Unknown context should return empty list."""
        model = NGramModel(n=2)
        model.build(["a", "b"])

        predictions = model.predict_next(["x"])
        assert predictions == []

    def test_predict_next_unigram_model(self) -> None:
        """Unigram model should not support prediction."""
        model = NGramModel(n=1)
        model.build(["a", "b"])

        with pytest.raises(ValueError, match="Cannot predict with unigram"):
            model.predict_next([])

    def test_predict_next_wrong_context_length(self) -> None:
        """Wrong context length should raise ValueError."""
        model = NGramModel(n=3)
        model.build(["a", "b", "c"])

        with pytest.raises(ValueError, match="context must have 2 tokens"):
            model.predict_next(["a"])

        with pytest.raises(ValueError, match="context must have 2 tokens"):
            model.predict_next(["a", "b", "c"])

    def test_predict_next_trigram(self) -> None:
        """Trigram model should predict based on 2-token context."""
        model = NGramModel(n=3)
        model.build(["a", "b", "c", "a", "b", "d"])

        predictions = model.predict_next(["a", "b"])
        tokens = [t for t, _ in predictions]
        assert "c" in tokens
        assert "d" in tokens


class TestAnalysis:
    """Test analysis methods."""

    def test_most_common(self) -> None:
        """Most common should return sorted n-grams by frequency."""
        model = NGramModel(n=1)
        model.build(["a", "a", "a", "b", "b", "c"])

        common = model.most_common(2)
        assert common[0] == (("a",), 3)
        assert common[1] == (("b",), 2)

    def test_most_common_k(self) -> None:
        """Most common should respect k parameter."""
        model = NGramModel(n=1)
        model.build(["a", "b", "c", "d", "e"])

        common = model.most_common(3)
        assert len(common) == 3

    def test_vocab_size(self) -> None:
        """Vocab size should be unique n-gram count."""
        model = NGramModel(n=1)
        model.build(["a", "b", "a", "c", "b"])

        assert model.vocab_size() == 3

    def test_total_ngrams(self) -> None:
        """Total n-grams should be sequence length - n + 1."""
        model = NGramModel(n=2)
        model.build(["a", "b", "c", "d"])

        assert model.total_ngrams() == 3  # 4 tokens -> 3 bigrams


class TestIntegration:
    """Integration tests with real text patterns."""

    def test_sentence_model(self) -> None:
        """Test on realistic sentence data."""
        tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        model = NGramModel(n=2)
        model.build(tokens)

        # "the" should appear twice
        assert model.count(["the", "quick"]) == 1
        assert model.count(["the", "lazy"]) == 1

        # Predictions from "the"
        predictions = model.predict_next(["the"])
        assert len(predictions) == 2

    def test_repeated_sequence(self) -> None:
        """Test on repeated patterns."""
        tokens = ["a", "b", "c"] * 3
        model = NGramModel(n=3)
        model.build(tokens)

        # "a b c" should appear 3 times
        assert model.count(["a", "b", "c"]) == 3
