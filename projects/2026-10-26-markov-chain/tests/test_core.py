"""Tests for Markov chain text generator."""

import pytest

from markov_chain.core import MarkovChain


def test_init_valid_order() -> None:
    chain = MarkovChain(order=1)
    assert chain.order == 1

    chain = MarkovChain(order=2)
    assert chain.order == 2


def test_init_invalid_order() -> None:
    with pytest.raises(ValueError, match="order must be >= 1"):
        MarkovChain(order=0)

    with pytest.raises(ValueError, match="order must be >= 1"):
        MarkovChain(order=-1)


def test_train_simple() -> None:
    chain = MarkovChain(order=1)
    chain.train("the cat sat on the mat")
    assert ("the",) in chain.transitions
    assert chain.transitions[("the",)] == ["cat", "mat"]


def test_train_order_two() -> None:
    chain = MarkovChain(order=2)
    chain.train("a b c a b d")
    assert ("a", "b") in chain.transitions
    assert chain.transitions[("a", "b")] == ["c", "d"]


def test_train_insufficient_tokens() -> None:
    chain = MarkovChain(order=2)
    chain.train("a b")
    assert len(chain.transitions) == 0


def test_train_single_sequence() -> None:
    chain = MarkovChain(order=1)
    chain.train("one two three")
    assert chain.transitions[("one",)] == ["two"]
    assert chain.transitions[("two",)] == ["three"]


def test_generate_without_training() -> None:
    chain = MarkovChain()
    with pytest.raises(ValueError, match="has not been trained"):
        chain.generate(5)


def test_generate_invalid_length() -> None:
    chain = MarkovChain()
    chain.train("a b c")
    with pytest.raises(ValueError, match="length must be >= 1"):
        chain.generate(0)

    with pytest.raises(ValueError, match="length must be >= 1"):
        chain.generate(-1)


def test_generate_from_specific_start() -> None:
    chain = MarkovChain(order=1, seed=42)
    chain.train("the quick brown fox jumps")
    result = chain.generate(2, start=("the",))
    assert result.startswith("the")


def test_generate_invalid_start() -> None:
    chain = MarkovChain(order=1)
    chain.train("a b c")
    with pytest.raises(ValueError, match="start context"):
        chain.generate(5, start=("z",))


def test_generate_reproducible_seed() -> None:
    text = "the cat sat on the mat and the cat ran"
    chain1 = MarkovChain(order=1, seed=123)
    chain1.train(text)
    result1 = chain1.generate(5, start=("the",))

    chain2 = MarkovChain(order=1, seed=123)
    chain2.train(text)
    result2 = chain2.generate(5, start=("the",))

    assert result1 == result2


def test_generate_different_without_seed() -> None:
    text = "a b a c a d a e a f"
    chain = MarkovChain(order=1)
    chain.train(text)
    # With multiple options from "a", results should vary across runs
    results = {chain.generate(2, start=("a",)) for _ in range(20)}
    assert len(results) > 1


def test_most_common_continuation() -> None:
    chain = MarkovChain(order=1)
    chain.train("a b a b c")
    # ("a",) -> ["b", "b"], so "b" is most common
    assert chain.most_common_continuation(("a",)) == "b"


def test_most_common_continuation_ties() -> None:
    chain = MarkovChain(order=1)
    chain.train("a b a c")
    # ("a",) -> ["b", "c"] (equal frequency)
    result = chain.most_common_continuation(("a",))
    assert result in ("b", "c")


def test_most_common_continuation_unseen() -> None:
    chain = MarkovChain(order=1)
    chain.train("a b c")
    assert chain.most_common_continuation(("z",)) is None


def test_generate_length_equals_context() -> None:
    chain = MarkovChain(order=1, seed=42)
    chain.train("a b c d e")
    result = chain.generate(1, start=("a",))
    assert result == "a"


def test_generate_order_two() -> None:
    chain = MarkovChain(order=2, seed=99)
    chain.train("the quick brown fox the quick red fox")
    result = chain.generate(4, start=("the", "quick"))
    tokens = result.split()
    assert len(tokens) == 4
    assert tokens[:2] == ["the", "quick"]


def test_train_multiple_times() -> None:
    chain = MarkovChain(order=1)
    chain.train("a b c")
    chain.train("a b d")
    # After second training, transitions should accumulate
    assert chain.transitions[("a",)] == ["b", "b"]
    assert set(chain.transitions[("b",)]) == {"c", "d"}
