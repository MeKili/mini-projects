"""Tests for Trie data structure."""

from trie import Trie


def test_insert_and_search() -> None:
    trie = Trie()
    trie.insert("hello")
    assert trie.search("hello")


def test_search_nonexistent() -> None:
    trie = Trie()
    trie.insert("hello")
    assert not trie.search("world")


def test_search_empty_string() -> None:
    trie = Trie()
    assert not trie.search("")


def test_insert_empty_string() -> None:
    trie = Trie()
    trie.insert("")
    assert not trie.search("")


def test_search_prefix_not_word() -> None:
    trie = Trie()
    trie.insert("hello")
    assert not trie.search("hel")


def test_starts_with_true() -> None:
    trie = Trie()
    trie.insert("hello")
    assert trie.starts_with("hel")


def test_starts_with_false() -> None:
    trie = Trie()
    trie.insert("hello")
    assert not trie.starts_with("world")


def test_starts_with_empty_prefix() -> None:
    trie = Trie()
    trie.insert("hello")
    assert trie.starts_with("")


def test_starts_with_empty_trie() -> None:
    trie = Trie()
    assert not trie.starts_with("hello")


def test_autocomplete_single_word() -> None:
    trie = Trie()
    trie.insert("hello")
    assert trie.autocomplete("hel") == ["hello"]


def test_autocomplete_multiple_words() -> None:
    trie = Trie()
    trie.insert("hello")
    trie.insert("help")
    trie.insert("hero")
    trie.insert("heap")
    results = trie.autocomplete("he")
    assert sorted(results) == ["heap", "hello", "help", "hero"]


def test_autocomplete_empty_prefix() -> None:
    trie = Trie()
    trie.insert("apple")
    trie.insert("app")
    results = trie.autocomplete("")
    assert sorted(results) == ["app", "apple"]


def test_autocomplete_no_match() -> None:
    trie = Trie()
    trie.insert("hello")
    assert trie.autocomplete("xyz") == []


def test_autocomplete_exact_match() -> None:
    trie = Trie()
    trie.insert("hello")
    assert trie.autocomplete("hello") == ["hello"]


def test_autocomplete_sorted() -> None:
    trie = Trie()
    trie.insert("zebra")
    trie.insert("apple")
    trie.insert("banana")
    results = trie.autocomplete("")
    assert results == ["apple", "banana", "zebra"]


def test_delete_existing_word() -> None:
    trie = Trie()
    trie.insert("hello")
    assert trie.delete("hello")
    assert not trie.search("hello")


def test_delete_nonexistent_word() -> None:
    trie = Trie()
    trie.insert("hello")
    assert not trie.delete("world")


def test_delete_empty_string() -> None:
    trie = Trie()
    assert not trie.delete("")


def test_delete_prefix_only() -> None:
    trie = Trie()
    trie.insert("hello")
    assert not trie.delete("hel")
    assert trie.search("hello")


def test_delete_with_sibling_branches() -> None:
    trie = Trie()
    trie.insert("hello")
    trie.insert("help")
    assert trie.delete("hello")
    assert not trie.search("hello")
    assert trie.search("help")
    assert trie.starts_with("hel")


def test_delete_preserves_longer_words() -> None:
    trie = Trie()
    trie.insert("app")
    trie.insert("apple")
    assert trie.delete("app")
    assert not trie.search("app")
    assert trie.search("apple")


def test_delete_shorter_word_in_chain() -> None:
    trie = Trie()
    trie.insert("app")
    trie.insert("apple")
    trie.insert("application")
    assert trie.delete("apple")
    assert trie.search("app")
    assert trie.search("application")
    assert not trie.search("apple")


def test_multiple_operations() -> None:
    trie = Trie()
    words = ["cat", "car", "card", "care", "careful", "dog", "dodge"]

    for word in words:
        trie.insert(word)

    assert len(trie.autocomplete("ca")) == 5
    assert len(trie.autocomplete("car")) == 4
    assert len(trie.autocomplete("do")) == 2

    trie.delete("car")
    assert len(trie.autocomplete("ca")) == 4
    assert not trie.search("car")

    assert trie.starts_with("dodge")
    assert trie.starts_with("cat")


def test_single_character_words() -> None:
    trie = Trie()
    trie.insert("a")
    trie.insert("b")
    assert trie.search("a")
    assert trie.search("b")
    assert not trie.search("c")
    assert trie.autocomplete("") == ["a", "b"]


def test_overlapping_words() -> None:
    trie = Trie()
    trie.insert("test")
    trie.insert("testing")
    trie.insert("tester")
    results = trie.autocomplete("test")
    assert sorted(results) == ["test", "tester", "testing"]
