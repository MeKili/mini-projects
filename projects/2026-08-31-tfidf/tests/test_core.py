"""Tests for TF-IDF vectorizer and similarity."""

import math

from tfidf.core import TfIdfVectorizer, cosine_similarity


def test_fit_empty_corpus() -> None:
    vec = TfIdfVectorizer()
    vec.fit([])
    assert vec.vocab == {}
    assert vec.idf_values == {}


def test_fit_single_document() -> None:
    vec = TfIdfVectorizer()
    vec.fit([["hello", "world"]])
    assert "hello" in vec.vocab
    assert "world" in vec.vocab
    assert len(vec.vocab) == 2


def test_fit_multiple_documents() -> None:
    vec = TfIdfVectorizer()
    docs = [["cat", "dog"], ["cat", "bird"], ["dog", "bird"]]
    vec.fit(docs)
    assert set(vec.vocab.keys()) == {"cat", "dog", "bird"}
    assert vec.num_docs == 3


def test_vocab_indices_ordered() -> None:
    vec = TfIdfVectorizer()
    docs = [["zebra", "apple"], ["apple", "banana"]]
    vec.fit(docs)
    assert vec.vocab["apple"] < vec.vocab["banana"] < vec.vocab["zebra"]


def test_transform_empty_vectorizer() -> None:
    vec = TfIdfVectorizer()
    result = vec.transform(["hello", "world"])
    assert result == {}


def test_transform_single_term() -> None:
    vec = TfIdfVectorizer()
    vec.fit([["hello"], ["goodbye"]])
    result = vec.transform(["hello", "hello"])
    assert "hello" in result
    assert result["hello"] > 0


def test_transform_unknown_terms_ignored() -> None:
    vec = TfIdfVectorizer()
    vec.fit([["hello"]])
    result = vec.transform(["hello", "unknown", "terms"])
    assert set(result.keys()) == {"hello"}


def test_tfidf_scoring() -> None:
    vec = TfIdfVectorizer()
    docs = [["cat", "cat", "dog"], ["bird", "cat"]]
    vec.fit(docs)

    tfidf = vec.transform(["cat", "cat", "cat"])
    tf_cat = 1.0
    idf_cat = vec.idf_values["cat"]
    expected_score = tf_cat * idf_cat
    assert math.isclose(tfidf["cat"], expected_score)


def test_fit_transform() -> None:
    vec = TfIdfVectorizer()
    docs = [["a", "b"], ["b", "c"], ["a", "c"]]
    result = vec.fit_transform(docs)

    assert len(result) == 3
    assert all(isinstance(r, dict) for r in result)
    assert all(isinstance(v, float) for r in result for v in r.values())


def test_cosine_similarity_identical() -> None:
    vec1 = {"a": 1.0, "b": 2.0}
    vec2 = {"a": 1.0, "b": 2.0}
    sim = cosine_similarity(vec1, vec2)
    assert math.isclose(sim, 1.0)


def test_cosine_similarity_orthogonal() -> None:
    vec1 = {"a": 1.0, "b": 0.0}
    vec2 = {"b": 1.0, "c": 0.0}
    sim = cosine_similarity(vec1, vec2)
    assert math.isclose(sim, 0.0)


def test_cosine_similarity_empty_vectors() -> None:
    sim = cosine_similarity({}, {})
    assert sim == 0.0


def test_cosine_similarity_one_empty() -> None:
    vec1 = {"a": 1.0}
    vec2: dict[str, float] = {}
    sim = cosine_similarity(vec1, vec2)
    assert sim == 0.0


def test_cosine_similarity_partial_overlap() -> None:
    vec1 = {"a": 3.0, "b": 4.0}
    vec2 = {"a": 3.0, "b": 4.0}
    sim = cosine_similarity(vec1, vec2)
    assert math.isclose(sim, 1.0)

    vec3 = {"c": 1.0}
    sim2 = cosine_similarity(vec1, vec3)
    assert math.isclose(sim2, 0.0)


def test_corpus_example() -> None:
    docs = [
        ["the", "cat", "sat"],
        ["the", "dog", "sat"],
        ["cat", "dog", "chase"],
    ]
    vec = TfIdfVectorizer()
    vec.fit_transform(docs)

    doc1_vec = vec.transform(["the", "cat", "sat"])
    doc2_vec = vec.transform(["the", "dog", "sat"])
    doc3_vec = vec.transform(["cat", "dog", "chase"])

    sim_1_2 = cosine_similarity(doc1_vec, doc2_vec)
    sim_1_3 = cosine_similarity(doc1_vec, doc3_vec)

    assert sim_1_2 > sim_1_3
