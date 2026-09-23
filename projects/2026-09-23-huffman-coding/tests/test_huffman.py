"""Tests for Huffman coding implementation."""

import pytest

from huffman_coding import HuffmanDecoder, HuffmanEncoder


def test_single_char() -> None:
    """Single character always encodes to '0'."""
    encoder = HuffmanEncoder()
    encoder.build("aaaa")
    assert encoder.get_codes() == {"a": "0"}
    assert encoder.encode("aaaa") == "0000"


def test_two_chars() -> None:
    """Two different characters get codes 0 and 1."""
    encoder = HuffmanEncoder()
    encoder.build("aab")
    codes = encoder.get_codes()
    assert len(codes) == 2
    assert set(codes.values()) == {"0", "1"}


def test_encode_simple() -> None:
    """Encode 'hello world' produces consistent output."""
    text = "hello world"
    encoder = HuffmanEncoder()
    encoder.build(text)
    encoded = encoder.encode(text)
    assert isinstance(encoded, str)
    assert all(c in "01" for c in encoded)
    assert len(encoded) > 0


def test_decode_simple() -> None:
    """Encode and decode 'hello' recovers original."""
    text = "hello"
    encoder = HuffmanEncoder()
    encoder.build(text)
    codes = encoder.get_codes()
    encoded = encoder.encode(text)
    decoder = HuffmanDecoder(codes)
    decoded = decoder.decode(encoded)
    assert decoded == text


def test_round_trip_pangram() -> None:
    """Encode-decode cycle preserves pangram."""
    text = "the quick brown fox jumps over the lazy dog"
    encoder = HuffmanEncoder()
    encoder.build(text)
    codes = encoder.get_codes()
    encoded = encoder.encode(text)
    decoder = HuffmanDecoder(codes)
    decoded = decoder.decode(encoded)
    assert decoded == text


def test_repeated_chars_compress() -> None:
    """Repeated characters should compress better with Huffman."""
    text = "aaabbc"
    encoder = HuffmanEncoder()
    encoder.build(text)
    codes = encoder.get_codes()
    assert codes["a"] == "0"
    assert len(codes["b"]) == 2
    assert len(codes["c"]) == 2


def test_compression_ratio() -> None:
    """Compression ratio is reasonable for skewed distribution."""
    text = "a" * 100 + "b" * 50 + "c" * 25
    encoder = HuffmanEncoder()
    encoder.build(text)
    encoded = encoder.encode(text)
    ratio = len(encoded) / (len(text) * 8)
    assert ratio < 1.0
    assert ratio < 0.6


def test_codes_are_prefix_free() -> None:
    """No code is a prefix of another (prefix-free property)."""
    text = "abracadabra"
    encoder = HuffmanEncoder()
    encoder.build(text)
    codes = list(encoder.get_codes().values())
    for i, code1 in enumerate(codes):
        for code2 in codes[i + 1 :]:
            assert not code1.startswith(code2)
            assert not code2.startswith(code1)


def test_encoder_empty_text_raises() -> None:
    """Empty text raises ValueError."""
    encoder = HuffmanEncoder()
    with pytest.raises(ValueError, match="cannot be empty"):
        encoder.build("")


def test_encoder_encode_before_build_raises() -> None:
    """Encoding before build raises ValueError."""
    encoder = HuffmanEncoder()
    with pytest.raises(ValueError, match="must be called first"):
        encoder.encode("test")


def test_decoder_empty_codes_raises() -> None:
    """Empty codes dict raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        HuffmanDecoder({})


def test_decoder_empty_encoded_raises() -> None:
    """Empty encoded string raises ValueError."""
    decoder = HuffmanDecoder({"a": "0"})
    with pytest.raises(ValueError, match="cannot be empty"):
        decoder.decode("")


def test_decoder_invalid_code_raises() -> None:
    """Invalid encoded data raises ValueError."""
    decoder = HuffmanDecoder({"a": "00", "b": "01"})
    with pytest.raises(ValueError, match="invalid encoded data"):
        decoder.decode("10")


def test_decoder_incomplete_code_raises() -> None:
    """Incomplete code at end raises ValueError."""
    decoder = HuffmanDecoder({"a": "00", "b": "01"})
    with pytest.raises(ValueError, match="incomplete code"):
        decoder.decode("0")


def test_special_chars() -> None:
    """Special characters are encoded correctly."""
    text = "hello\nworld\t!"
    encoder = HuffmanEncoder()
    encoder.build(text)
    codes = encoder.get_codes()
    encoded = encoder.encode(text)
    decoder = HuffmanDecoder(codes)
    decoded = decoder.decode(encoded)
    assert decoded == text


def test_unicode_chars() -> None:
    """Unicode characters are encoded correctly."""
    text = "hello 世界 🌍"
    encoder = HuffmanEncoder()
    encoder.build(text)
    codes = encoder.get_codes()
    encoded = encoder.encode(text)
    decoder = HuffmanDecoder(codes)
    decoded = decoder.decode(encoded)
    assert decoded == text


def test_long_text() -> None:
    """Long text encodes and decodes correctly."""
    text = "abc" * 1000
    encoder = HuffmanEncoder()
    encoder.build(text)
    codes = encoder.get_codes()
    encoded = encoder.encode(text)
    decoder = HuffmanDecoder(codes)
    decoded = decoder.decode(encoded)
    assert decoded == text
    assert len(encoded) < len(text) * 8


def test_skewed_distribution() -> None:
    """Highly skewed distribution compresses significantly."""
    text = "a" * 100 + "b" * 10 + "c"
    encoder = HuffmanEncoder()
    encoder.build(text)
    codes = encoder.get_codes()
    assert len(codes["a"]) == 1
    encoded = encoder.encode(text)
    assert len(encoded) < len(text) * 2


def test_uniform_distribution() -> None:
    """Uniform distribution has longer codes."""
    text = "abcd"
    encoder = HuffmanEncoder()
    encoder.build(text)
    codes = encoder.get_codes()
    max_code_len = max(len(code) for code in codes.values())
    assert max_code_len >= 2


def test_multiple_encode_calls() -> None:
    """Multiple encode calls with same encoder work."""
    encoder = HuffmanEncoder()
    encoder.build("hello")
    enc1 = encoder.encode("hello")
    enc2 = encoder.encode("hello")
    assert enc1 == enc2


def test_different_texts_same_alphabet() -> None:
    """Encoding different texts with same alphabet."""
    encoder = HuffmanEncoder()
    encoder.build("aabbcc")
    codes = encoder.get_codes()
    encoded1 = encoder.encode("abc")
    encoded2 = encoder.encode("cba")
    decoder = HuffmanDecoder(codes)
    assert decoder.decode(encoded1) == "abc"
    assert decoder.decode(encoded2) == "cba"
