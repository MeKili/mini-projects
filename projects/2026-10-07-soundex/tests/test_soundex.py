"""Tests for Soundex phonetic algorithm."""

from soundex import soundex, soundex_match


class TestSoundex:
    """Test cases for soundex function."""

    def test_classic_examples(self) -> None:
        """Test well-known Soundex examples."""
        assert soundex("Robert") == "R163"
        assert soundex("Rubin") == "R150"
        assert soundex("Rupert") == "R163"
        assert soundex("Smith") == "S530"
        assert soundex("Smythe") == "S530"

    def test_case_insensitive(self) -> None:
        """Test that soundex is case-insensitive."""
        assert soundex("Robert") == soundex("robert")
        assert soundex("JOHN") == soundex("John")

    def test_whitespace_handling(self) -> None:
        """Test that leading/trailing whitespace is trimmed."""
        assert soundex("  Robert  ") == soundex("Robert")
        assert soundex("\tSmith\n") == soundex("Smith")

    def test_first_letter_preserved(self) -> None:
        """Test that the first letter is always preserved."""
        result = soundex("Robert")
        assert result[0] == "R"
        result = soundex("Smith")
        assert result[0] == "S"

    def test_length_always_four(self) -> None:
        """Test that result is always 4 characters."""
        names = ["A", "AB", "John", "Bartholomew", "X", "Zz"]
        for name in names:
            assert len(soundex(name)) == 4

    def test_padding_with_zeros(self) -> None:
        """Test that short codes are padded with zeros."""
        assert soundex("A") == "A000"
        assert soundex("B") == "B000"
        assert soundex("Go") == "G000"

    def test_consonant_grouping(self) -> None:
        """Test that consonants are grouped correctly."""
        # B, F, P, V -> 1
        assert soundex("Bob")[1] == soundex("Fob")[1]
        # C, G, J, K, Q, S, X, Z -> 2
        assert soundex("Cat")[1] == soundex("Sat")[1]
        # D, T -> 3
        assert soundex("Dot")[1] == soundex("Tot")[1]
        # L -> 4
        assert soundex("Bill")[1] == "4"

    def test_vowels_and_y_ignored(self) -> None:
        """Test that vowels (A, E, I, O, U) and H, W, Y are ignored."""
        # Vowels don't produce codes
        assert soundex("Lloyd") == soundex("Laod")
        # H, W, Y don't produce codes but act as separators
        assert soundex("Henry") == soundex("Henery")

    def test_duplicate_digit_suppression(self) -> None:
        """Test that adjacent duplicate codes are suppressed."""
        # Adjacent same consonants should produce one digit
        assert soundex("Jackson") == "J250"
        # Doubled F produces same code (duplicates suppressed)
        assert soundex("Pfister") == soundex("Pffister")

    def test_empty_string(self) -> None:
        """Test behavior with empty input."""
        assert soundex("") == ""

    def test_vowels_only(self) -> None:
        """Test strings of vowels only."""
        assert soundex("A") == "A000"
        assert soundex("AEI") == "A000"

    def test_repeated_consonants(self) -> None:
        """Test repeated consonants."""
        # Consecutive repeats should be one digit
        assert soundex("Otto") == "O300"
        assert soundex("Attic") == "A320"

    def test_h_w_y_as_separators(self) -> None:
        """Test H, W, Y as separators between consonants."""
        # H, W, Y between consonants allow duplicate codes
        assert soundex("Lloyd") == "L300"
        assert soundex("Johnson") == "J525"


class TestSoundexMatch:
    """Test cases for soundex_match function."""

    def test_matching_pairs(self) -> None:
        """Test names that should match."""
        assert soundex_match("Robert", "Rupert")
        assert soundex_match("Smith", "Smythe")
        assert soundex_match("Lloyd", "Laod")

    def test_non_matching_pairs(self) -> None:
        """Test names that should not match."""
        assert not soundex_match("Robert", "Richard")
        assert not soundex_match("Smith", "Johnson")
        assert not soundex_match("Alice", "Bob")

    def test_same_name_matches_itself(self) -> None:
        """Test that a name always matches itself."""
        names = ["Robert", "Smith", "John", "Elizabeth"]
        for name in names:
            assert soundex_match(name, name)

    def test_case_insensitive_matching(self) -> None:
        """Test case-insensitive matching."""
        assert soundex_match("ROBERT", "robert")
        assert soundex_match("Smith", "SMITH")

    def test_whitespace_insensitive(self) -> None:
        """Test whitespace-insensitive matching."""
        assert soundex_match("  Smith  ", "Smith")
        assert soundex_match("John", "\tJohn\n")
