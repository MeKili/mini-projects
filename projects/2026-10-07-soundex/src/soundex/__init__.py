"""Soundex phonetic algorithm for fuzzy string matching.

Soundex converts names into a 4-character code where similar-sounding names
share the same code. Useful for approximate name matching and duplicate detection.
"""

__all__ = ["soundex", "soundex_match"]


def soundex(name: str) -> str:
    """Encode a name using the Soundex algorithm.

    Args:
        name: The name to encode (whitespace trimmed, case-insensitive).

    Returns:
        A 4-character Soundex code.
    """
    name = name.strip().upper()
    if not name:
        return ""

    # Soundex digit mapping (consonant -> digit)
    mapping = {
        "B": "1",
        "F": "1",
        "P": "1",
        "V": "1",
        "C": "2",
        "G": "2",
        "J": "2",
        "K": "2",
        "Q": "2",
        "S": "2",
        "X": "2",
        "Z": "2",
        "D": "3",
        "T": "3",
        "L": "4",
        "M": "5",
        "N": "5",
        "R": "6",
    }

    # Keep first letter
    code = name[0]

    # Encode remaining letters, skipping vowels, H, W, Y
    prev_digit = mapping.get(name[0], "")
    for char in name[1:]:
        digit = mapping.get(char, "")
        if digit and digit != prev_digit:
            code += digit
            prev_digit = digit
        elif not digit:
            # Reset prev_digit for vowels, H, W, Y (adjacent repeats ignored)
            prev_digit = ""

    # Pad with zeros or truncate to 4 characters
    code = (code + "000")[:4]
    return code


def soundex_match(name1: str, name2: str) -> bool:
    """Check if two names have the same Soundex code.

    Args:
        name1: First name.
        name2: Second name.

    Returns:
        True if both names encode to the same Soundex code.
    """
    return soundex(name1) == soundex(name2)
