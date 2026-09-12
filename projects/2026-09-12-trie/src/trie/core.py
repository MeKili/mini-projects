"""Core Trie implementation."""


class TrieNode:
    """Node in the Trie tree."""

    __slots__ = ("children", "is_end_of_word")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False


class Trie:
    """Prefix tree for efficient string storage and search.

    Supports O(m) insert, search, startsWith, and delete operations,
    where m is the string length. Provides prefix-based autocomplete.
    """

    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a word into the trie.

        Time: O(m) where m is the length of word.
        Space: O(m) worst case (new path).
        """
        if not word:
            return

        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """Return True if word exists in the trie.

        Time: O(m) where m is the length of word.
        """
        if not word:
            return False

        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Return True if any word in trie starts with prefix.

        Time: O(m) where m is the length of prefix.
        """
        if not prefix:
            return True

        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def autocomplete(self, prefix: str) -> list[str]:
        """Return all words in the trie starting with prefix.

        Time: O(n) where n is the total number of words in trie.
        Space: O(n) for the result list.
        """
        words: list[str] = []

        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        self._collect_words(node, prefix, words)
        return sorted(words)

    def delete(self, word: str) -> bool:
        """Delete a word from the trie. Return True if deleted, False if not found.

        Time: O(m) where m is the length of word.
        """
        if not word:
            return False

        def _delete_helper(node: TrieNode, word: str, idx: int) -> tuple[bool, bool]:
            if idx == len(word):
                if not node.is_end_of_word:
                    return False, False
                node.is_end_of_word = False
                return True, len(node.children) == 0

            char = word[idx]
            if char not in node.children:
                return False, False

            child = node.children[char]
            was_deleted, should_delete_child = _delete_helper(child, word, idx + 1)

            if should_delete_child:
                del node.children[char]
                return was_deleted, len(node.children) == 0 and not node.is_end_of_word

            return was_deleted, False

        was_deleted, _ = _delete_helper(self.root, word, 0)
        return was_deleted

    def _collect_words(self, node: TrieNode, prefix: str, words: list[str]) -> None:
        """Helper to collect all words from a node."""
        if node.is_end_of_word:
            words.append(prefix)

        for char, child in node.children.items():
            self._collect_words(child, prefix + char, words)
