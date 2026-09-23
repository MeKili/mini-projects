"""Huffman coding for lossless data compression using variable-length prefix codes."""

from heapq import heappop, heappush
from typing import Optional


class _TreeNode:
    """Internal node in Huffman tree."""

    __slots__ = ("freq", "char", "left", "right")

    def __init__(
        self,
        freq: int,
        char: str | None = None,
        left: Optional["_TreeNode"] = None,
        right: Optional["_TreeNode"] = None,
    ):
        self.freq = freq
        self.char = char
        self.left = left
        self.right = right

    def __lt__(self, other: "_TreeNode") -> bool:
        return self.freq < other.freq

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _TreeNode):
            return NotImplemented
        return self.freq == other.freq and self.char == other.char


class HuffmanEncoder:
    """Huffman encoder: build tree from frequencies and generate codes."""

    def __init__(self) -> None:
        self.root: _TreeNode | None = None
        self.codes: dict[str, str] = {}

    def build(self, text: str) -> None:
        """Build Huffman tree and generate codes from text."""
        if not text:
            raise ValueError("text cannot be empty")

        freq: dict[str, int] = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        heap: list[_TreeNode] = [_TreeNode(count, char) for char, count in freq.items()]
        heap.sort()

        while len(heap) > 1:
            left = heappop(heap)
            right = heappop(heap)
            parent = _TreeNode(left.freq + right.freq, left=left, right=right)
            heappush(heap, parent)

        self.root = heap[0]
        self.codes = {}
        self._generate_codes(self.root, "")

    def _generate_codes(self, node: _TreeNode | None, code: str) -> None:
        """Recursively generate codes by traversing tree."""
        if node is None:
            return
        if node.char is not None:
            self.codes[node.char] = code if code else "0"
            return
        self._generate_codes(node.left, code + "0")
        self._generate_codes(node.right, code + "1")

    def encode(self, text: str) -> str:
        """Encode text using Huffman codes."""
        if not self.codes:
            raise ValueError("build() must be called first")
        return "".join(self.codes[char] for char in text)

    def get_codes(self) -> dict[str, str]:
        """Return generated Huffman codes."""
        return dict(self.codes)


class HuffmanDecoder:
    """Huffman decoder: reconstruct text using code tree."""

    def __init__(self, codes: dict[str, str]) -> None:
        if not codes:
            raise ValueError("codes cannot be empty")
        self.tree: _TreeNode | None = self._build_tree(codes)

    def _build_tree(self, codes: dict[str, str]) -> _TreeNode:
        """Reconstruct tree from code dictionary."""
        root = _TreeNode(0)
        for char, code in codes.items():
            node = root
            for i, bit in enumerate(code):
                if i == len(code) - 1:
                    if bit == "0":
                        node.left = _TreeNode(0, char)
                    else:
                        node.right = _TreeNode(0, char)
                else:
                    if bit == "0":
                        if node.left is None:
                            node.left = _TreeNode(0)
                        node = node.left
                    else:
                        if node.right is None:
                            node.right = _TreeNode(0)
                        node = node.right
        return root

    def decode(self, encoded: str) -> str:
        """Decode binary string using Huffman tree."""
        if not encoded:
            raise ValueError("encoded cannot be empty")
        if self.tree is None:
            raise ValueError("tree not initialized")

        result = []
        node = self.tree
        assert node is not None
        for bit in encoded:
            next_node = node.left if bit == "0" else node.right
            if next_node is None:
                raise ValueError("invalid encoded data")
            if next_node.char is not None:
                result.append(next_node.char)
                node = self.tree
            else:
                node = next_node
        if node is not self.tree:
            raise ValueError("incomplete code at end of data")
        return "".join(result)


__all__ = ["HuffmanEncoder", "HuffmanDecoder"]
