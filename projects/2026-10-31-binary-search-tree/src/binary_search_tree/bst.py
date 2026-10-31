"""Binary search tree with insert, search, delete, and in-order traversal.

A simple BST without self-balancing. Supports integer keys and generic values.
"""


class Node[T]:
    """A node in the binary search tree."""

    def __init__(self, key: int, value: T) -> None:
        self.key = key
        self.value = value
        self.left: Node[T] | None = None
        self.right: Node[T] | None = None


class BinarySearchTree[T]:
    """A generic binary search tree with insert, search, delete, and traversal."""

    def __init__(self) -> None:
        self.root: Node[T] | None = None

    def insert(self, key: int, value: T) -> None:
        """Insert a key-value pair into the tree. Overwrites if key exists."""
        if self.root is None:
            self.root = Node(key, value)
        else:
            self._insert_recursive(self.root, key, value)

    def _insert_recursive(self, node: Node[T], key: int, value: T) -> None:
        if key < node.key:
            if node.left is None:
                node.left = Node(key, value)
            else:
                self._insert_recursive(node.left, key, value)
        elif key > node.key:
            if node.right is None:
                node.right = Node(key, value)
            else:
                self._insert_recursive(node.right, key, value)
        else:
            node.value = value

    def search(self, key: int) -> T | None:
        """Search for a key and return its value, or None if not found."""
        node = self._search_node(self.root, key)
        return node.value if node is not None else None

    def _search_node(self, node: Node[T] | None, key: int) -> Node[T] | None:
        if node is None:
            return None
        if key == node.key:
            return node
        elif key < node.key:
            return self._search_node(node.left, key)
        else:
            return self._search_node(node.right, key)

    def delete(self, key: int) -> bool:
        """Delete a key from the tree. Returns True if deleted, False if not found."""
        original_size = self.size()
        self.root = self._delete_recursive(self.root, key)
        return self.size() < original_size

    def _delete_recursive(self, node: Node[T] | None, key: int) -> Node[T] | None:
        if node is None:
            return None

        if key < node.key:
            node.left = self._delete_recursive(node.left, key)
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key)
        else:
            # Node to delete found
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left

            # Node has two children: find in-order successor
            successor = self._find_min(node.right)
            node.key = successor.key
            node.value = successor.value
            node.right = self._delete_recursive(node.right, successor.key)

        return node

    def _find_min(self, node: Node[T]) -> Node[T]:
        current = node
        while current.left is not None:
            current = current.left
        return current

    def find_min(self) -> tuple[int, T] | None:
        """Find the minimum key-value pair in the tree."""
        if self.root is None:
            return None
        node = self._find_min(self.root)
        return (node.key, node.value)

    def find_max(self) -> tuple[int, T] | None:
        """Find the maximum key-value pair in the tree."""
        if self.root is None:
            return None
        node = self._find_max(self.root)
        return (node.key, node.value)

    def _find_max(self, node: Node[T]) -> Node[T]:
        current = node
        while current.right is not None:
            current = current.right
        return current

    def in_order(self) -> list[tuple[int, T]]:
        """Return keys and values in in-order (sorted by key)."""
        result: list[tuple[int, T]] = []
        self._in_order_recursive(self.root, result)
        return result

    def _in_order_recursive(self, node: Node[T] | None, result: list[tuple[int, T]]) -> None:
        if node is not None:
            self._in_order_recursive(node.left, result)
            result.append((node.key, node.value))
            self._in_order_recursive(node.right, result)

    def height(self) -> int:
        """Return the height of the tree (0 for single node, -1 for empty tree)."""
        return self._height_recursive(self.root)

    def _height_recursive(self, node: Node[T] | None) -> int:
        if node is None:
            return -1
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        return 1 + max(left_height, right_height)

    def size(self) -> int:
        """Return the number of nodes in the tree."""
        return self._size_recursive(self.root)

    def _size_recursive(self, node: Node[T] | None) -> int:
        if node is None:
            return 0
        return 1 + self._size_recursive(node.left) + self._size_recursive(node.right)

    def is_valid(self) -> bool:
        """Validate that the tree maintains BST property."""

        def validate(node: Node[T] | None, min_key: int, max_key: int) -> bool:
            if node is None:
                return True
            if node.key <= min_key or node.key >= max_key:
                return False
            return validate(node.left, min_key, node.key) and validate(
                node.right, node.key, max_key
            )

        return validate(self.root, float("-inf"), float("inf"))  # type: ignore
