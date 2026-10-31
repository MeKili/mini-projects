"""Tests for binary search tree implementation."""


from binary_search_tree import BinarySearchTree


def test_empty_tree() -> None:
    """Test operations on empty tree."""
    bst: BinarySearchTree[str] = BinarySearchTree()
    assert bst.search(1) is None
    assert bst.in_order() == []
    assert bst.height() == -1
    assert bst.size() == 0
    assert bst.find_min() is None
    assert bst.find_max() is None
    assert bst.is_valid()


def test_single_node() -> None:
    """Test tree with single node."""
    bst: BinarySearchTree[str] = BinarySearchTree()
    bst.insert(5, "five")
    assert bst.search(5) == "five"
    assert bst.height() == 0
    assert bst.size() == 1
    assert bst.find_min() == (5, "five")
    assert bst.find_max() == (5, "five")
    assert bst.is_valid()


def test_insert_and_search() -> None:
    """Test basic insert and search operations."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    bst.insert(10, 100)
    bst.insert(5, 50)
    bst.insert(15, 150)
    bst.insert(3, 30)
    bst.insert(7, 70)

    assert bst.search(10) == 100
    assert bst.search(5) == 50
    assert bst.search(15) == 150
    assert bst.search(3) == 30
    assert bst.search(7) == 70
    assert bst.search(999) is None


def test_in_order_traversal() -> None:
    """Test in-order traversal returns sorted keys."""
    bst: BinarySearchTree[str] = BinarySearchTree()
    keys = [10, 5, 15, 3, 7, 12, 18]
    for key in keys:
        bst.insert(key, f"val_{key}")

    result = bst.in_order()
    result_keys = [k for k, _ in result]
    assert result_keys == sorted(keys)


def test_min_max() -> None:
    """Test finding minimum and maximum."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    keys = [10, 5, 15, 3, 20, 1, 8]
    for key in keys:
        bst.insert(key, key * 10)

    assert bst.find_min() == (1, 10)
    assert bst.find_max() == (20, 200)


def test_delete_leaf_node() -> None:
    """Test deleting a leaf node."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    bst.insert(10, 100)
    bst.insert(5, 50)
    bst.insert(15, 150)

    assert bst.delete(5)
    assert bst.search(5) is None
    assert bst.in_order() == [(10, 100), (15, 150)]
    assert bst.is_valid()


def test_delete_node_with_one_child() -> None:
    """Test deleting a node with one child."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    bst.insert(10, 100)
    bst.insert(5, 50)
    bst.insert(15, 150)
    bst.insert(3, 30)

    assert bst.delete(5)
    assert bst.search(5) is None
    assert bst.search(3) == 30
    assert bst.search(10) == 100
    assert bst.in_order() == [(3, 30), (10, 100), (15, 150)]
    assert bst.is_valid()


def test_delete_node_with_two_children() -> None:
    """Test deleting a node with two children."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    bst.insert(10, 100)
    bst.insert(5, 50)
    bst.insert(15, 150)
    bst.insert(3, 30)
    bst.insert(7, 70)
    bst.insert(12, 120)
    bst.insert(18, 180)

    assert bst.delete(5)
    assert bst.search(5) is None
    assert bst.in_order() == [(3, 30), (7, 70), (10, 100), (12, 120), (15, 150), (18, 180)]
    assert bst.is_valid()


def test_delete_root_node() -> None:
    """Test deleting the root node."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    bst.insert(10, 100)
    bst.insert(5, 50)
    bst.insert(15, 150)

    assert bst.delete(10)
    assert bst.search(10) is None
    assert bst.size() == 2
    assert bst.is_valid()


def test_delete_nonexistent_key() -> None:
    """Test deleting a key that doesn't exist."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    bst.insert(10, 100)

    assert not bst.delete(999)
    assert bst.size() == 1


def test_update_existing_key() -> None:
    """Test updating value of existing key."""
    bst: BinarySearchTree[str] = BinarySearchTree()
    bst.insert(5, "old")
    assert bst.search(5) == "old"

    bst.insert(5, "new")
    assert bst.search(5) == "new"
    assert bst.size() == 1


def test_height_calculation() -> None:
    """Test height calculation for various tree structures."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    assert bst.height() == -1

    bst.insert(10, 1)
    assert bst.height() == 0

    bst.insert(5, 1)
    bst.insert(15, 1)
    assert bst.height() == 1

    bst.insert(3, 1)
    bst.insert(7, 1)
    assert bst.height() == 2


def test_size_calculation() -> None:
    """Test size calculation."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    assert bst.size() == 0

    for i in range(1, 6):
        bst.insert(i, i)
        assert bst.size() == i


def test_is_valid() -> None:
    """Test BST validity checking."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    assert bst.is_valid()

    bst.insert(10, 1)
    bst.insert(5, 1)
    bst.insert(15, 1)
    bst.insert(3, 1)
    bst.insert(7, 1)
    bst.insert(12, 1)
    bst.insert(18, 1)

    assert bst.is_valid()


def test_generic_values() -> None:
    """Test BST with different value types."""
    bst_int: BinarySearchTree[int] = BinarySearchTree()
    bst_int.insert(1, 100)
    assert bst_int.search(1) == 100

    bst_str: BinarySearchTree[str] = BinarySearchTree()
    bst_str.insert(1, "one")
    assert bst_str.search(1) == "one"

    bst_tuple: BinarySearchTree[tuple[int, str]] = BinarySearchTree()
    bst_tuple.insert(1, (42, "answer"))
    assert bst_tuple.search(1) == (42, "answer")


def test_large_tree() -> None:
    """Test with a larger number of nodes."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    for i in range(100):
        bst.insert(i, i * 10)

    assert bst.size() == 100
    assert bst.find_min() == (0, 0)
    assert bst.find_max() == (99, 990)
    assert len(bst.in_order()) == 100
    assert bst.is_valid()

    # Delete every other element
    for i in range(0, 100, 2):
        assert bst.delete(i)

    assert bst.size() == 50
    assert bst.is_valid()


def test_ascending_insertion() -> None:
    """Test inserting in ascending order (degenerate case)."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    for i in range(10):
        bst.insert(i, i)

    assert bst.size() == 10
    assert bst.is_valid()


def test_descending_insertion() -> None:
    """Test inserting in descending order (degenerate case)."""
    bst: BinarySearchTree[int] = BinarySearchTree()
    for i in range(10, 0, -1):
        bst.insert(i, i)

    assert bst.size() == 10
    assert bst.is_valid()
