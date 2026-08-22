"""Tests for union-find data structure."""

import pytest

from union_find import UnionFind


class TestUnionFindBasics:
    """Test basic initialization and operations."""

    def test_init_creates_n_components(self) -> None:
        """Test that initialization creates n disjoint sets."""
        uf = UnionFind(5)
        assert uf.components() == 5

    def test_init_zero_components(self) -> None:
        """Test that zero-sized union-find is valid."""
        uf = UnionFind(0)
        assert uf.components() == 0

    def test_init_negative_size_raises(self) -> None:
        """Test that negative size raises ValueError."""
        with pytest.raises(ValueError, match="n must be non-negative"):
            UnionFind(-1)

    def test_find_self(self) -> None:
        """Test that find(i) initially returns i."""
        uf = UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_find_out_of_range_raises(self) -> None:
        """Test that find with out-of-range index raises IndexError."""
        uf = UnionFind(5)
        with pytest.raises(IndexError):
            uf.find(5)
        with pytest.raises(IndexError):
            uf.find(-1)

    def test_connected_initially_false(self) -> None:
        """Test that different elements are initially not connected."""
        uf = UnionFind(5)
        assert not uf.connected(0, 1)
        assert not uf.connected(2, 4)

    def test_element_connected_to_itself(self) -> None:
        """Test that each element is connected to itself."""
        uf = UnionFind(5)
        for i in range(5):
            assert uf.connected(i, i)


class TestUnionFindUnion:
    """Test union operations."""

    def test_union_basic(self) -> None:
        """Test basic union operation."""
        uf = UnionFind(5)
        assert uf.union(0, 1)
        assert uf.connected(0, 1)
        assert uf.components() == 4

    def test_union_returns_false_when_already_connected(self) -> None:
        """Test that union returns False if elements already connected."""
        uf = UnionFind(5)
        assert uf.union(0, 1)
        assert not uf.union(0, 1)
        assert uf.components() == 4

    def test_union_transitive(self) -> None:
        """Test that connectivity is transitive."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.connected(0, 2)
        assert uf.connected(1, 0)

    def test_union_multiple_components(self) -> None:
        """Test merging multiple components."""
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(4, 5)
        assert uf.components() == 3

    def test_union_all_into_one(self) -> None:
        """Test merging all elements into a single component."""
        uf = UnionFind(5)
        for i in range(1, 5):
            uf.union(0, i)
        assert uf.components() == 1
        for i in range(5):
            for j in range(5):
                assert uf.connected(i, j)

    def test_union_symmetric(self) -> None:
        """Test that union(a, b) and union(b, a) are equivalent."""
        uf1 = UnionFind(5)
        uf2 = UnionFind(5)
        uf1.union(0, 1)
        uf2.union(1, 0)
        for i in range(5):
            for j in range(5):
                assert uf1.connected(i, j) == uf2.connected(i, j)


class TestUnionFindPathCompression:
    """Test path compression behavior."""

    def test_path_compression_long_chain(self) -> None:
        """Test that path compression works on long chains."""
        uf = UnionFind(10)
        for i in range(9):
            uf.union(i, i + 1)
        root = uf.find(0)
        for i in range(10):
            assert uf.find(i) == root

    def test_path_compression_reduces_depth(self) -> None:
        """Test that repeated find calls compress paths."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        uf.find(0)
        uf.find(0)
        assert uf.connected(0, 4)


class TestUnionFindUnionByRank:
    """Test union by rank behavior."""

    def test_union_by_rank_small_to_large(self) -> None:
        """Test that rank is properly maintained."""
        uf = UnionFind(10)
        for i in range(4):
            uf.union(0, i + 1)
        for i in range(4, 9):
            uf.union(1, i + 1)
        for i in range(10):
            assert uf.connected(0, i)

    def test_union_maintains_connectivity_regardless_of_order(self) -> None:
        """Test that connectivity is maintained regardless of union order."""
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(4, 5)
        uf.union(1, 2)
        uf.union(3, 4)
        for i in range(6):
            for j in range(6):
                assert uf.connected(i, j)


class TestUnionFindComponents:
    """Test component counting."""

    def test_components_after_unions(self) -> None:
        """Test component count decreases correctly after unions."""
        uf = UnionFind(10)
        assert uf.components() == 10
        uf.union(0, 1)
        assert uf.components() == 9
        uf.union(2, 3)
        assert uf.components() == 8
        uf.union(1, 2)
        assert uf.components() == 7

    def test_components_no_change_on_redundant_union(self) -> None:
        """Test that component count doesn't change on redundant union."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        count = uf.components()
        uf.union(0, 2)
        assert uf.components() == count


class TestUnionFindEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_element(self) -> None:
        """Test union-find with single element."""
        uf = UnionFind(1)
        assert uf.components() == 1
        assert uf.connected(0, 0)

    def test_large_n(self) -> None:
        """Test with larger dataset."""
        uf = UnionFind(1000)
        assert uf.components() == 1000
        for i in range(999):
            uf.union(i, i + 1)
        assert uf.components() == 1
        assert uf.connected(0, 999)

    def test_union_with_out_of_range_raises(self) -> None:
        """Test that union with out-of-range indices raises IndexError."""
        uf = UnionFind(5)
        with pytest.raises(IndexError):
            uf.union(0, 5)
        with pytest.raises(IndexError):
            uf.union(-1, 0)
