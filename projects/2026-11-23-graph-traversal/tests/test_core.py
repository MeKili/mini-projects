"""Tests for graph traversal algorithms."""

import pytest

from graph_traversal.core import Graph


def test_graph_creation() -> None:
    g = Graph(5)
    assert g.vertices == 5


def test_graph_invalid_vertices() -> None:
    with pytest.raises(ValueError):
        Graph(0)
    with pytest.raises(ValueError):
        Graph(-1)


def test_add_edge() -> None:
    g = Graph(3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    assert 1 in g.adj_list[0]
    assert 0 in g.adj_list[1]
    assert 2 in g.adj_list[1]


def test_add_edge_out_of_range() -> None:
    g = Graph(3)
    with pytest.raises(ValueError):
        g.add_edge(0, 5)
    with pytest.raises(ValueError):
        g.add_edge(-1, 1)


def test_add_edge_duplicate() -> None:
    g = Graph(3)
    g.add_edge(0, 1)
    g.add_edge(0, 1)
    assert g.adj_list[0].count(1) == 1


def test_bfs_linear_graph() -> None:
    g = Graph(4)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    result = g.bfs(0)
    assert result == [0, 1, 2, 3]


def test_bfs_branching() -> None:
    g = Graph(5)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    result = g.bfs(0)
    assert result[0] == 0
    assert set(result[1:3]) == {1, 2}
    assert set(result[3:5]) == {3, 4}


def test_bfs_single_vertex() -> None:
    g = Graph(1)
    result = g.bfs(0)
    assert result == [0]


def test_bfs_out_of_range() -> None:
    g = Graph(3)
    with pytest.raises(ValueError):
        g.bfs(5)
    with pytest.raises(ValueError):
        g.bfs(-1)


def test_dfs_recursive_linear() -> None:
    g = Graph(4)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    result = g.dfs_recursive(0)
    assert result == [0, 1, 2, 3]


def test_dfs_iterative_linear() -> None:
    g = Graph(4)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    result = g.dfs_iterative(0)
    assert result == [0, 1, 2, 3]


def test_dfs_recursive_branching() -> None:
    g = Graph(5)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    result = g.dfs_recursive(0)
    assert result[0] == 0
    assert set(result) == {0, 1, 2, 3, 4}


def test_dfs_iterative_branching() -> None:
    g = Graph(5)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    result = g.dfs_iterative(0)
    assert result[0] == 0
    assert set(result) == {0, 1, 2, 3, 4}


def test_dfs_single_vertex() -> None:
    g = Graph(1)
    result_rec = g.dfs_recursive(0)
    result_iter = g.dfs_iterative(0)
    assert result_rec == [0]
    assert result_iter == [0]


def test_dfs_out_of_range() -> None:
    g = Graph(3)
    with pytest.raises(ValueError):
        g.dfs_recursive(5)
    with pytest.raises(ValueError):
        g.dfs_iterative(-1)


def test_find_path_exists() -> None:
    g = Graph(5)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    path = g.find_path(0, 4)
    assert path == [0, 1, 2, 3, 4]


def test_find_path_same_vertex() -> None:
    g = Graph(3)
    path = g.find_path(1, 1)
    assert path == [1]


def test_find_path_adjacent() -> None:
    g = Graph(3)
    g.add_edge(0, 1)
    path = g.find_path(0, 1)
    assert path == [0, 1]


def test_find_path_nonexistent() -> None:
    g = Graph(4)
    g.add_edge(0, 1)
    g.add_edge(2, 3)
    path = g.find_path(0, 3)
    assert path is None


def test_find_path_out_of_range() -> None:
    g = Graph(3)
    with pytest.raises(ValueError):
        g.find_path(0, 5)
    with pytest.raises(ValueError):
        g.find_path(-1, 2)


def test_has_path_true() -> None:
    g = Graph(4)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    assert g.has_path(0, 3)


def test_has_path_false() -> None:
    g = Graph(4)
    g.add_edge(0, 1)
    g.add_edge(2, 3)
    assert not g.has_path(0, 3)


def test_has_path_same() -> None:
    g = Graph(2)
    assert g.has_path(0, 0)


def test_connected_components_single() -> None:
    g = Graph(4)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    components = g.connected_components()
    assert len(components) == 1
    assert set(components[0]) == {0, 1, 2, 3}


def test_connected_components_multiple() -> None:
    g = Graph(6)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    components = g.connected_components()
    assert len(components) == 2
    component_sets = [set(c) for c in components]
    assert {0, 1, 2} in component_sets
    assert {3, 4, 5} in component_sets


def test_connected_components_isolated() -> None:
    g = Graph(3)
    g.add_edge(0, 1)
    components = g.connected_components()
    assert len(components) == 2
    component_sets = [set(c) for c in components]
    assert {0, 1} in component_sets
    assert {2} in component_sets


def test_connected_components_all_isolated() -> None:
    g = Graph(3)
    components = g.connected_components()
    assert len(components) == 3
    for c in components:
        assert len(c) == 1
