"""Tests for Tarjan's strongly connected components algorithm."""

from tarjan_scc.core import find_sccs


def test_empty_graph() -> None:
    assert find_sccs({}) == []


def test_single_vertex() -> None:
    result = find_sccs({0: []})
    assert result == [[0]]


def test_two_vertex_self_cycle() -> None:
    graph = {0: [0], 1: []}
    result = find_sccs(graph)
    assert sorted(result) == [[0], [1]]


def test_simple_cycle() -> None:
    graph = {0: [1], 1: [2], 2: [0]}
    result = find_sccs(graph)
    assert len(result) == 1
    assert set(result[0]) == {0, 1, 2}


def test_two_separate_cycles() -> None:
    graph = {0: [1], 1: [0], 2: [3], 3: [2]}
    result = find_sccs(graph)
    assert len(result) == 2
    assert {0, 1} in [set(scc) for scc in result]
    assert {2, 3} in [set(scc) for scc in result]


def test_linear_chain() -> None:
    graph = {0: [1], 1: [2], 2: []}
    result = find_sccs(graph)
    assert sorted(result) == [[0], [1], [2]]


def test_complex_graph() -> None:
    graph = {
        0: [1],
        1: [2],
        2: [0, 3],
        3: [4],
        4: [5],
        5: [3],
        6: [5, 7],
        7: [6],
    }
    result = find_sccs(graph)
    assert len(result) == 3
    sccs = [set(scc) for scc in result]
    assert {0, 1, 2} in sccs
    assert {3, 4, 5} in sccs
    assert {6, 7} in sccs


def test_all_strongly_connected() -> None:
    graph = {0: [1], 1: [2], 2: [0]}
    result = find_sccs(graph)
    assert len(result) == 1
    assert set(result[0]) == {0, 1, 2}


def test_complete_graph() -> None:
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    result = find_sccs(graph)
    assert len(result) == 1
    assert set(result[0]) == {0, 1, 2}


def test_diamond_graph() -> None:
    graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
    result = find_sccs(graph)
    assert sorted(result) == [[0], [1], [2], [3]]


def test_scc_with_missing_nodes() -> None:
    graph = {0: [1], 1: [0], 3: [4], 4: [3]}
    result = find_sccs(graph)
    assert len(result) == 2
    assert {0, 1} in [set(scc) for scc in result]
    assert {3, 4} in [set(scc) for scc in result]


def test_self_loop_only() -> None:
    result = find_sccs({5: [5]})
    assert result == [[5]]


def test_output_sorted() -> None:
    graph = {3: [1], 1: [2], 2: [3]}
    result = find_sccs(graph)
    assert result == [[1, 2, 3]]
