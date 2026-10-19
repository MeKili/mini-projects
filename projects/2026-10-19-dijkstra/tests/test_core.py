"""Tests for Dijkstra's algorithm implementation."""

import pytest

from dijkstra.core import dijkstra, dijkstra_path


@pytest.fixture
def simple_graph() -> dict[str, list[tuple[str, int]]]:
    """Simple 4-node graph with weights."""
    return {
        "A": [("B", 1), ("C", 4)],
        "B": [("C", 2), ("D", 5)],
        "C": [("D", 1)],
        "D": [],
    }


@pytest.fixture
def complex_graph() -> dict[str, list[tuple[str, int]]]:
    """Graph with multiple paths and cycles."""
    return {
        "A": [("B", 7), ("C", 9), ("F", 14)],
        "B": [("A", 7), ("C", 10), ("D", 15)],
        "C": [("A", 9), ("B", 10), ("D", 11), ("F", 2)],
        "D": [("B", 15), ("C", 11), ("E", 6)],
        "E": [("D", 6), ("F", 9)],
        "F": [("A", 14), ("C", 2), ("E", 9)],
    }


@pytest.fixture
def disconnected_graph() -> dict[str, list[tuple[str, int]]]:
    """Graph with disconnected components."""
    return {
        "A": [("B", 1)],
        "B": [("A", 1)],
        "C": [("D", 2)],
        "D": [("C", 2)],
    }


def test_dijkstra_simple(simple_graph: dict[str, list[tuple[str, int]]]) -> None:
    """Test dijkstra on simple graph."""
    distances, predecessors = dijkstra(simple_graph, "A")

    assert distances["A"] == 0
    assert distances["B"] == 1
    assert distances["C"] == 3
    assert distances["D"] == 4

    assert predecessors["A"] is None
    assert predecessors["B"] == "A"
    assert predecessors["C"] == "B"
    assert predecessors["D"] == "C"


def test_dijkstra_complex(complex_graph: dict[str, list[tuple[str, int]]]) -> None:
    """Test dijkstra on more complex graph."""
    distances, predecessors = dijkstra(complex_graph, "A")

    assert distances["A"] == 0
    assert distances["B"] == 7
    assert distances["C"] == 9
    assert distances["D"] == 20
    assert distances["E"] == 20  # A->C->F->E = 9+2+9 = 20
    assert distances["F"] == 11


def test_dijkstra_single_node() -> None:
    """Test dijkstra with single node."""
    graph: dict[str, list[tuple[str, int]]] = {"A": []}
    distances, predecessors = dijkstra(graph, "A")

    assert distances["A"] == 0
    assert predecessors["A"] is None


def test_dijkstra_path_simple(simple_graph: dict[str, list[tuple[str, int]]]) -> None:
    """Test dijkstra_path finds correct path."""
    path, distance = dijkstra_path(simple_graph, "A", "D")

    assert path == ["A", "B", "C", "D"]
    assert distance == 4


def test_dijkstra_path_direct_edge(simple_graph: dict[str, list[tuple[str, int]]]) -> None:
    """Test dijkstra_path with direct edge."""
    path, distance = dijkstra_path(simple_graph, "A", "B")

    assert path == ["A", "B"]
    assert distance == 1


def test_dijkstra_path_start_equals_end(simple_graph: dict[str, list[tuple[str, int]]]) -> None:
    """Test dijkstra_path when start equals end."""
    path, distance = dijkstra_path(simple_graph, "A", "A")

    assert path == ["A"]
    assert distance == 0


def test_dijkstra_path_unreachable(disconnected_graph: dict[str, list[tuple[str, int]]]) -> None:
    """Test dijkstra_path returns empty when no path exists."""
    path, distance = dijkstra_path(disconnected_graph, "A", "C")

    assert path == []
    assert distance == float("inf")


def test_dijkstra_path_missing_start_node(simple_graph: dict[str, list[tuple[str, int]]]) -> None:
    """Test dijkstra_path raises when start node missing."""
    with pytest.raises(KeyError):
        dijkstra_path(simple_graph, "Z", "A")


def test_dijkstra_path_missing_end_node(simple_graph: dict[str, list[tuple[str, int]]]) -> None:
    """Test dijkstra_path raises when end node missing."""
    with pytest.raises(KeyError):
        dijkstra_path(simple_graph, "A", "Z")


def test_dijkstra_path_complex(complex_graph: dict[str, list[tuple[str, int]]]) -> None:
    """Test dijkstra_path on complex graph."""
    path, distance = dijkstra_path(complex_graph, "A", "E")

    assert distance == 20  # A->C->F->E = 9+2+9 = 20
    assert len(path) == 4
    assert path[0] == "A"
    assert path[-1] == "E"


def test_dijkstra_all_infinite() -> None:
    """Test dijkstra when no paths exist."""
    graph: dict[str, list[tuple[str, int]]] = {
        "A": [],
        "B": [],
    }
    distances, predecessors = dijkstra(graph, "A")

    assert distances["A"] == 0
    assert distances["B"] == float("inf")
    assert predecessors["B"] is None


def test_dijkstra_weighted_alternatives() -> None:
    """Test dijkstra chooses minimum weight path."""
    graph: dict[str, list[tuple[str, int]]] = {
        "A": [("B", 10), ("C", 1)],
        "B": [("C", 1), ("D", 1)],
        "C": [("D", 10)],
        "D": [],
    }
    distances, _ = dijkstra(graph, "A")

    assert distances["D"] == 11  # Min of A->B->D (10+1) and A->C->D (1+10)
    assert distances["B"] == 10  # A -> B


def test_dijkstra_graph_with_self_loops() -> None:
    """Test dijkstra ignores self-loops with positive weight."""
    graph: dict[str, list[tuple[str, int]]] = {
        "A": [("A", 5), ("B", 1)],
        "B": [("A", 1), ("B", 5)],
    }
    distances, _ = dijkstra(graph, "A")

    assert distances["A"] == 0
    assert distances["B"] == 1
