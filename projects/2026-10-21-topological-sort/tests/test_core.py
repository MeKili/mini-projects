"""Tests for topological sort."""

import pytest

from topological_sort.core import topological_sort


def test_simple_linear_graph() -> None:
    """Test a simple linear ordering: 1 -> 2 -> 3."""
    graph = {"1": ["2"], "2": ["3"], "3": []}
    result = topological_sort(graph)
    assert result == ["1", "2", "3"]


def test_multiple_branches() -> None:
    """Test a graph with multiple branches."""
    graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    result = topological_sort(graph)
    # 'a' must come before 'b' and 'c', both must come before 'd'.
    assert result.index("a") < result.index("b")
    assert result.index("a") < result.index("c")
    assert result.index("b") < result.index("d")
    assert result.index("c") < result.index("d")


def test_disconnected_components() -> None:
    """Test a graph with multiple disconnected components."""
    graph = {"a": ["b"], "b": [], "c": ["d"], "d": []}
    result = topological_sort(graph)
    assert len(result) == 4
    assert result.index("a") < result.index("b")
    assert result.index("c") < result.index("d")


def test_empty_graph() -> None:
    """Test with no nodes and no edges."""
    graph: dict[str, list[str]] = {}
    result = topological_sort(graph)
    assert result == []


def test_single_node() -> None:
    """Test with a single isolated node."""
    graph = {"x": []}
    result = topological_sort(graph)
    assert result == ["x"]


def test_node_with_no_outgoing_edges() -> None:
    """Test that nodes with incoming edges but no outgoing edges are included."""
    graph = {"a": ["b"], "b": []}
    result = topological_sort(graph)
    assert set(result) == {"a", "b"}


def test_cycle_detection_simple() -> None:
    """Test detection of a simple cycle: a -> b -> a."""
    graph = {"a": ["b"], "b": ["a"]}
    with pytest.raises(ValueError, match="cycle"):
        topological_sort(graph)


def test_cycle_detection_self_loop() -> None:
    """Test detection of a self-loop."""
    graph = {"a": ["a"]}
    with pytest.raises(ValueError, match="cycle"):
        topological_sort(graph)


def test_cycle_detection_triangle() -> None:
    """Test detection of a cycle in a larger graph: a -> b -> c -> a."""
    graph = {"a": ["b"], "b": ["c"], "c": ["a"]}
    with pytest.raises(ValueError, match="cycle"):
        topological_sort(graph)


def test_deterministic_ordering() -> None:
    """Test that the result is deterministic with tie-breaking by name."""
    graph = {"a": ["c"], "b": ["c"], "c": []}
    result = topological_sort(graph)
    # Both 'a' and 'b' have in-degree 0, so they appear before 'c'.
    # Sorting ensures 'a' comes before 'b' for determinism.
    assert result == ["a", "b", "c"]


def test_complex_dag() -> None:
    """Test a more complex DAG (course prerequisites example)."""
    graph = {
        "intro": ["dsa", "webdev"],
        "dsa": ["algorithms", "db"],
        "webdev": ["frontend", "backend"],
        "algorithms": ["ml"],
        "db": [],
        "frontend": [],
        "backend": [],
        "ml": [],
    }
    result = topological_sort(graph)
    # Check that prerequisites come before their dependents.
    assert result.index("intro") < result.index("dsa")
    assert result.index("intro") < result.index("webdev")
    assert result.index("dsa") < result.index("algorithms")
    assert result.index("dsa") < result.index("db")
    assert result.index("webdev") < result.index("frontend")
    assert result.index("webdev") < result.index("backend")
    assert result.index("algorithms") < result.index("ml")


def test_isolated_nodes_implicit() -> None:
    """Test that nodes appearing only as targets are included."""
    # Node 'b' is not a key, only appears as a target.
    graph = {"a": ["b"]}
    result = topological_sort(graph)
    assert set(result) == {"a", "b"}
