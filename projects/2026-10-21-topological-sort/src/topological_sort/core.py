"""Topological sorting of directed acyclic graphs using Kahn's algorithm."""

from __future__ import annotations

from collections import deque


def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Return a topological ordering of nodes in a directed acyclic graph.

    Uses Kahn's algorithm (BFS-based). Nodes are sorted by decreasing in-degree
    to break ties deterministically.

    Args:
        graph: Adjacency list where graph[u] = [v1, v2, ...] means edges u→v1, u→v2, ...

    Returns:
        List of nodes in topological order.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    # Compute in-degrees for all nodes.
    in_degree: dict[str, int] = {}
    all_nodes = set(graph.keys())
    for node in graph:
        all_nodes.update(graph[node])
    for node in all_nodes:
        in_degree[node] = 0
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    # Initialize queue with nodes of zero in-degree, sorted by name for determinism.
    queue: deque[str] = deque(sorted(node for node in all_nodes if in_degree[node] == 0))
    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)

        # Process neighbors and update in-degrees.
        neighbors_sorted = sorted(graph.get(node, []))
        for neighbor in neighbors_sorted:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles.
    if len(result) != len(all_nodes):
        msg = "Graph contains a cycle"
        raise ValueError(msg)

    return result
