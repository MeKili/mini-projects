"""Dijkstra's algorithm implementation for shortest paths."""

import heapq


def dijkstra(
    graph: dict[str, list[tuple[str, int]]], start: str
) -> tuple[dict[str, float], dict[str, str | None]]:
    """
    Compute shortest distances and predecessors from start node using Dijkstra's algorithm.

    Args:
        graph: Adjacency list where graph[u] = [(v, weight), ...] for edge u -> v
        start: Starting node

    Returns:
        (distances, predecessors) where:
        - distances[node] = shortest distance from start to node
        - predecessors[node] = previous node in shortest path (None for start)
    """
    distances: dict[str, float] = {node: float("inf") for node in graph}
    distances[start] = 0

    predecessors: dict[str, str | None] = {node: None for node in graph}

    visited: set[str] = set()
    heap: list[tuple[int, str]] = [(0, start)]

    while heap:
        current_dist, current = heapq.heappop(heap)

        if current in visited:
            continue

        visited.add(current)

        if current_dist > distances[current]:
            continue

        for neighbor, weight in graph[current]:
            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = current
                heapq.heappush(heap, (new_dist, neighbor))

    return distances, predecessors


def dijkstra_path(
    graph: dict[str, list[tuple[str, int]]], start: str, end: str
) -> tuple[list[str], float]:
    """
    Find shortest path from start to end node.

    Args:
        graph: Adjacency list where graph[u] = [(v, weight), ...] for edge u -> v
        start: Starting node
        end: Ending node

    Returns:
        (path, distance) where path is list of nodes from start to end,
        and distance is the total path length. Returns ([], float('inf')) if
        no path exists.

    Raises:
        KeyError: If start or end node not in graph
    """
    if start not in graph or end not in graph:
        raise KeyError("Node not in graph")

    distances, predecessors = dijkstra(graph, start)

    if distances[end] == float("inf"):
        return [], float("inf")

    path: list[str] = []
    current: str | None = end

    while current is not None:
        path.append(current)
        current = predecessors[current]

    path.reverse()
    return path, distances[end]
