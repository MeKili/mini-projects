"""A* pathfinding algorithm implementation."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Hashable, Sequence

Node = Hashable


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Return the Manhattan distance between two (x, y) grid points."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def astar(
    start: Node,
    goal: Node,
    neighbors: Callable[[Node], Sequence[tuple[Node, float]]],
    heuristic: Callable[[Node], float],
) -> tuple[list[Node], float] | None:
    """Find shortest path from start to goal using A* algorithm.

    Args:
        start: Starting node.
        goal: Target node.
        neighbors: Function that returns list of (neighbor_node, cost_to_neighbor) tuples.
        heuristic: Function that returns estimated cost from node to goal.

    Returns:
        Tuple of (path, total_cost) if goal is reachable, None otherwise.
        Path is ordered from start to goal.
    """
    # Open set: priority queue of (f_score, counter, node)
    # Counter ensures FIFO for ties to avoid heap comparison issues
    open_set: list[tuple[float, int, Node]] = [(heuristic(start), 0, start)]
    counter = 1

    # Track visited nodes and their costs
    g_score: dict[Node, float] = {start: 0.0}
    came_from: dict[Node, Node] = {}
    closed: set[Node] = set()

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            path: list[Node] = []
            node: Node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        if current in closed:
            continue
        closed.add(current)

        for neighbor, cost in neighbors(current):
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, counter, neighbor))
                counter += 1

    return None
