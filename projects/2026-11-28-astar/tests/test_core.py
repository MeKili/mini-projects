"""Tests for A* pathfinding algorithm."""

import math

from astar.core import astar, manhattan_distance


def test_manhattan_distance() -> None:
    """Test Manhattan distance calculation."""
    assert manhattan_distance((0, 0), (3, 4)) == 7.0
    assert manhattan_distance((0, 0), (0, 0)) == 0.0
    assert manhattan_distance((-1, -1), (1, 1)) == 4.0


def test_astar_simple_grid() -> None:
    """Test A* on a simple 3x3 grid with no obstacles."""

    def neighbors(pos: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        x, y = pos
        result = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                result.append(((nx, ny), 1.0))
        return result

    def heuristic(pos: tuple[int, int]) -> float:
        return manhattan_distance(pos, (2, 2))

    result = astar((0, 0), (2, 2), neighbors, heuristic)
    assert result is not None
    path, cost = result
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert cost == 4.0  # Minimum steps: right, right, down, down


def test_astar_with_obstacles() -> None:
    """Test A* avoiding obstacles."""
    obstacles = {(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)}

    def neighbors(pos: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        x, y = pos
        result = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 5 and 0 <= ny < 5 and (nx, ny) not in obstacles:
                result.append(((nx, ny), 1.0))
        return result

    def heuristic(pos: tuple[int, int]) -> float:
        return manhattan_distance(pos, (4, 4))

    result = astar((0, 0), (4, 4), neighbors, heuristic)
    assert result is not None
    path, cost = result
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    assert all(pos not in obstacles for pos in path)
    assert cost >= 8.0  # Must detour around obstacles


def test_astar_unreachable() -> None:
    """Test A* when goal is unreachable."""
    obstacles = {(1, 0), (1, 1), (1, 2)}  # Wall blocking path

    def neighbors(pos: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        x, y = pos
        result = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3 and (nx, ny) not in obstacles:
                result.append(((nx, ny), 1.0))
        return result

    def heuristic(pos: tuple[int, int]) -> float:
        return manhattan_distance(pos, (2, 1))

    result = astar((0, 1), (2, 1), neighbors, heuristic)
    assert result is None


def test_astar_weighted_edges() -> None:
    """Test A* with varying edge weights."""

    def neighbors(pos: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        x, y = pos
        # Prefer horizontal movement (cost 1.0) over vertical (cost 2.0)
        result = []
        if x < 2:
            result.append(((x + 1, y), 1.0))
        if y < 2:
            result.append(((x, y + 1), 2.0))
        if x > 0:
            result.append(((x - 1, y), 1.0))
        if y > 0:
            result.append(((x, y - 1), 2.0))
        return result

    def heuristic(pos: tuple[int, int]) -> float:
        return manhattan_distance(pos, (2, 2))

    result = astar((0, 0), (2, 2), neighbors, heuristic)
    assert result is not None
    path, cost = result
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Optimal path prefers horizontal movement
    assert cost == 6.0  # Right, right (cost 1+1), down, down (cost 2+2)


def test_astar_start_equals_goal() -> None:
    """Test A* when start and goal are the same."""

    def neighbors(pos: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        return []

    def heuristic(pos: tuple[int, int]) -> float:
        return 0.0

    result = astar((0, 0), (0, 0), neighbors, heuristic)
    assert result is not None
    path, cost = result
    assert path == [(0, 0)]
    assert cost == 0.0


def test_astar_string_nodes() -> None:
    """Test A* with string nodes (graph representation)."""

    def neighbors(node: str) -> list[tuple[str, float]]:
        graph: dict[str, list[tuple[str, float]]] = {
            "A": [("B", 1.0), ("C", 4.0)],
            "B": [("A", 1.0), ("D", 2.0)],
            "C": [("A", 4.0), ("D", 1.0)],
            "D": [("B", 2.0), ("C", 1.0)],
        }
        return graph.get(node, [])

    def heuristic(node: str) -> float:
        # Rough estimates to goal D
        estimates = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
        return estimates.get(node, 0.0)

    result = astar("A", "D", neighbors, heuristic)
    assert result is not None
    path, cost = result
    assert path[0] == "A"
    assert path[-1] == "D"
    # Optimal path: A -> B -> D (cost 1 + 2 = 3) is better than A -> C -> D (4 + 1 = 5)
    assert math.isclose(cost, 3.0)
