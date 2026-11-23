"""Graph traversal algorithms: BFS and DFS with path finding and connected components."""

from __future__ import annotations

from collections import deque
from typing import TypeVar

V = TypeVar("V")


class Graph:
    """Undirected graph using adjacency list representation."""

    def __init__(self, vertices: int) -> None:
        if vertices <= 0:
            raise ValueError("vertices must be positive")
        self.vertices = vertices
        self.adj_list: dict[int, list[int]] = {i: [] for i in range(vertices)}

    def add_edge(self, u: int, v: int) -> None:
        """Add an undirected edge between vertices u and v."""
        if u < 0 or u >= self.vertices or v < 0 or v >= self.vertices:
            raise ValueError("vertex out of range")
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def bfs(self, start: int) -> list[int]:
        """Return vertices in breadth-first order from start vertex."""
        if start < 0 or start >= self.vertices:
            raise ValueError("start vertex out of range")
        visited: set[int] = set()
        result: list[int] = []
        queue: deque[int] = deque([start])
        visited.add(start)
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            for neighbor in self.adj_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return result

    def dfs_recursive(self, start: int) -> list[int]:
        """Return vertices in depth-first order from start vertex (recursive)."""
        if start < 0 or start >= self.vertices:
            raise ValueError("start vertex out of range")
        visited: set[int] = set()
        result: list[int] = []

        def _dfs(v: int) -> None:
            visited.add(v)
            result.append(v)
            for neighbor in self.adj_list[v]:
                if neighbor not in visited:
                    _dfs(neighbor)

        _dfs(start)
        return result

    def dfs_iterative(self, start: int) -> list[int]:
        """Return vertices in depth-first order from start vertex (iterative)."""
        if start < 0 or start >= self.vertices:
            raise ValueError("start vertex out of range")
        visited: set[int] = set()
        result: list[int] = []
        stack: list[int] = [start]
        visited.add(start)
        while stack:
            vertex = stack.pop()
            result.append(vertex)
            for neighbor in reversed(self.adj_list[vertex]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return result

    def find_path(self, start: int, end: int) -> list[int] | None:
        """Return shortest path from start to end using BFS, or None if unreachable."""
        if start < 0 or start >= self.vertices or end < 0 or end >= self.vertices:
            raise ValueError("vertex out of range")
        if start == end:
            return [start]
        visited: set[int] = {start}
        parent: dict[int, int] = {}
        queue: deque[int] = deque([start])
        while queue:
            vertex = queue.popleft()
            if vertex == end:
                path: list[int] = []
                current = end
                while current != start:
                    path.append(current)
                    current = parent[current]
                path.append(start)
                return path[::-1]
            for neighbor in self.adj_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = vertex
                    queue.append(neighbor)
        return None

    def connected_components(self) -> list[list[int]]:
        """Return list of connected components, each as a list of vertices."""
        visited: set[int] = set()
        components: list[list[int]] = []
        for v in range(self.vertices):
            if v not in visited:
                component = self.bfs(v)
                components.append(component)
                visited.update(component)
        return components

    def has_path(self, start: int, end: int) -> bool:
        """Return whether there is a path from start to end."""
        return self.find_path(start, end) is not None
