"""astar — A* pathfinding algorithm for grid-based and graph-based search.

Given a start and goal position, heuristic function, and cost function, find the
shortest path using the A* algorithm. Supports both grid-based and arbitrary graph
traversal with customizable heuristics.
"""

from astar.core import astar, manhattan_distance

__all__ = ["astar", "manhattan_distance"]
__version__ = "0.1.0"
