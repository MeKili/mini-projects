"""Tarjan's algorithm for finding strongly connected components in directed graphs.

A strongly connected component is a maximal subgraph where every vertex is reachable from
every other vertex. This implementation uses Tarjan's algorithm with a single DFS pass to
identify all SCCs in O(V + E) time.
"""

from tarjan_scc.core import find_sccs

__all__ = ["find_sccs"]
