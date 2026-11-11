"""Tarjan's strongly connected components algorithm implementation."""

from __future__ import annotations


def find_sccs(graph: dict[int, list[int]]) -> list[list[int]]:
    """Find all strongly connected components in a directed graph.

    Args:
        graph: Adjacency list representation where graph[u] = [v1, v2, ...] means
               edges from u to v1, v2, etc. Graph must have non-negative integer vertices.

    Returns:
        List of SCCs, each SCC is a sorted list of vertex indices. SCCs are sorted by
        their minimum vertex index.
    """
    if not graph:
        return []

    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    index_counter = [0]
    stack: list[int] = []
    lowlinks = {v: -1 for v in all_vertices}
    index = {v: -1 for v in all_vertices}
    on_stack = {v: False for v in all_vertices}
    sccs: list[list[int]] = []

    def strongconnect(v: int) -> None:
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in graph.get(v, []):
            if index[w] == -1:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                lowlinks[v] = min(lowlinks[v], index[w])

        if lowlinks[v] == index[v]:
            component: list[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            component.sort()
            sccs.append(component)

    for v in sorted(all_vertices):
        if index[v] == -1:
            strongconnect(v)

    sccs.sort(key=lambda c: c[0])
    return sccs
