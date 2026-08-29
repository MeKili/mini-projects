# mini-projects

A curated, growing collection of small, **finished** engineering mini-projects — each one
self-contained, typed and tested. A new one is added every few days, spanning algorithms and
data structures, ML/DS, NLP, data engineering, and developer tooling.

Every project lives in its own folder under [`projects/`](projects) as a standalone
[uv](https://docs.astral.sh/uv/) project with the same quality bar (ruff + mypy `--strict` +
pytest). CI runs each project's checks on every push, so everything here stays green.

## Projects

| Date | Project | Description |
|---|---|---|
| 2026-08-29 | [kmeans](projects/2026-08-29-kmeans) | K-means clustering from scratch with k-means++ initialization |
| 2026-08-26 | [reservoir-sampling](projects/2026-08-26-reservoir-sampling) | Select k random items from a stream in a single pass (Algorithm R) |
| 2026-08-24 | [edit-distance](projects/2026-08-24-edit-distance) | Levenshtein distance and fuzzy string matching |
| 2026-08-22 | [union-find](projects/2026-08-22-union-find) | Fast, typed union-find with path compression and union by rank |
| 2026-08-19 | [lru-cache](projects/2026-08-19-lru-cache) | Fast, typed LRU cache with O(1) get and put operations |
| 2026-08-16 | [cosine-topk](projects/2026-08-16-cosine-topk) | In-memory top-k vector similarity search (cosine), pure Python |
