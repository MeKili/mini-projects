#!/usr/bin/env bash
# Run the quality gates for every mini-project under projects/.
# Each project is a standalone uv project (ruff + mypy strict + pytest).
set -euo pipefail
shopt -s nullglob

found=0
for dir in projects/*/; do
  [ -f "${dir}pyproject.toml" ] || continue
  found=1
  echo "::group::${dir}"
  (
    cd "$dir"
    uv sync
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src
    uv run pytest -q
  )
  echo "::endgroup::"
done

if [ "$found" -eq 0 ]; then
  echo "No projects yet."
fi
echo "All projects passed."
