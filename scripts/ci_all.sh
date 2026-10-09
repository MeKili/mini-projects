#!/usr/bin/env bash
# Run the quality gates for every mini-project under projects/.
# Each project is a standalone uv project (ruff + mypy strict + pytest).
# Every project is checked even if an earlier one fails; failures are listed at the end.
set -uo pipefail
shopt -s nullglob

found=0
failed=()
for dir in projects/*/; do
  [ -f "${dir}pyproject.toml" ] || continue
  found=1
  echo "::group::${dir}"
  if ! (
    set -e
    cd "$dir"
    uv sync
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src
    uv run pytest -q
  ); then
    failed+=("$dir")
  fi
  echo "::endgroup::"
done

if [ "$found" -eq 0 ]; then
  echo "No projects yet."
fi
if [ "${#failed[@]}" -gt 0 ]; then
  echo "FAILED projects:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "All projects passed."
