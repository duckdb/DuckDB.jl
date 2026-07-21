#!/usr/bin/env bash
# Regenerates src/api.jl and src/ctypes_generated.jl from the DuckDB C API v1
# spec via the capigen Julia adapter, then runs JuliaFormatter.
#
# By default the spec is fetched (sparse, shallow) from the DuckDB repository.
# Override the source with SPEC_REPO / SPEC_REF, or set SPEC_DIR to a local
# api_spec/v1 checkout to skip fetching entirely. capigen is pulled from PyPI via
# uv (see pyproject.toml).
set -euo pipefail

SPEC_DIR="${SPEC_DIR:-${1:-}}"
if [[ -n "$SPEC_DIR" ]]; then
  [[ -d "$SPEC_DIR" ]] || { echo "SPEC_DIR '$SPEC_DIR' is not a directory"; exit 1; }
  SPEC_DIR="$(cd "$SPEC_DIR" && pwd)"  # absolutize before we cd to the repo root
fi

cd "$(git rev-parse --show-toplevel)"

if [[ -z "$SPEC_DIR" ]]; then
  SPEC_REPO="${SPEC_REPO:-https://github.com/duckdb/duckdb}"
  SPEC_REF="${SPEC_REF:-main}"
  echo "Fetching api_spec from $SPEC_REPO @ $SPEC_REF ..."
  rm -rf .spec
  git clone --quiet --filter=blob:none --sparse --depth 1 --branch "$SPEC_REF" "$SPEC_REPO" .spec
  git -C .spec sparse-checkout set api_spec
  SPEC_DIR=".spec/api_spec/v1"
  [[ -d "$SPEC_DIR" ]] || {
    echo "api_spec/v1 not found in $SPEC_REPO @ $SPEC_REF; the C API spec may not be present at that ref yet."
    exit 1
  }
fi

echo "Regenerating api.jl + ctypes_generated.jl from $SPEC_DIR ..."
PYTHONPATH=scripts uv run capigen julia_adapter --spec-dir "$SPEC_DIR" -o src/api.jl

echo "Formatting..."
./format.sh
