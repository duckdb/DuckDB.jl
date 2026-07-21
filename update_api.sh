#!/usr/bin/env bash
# Regenerates src/api.jl and src/ctypes_generated.jl from the DuckDB C API v1
# spec via the capigen Julia adapter, then runs JuliaFormatter.
#
# The spec (api_spec/v1) lives in the DuckDB C API repo, not here. Point SPEC_DIR
# at a checkout of it. capigen is pulled from PyPI via uv (see pyproject.toml).
set -euo pipefail

SPEC_DIR="${SPEC_DIR:-${1:-}}"
if [[ -z "$SPEC_DIR" || ! -d "$SPEC_DIR" ]]; then
  echo "Usage: SPEC_DIR=/path/to/api_spec/v1 ./update_api.sh   (or pass it as arg 1)"
  echo "SPEC_DIR must point at a checkout of the DuckDB C API v1 spec."
  exit 1
fi

cd "$(git rev-parse --show-toplevel)"
echo "Regenerating api.jl + ctypes_generated.jl from $SPEC_DIR ..."
PYTHONPATH=scripts uv run capigen julia_adapter --spec-dir "$SPEC_DIR" -o src/api.jl

echo "Formatting..."
./format.sh
