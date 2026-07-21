#!/usr/bin/env bash
# Regenerates tools/juliapkg/src/api.jl from api_spec/v1 via the capigen Julia
# adapter, then runs JuliaFormatter for the local dev loop. The raw adapter output
# is already formatter clean, so the format step is a no-op; CI regenerates
# without Julia (scripts/capi_v1_julia_regen.sh) and checks for drift.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "Regenerating api.jl..."
scripts/capi_v1_julia_regen.sh

echo "Formatting..."
cd tools/juliapkg && ./format.sh
