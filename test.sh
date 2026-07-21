set -e

# Tests run against DuckDB_jll by default. To test against a locally built
# libduckdb, set JULIA_DUCKDB_LIBRARY to its path before running.
export JULIA_NUM_THREADS=1
julia --project -e "import Pkg; Pkg.test(; test_args = [\"$1\"])"
