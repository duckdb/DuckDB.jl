module DuckDBExcelJLLExt

using DuckDB
using DuckDB_excel_jll

function __init__()
    DuckDB_excel_jll.is_available() || return nothing
    DuckDB.add_extension_directory!(DuckDB_excel_jll.duckdb_extensions_dir)
    return nothing
end

end # module
