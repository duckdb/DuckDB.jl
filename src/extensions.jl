"""
Directories holding DuckDB extensions that are distributed as Julia packages,
typically `DuckDB_<name>_jll` packages built alongside `DuckDB_jll`.

Each directory has the layout DuckDB expects for an extension directory:

    <dir>/<duckdb version>/<duckdb platform>/<name>.duckdb_extension

Use [`add_extension_directory!`](@ref) to register a directory. Every `DB` opened
afterwards

  * allows unsigned extensions (extensions built outside DuckDB's own release
    pipeline carry no signature) unless `allow_unsigned_extensions` was set
    explicitly in the configuration,
  * appends the registered directories to the `extension_directories` setting
    so that `LOAD <name>` and `duckdb_extensions()` see them, unless
    `extension_directories` was set explicitly in the configuration, and
  * loads every extension found in them for the running DuckDB version and
    platform, so the extension is usable without a `LOAD` statement and cannot
    be shadowed by a copy that was downloaded into the default extension
    directory.
"""
const EXTENSION_DIRECTORIES = String[]
const EXTENSION_DIRECTORIES_LOCK = ReentrantLock()

"""
    add_extension_directory!(dir::AbstractString)

Register `dir` as a directory containing DuckDB extensions (see
[`EXTENSION_DIRECTORIES`](@ref) for the expected layout). Registering the same
directory twice has no effect. Returns the normalized directory path.

Packages that ship DuckDB extensions call this from their initialization code;
end users normally do not need to call it.
"""
function add_extension_directory!(dir::AbstractString)
    dir = abspath(expanduser(String(dir)))
    lock(EXTENSION_DIRECTORIES_LOCK) do
        return dir in EXTENSION_DIRECTORIES || push!(EXTENSION_DIRECTORIES, dir)
    end
    return dir
end

"""
    remove_extension_directory!(dir::AbstractString)

Undo [`add_extension_directory!`](@ref) for `dir`. Databases that are already
open are not affected. Returns `true` if the directory was registered.
"""
function remove_extension_directory!(dir::AbstractString)
    dir = abspath(expanduser(String(dir)))
    return lock(EXTENSION_DIRECTORIES_LOCK) do
        idx = findfirst(==(dir), EXTENSION_DIRECTORIES)
        idx === nothing && return false
        deleteat!(EXTENSION_DIRECTORIES, idx)
        return true
    end
end

"""
    extension_directories()

The directories registered with [`add_extension_directory!`](@ref), in
registration order.
"""
extension_directories() = lock(() -> copy(EXTENSION_DIRECTORIES), EXTENSION_DIRECTORIES_LOCK)

# SQL string literal; DuckDB does not treat backslashes as escapes in plain
# string literals, so only quotes need doubling.
_sql_string_literal(s::AbstractString) = string('\'', replace(s, "'" => "''"), '\'')

# The directory name DuckDB uses for the running version: the release tag for
# releases, the source id for development builds (see
# ExtensionHelper::GetVersionDirectoryName).
function _extension_version_directory(db)
    row = first(Tables.rows(DBInterface.execute(db, "PRAGMA version")))
    version = String(row.library_version)
    return occursin("-dev", version) ? String(row.source_id) : version
end

function _extension_platform(db)
    row = first(Tables.rows(DBInterface.execute(db, "PRAGMA platform")))
    return String(row.platform)
end

# Called while opening a database, after the main connection exists.
function _load_registered_extensions(db, config::Config, dirs::Vector{String})
    isempty(dirs) && return

    if !haskey(config, "extension_directories")
        # DuckDB drops its default directory as soon as extension_directories
        # is set, and INSTALL writes into the first directory it knows about.
        # Keep the default in front so INSTALL keeps working as before. If the
        # user picked an explicit extension_directory, DuckDB already searches
        # it first, so there is nothing to add.
        search = haskey(config, "extension_directory") ? String[] : ["~/.duckdb/extensions"]
        append!(search, dirs)
        DBInterface.execute(db, "SET extension_directories = [" * join(_sql_string_literal.(search), ", ") * "]")
    end

    version = _extension_version_directory(db)
    platform = _extension_platform(db)
    for dir in dirs
        ext_dir = joinpath(dir, version, platform)
        isdir(ext_dir) || continue
        for file in sort!(readdir(ext_dir))
            endswith(file, ".duckdb_extension") || continue
            path = joinpath(ext_dir, file)
            try
                DBInterface.execute(db, "LOAD " * _sql_string_literal(path))
            catch e
                e isa QueryException || rethrow()
                throw(
                    ConnectionException(
                        "Failed to load DuckDB extension \"$path\" from the extension directory \"$dir\" " *
                        "registered with DuckDB.add_extension_directory!: $(e.var)"
                    )
                )
            end
        end
    end
    return
end
