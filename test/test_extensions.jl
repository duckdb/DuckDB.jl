# test_extensions.jl

@testset "Extension directories" begin
    setting(con, name) = first(Tables.rows(DBInterface.execute(con, "SELECT current_setting('$name') AS v"))).v

    @test isempty(DuckDB.extension_directories())

    # without registered directories DuckDB.jl leaves the signature check alone
    con = DBInterface.connect(DuckDB.DB, ":memory:")
    @test setting(con, "allow_unsigned_extensions") == false
    version = String(first(Tables.rows(DBInterface.execute(con, "PRAGMA version"))).library_version)
    platform = String(first(Tables.rows(DBInterface.execute(con, "PRAGMA platform"))).platform)
    DBInterface.close!(con)

    mktempdir() do dir
        registered = DuckDB.add_extension_directory!(dir)
        @test DuckDB.extension_directories() == [registered]
        # registering twice is a no-op
        @test DuckDB.add_extension_directory!(dir) == registered
        @test DuckDB.extension_directories() == [registered]

        # an empty directory: the settings are adjusted, nothing is loaded
        con = DBInterface.connect(DuckDB.DB, ":memory:")
        @test setting(con, "allow_unsigned_extensions") == true
        dirs = setting(con, "extension_directories")
        # DuckDB's default directory stays in front so INSTALL keeps working
        @test dirs[1] == "~/.duckdb/extensions"
        @test registered in dirs
        DBInterface.close!(con)

        # explicit user configuration wins
        con = DBInterface.connect(DuckDB.DB, ":memory:"; config = ["allow_unsigned_extensions" => "false"])
        @test setting(con, "allow_unsigned_extensions") == false
        DBInterface.close!(con)

        con = DBInterface.connect(DuckDB.DB, ":memory:"; config = ["extension_directory" => registered])
        @test setting(con, "extension_directory") == registered
        @test setting(con, "extension_directories") == [registered]
        DBInterface.close!(con)

        # a file that is not a valid extension for this build fails loudly
        ext_dir = joinpath(dir, version, platform)
        mkpath(ext_dir)
        bogus = joinpath(ext_dir, "bogus.duckdb_extension")
        write(bogus, repeat("not an extension\n", 64))
        @test_throws DuckDB.ConnectionException DBInterface.connect(DuckDB.DB, ":memory:")
        rm(bogus)

        # files for other versions or platforms are ignored
        other_dir = joinpath(dir, "v0.0.0", platform)
        mkpath(other_dir)
        write(joinpath(other_dir, "bogus.duckdb_extension"), repeat("not an extension\n", 64))
        con = DBInterface.connect(DuckDB.DB, ":memory:")
        @test setting(con, "allow_unsigned_extensions") == true
        DBInterface.close!(con)

        @test DuckDB.remove_extension_directory!(dir)
        @test !DuckDB.remove_extension_directory!(dir)
        @test isempty(DuckDB.extension_directories())
    end

    con = DBInterface.connect(DuckDB.DB, ":memory:")
    @test setting(con, "allow_unsigned_extensions") == false
    DBInterface.close!(con)
end
