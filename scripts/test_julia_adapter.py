"""Tests for the Julia binding generator (julia_adapter.generate).

Black-box: build spec dicts in Python (supplying the fields the loader would
have defaulted) or load the checked-in testspec, run generate, and assert on
the emitted lines. The golden tests exercise the real load path.
"""

from pathlib import Path

import pytest

from capigen.loader import load_metadata, load_modules
from capigen.validate import validate_semantics

import julia_adapter

TESTSPEC = Path(__file__).parent / "testspec"

UNSTABLE = [["unstable", "v1.2.0", "2026-01-01"]]
REMOVED = [["removed", "v1.2.0", "2026-01-01"]]


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _meta(**over):
    m = {
        "schema_version": "0.5",
        "prefix": "duckdb_",
        "versions": ["v1.2.0"],
        "lifecycle_states": {
            "unstable": {"visibility": "opt_in", "guard": "DUCKDB_API_UNSTABLE"},
            "frozen": {"visibility": "always"},
            "deprecated": {"visibility": "opt_out", "guard": "DUCKDB_API_NO_DEPRECATED"},
            "removed": {"visibility": "never"},
        },
        "suffixes": {"handles": "", "callbacks": "", "aliases": ""},
        "primitives": [
            {"name": "void", "c_type": "void"},
            {"name": "opaque", "c_type": "void"},
            {"name": "char", "c_type": "char"},
            {"name": "bool", "c_type": "bool"},
            {"name": "idx", "c_type": "idx_t"},
            {"name": "i32", "c_type": "int32_t"},
            {"name": "u32", "c_type": "uint32_t"},
            {"name": "state", "c_type": "duckdb_state"},
        ],
    }
    m.update(over)
    return m


def _mod(name="m", **over):
    mod = {
        "module": name,
        "handles": {},
        "callbacks": {},
        "aliases": {},
        "structs": {},
        "enums": {},
        "functions": {},
    }
    mod.update(over)
    return mod


def _fn(return_type="opaque", return_pointer=0, parameters=None, **over):
    f = {"return_type": return_type, "return_pointer": return_pointer, "parameters": parameters or {}}
    f.update(over)
    return f


def _p(type, indirection=0, **over):
    p = {"type": type, "indirection": indirection}
    p.update(over)
    return p


def _cb(return_type="opaque", parameters=None, **over):
    c = {"return_type": return_type, "parameters": parameters or {}}
    c.update(over)
    return c


def _gen(tmp_path, modules, metadata=None):
    out = tmp_path / "api.jl"
    julia_adapter.generate(modules, metadata if metadata is not None else _meta(), out)
    return out.read_text()


def _gen_types(tmp_path, modules, metadata=None):
    """Run generate and return the sibling ctypes_generated.jl text."""
    julia_adapter.generate(modules, metadata if metadata is not None else _meta(), tmp_path / "api.jl")
    return (tmp_path / "ctypes_generated.jl").read_text()


def _block(text, name):
    """Return the docstring + `function <name> ... end` block as one string."""
    lines = text.splitlines()
    fi = next(i for i, ln in enumerate(lines) if ln.startswith(f"function {name}("))
    op = fi - 1  # closing docstring quote
    assert lines[op] == '"""', f"no docstring above {name}"
    op -= 1
    while lines[op] != '"""':  # opening docstring quote
        op -= 1
    ei = fi
    while lines[ei] != "end":
        ei += 1
    return "\n".join(lines[op : ei + 1])


# ---------------------------------------------------------------------------
# Deprecation (the load-bearing behavior)
# ---------------------------------------------------------------------------


class TestDeprecation:
    def test_structured_lifecycle_deprecated_emits_depwarn(self, tmp_path):
        mod = _mod(functions={"gone": _fn(lifecycle=[["deprecated", "v1.5.0", "2026-01-01"]], description="Old.")})
        body = _block(_gen(tmp_path, [mod]), "duckdb_gone")
        assert "Base.depwarn(" in body
        assert ":duckdb_gone" in body

    def test_deprecated_field_emits_depwarn(self, tmp_path):
        mod = _mod(functions={"flagged": _fn(deprecated="v1.5.0", description="Old.")})
        body = _block(_gen(tmp_path, [mod]), "duckdb_flagged")
        assert "Base.depwarn(" in body
        assert ":duckdb_flagged" in body

    def test_frozen_with_notice_prose_emits_no_depwarn(self, tmp_path):
        # LOAD-BEARING. This function is frozen (not structurally deprecated) yet its
        # description contains "**DEPRECATION NOTICE**:". Deprecation is driven by the
        # structured lifecycle/deprecated field, not the prose. A naive implementation
        # that scanned the description text would WRONGLY emit a depwarn and fail here.
        mod = _mod(
            functions={
                "kept": _fn(
                    lifecycle=[["frozen", "v1.5.4", "2026-05-18"]],
                    description="**DEPRECATION NOTICE**: scheduled for removal.",
                )
            }
        )
        body = _block(_gen(tmp_path, [mod]), "duckdb_kept")
        assert "Base.depwarn" not in body
        # The prose still reaches the docstring; only the depwarn is suppressed.
        assert "**DEPRECATION NOTICE**: scheduled for removal." in body


# ---------------------------------------------------------------------------
# Lifecycle states
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_removed_function_is_skipped(self, tmp_path):
        mod = _mod(functions={"live": _fn(return_type="i32"), "dead": _fn(lifecycle=REMOVED)})
        text = _gen(tmp_path, [mod])
        assert "duckdb_live" in text
        assert "duckdb_dead" not in text

    def test_unstable_function_is_skipped(self, tmp_path):
        """Julia has no preprocessor: an opt-in construct cannot be opted into."""
        mod = _mod(functions={"live": _fn(return_type="i32"), "exp": _fn(lifecycle=UNSTABLE)})
        text = _gen(tmp_path, [mod])
        assert "duckdb_live" in text
        assert "duckdb_exp" not in text

    def test_removed_and_unstable_types_are_skipped(self, tmp_path):
        mod = _mod(
            handles={"kept": {}, "gone": {"lifecycle": REMOVED}},
            enums={
                "color": {"values": {"COLOR_RED": {}}},
                "mood": {"values": {"MOOD_A": {}}, "lifecycle": UNSTABLE},
            },
            functions={"f": _fn(return_type="i32")},
        )
        types = _gen_types(tmp_path, [mod])
        assert "const duckdb_kept = Ptr{Cvoid}" in types
        assert "duckdb_gone" not in types
        assert "const duckdb_color" in types
        assert "duckdb_mood" not in types

    def test_module_with_only_skipped_functions_emits_no_group(self, tmp_path):
        mod = _mod("ghost", functions={"dead": _fn(lifecycle=REMOVED)})
        text = _gen(tmp_path, [mod])
        assert "# Ghost" not in text


# ---------------------------------------------------------------------------
# Blob overrides (regression guard)
# ---------------------------------------------------------------------------


class TestBlobOverrides:
    def test_free_uses_ptr_cvoid(self, tmp_path):
        mod = _mod(functions={"free": _fn(return_type="void", parameters={"ptr": _p("opaque", 1)})})
        body = _block(_gen(tmp_path, [mod]), "duckdb_free")
        assert "(Ptr{Cvoid},)" in body
        assert "Ref{Cvoid}" not in body

    def test_bind_blob_uses_ptr_cvoid(self, tmp_path):
        mod = _mod(
            functions={
                "bind_blob": _fn(
                    return_type="state",
                    parameters={
                        "prepared_statement": _p("opaque"),
                        "param_idx": _p("idx"),
                        "data": _p("opaque", 1),
                        "length": _p("idx"),
                    },
                )
            }
        )
        body = _block(_gen(tmp_path, [mod]), "duckdb_bind_blob")
        assert "(duckdb_prepared_statement, idx_t, Ptr{Cvoid}, idx_t)" in body
        assert "Ref{Cvoid}" not in body

    def test_append_blob_uses_ptr_cvoid(self, tmp_path):
        # Regression: append_blob was missing from the override and emitted Ref{Cvoid};
        # appender.jl passes a Vector{UInt8}, which needs Ptr{Cvoid}.
        mod = _mod(
            functions={
                "append_blob": _fn(
                    return_type="state",
                    parameters={"appender": _p("opaque"), "data": _p("opaque", 1), "length": _p("idx")},
                )
            }
        )
        body = _block(_gen(tmp_path, [mod]), "duckdb_append_blob")
        assert "(duckdb_appender, Ptr{Cvoid}, idx_t)" in body
        assert "Ref{Cvoid}" not in body


# ---------------------------------------------------------------------------
# 1-based index conversion
# ---------------------------------------------------------------------------


class TestOneBasedIndex:
    def test_index_arg_subtracts_one(self, tmp_path):
        mod = _mod(functions={"at": _fn(return_type="i32", parameters={"col": _p("idx", description="The column.")})})
        body = _block(_gen(tmp_path, [mod]), "duckdb_at")
        assert "col - 1" in body  # call site
        assert "(1-based index)" in body  # docstring marker

    def test_return_index_adds_one(self, tmp_path):
        # duckdb_init_get_column_index is in the auto-1base return set.
        mod = _mod(functions={"init_get_column_index": _fn(return_type="idx", parameters={"info": _p("opaque")})})
        body = _block(_gen(tmp_path, [mod]), "duckdb_init_get_column_index")
        assert ") + 1" in body

    def test_ignore_set_leaves_index_arg_untouched(self, tmp_path):
        # duckdb_param_type is in the ignore set: its index-looking `col` stays 0-based.
        mod = _mod(functions={"param_type": _fn(return_type="i32", parameters={"col": _p("idx", description="c")})})
        body = _block(_gen(tmp_path, [mod]), "duckdb_param_type")
        assert "col - 1" not in body
        assert "(1-based index)" not in body


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


class TestTypeMapping:
    def test_char_pointer_is_cstring(self, tmp_path):
        mod = _mod(functions={"f": _fn(parameters={"path": _p("char", 1)})})
        body = _block(_gen(tmp_path, [mod]), "duckdb_f")
        assert "(Cstring,)" in body

    def test_void_pointer_return_is_ptr_cvoid(self, tmp_path):
        mod = _mod(functions={"f": _fn(return_type="void", return_pointer=1)})
        body = _block(_gen(tmp_path, [mod]), "duckdb_f")
        assert "libduckdb), Ptr{Cvoid}, ()" in body  # return-type slot

    def test_handle_out_param_is_ref(self, tmp_path):
        mod = _mod(handles={"database": {}}, functions={"f": _fn(parameters={"out_db": _p("database", 1, kind="OUT")})})
        body = _block(_gen(tmp_path, [mod]), "duckdb_f")
        assert "(Ref{duckdb_database},)" in body

    def test_bare_handle_param_is_c_name(self, tmp_path):
        mod = _mod(handles={"database": {}}, functions={"f": _fn(parameters={"db": _p("database")})})
        body = _block(_gen(tmp_path, [mod]), "duckdb_f")
        assert "(duckdb_database,)" in body

    def test_qualified_alias_resolves_to_underlying(self, tmp_path):
        # sel_t is a qualified alias to u32; a pointer to it must be Ptr{UInt32}, not
        # Ptr{sel} (the old _t-stripped name, which was never a defined Julia type).
        mod = _mod(
            aliases={"sel_t": {"underlying": "u32", "qualified": True}},
            functions={"f": _fn(return_type="sel_t", return_pointer=1)},
        )
        body = _block(_gen(tmp_path, [mod]), "duckdb_f")
        assert "libduckdb), Ptr{UInt32}, ()" in body


# ---------------------------------------------------------------------------
# Reserved-keyword argument escaping
# ---------------------------------------------------------------------------


class TestReservedKeyword:
    def test_reserved_args_escaped_in_signature_and_call(self, tmp_path):
        mod = _mod(functions={"f": _fn(parameters={"type": _p("i32"), "function": _p("i32")})})
        body = _block(_gen(tmp_path, [mod]), "duckdb_f")
        assert "function duckdb_f(_type, _function)" in body
        assert ", _type, _function)" in body  # ccall call args
        assert "duckdb_f(type, function)" not in body  # never the raw reserved names


# ---------------------------------------------------------------------------
# Zero-argument function
# ---------------------------------------------------------------------------


class TestZeroArg:
    def test_zero_arg_emits_empty_tuple(self, tmp_path):
        mod = _mod(functions={"f": _fn(return_type="i32", parameters={})})
        body = _block(_gen(tmp_path, [mod]), "duckdb_f")
        assert "ccall((:duckdb_f, libduckdb), Int32, ())" in body


# ---------------------------------------------------------------------------
# API version header
# ---------------------------------------------------------------------------


class TestApiVersionHeader:
    def test_version_is_the_latest_spec_version(self, tmp_path):
        text = _gen(tmp_path, [_mod(functions={"f": _fn(return_type="i32")})])
        assert 'DUCKDB_API_VERSION = v"1.2.0"' in text

    def test_latest_version_wins_numerically(self, tmp_path):
        """Numeric comparison, not lexicographic: 1.10.0 beats 1.9.0."""
        meta = _meta(versions=["v1.9.0", "v1.10.0"])
        text = _gen(tmp_path, [_mod(functions={"f": _fn(return_type="i32")})], meta)
        assert 'DUCKDB_API_VERSION = v"1.10.0"' in text

    def test_empty_versions_errors(self, tmp_path):
        meta = _meta(versions=[])
        with pytest.raises(ValueError, match="non-empty 'versions' list"):
            julia_adapter.generate([_mod(functions={"f": _fn(return_type="i32")})], meta, tmp_path / "api.jl")


# ---------------------------------------------------------------------------
# Determinism and ordering
# ---------------------------------------------------------------------------


class TestDeterminismAndOrdering:
    def _modules(self):
        # Passed in reverse-alphabetical order on purpose.
        return [
            _mod("zeta", functions={"z": _fn(return_type="i32")}),
            _mod("alpha", functions={"a": _fn(return_type="i32")}),
        ]

    def test_modules_grouped_alphabetically(self, tmp_path):
        text = _gen(tmp_path, self._modules())
        assert text.index("# Alpha") < text.index("# Zeta")

    def test_output_is_deterministic(self, tmp_path):
        out1, out2 = tmp_path / "a.jl", tmp_path / "b.jl"
        julia_adapter.generate(self._modules(), _meta(), out1)
        julia_adapter.generate(self._modules(), _meta(), out2)
        assert out1.read_text() == out2.read_text()


# ---------------------------------------------------------------------------
# Generated type layer (ctypes_generated.jl)
# ---------------------------------------------------------------------------


class TestGeneratedTypes:
    def test_handle_becomes_pointer_const(self, tmp_path):
        mod = _mod(handles={"gadget": {}}, functions={"f": _fn(return_type="i32")})
        types = _gen_types(tmp_path, [mod])
        assert "const duckdb_gadget = Ptr{Cvoid}" in types

    def test_callback_keeps_t_suffix_and_never_collides_with_handle(self, tmp_path):
        # Callbacks keep their C typedef name (no _t strip), so a callback and a
        # same-stemmed handle stay distinct without any hardcoded disambiguation.
        mod = _mod(
            handles={"widget": {}},
            callbacks={"widget_cb_t": _cb()},
            functions={"f": _fn(return_type="i32")},
        )
        types = _gen_types(tmp_path, [mod])
        assert "const duckdb_widget = Ptr{Cvoid}" in types
        assert "const duckdb_widget_cb_t = Ptr{Cvoid}" in types

    def test_enum_variant_names_and_values(self, tmp_path):
        enum = {"values": {"COLOR_RED": {}, "COLOR_GREEN": {}, "COLOR_BLUE": {"value": 5}}}
        mod = _mod(enums={"color": enum}, functions={"f": _fn(return_type="i32")})
        types = _gen_types(tmp_path, [mod])
        assert "@enum DUCKDB_COLOR_::Cint begin" in types
        assert "    DUCKDB_COLOR_RED = 0" in types  # auto-numbered from 0
        assert "    DUCKDB_COLOR_GREEN = 1" in types  # auto-numbered
        assert "    DUCKDB_COLOR_BLUE = 5" in types  # explicit value from spec
        assert "const duckdb_color = DUCKDB_COLOR_" in types

    def test_enum_public_alias_is_lowercase_even_for_uppercase_spec_name(self, tmp_path):
        # The uppercase spec name TYPE still yields the consistent lowercase public
        # const duckdb_type (no per-enum special-case).
        mod = _mod(enums={"TYPE": {"values": {"TYPE_A": {"value": 0}}}}, functions={"f": _fn(return_type="i32")})
        types = _gen_types(tmp_path, [mod])
        assert "@enum DUCKDB_TYPE_::Cint begin" in types
        assert "    DUCKDB_TYPE_A = 0" in types
        assert "const duckdb_type = DUCKDB_TYPE_" in types

    def test_no_width_sentinel_in_julia_enums(self, tmp_path):
        """Julia enums are ::Cint by construction; the C sentinel is not mirrored."""
        mod = _mod(enums={"color": {"values": {"COLOR_RED": {}}}}, functions={"f": _fn(return_type="i32")})
        types = _gen_types(tmp_path, [mod])
        assert "MAX_ENUM" not in types


# ---------------------------------------------------------------------------
# Loader-backed golden tests
# ---------------------------------------------------------------------------


class TestGolden:
    def test_matches_checked_in_golden(self, tmp_path):
        # Byte-for-byte against the committed golden: a changed banner, spacing, or
        # ccall wrap fails here, not just a substring check. Exercises the real load
        # path (schema defaults) that the direct-dict tests bypass.
        metadata = load_metadata(TESTSPEC)
        modules = load_modules(TESTSPEC)
        out = tmp_path / "api.jl"
        julia_adapter.generate(modules, metadata, out)
        assert out.read_text() == (TESTSPEC / "expected_api.jl").read_text()

    def test_matches_ctypes_generated_golden(self, tmp_path):
        # Byte-for-byte golden for the generated type layer (handles, callbacks, enums).
        metadata = load_metadata(TESTSPEC)
        modules = load_modules(TESTSPEC)
        julia_adapter.generate(modules, metadata, tmp_path / "api.jl")
        got = (tmp_path / "ctypes_generated.jl").read_text()
        assert got == (TESTSPEC / "expected_ctypes_generated.jl").read_text()

    def test_fixture_validates(self):
        metadata = load_metadata(TESTSPEC)
        modules = load_modules(TESTSPEC)
        assert validate_semantics(modules, metadata) == []
