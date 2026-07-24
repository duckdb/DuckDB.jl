"""DuckDB.jl's binding generator: regenerate the Julia layer from the C API spec.

One generate() call writes two sibling files:
  - api.jl (at output_path): the ccall function wrappers.
  - ctypes_generated.jl (sibling): the mechanical type layer api.jl needs, i.e.
    every spec handle and callback as `const duckdb_<name> = Ptr{Cvoid}` and every
    spec enum as an `@enum`. It is included before the hand-written ctypes.jl.

This is a binding generator, not a capigen adapter: the generated Julia code is a
consumer of the C ABI that duckdb.h defines, and it lives here, with the binding
it serves. It reads the spec through capigen's public library surface (loader,
validate, states, tools), pinned by the capigen dependency in pyproject.toml,
which also pins the spec schema line it can read. Correctness is proven by this
repository's test suite against the real library.

Lifecycle mapping. Julia has no preprocessor, so visibilities translate:
  - always and opt_out constructs are emitted; a deprecated function gets a
    Base.depwarn (the runtime analog of the C attribute).
  - opt_in constructs are skipped: a Julia consumer has no way to opt in.
  - never constructs are skipped.

The generator's declared version (DUCKDB_API_VERSION) is the latest entry in the
spec's `versions`: the spec describes the API as of that version.

The ccall and Base.depwarn are emitted pre-wrapped in JuliaFormatter's canonical
form (margin 120), so the committed api.jl is the raw generator output and stays
clean without a Julia toolchain in CI.

The 1-based-index and argument-type override tables below are DuckDB.jl policy,
carried from the original generator lineage.
"""

import argparse
import sys
from pathlib import Path

import capigen
from capigen.states import State, current_state, resolve_states
from capigen.tools import build_registry, chase, resolve_enum_values, version_key

# ---------------------------------------------------------------------------
# DuckDB.jl policy
# ---------------------------------------------------------------------------

# The Julia package that provides libduckdb, used when the env var is unset.
JLL_PACKAGE = "DuckDB_jll"
# Banner title of the generated files.
TITLE = "DuckDB Julia API"

JULIA_RESERVED_KEYWORDS = {
    "function",
    "if",
    "else",
    "while",
    "for",
    "try",
    "catch",
    "finally",
    "return",
    "break",
    "continue",
    "end",
    "begin",
    "quote",
    "let",
    "local",
    "global",
    "const",
    "do",
    "struct",
    "mutable",
    "abstract",
    "type",
    "module",
    "using",
    "import",
    "export",
    "public",
}

# C spelling -> Julia spelling, for the fundamental primitives whose names differ.
# Everything else (handles, callbacks, enums, structs, idx_t, ...) keeps its C name.
JULIA_BASE_TYPE_MAP = {
    "char": "Char",
    "int": "Int",
    "int8_t": "Int8",
    "int16_t": "Int16",
    "int32_t": "Int32",
    "int64_t": "Int64",
    "uint8_t": "UInt8",
    "uint16_t": "UInt16",
    "uint32_t": "UInt32",
    "uint64_t": "UInt64",
    "double": "Float64",
    "float": "Float32",
    "bool": "Bool",
    "void": "Cvoid",
    "size_t": "Csize_t",
}

# Argument names that (when integer-typed) denote a 0-based index and are exposed
# 1-based in Julia (call site subtracts 1).
INDEX_ARG_NAMES = {
    "index",
    "idx",
    "i",
    "row",
    "col",
    "column",
    "col_idx",
    "column_idx",
    "column_index",
    "row_idx",
    "row_index",
    "chunk_index",
}

# Integer Julia types that qualify an index argument for 1-based conversion. The
# "idxInt32" entry preserves an implicit string concatenation in the old generator.
INDEX_ARG_JULIA_TYPES = {
    "Int",
    "Int64",
    "UInt",
    "UInt64",
    "idx_t",
    "idxInt32",
    "UInt32",
    "Csize_t",
}

# Functions whose return value is a 0-based index exposed 1-based (call site adds 1).
AUTO_1BASE_RETURN_FUNCTIONS = {"duckdb_init_get_column_index"}

# Functions whose index-looking arguments must stay 0-based.
AUTO_1BASE_IGNORE_FUNCTIONS = {
    "duckdb_parameter_name",  # Parameter names start at 1
    "duckdb_param_type",  # Parameter types (like names) start at 1
    "duckdb_param_logical_type",
    "duckdb_bind_get_parameter",  # Would be a breaking API change
}

# Hand-written Julia argument-type tuples for the few functions whose faithful C
# mapping would be wrong or unsafe. Only the argument types are overridden; the
# return type is still derived from the spec.
OVERWRITE_ARG_TYPES = {
    # Must be Ptr{Cvoid} and not Ref, so a Vector can be passed as the blob buffer.
    "duckdb_free": ("Ptr{Cvoid}",),
    "duckdb_bind_blob": ("duckdb_prepared_statement", "idx_t", "Ptr{Cvoid}", "idx_t"),
    "duckdb_append_blob": ("duckdb_appender", "Ptr{Cvoid}", "idx_t"),
    # Must be Ptr{UInt8} instead of Cstring to allow a '\0' in the middle.
    "duckdb_vector_assign_string_element_len": ("duckdb_vector", "idx_t", "Ptr{UInt8}", "idx_t"),
}

MARGIN = 120
INDENT = "    "
# Continuation column for a binary "+" that wraps: aligned with the first operand.
PLUS_CONT = " " * len(f"{INDENT}return ")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _included(d: dict, states: dict[str, State]) -> bool:
    """Whether a construct is part of the emitted Julia surface."""
    name = current_state(d)
    state = states.get(name) if name else None
    if state is None:
        return True
    return state.visibility in ("always", "opt_out")


def _is_deprecated(func: dict) -> bool:
    """Deprecated by current state name or by the legacy field."""
    if current_state(func) == "deprecated":
        return True
    return bool(func.get("deprecated"))


# ---------------------------------------------------------------------------
# Type resolution (registry shared with capigen's C adapter)
# ---------------------------------------------------------------------------


def _julia_enum_name(prefix: str, spec_name: str) -> str:
    """The public Julia name of an enum: prefix + lowercased spec name (duckdb_type)."""
    return f"{prefix}{spec_name.lower()}"


class _Resolver:
    """Resolves a spec type reference to its Julia type."""

    def __init__(self, modules: list[dict], metadata: dict) -> None:
        prefix = metadata.get("prefix", "")
        self.primitives = {p["name"]: p["c_type"] for p in metadata["primitives"]}
        self.registry = build_registry(modules, metadata["suffixes"], prefix)
        # Enums use one uniform Julia name (const duckdb_<name>), overriding the C
        # registry which uppercases a leading-uppercase spec name (TYPE -> DUCKDB_TYPE).
        for mod in modules:
            for ename in mod.get("enums", {}):
                self.registry[ename] = _julia_enum_name(prefix, ename)
        # Aliases resolve to their underlying type: sel_t -> u32, type -> TYPE.
        self.aliases = {name: a["underlying"] for mod in modules for name, a in mod.get("aliases", {}).items()}

    def c_name(self, symbol: str) -> str:
        symbol = chase(self.aliases, symbol)
        if symbol in self.primitives:
            return self.primitives[symbol]
        if symbol in self.registry:
            return self.registry[symbol]
        raise ValueError(f"unknown type '{symbol}'")

    def julia_type(self, symbol: str, indirection: int, is_return: bool) -> str:
        """Map a resolved C base plus a pointer depth to a Julia type.

        A single pointer to char is Cstring; other pointers wrap in Ptr{} for
        returns and Ref{} for arguments; a fundamental C primitive maps through the
        base map; every other type keeps its C name.
        """
        base = self.c_name(symbol)

        def reduce(depth: int) -> str:
            if depth == 0:
                t = base
                if t.startswith("const "):
                    t = t[len("const ") :]
                if t.startswith("struct "):
                    t = t[len("struct ") :]  # C "struct Foo" -> Julia "Foo"
                if t in JULIA_BASE_TYPE_MAP:
                    return JULIA_BASE_TYPE_MAP[t]
                if " " in t:
                    raise ValueError(f"Unknown type: {t}")
                return t
            if depth == 1 and base in ("char", "const char"):
                return "Cstring"
            inner = reduce(depth - 1)
            return f"Ptr{{{inner}}}" if is_return else f"Ref{{{inner}}}"

        return reduce(indirection)


# ---------------------------------------------------------------------------
# Per-function inspection
# ---------------------------------------------------------------------------


def _arg_name(name: str) -> str:
    return f"_{name}" if name in JULIA_RESERVED_KEYWORDS else name


def _julia_tuple(items: list[str]) -> str:
    if len(items) == 0:
        return "()"
    if len(items) == 1:
        return f"({items[0]},)"
    return f"({', '.join(items)})"


class _Function:
    """A single spec function resolved into everything the templates need."""

    def __init__(self, name: str, func: dict, resolver: _Resolver) -> None:
        self.name = name
        self.func = func
        self.params = func["parameters"]  # ordered dict: pname -> param
        self.arg_names = [_arg_name(p) for p in self.params]

        if name in OVERWRITE_ARG_TYPES:
            self.arg_types = list(OVERWRITE_ARG_TYPES[name])
        else:
            self.arg_types = [
                resolver.julia_type(p["type"], p["indirection"], is_return=False) for p in self.params.values()
            ]

        self.return_type = resolver.julia_type(func["return_type"], func["return_pointer"], is_return=True)
        self.deprecated = _is_deprecated(func)
        self.index_args, self.index_return = self._index_info(resolver)

    def _index_info(self, resolver: _Resolver) -> tuple[list[bool], bool]:
        if self.name in AUTO_1BASE_IGNORE_FUNCTIONS:
            return [False] * len(self.params), False
        index_args = []
        for pname, p in self.params.items():
            if pname not in INDEX_ARG_NAMES:
                index_args.append(False)
                continue
            jt = resolver.julia_type(p["type"], p["indirection"], is_return=False)
            index_args.append(jt in INDEX_ARG_JULIA_TYPES)
        return index_args, self.name in AUTO_1BASE_RETURN_FUNCTIONS


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_docstring(fn: _Function) -> list[str]:
    func = fn.func
    description = (func.get("description") or "").strip().replace('"', '\\"')

    arg_comments = []
    for name, param, t, is_index in zip(fn.arg_names, fn.params.values(), fn.arg_types, fn.index_args):
        param_comment = param.get("description") or ""
        if is_index:
            parts = [f"`{name}`:", f"`{t}`", "(1-based index)", param_comment]
        else:
            parts = [f"`{name}`:", f"`{t}`", param_comment]
        arg_comments.append(" ".join(p for p in parts if p))

    return_type = "Nothing" if fn.return_type == "Cvoid" else fn.return_type
    return_parts = [f"`{return_type}`", func.get("return_description") or ""]
    if fn.index_return:
        return_parts.append("(1-based index)")
    return_comment = " ".join(p for p in return_parts if p)

    lines = ['"""', f"{INDENT}{fn.name}({', '.join(fn.arg_names)})", "", description, "", "# Arguments"]
    lines += [f"- {c}" for c in arg_comments]
    lines += ["", f"Returns: {return_comment}", '"""']
    return lines


def _depwarn_notice(func: dict) -> str:
    description = func.get("description") or ""
    if not description.startswith("**DEPRECATION NOTICE**:"):
        description = f"**DEPRECATION NOTICE**: {description}"
    notice = description.split("\n")[0]
    return notice.replace("\n", " ").replace('"', '\\"').strip()


def _render_depwarn(fn: _Function) -> list[str]:
    notice = _depwarn_notice(fn.func)
    single = f'{INDENT}Base.depwarn("{notice}", :{fn.name})'
    if len(single) <= MARGIN:
        return [single]
    return [
        f"{INDENT}Base.depwarn(",
        f'{INDENT}{INDENT}"{notice}",',
        f"{INDENT}{INDENT}:{fn.name}",
        f"{INDENT})",
    ]


def _wrap_type_tuple(fn: _Function) -> list[str]:
    """Emit the ccall argument-type tuple, breaking it across lines if it is long."""
    tuple_str = _julia_tuple(fn.arg_types)
    line = f"{INDENT}{INDENT}{tuple_str},"
    if len(line) <= MARGIN:
        return [line]
    inner = INDENT * 3
    lines = [f"{INDENT}{INDENT}("]
    lines += [f"{inner}{t}," for t in fn.arg_types[:-1]]
    lines.append(f"{inner}{fn.arg_types[-1]}")
    lines.append(f"{INDENT}{INDENT}),")
    return lines


def _render_ccall(fn: _Function, lib: str) -> list[str]:
    sym = f"(:{fn.name}, {lib})"
    tuple_str = _julia_tuple(fn.arg_types)
    call_args = [f"{name} - 1" if is_index else name for name, is_index in zip(fn.arg_names, fn.index_args)]
    suffix = " + 1" if fn.index_return else ""

    head_parts = [sym, fn.return_type, tuple_str, *call_args]
    single = f"{INDENT}return ccall({', '.join(head_parts)})"

    if fn.index_return:
        # Wrap at the binary "+" first, matching JuliaFormatter.
        if len(single + suffix) <= MARGIN:
            return [single + suffix]
        if len(single + " +") <= MARGIN:
            return [single + " +", f"{PLUS_CONT}1"]
        # ccall itself must wrap too; append the "+ 1" to the closing line.
        return _wrap_ccall(fn, sym, call_args, close_suffix=suffix)

    if len(single) <= MARGIN:
        return [single]
    return _wrap_ccall(fn, sym, call_args, close_suffix="")


def _wrap_ccall(fn: _Function, sym: str, call_args: list[str], close_suffix: str) -> list[str]:
    lines = [f"{INDENT}return ccall("]
    lines.append(f"{INDENT}{INDENT}{sym},")
    lines.append(f"{INDENT}{INDENT}{fn.return_type},")
    lines += _wrap_type_tuple(fn)
    for arg in call_args[:-1]:
        lines.append(f"{INDENT}{INDENT}{arg},")
    if call_args:
        lines.append(f"{INDENT}{INDENT}{call_args[-1]}")
    lines.append(f"{INDENT}){close_suffix}")
    return lines


def _render_function(fn: _Function, lib: str) -> list[str]:
    lines = _render_docstring(fn)
    lines.append(f"function {fn.name}({', '.join(fn.arg_names)})")
    if fn.deprecated:
        lines += _render_depwarn(fn)
    lines += _render_ccall(fn, lib)
    lines.append("end")
    return lines


def _render_group_start(module_name: str) -> list[str]:
    title = " ".join(word.capitalize() for word in module_name.replace("_", " ").strip().split(" "))
    rule = f"# {'-' * 80}"
    return [rule, f"# {title}", rule]


def _banner(invocation: str | None) -> list[str]:
    lines = [
        "# !!!!!!!!!!!!",
        "# WARNING: this file is autogenerated by scripts/julia_adapter.py, manual changes will be overwritten",
        "# Regenerate with ./update_api.sh",
    ]
    if invocation:
        lines.append(f"# Re-run: {invocation}")
    lines.append("# !!!!!!!!!!!!")
    return lines


def _names(prefix: str) -> dict:
    """Generated identifier names, derived from the spec prefix."""
    uprefix = prefix.upper()
    return {
        "lib": f"lib{prefix.rstrip('_')}",
        "env_var": f"JULIA_{uprefix}LIBRARY",
        "version_const": f"{uprefix}API_VERSION",
    }


def _render_header(version: str, invocation: str | None, names: dict) -> list[str]:
    if version and version[0] == "v":
        version = version[1:]
    return [
        "",
        "###############################################################################",
        "#",
        f"# {TITLE}",
        "#",
        *_banner(invocation),
        "#",
        "###############################################################################",
        "",
        "using Base.Libc",
        "",
        f'if "{names["env_var"]}" in keys(ENV)',
        f'    {names["lib"]} = ENV["{names["env_var"]}"]',
        "else",
        f"    using {JLL_PACKAGE}",
        "end",
        "",
        f'{names["version_const"]} = v"{version}"',
        "",
    ]


def _render_footer(invocation: str | None) -> list[str]:
    # Two leading blanks: with the three the group loop already emitted this makes
    # five. No trailing blank: the file ends on the banner.
    return ["", "", *_banner(invocation)]


# ---------------------------------------------------------------------------
# Generated type layer (ctypes_generated.jl)
# ---------------------------------------------------------------------------


def _pointer_type_names(modules: list[dict], resolver: _Resolver, states: dict[str, State]) -> list[str]:
    """Every included handle and callback, as the Julia name api.jl references."""
    names = set()
    for mod in modules:
        for handle, d in mod.get("handles", {}).items():
            if _included(d, states):
                names.add(resolver.julia_type(handle, 0, is_return=False))
        for callback, d in mod.get("callbacks", {}).items():
            if _included(d, states):
                names.add(resolver.julia_type(callback, 0, is_return=False))
    return sorted(names)


def _render_enum(spec_name: str, enum: dict, prefix: str) -> list[str]:
    # One uniform rule for all enums: public const duckdb_<name>, inner @enum
    # DUCKDB_<NAME>_. Variant names/values come straight from the spec.
    alias = _julia_enum_name(prefix, spec_name)
    inner = f"{alias.upper()}_"
    lines = [f"@enum {inner}::Cint begin"]
    lines += [f"{INDENT}{prefix.upper()}{vname} = {value}" for vname, value in resolve_enum_values(enum)]
    lines.append("end")
    lines.append(f"const {alias} = {inner}")
    return lines


def _render_ctypes_generated(
    modules: list[dict],
    prefix: str,
    resolver: _Resolver,
    states: dict[str, State],
    invocation: str | None,
) -> list[str]:
    out = [
        "",
        "###############################################################################",
        "#",
        f"# {TITLE} - generated C type layer (handles, callbacks, enums)",
        "#",
        *_banner(invocation),
        "# It is a sibling of api.jl, written together with it, and is",
        "# included before the hand-written ctypes.jl.",
        "#",
        "###############################################################################",
        "",
    ]
    out += [f"const {name} = Ptr{{Cvoid}}" for name in _pointer_type_names(modules, resolver, states)]
    out.append("")
    blocks = [
        _render_enum(ename, enum, prefix)
        for mod in sorted(modules, key=lambda m: m["module"])
        for ename, enum in mod.get("enums", {}).items()
        if _included(enum, states)
    ]
    for i, block in enumerate(blocks):
        out += block
        if i != len(blocks) - 1:
            out.append("")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _latest_version(metadata: dict) -> str:
    """The spec describes the API as of the latest entry in `versions`."""
    versions = metadata.get("versions") or []
    if not versions:
        raise ValueError("the julia generator requires a non-empty 'versions' list")
    return max(versions, key=version_key)


def generate(
    modules: list[dict],
    metadata: dict,
    output_path: Path,
    invocation: str | None = None,
) -> None:
    """Render api.jl and its sibling ctypes_generated.jl from the spec."""
    resolver = _Resolver(modules, metadata)
    states = resolve_states(metadata)
    prefix = metadata.get("prefix", "")
    version = _latest_version(metadata)
    names = _names(prefix)

    out: list[str] = []
    out += _render_header(version, invocation, names)
    out.append("")  # blank line before the first group

    for mod in sorted(modules, key=lambda m: m["module"]):
        functions = {fname: func for fname, func in mod.get("functions", {}).items() if _included(func, states)}
        if not functions:
            continue  # type-only module: nothing to emit
        out += _render_group_start(mod["module"])
        out.append("")
        for fname, func in functions.items():
            fn = _Function(f"{prefix}{fname}", func, resolver)
            out += _render_function(fn, names["lib"])
            out.append("")
        out.append("")
        out.append("")

    out += _render_footer(invocation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out) + "\n")

    # Sibling type layer, written together with api.jl.
    types = _render_ctypes_generated(modules, prefix, resolver, states, invocation)
    types_path = output_path.parent / "ctypes_generated.jl"
    types_path.write_text("\n".join(types) + "\n")

    n_functions = sum(1 for m in modules for func in m.get("functions", {}).values() if _included(func, states))
    n_types = len(_pointer_type_names(modules, resolver, states)) + sum(
        1 for m in modules for enum in m.get("enums", {}).values() if _included(enum, states)
    )
    print(f"Generated {output_path} ({n_functions} functions) and {types_path} ({n_types} types)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the Julia layer from the C API spec")
    parser.add_argument("--spec-dir", required=True, help="Directory with metadata.yaml and module YAMLs")
    parser.add_argument("--output", "-o", required=True, help="Path for api.jl (ctypes_generated.jl is its sibling)")
    args = parser.parse_args()

    try:
        spec = capigen.load(args.spec_dir)
    except capigen.SpecError as e:
        print(f"Spec validation failed:\n{e}", file=sys.stderr)
        sys.exit(1)
    generate(spec.modules, spec.metadata, Path(args.output), invocation=None)


if __name__ == "__main__":
    main()
