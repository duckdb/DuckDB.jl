"""capigen external adapter: regenerate the Julia layer from api_spec/v1.

One generate() call writes two sibling files:
  - api.jl (at output_path): the ccall function wrappers.
  - ctypes_generated.jl (sibling): the mechanical type layer api.jl needs, i.e.
    every spec handle and callback as `const duckdb_<name> = Ptr{Cvoid}` and every
    spec enum as an `@enum`. It is included before the hand-written ctypes.jl.

Behavior-preserving port of the bespoke Julia generator it replaces. It reads the
structured capigen spec instead of the legacy JSON, so C types arrive as
(type, indirection, const) and resolve through a name registry rather than being
re-parsed from C type strings.

Three deliberate differences from the old generator:
  1. Input is the capigen spec (modules + metadata), not the legacy JSON.
  2. Order is deterministic: modules sorted by name ascending, functions within a
     module in YAML-declared (insertion) order. Type-only modules (no functions)
     emit nothing. The old JULIA_API_ORIGINAL_ORDER pinning is retired.
  3. DUCKDB_API_VERSION is derived from the spec (options.extension.api_version),
     the same source the extension header uses.

Deprecation follows the spec, mirroring the C header: a function is deprecated
when its status begins with "deprecated" or it carries a deprecated flag. This is
narrower than the legacy JSON, which also deprecated whole groups.

The ccall and Base.depwarn are emitted pre-wrapped in JuliaFormatter's canonical
form (margin 120), so the committed api.jl is the raw adapter output and stays
clean without a Julia toolchain in CI.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Verbatim tables from the old generator
# ---------------------------------------------------------------------------

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
# Everything else (handles, callbacks, enums, structs, idx_t, duckdb_string_t, ...)
# keeps its C name, so the map holds no DuckDB-specific entries.
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

_DEFAULT_INVOCATION = "capigen julia_adapter --spec-dir api_spec/v1 -o tools/juliapkg/src/api.jl"


# ---------------------------------------------------------------------------
# Type registry and resolution (mirrors capigen.adapters.c.resolve)
# ---------------------------------------------------------------------------


def _apply_prefix(prefix: str, name: str) -> str:
    if name and name[0].isupper():
        return f"{prefix.upper()}{name}"
    return f"{prefix}{name}"


def _build_registry(modules: list[dict], suffixes: dict[str, str], prefix: str) -> dict[str, str]:
    """Map every declared spec name (and its prefixed form) to its C type name."""
    registry: dict[str, str] = {}
    for mod in modules:
        for name in mod.get("handles", {}):
            canonical = f"{_apply_prefix(prefix, name)}{suffixes['handles']}"
            registry[name] = canonical
            registry[canonical] = canonical
        for name in mod.get("callbacks", {}):
            canonical = f"{_apply_prefix(prefix, name)}{suffixes['callbacks']}"
            registry[name] = canonical
            registry[canonical] = canonical
        for name, a in mod.get("aliases", {}).items():
            if a.get("qualified"):
                registry[name] = name
            else:
                canonical = f"{_apply_prefix(prefix, name)}{suffixes['aliases']}"
                registry[name] = canonical
                registry[canonical] = canonical
        for name, s in mod.get("structs", {}).items():
            prefixed = _apply_prefix(prefix, name)
            alias = f"{prefixed}{suffixes['aliases']}" if s.get("pointer_alias") else prefixed
            registry[name] = alias
            registry[prefixed] = alias
            if alias != prefixed:
                registry[alias] = alias
        for name in mod.get("enums", {}):
            prefixed = _apply_prefix(prefix, name)
            registry[name] = prefixed
            registry[prefixed] = prefixed
    return registry


def _julia_enum_name(prefix: str, spec_name: str) -> str:
    """The public Julia name of an enum: prefix + lowercased spec name (duckdb_type)."""
    return f"{prefix}{spec_name.lower()}"


class _Resolver:
    """Resolves a spec type reference to its Julia type."""

    def __init__(self, modules: list[dict], metadata: dict) -> None:
        prefix = metadata.get("prefix", "")
        self.primitives = {p["name"]: p["c_type"] for p in metadata["primitives"]}
        self.registry = _build_registry(modules, metadata["suffixes"], prefix)
        # Enums use one uniform Julia name (const duckdb_<name>), overriding the C
        # registry which uppercases a leading-uppercase spec name (TYPE -> DUCKDB_TYPE).
        for mod in modules:
            for ename in mod.get("enums", {}):
                self.registry[ename] = _julia_enum_name(prefix, ename)
        # Aliases resolve to their underlying type: sel_t -> u32, type -> TYPE.
        self.aliases = {name: a["underlying"] for mod in modules for name, a in mod.get("aliases", {}).items()}

    def _follow(self, symbol: str) -> str:
        seen: set[str] = set()
        while symbol in self.aliases and symbol not in seen:
            seen.add(symbol)
            symbol = self.aliases[symbol]
        return symbol

    def c_name(self, symbol: str) -> str:
        symbol = self._follow(symbol)
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


def _is_deprecated(func: dict) -> bool:
    status = func.get("status", [])
    if status and status[0][0] == "deprecated":
        return bool(status[0][1])
    return bool(func.get("deprecated"))


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
    for name, (pname, param), t, is_index in zip(fn.arg_names, fn.params.items(), fn.arg_types, fn.index_args):
        param_comment = param.get("description") or ""
        if is_index:
            parts = [f"`{name}`:", f"`{t}`", "(1-based index)", param_comment]
        else:
            parts = [f"`{name}`:", f"`{t}`", param_comment]
        arg_comments.append(" ".join(parts))

    return_type = "Nothing" if fn.return_type == "Cvoid" else fn.return_type
    return_parts = [f"`{return_type}`", func.get("return_description") or ""]
    if fn.index_return:
        return_parts.append("(1-based index)")
    return_comment = " ".join(return_parts)

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


def _render_ccall(fn: _Function) -> list[str]:
    sym = f"(:{fn.name}, libduckdb)"
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


def _render_function(fn: _Function) -> list[str]:
    lines = _render_docstring(fn)
    lines.append(f"function {fn.name}({', '.join(fn.arg_names)})")
    if fn.deprecated:
        lines += _render_depwarn(fn)
    lines += _render_ccall(fn)
    lines.append("end")
    return lines


def _render_group_start(module_name: str) -> list[str]:
    title = " ".join(word.capitalize() for word in module_name.replace("_", " ").strip().split(" "))
    rule = f"# {'-' * 80}"
    return [rule, f"# {title}", rule]


def _render_header(version: str, invocation: str) -> list[str]:
    if version and version[0] == "v":
        version = version[1:]
    return [
        "",
        "###############################################################################",
        "# ",
        "# DuckDB Julia API",
        "# ",
        "# !!!!!!!!!!!!",
        "# WARNING: this file is autogenerated from api_spec/v1 by the capigen julia_adapter, manual changes will be overwritten",
        f"# Regenerate: {invocation}",
        "# !!!!!!!!!!!!",
        "#",
        "###############################################################################",
        "",
        "using Base.Libc",
        "",
        'if "JULIA_DUCKDB_LIBRARY" in keys(ENV)',
        '    libduckdb = ENV["JULIA_DUCKDB_LIBRARY"]',
        "else",
        "    using DuckDB_jll",
        "end",
        "",
        f'DUCKDB_API_VERSION = v"{version}"',
        "",
    ]


def _render_footer() -> list[str]:
    # Two leading blanks: with the three the group loop already emitted this makes
    # five, matching the old output. No trailing blank: the file ends on the banner.
    return [
        "",
        "",
        "# !!!!!!!!!!!!",
        "# WARNING: this file is autogenerated from api_spec/v1 by the capigen julia_adapter, manual changes will be overwritten",
        "# !!!!!!!!!!!!",
    ]


# ---------------------------------------------------------------------------
# Generated type layer (ctypes_generated.jl)
# ---------------------------------------------------------------------------


def _pointer_type_names(modules: list[dict], resolver: _Resolver) -> list[str]:
    """Every handle and callback, as the Julia name api.jl references, sorted."""
    names = set()
    for mod in modules:
        for handle in mod.get("handles", {}):
            names.add(resolver.julia_type(handle, 0, is_return=False))
        for callback in mod.get("callbacks", {}):
            names.add(resolver.julia_type(callback, 0, is_return=False))
    return sorted(names)


def _resolve_enum_values(enum: dict) -> list[tuple[str, int]]:
    """Auto-number from 0, reset on an explicit value, prefix DUCKDB_ (mirrors the C adapter)."""
    values = []
    current = 0
    for vname, entry in enum["values"].items():
        if entry.get("value") is not None:
            current = entry["value"]
        values.append((f"DUCKDB_{vname}", current))
        current += 1
    return values


def _render_enum(spec_name: str, enum: dict, prefix: str) -> list[str]:
    # One uniform rule for all enums: public const duckdb_<name>, inner @enum
    # DUCKDB_<NAME>_. Variant names/values come straight from the spec.
    alias = _julia_enum_name(prefix, spec_name)
    inner = f"{alias.upper()}_"
    lines = [f"@enum {inner}::Cint begin"]
    lines += [f"{INDENT}{vname} = {value}" for vname, value in _resolve_enum_values(enum)]
    lines.append("end")
    lines.append(f"const {alias} = {inner}")
    return lines


def _render_ctypes_generated(modules: list[dict], prefix: str, resolver: _Resolver, invocation: str) -> list[str]:
    out = [
        "",
        "###############################################################################",
        "# ",
        "# DuckDB Julia API - generated C type layer (handles, callbacks, enums)",
        "# ",
        "# !!!!!!!!!!!!",
        "# WARNING: this file is autogenerated from api_spec/v1 by the capigen julia_adapter, manual changes will be overwritten",
        f"# Regenerate: {invocation}",
        "# It is a sibling of api.jl, written by the same capigen invocation, and is",
        "# included before the hand-written ctypes.jl.",
        "# !!!!!!!!!!!!",
        "#",
        "###############################################################################",
        "",
    ]
    out += [f"const {name} = Ptr{{Cvoid}}" for name in _pointer_type_names(modules, resolver)]
    out.append("")
    blocks = [
        _render_enum(ename, enum, prefix)
        for mod in sorted(modules, key=lambda m: m["module"])
        for ename, enum in mod.get("enums", {}).items()
    ]
    for i, block in enumerate(blocks):
        out += block
        if i != len(blocks) - 1:
            out.append("")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(
    modules: list[dict],
    metadata: dict,
    output_path: Path,
    invocation: str | None = None,
) -> None:
    """Render tools/juliapkg/src/api.jl from the capigen v1 spec."""
    resolver = _Resolver(modules, metadata)
    prefix = metadata.get("prefix", "")
    version = metadata["options"]["extension"]["api_version"]

    out: list[str] = []
    out += _render_header(version, invocation or _DEFAULT_INVOCATION)
    out.append("")  # blank line before the first group (matches the old generator)

    for mod in sorted(modules, key=lambda m: m["module"]):
        functions = mod.get("functions", {})
        if not functions:
            continue  # type-only module: nothing to emit
        out += _render_group_start(mod["module"])
        out.append("")
        for fname, func in functions.items():
            fn = _Function(f"{prefix}{fname}", func, resolver)
            out += _render_function(fn)
            out.append("")
        out.append("")
        out.append("")

    out += _render_footer()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out) + "\n")

    # Sibling type layer, written by the same invocation.
    types = _render_ctypes_generated(modules, prefix, resolver, invocation or _DEFAULT_INVOCATION)
    types_path = output_path.parent / "ctypes_generated.jl"
    types_path.write_text("\n".join(types) + "\n")

    n_functions = sum(len(m.get("functions", {})) for m in modules)
    n_types = len(_pointer_type_names(modules, resolver)) + sum(len(m.get("enums", {})) for m in modules)
    print(f"Generated {output_path} ({n_functions} functions) and {types_path} ({n_types} types)")
