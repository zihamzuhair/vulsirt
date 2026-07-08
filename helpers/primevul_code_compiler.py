#!/usr/bin/env python3
"""PrimeVul code-to-LLVM compiler pipeline.

This is the main LLVM-friendly PrimeVul processing script. It reads isolated
C functions, builds a minimal translation unit around each function,
compiles it with Clang, extracts exactly one target LLVM function, and writes
JSONL records for model training.

Successful records keep the raw PrimeVul function in ``source_code``. Repairs
are used only inside the temporary translation unit needed to produce LLVM IR.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence


PREPROCESSOR_VERSION = "7.0.0"


# ---------------------------------------------------------------------------
# Static run configuration
# ---------------------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Edit these variables for each run. Then execute:
#   python compiler.py --dataset primevul
INPUT_JSONL = DATA_DIR / "processed" / "primevul_dataset.jsonl"
OUTPUT_JSONL = DATA_DIR / "processed" / "primevul_dataset_with_llvm.jsonl"
WORKERS = 4

CLANG = "clang"
LLVM_EXTRACT = "llvm-extract"
C_STANDARD = "gnu11"
TIMEOUT_SECONDS = 20
MAX_REPAIR_ROUNDS = 8
MAX_SOURCE_BYTES = 1_000_000
KEEP_COMPILER_COMMENTS = False
VALIDATE_LLVM_OBJECT = True
MAX_TARGET_IR_FUNCTIONS = 1
ALLOW_LOW_CONFIDENCE_STUBS = False
MAX_STUB_COUNT = 12
ALLOW_SEMANTIC_MACRO_FALLBACKS = True
EXTRA_C_FLAGS: tuple[str, ...] = ()


@dataclass
class CompileAttempt:
    success: bool
    language: str
    llvm_ir: str = ""
    compilable_source: str = ""
    target_symbol: str = ""
    stderr: str = ""
    rounds: int = 0
    generated_stubs: list[str] = field(default_factory=list)
    failure_status: str = ""
    repaired_return_type: str = ""


@dataclass(frozen=True)
class TypeInfo:
    base: str
    pointer_depth: int = 0
    reference: bool = False


@dataclass
class FunctionSignature:
    qualified_name: str = ""
    function_name: str = ""
    owner: str = ""
    return_prefix: str = ""
    params_text: str = ""
    trailing_qualifiers: str = ""
    initializer_list: str = ""
    body_start: int = -1
    declaration_start: int = 0
    is_constructor: bool = False
    is_destructor: bool = False


@dataclass
class FieldSpec:
    kind: str  # scalar, pointer, value, function, array, unresolved_array, typed
    target: str = ""


@dataclass
class Analysis:
    source: str
    masked: str
    signature: FunctionSignature
    variables: dict[str, TypeInfo] = field(default_factory=dict)
    root_first_operator: dict[str, str] = field(default_factory=dict)
    member_chains: list[tuple[str, list[tuple[str, str]], str]] = field(default_factory=list)
    defined_types: set[str] = field(default_factory=set)
    defined_identifiers: set[str] = field(default_factory=set)


@dataclass
class RepairState:
    scalar_types: set[str] = field(default_factory=set)
    opaque_types: set[str] = field(default_factory=set)
    pointer_aliases: dict[str, str] = field(default_factory=dict)
    structs: dict[str, dict[str, FieldSpec]] = field(default_factory=dict)
    macro_constants: set[str] = field(default_factory=set)
    empty_macros: set[str] = field(default_factory=set)
    semantic_macro_fallbacks: set[str] = field(default_factory=set)
    functions: dict[str, str] = field(default_factory=dict)
    variables: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)

    def note(self, value: str) -> None:
        if value not in self.notes:
            self.notes.append(value)

    def mark_uncertain(self, value: str) -> None:
        if value not in self.uncertain:
            self.uncertain.append(value)
        self.note(f"uncertain:{value}")


BUILTIN_TYPES = {
    "void", "char", "signed", "unsigned", "short", "int", "long", "float",
    "double", "bool", "_Bool", "size_t", "ssize_t", "ptrdiff_t", "intptr_t",
    "uintptr_t", "int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t",
    "uint32_t", "int64_t", "uint64_t", "wchar_t", "FILE", "time_t", "off_t",
    "va_list", "intmax_t", "uintmax_t", "mode_t", "uid_t", "gid_t",
    "socklen_t", "loff_t", "dev_t", "ino_t", "nlink_t", "blksize_t",
    "blkcnt_t", "std::string", "std::size_t", "std::nullptr_t", "auto",
}

TYPE_WORDS = {
    "const", "volatile", "restrict", "__restrict", "__restrict__", "static",
    "extern", "register", "inline", "__inline", "__inline__", "signed",
    "unsigned", "short", "long", "struct", "union", "enum", "class",
    "typename", "mutable", "constexpr", "consteval", "constinit", "virtual",
}

CONTROL_WORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "alignof", "typeof",
    "__typeof__", "do", "case", "new", "delete", "throw", "catch", "static_cast",
    "reinterpret_cast", "const_cast", "dynamic_cast", "decltype", "noexcept",
}

KNOWN_CALLS = CONTROL_WORDS | {
    "assert", "offsetof", "va_start", "va_end", "va_arg", "va_copy",
}

SAFE_FUNCTION_MACROS = {
    "likely",
    "unlikely",
    "MIN",
    "MAX",
    "ARRAY_SIZE",
}

UNSAFE_MACRO_PATTERNS = (
    "FOREACH",
    "FOR_EACH",
    "CONTAINEROF",
    "CONTAINER_OF",
    "LIST_",
    "HLIST_",
    "RING_",
    "TAILQ_",
    "STAILQ_",
    "RB_",
    "HASH_",
    "VIRGL_OBJ_",
)

UNSAFE_SEMANTIC_MACROS = {
    "fz_try",
    "fz_always",
    "fz_catch",
    "container_of",
    "list_for_each",
    "list_for_each_entry",
    "hlist_for_each_entry",
    "do_div",
    "INIT_LIST_HEAD",
    "INIT_HLIST_HEAD",
    "list_add",
    "list_add_tail",
    "list_del",
    "list_del_init",
}

# Syntax-only fallbacks for common project macros. These are deliberately
# marked in generated_stubs/ir_status. They make the temporary translation unit
# compileable; they are not treated as clean semantic reconstructions.
SEMANTIC_MACRO_FALLBACKS = {
    "fz_try": "#ifndef fz_try\n#define fz_try(ctx) if (1)\n#endif",
    "fz_always": "#ifndef fz_always\n#define fz_always(ctx) if (1)\n#endif",
    "fz_catch": "#ifndef fz_catch\n#define fz_catch(ctx) if (0)\n#endif",
    "container_of": "#ifndef container_of\n#define container_of(ptr, type, member) ((type *)0)\n#endif",
    "do_div": "#ifndef do_div\n#define do_div(n, base) ((n) % (base))\n#endif",
    "list_for_each": "#ifndef list_for_each\n#define list_for_each(pos, head) for (; 0; )\n#endif",
    "list_for_each_entry": "#ifndef list_for_each_entry\n#define list_for_each_entry(pos, head, member) for (; 0; )\n#endif",
    "hlist_for_each_entry": "#ifndef hlist_for_each_entry\n#define hlist_for_each_entry(pos, head, member) for (; 0; )\n#endif",
    "INIT_LIST_HEAD": "#ifndef INIT_LIST_HEAD\n#define INIT_LIST_HEAD(ptr) do { (void)(ptr); } while (0)\n#endif",
    "INIT_HLIST_HEAD": "#ifndef INIT_HLIST_HEAD\n#define INIT_HLIST_HEAD(ptr) do { (void)(ptr); } while (0)\n#endif",
    "list_add": "#ifndef list_add\n#define list_add(new_entry, head) do { (void)(new_entry); (void)(head); } while (0)\n#endif",
    "list_add_tail": "#ifndef list_add_tail\n#define list_add_tail(new_entry, head) do { (void)(new_entry); (void)(head); } while (0)\n#endif",
    "list_del": "#ifndef list_del\n#define list_del(entry) do { (void)(entry); } while (0)\n#endif",
    "list_del_init": "#ifndef list_del_init\n#define list_del_init(entry) do { (void)(entry); } while (0)\n#endif",
}

SOURCE_EXTENSIONS_C = {".c", ".h"}
DISCARD_SOURCE_EXTENSIONS = {".cc", ".cp", ".cpp", ".cxx", ".c++", ".cppm", ".ixx", ".hpp", ".hh", ".hxx"}

DIAGNOSTIC_PATTERNS = {
    "unknown_type": re.compile(r"(?:unknown type name|unknown type) ['‘]([^'’]+)['’]"),
    "undeclared": re.compile(r"use of undeclared identifier ['‘]([^'’]+)['’]"),
    "undeclared_function": re.compile(r"(?:call to undeclared function|implicit declaration of function) ['‘]([^'’]+)['’]"),
    "no_member": re.compile(r"no member named ['‘]([^'’]+)['’] in ['‘]([^'’]+)['’]"),
    "incomplete": re.compile(r"incomplete (?:definition of )?type ['‘](?:struct |class |union )?([^'’]+)['’]"),
    "unknown_namespace": re.compile(r"use of undeclared identifier ['‘]([^'’]+)['’]"),
    "must_tag": re.compile(r"must use ['‘](struct|union|enum)['’] tag to refer to type ['‘]([^'’]+)['’]"),
    "redefinition": re.compile(r"redefinition of ['‘]([^'’]+)['’]"),
}


C_PRELUDE = r"""
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>

#ifndef NULL
#define NULL ((void *)0)
#endif
#ifndef likely
#define likely(x) (x)
#endif
#ifndef unlikely
#define unlikely(x) (x)
#endif
#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif
#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif
#ifndef _PUBLIC_
#define _PUBLIC_
#endif
"""

def build_conditional_prelude(source: str, language: str) -> str:
    """Include only headers that the isolated function actually needs.

    This avoids emitting hundreds of unrelated Windows CRT inline functions.
    Semantic project macros are deliberately not invented here.
    """
    masked = mask_comments_and_literals(source)
    headers: list[str] = []

    def uses(pattern: str) -> bool:
        return bool(re.search(pattern, masked))

    common_rules = (
        (r"\b(?:malloc|calloc|realloc|free|abort|exit|qsort|bsearch|"
         r"atoi|atol|strtol|strtoul|strtoll|strtoull|abs|labs|mbtowc)\s*\(", "stdlib.h"),
        (r"\b(?:memcpy|memmove|memset|memcmp|strlen|strcmp|strncmp|"
         r"strcpy|strncpy|strcat|strncat|strchr|strrchr|strstr|strtok|"
         r"strspn|strcspn|strerror|strdup)\s*\(", "string.h"),
        (r"\b(?:FILE|printf|fprintf|sprintf|snprintf|vsprintf|vsnprintf|"
         r"scanf|sscanf|fscanf|fopen|fclose|fread|fwrite|fseek|ftell|"
         r"fflush|putc|fputc|fputs|getc|fgetc|fgets)\b", "stdio.h"),
        (r"\b(?:EINVAL|ENOMEM|ENOENT|EIO|ERANGE|errno)\b", "errno.h"),
        (r"\bassert\s*\(", "assert.h"),
        (r"\b(?:sin|cos|tan|sqrt|pow|fabs|floor|ceil|isnan|isinf|"
         r"ldexp|frexp|fmod)\s*\(", "math.h"),
        (r"\b(?:time_t|clock_t|time|clock|difftime|localtime|gmtime|mktime)\b", "time.h"),
        (r"\b(?:wchar_t|mbtowc|mbrtowc|wcstombs|wcslen|wcscmp)\b", "wchar.h"),
        (r"\b(?:iswprint|iswalpha|iswdigit|iswspace|iswalnum)\s*\(", "wctype.h"),
        (r"\b(?:isalnum|isalpha|isblank|iscntrl|isdigit|isgraph|islower|"
         r"isprint|ispunct|isspace|isupper|isxdigit|tolower|toupper)\s*\(", "ctype.h"),
        (r"\b(?:off_t|ssize_t|pid_t|mode_t|uid_t|gid_t|dev_t|ino_t|"
         r"nlink_t|blksize_t|blkcnt_t)\b", "sys/types.h"),
        (r"\b(?:INT_MAX|INT_MIN|CHAR_BIT|SCHAR_MAX|UCHAR_MAX|SHRT_MAX|"
         r"USHRT_MAX|UINT_MAX|LONG_MAX|ULONG_MAX|LLONG_MAX|ULLONG_MAX)\b", "limits.h"),
        (r"\b(?:setlocale|localeconv|LC_ALL|LC_CTYPE|LC_NUMERIC)\b", "locale.h"),
        (r"\b(?:intmax_t|uintmax_t|imaxabs|strtoimax|strtoumax)\b", "inttypes.h"),
    )
    for pattern, header in common_rules:
        if uses(pattern) and header not in headers:
            headers.append(header)

    base = C_PRELUDE
    include_lines = [f"#include <{header}>" for header in headers]

    return base + "\n" + "\n".join(include_lines) + "\n"



# ---------------------------------------------------------------------------
# Lexical helpers
# ---------------------------------------------------------------------------


def normalize_source(source: Any) -> str:
    """Normalize encoding/newlines without changing the function's code."""
    if not isinstance(source, str):
        return ""
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    source = source.replace("\x00", "").lstrip("\ufeff")
    return source.strip()


def sanitize_for_compilation(source: str) -> str:
    """Block file-reading directives only in the temporary compiler input."""
    return re.sub(
        r"^[ \t]*#[ \t]*(?:include|include_next|import|embed)[^\n]*$",
        lambda m: "/* PrimeVul preprocessor removed: " + m.group(0).replace("*/", "* /" ) + " */",
        source,
        flags=re.MULTILINE,
    )


def mask_comments_and_literals(source: str) -> str:
    """Replace comments/string contents with spaces while preserving positions."""
    out = list(source)
    i = 0
    n = len(source)
    state = "code"
    quote = ""
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "line_comment"
                continue
            if ch == "/" and nxt == "*":
                out[i] = out[i + 1] = " "
                i += 2
                state = "block_comment"
                continue
            if ch in {'"', "'"}:
                quote = ch
                out[i] = " "
                i += 1
                state = "literal"
                continue
        elif state == "line_comment":
            if ch == "\n":
                state = "code"
            else:
                out[i] = " "
            i += 1
            continue
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "code"
                continue
            if ch != "\n":
                out[i] = " "
            i += 1
            continue
        elif state == "literal":
            if ch == "\\":
                out[i] = " "
                if i + 1 < n:
                    if source[i + 1] != "\n":
                        out[i + 1] = " "
                    i += 2
                    continue
            if ch == quote:
                out[i] = " "
                i += 1
                state = "code"
                continue
            if ch != "\n":
                out[i] = " "
            i += 1
            continue
        i += 1
    return "".join(out)


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depth = {"(": 0, "[": 0, "{": 0, "<": 0}
    pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
    for i, ch in enumerate(text):
        if ch in depth:
            depth[ch] += 1
        elif ch in pairs and depth[pairs[ch]] > 0:
            depth[pairs[ch]] -= 1
        elif ch == delimiter and all(v == 0 for v in depth.values()):
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def find_matching(text: str, open_pos: int, opening: str = "(", closing: str = ")") -> int:
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    return -1


def safe_identifier(value: str) -> str:
    return re.sub(r"\W+", "_", value).strip("_") or "anonymous"


def unqualified_type(type_name: str) -> str:
    return type_name.split("::")[-1].replace("struct ", "").replace("class ", "").replace("union ", "").strip()



def is_known_type(name: str) -> bool:
    normalized = re.sub(r"\s+", " ", name.strip())
    if (
        normalized in BUILTIN_TYPES
        or normalized.startswith("std::")
        or normalized in {"__int128", "unsigned __int128"}
        or normalized.startswith("__builtin_")
    ):
        return True
    tokens = normalized.split()
    return bool(tokens) and all(
        token in {
            "void", "char", "signed", "unsigned", "short", "int", "long",
            "float", "double", "bool", "_Bool", "const", "volatile",
        }
        for token in tokens
    )

def scalar_underlying_type(name: str) -> Optional[str]:
    """Return a conservative C backing type for common scalar/pointer aliases.

    Only exact names, fixed-width patterns, and unambiguous suffix components
    are accepted. Raw substring matching is deliberately forbidden because it
    previously classified ``gpointer`` as an integer merely because the word
    ``pointer`` contains the letters ``int``.
    """
    raw = unqualified_type(name).strip()
    low = raw.lower()
    compact = re.sub(r"\s+", "", low)

    # Linux/kernel/project aliases: u8, __u8, s32, uint32, uint32_t, etc.
    fixed = re.fullmatch(r"_*([us])(?:int)?(8|16|32|64)(?:_t)?", compact)
    if fixed:
        signedness, width = fixed.groups()
        return f"{'u' if signedness == 'u' else ''}int{width}_t"

    exact = {
        # C / GLib-style scalars.
        "u_char": "unsigned char",
        "uchar": "unsigned char",
        "guchar": "unsigned char",
        "gchar": "char",
        "char_type": "unsigned char",
        "byte": "unsigned char",
        "u_short": "unsigned short",
        "ushort": "unsigned short",
        "gushort": "unsigned short",
        "gshort": "short",
        "u_int": "unsigned int",
        "uint": "unsigned int",
        "guint": "unsigned int",
        "gluint": "unsigned int",
        "gint": "int",
        "glint": "int",
        "u_long": "unsigned long",
        "ulong": "unsigned long",
        "gulong": "unsigned long",
        "glong": "long",
        "gboolean": "int",
        "gbool": "int",
        "boolean": "int",
        "bool_t": "int",
        "gsize": "size_t",
        "gssize": "ssize_t",
        "wgint": "long long",
        "lin": "long",
        "file_offset": "long long",
        "code_int": "int",

        # Pointer typedefs whose ABI category is unambiguous.
        "gpointer": "void *",
        "gconstpointer": "const void *",
        "lpvoid": "void *",
        "lpcvoid": "const void *",
        "pvoid": "void *",

        # FreeType scalar aliases. Pointer-like FT_* handles are inferred from
        # their actual -> use and are not listed here.
        "ft_byte": "unsigned char",
        "ft_char": "signed char",
        "ft_ushort": "unsigned short",
        "ft_short": "short",
        "ft_uint": "unsigned int",
        "ft_int": "int",
        "ft_uint32": "uint32_t",
        "ft_int32": "int32_t",
        "ft_ulong": "unsigned long",
        "ft_long": "long",
        "ft_fixed": "long",
        "ft_f26dot6": "long",
        "ft_pos": "long",
        "ft_offset": "unsigned long",
        "ft_error": "int",
        "ft_bool": "unsigned char",
        "ft_tag": "uint32_t",

        # Common X11/Windows-compatible integer aliases.
        "card8": "uint8_t",
        "card16": "uint16_t",
        "card32": "uint32_t",
        "card64": "uint64_t",
        "cardinal": "unsigned int",
        "xid": "unsigned long",
        "atom": "unsigned long",
        "window": "unsigned long",
        "drawable": "unsigned long",
        "byte_t": "unsigned char",
        "word": "unsigned short",
        "dword": "unsigned long",
        "bool": "int",

        # Common graphics / crypto scalar aliases seen in PrimeVul.
        "gx_color_index": "unsigned long",
        "gs_glyph": "unsigned long",
        "bn_ulong": "unsigned long",
        "magickbooleantype": "int",
        "magickoffsettype": "long long",
        "magicksize_type": "size_t",
        "booleantype": "int",
        "boolean_type": "int",
    }
    if low in exact:
        return exact[low]

    # GLib fixed-width names: gint8, guint32, etc.
    glib_fixed = re.fullmatch(r"g(u?)int(8|16|32|64)", compact)
    if glib_fixed:
        unsigned_marker, width = glib_fixed.groups()
        return f"{'u' if unsigned_marker else ''}int{width}_t"

    # Conservative project typedef patterns. Do not classify names merely
    # because they contain a token somewhere in the middle.
    if low.endswith("_t"):
        components = [p for p in re.split(r"[^a-z0-9]+", low[:-2]) if p]
        if components and components[-1] in {
            "size", "ssize", "offset", "count", "index", "flag", "status",
            "error", "bool", "byte", "word", "int", "uint",
        }:
            return "int"

    scalar_suffixes = {
        "bool": "int",
        "count": "int",
        "index": "int",
        "status": "int",
        "error": "int",
        "offset": "long",
        "size": "size_t",
    }
    for suffix, backing in scalar_suffixes.items():
        if low == suffix or low.endswith("_" + suffix):
            return backing

    # Many C libraries use FooType/FooKind/FooMode as enum typedefs. Only use
    # this for unqualified names; actual member-access use is handled earlier as
    # struct/class evidence.
    if re.search(r"(?:type|kind|mode)$", low) and "::" not in name:
        return "int"

    return None

def is_scalar_like_type(name: str) -> bool:
    return scalar_underlying_type(name) is not None


def language_candidates(record: Mapping[str, Any], source: str) -> list[str]:
    """Return compiler language candidates without throwing away C rows.

    PrimeVul-derived processed files often keep a ``language`` field but no
    original filename. The previous version only trusted file extensions, so
    valid C records with ``language: "c"`` were rejected as non-C.

    C++ is still rejected here because this pipeline builds C translation units.
    Compiling C++ snippets as C would produce invalid, low-confidence IR.
    """
    filename = str(
        record.get("file_name")
        or record.get("filename")
        or record.get("path")
        or ""
    )
    ext = Path(filename).suffix.lower()

    if ext in DISCARD_SOURCE_EXTENSIONS:
        return []
    if ext in SOURCE_EXTENSIONS_C:
        return ["c"]

    language = str(record.get("language") or record.get("lang") or "").strip().lower()
    if language in {"c", "c11", "gnu11"}:
        return ["c"]
    if language in {"cpp", "c++", "cc", "cxx", "hpp", "h++"}:
        return []

    # Conservative fallback for processed PrimeVul rows that lost filename
    # metadata. Reject obvious C++ constructs rather than forcing them through
    # the C compiler.
    masked = mask_comments_and_literals(source)
    if re.search(r"\b(?:class|template|namespace|public|private|protected)\b|::|\bnew\s+", masked):
        return []
    return ["c"]


# ---------------------------------------------------------------------------
# Source analysis and stub generation
# ---------------------------------------------------------------------------


def analyse_signature(source: str, masked: str) -> FunctionSignature:
    """Extract the outer function signature from an isolated PrimeVul body.

    Older versions selected the *first* top-level parenthesis before the body.
    That is fragile for real project code because attributes/macros such as
    ``__attribute__((...))`` or calling-convention wrappers can appear before
    the function name. V7 records all plausible top-level parameter lists and
    selects the last one immediately preceding the body. This substantially
    reduces "target function not emitted" failures caused by extracting the
    wrong token as the target name.
    """
    body_start = masked.find("{")
    if body_start < 0:
        return FunctionSignature(body_start=-1)

    prefix = masked[:body_start]
    candidates: list[tuple[int, int, str, int]] = []
    depth = 0
    for i, ch in enumerate(prefix):
        if ch == "(":
            if depth == 0:
                before = prefix[:i]
                match = re.search(
                    r"([A-Za-z_]\w*(?:::[A-Za-z_~]\w*)*)\s*$",
                    before,
                )
                if match:
                    token = match.group(1)
                    leaf = token.split("::")[-1]
                    if leaf not in CONTROL_WORDS and leaf not in {
                        "__attribute__", "__declspec", "alignas", "if", "while", "for", "switch",
                    }:
                        close = find_matching(prefix, i)
                        if close >= 0:
                            tail = prefix[close + 1:].strip()
                            # Keep only candidates that could be the declaration
                            # directly attached to this body. Attributes/macros
                            # earlier in the declaration leave real return/type
                            # text after their close and are rejected here.
                            if not tail or re.fullmatch(
                                r"(?:(?:const|volatile|noexcept|throw\s*\([^)]*\)|override|final|&|&&)\s*)*"
                                r"(?::\s*[^{}]*)?",
                                tail,
                                flags=re.DOTALL,
                            ):
                                candidates.append((i, close, token, match.start(1)))
            depth += 1
        elif ch == ")" and depth:
            depth -= 1

    if not candidates:
        return FunctionSignature(body_start=body_start)

    candidate_open, close, qname, qstart = candidates[-1]
    params = source[candidate_open + 1:close]
    between = source[close + 1:body_start].strip()
    trailing = between
    initializer_list = ""
    colon_match = re.search(r"(?<!:):(?!:)", between)
    if colon_match:
        trailing = between[:colon_match.start()].strip()
        initializer_list = between[colon_match.end():].strip()
    trailing = re.sub(r"\b(?:override|final)\b", "", trailing).strip()

    parts = qname.split("::")
    function_name = parts[-1]
    owner = "::".join(parts[:-1])
    owner_leaf = parts[-2] if len(parts) > 1 else ""
    is_ctor = bool(owner_leaf and function_name == owner_leaf)
    is_dtor = bool(owner_leaf and function_name == f"~{owner_leaf}")
    return_prefix = source[:qstart].strip()

    return FunctionSignature(
        qualified_name=qname,
        function_name=function_name,
        owner=owner,
        return_prefix=return_prefix,
        params_text=params,
        trailing_qualifiers=trailing,
        initializer_list=initializer_list,
        body_start=body_start,
        declaration_start=max(0, source.rfind("\n", 0, qstart) + 1),
        is_constructor=is_ctor,
        is_destructor=is_dtor,
    )

def strip_default_value(param: str) -> str:
    pieces = split_top_level(param, "=")
    return pieces[0].strip() if pieces else param.strip()


def parse_type_and_name(declaration: str) -> tuple[Optional[TypeInfo], str]:
    declaration = strip_default_value(declaration.strip())
    if not declaration or declaration == "void" or declaration == "...":
        return None, ""

    fp = re.search(r"\(\s*[*&]\s*([A-Za-z_]\w*)\s*\)", declaration)
    if fp:
        return TypeInfo(base="__pv_function_pointer", pointer_depth=1), fp.group(1)

    cleaned = re.sub(r"\[[^\]]*\]\s*$", "", declaration).strip()
    identifiers = list(re.finditer(r"[A-Za-z_]\w*", cleaned))
    if not identifiers:
        return None, ""

    name_match = identifiers[-1]
    name = name_match.group(0)
    if name in TYPE_WORDS or (len(identifiers) == 1 and "*" not in cleaned and "&" not in cleaned):
        return None, ""

    type_part = cleaned[:name_match.start()].strip()
    pointer_depth = type_part.count("*")
    reference = "&" in type_part

    tagged = re.search(r"\b(?:struct|union|class|enum)\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)", type_part)
    if tagged:
        base = tagged.group(1)
    else:
        type_without_symbols = re.sub(r"[*&]", " ", type_part)
        tokens = re.findall(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", type_without_symbols)
        # Preserve integer modifiers. Older code removed "unsigned"/"long"
        # because TYPE_WORDS included them, turning "unsigned int" into "int"
        # and damaging ABI/field inference.
        removable_type_words = TYPE_WORDS - {"signed", "unsigned", "short", "long"}
        tokens = [t for t in tokens if t not in removable_type_words]
        if not tokens:
            base = "int"
        else:
            # Preserve combined built-ins such as unsigned long as a known scalar.
            combined = " ".join(tokens)
            if all(t in BUILTIN_TYPES or t in {"signed", "unsigned", "short", "long"} for t in tokens):
                base = combined
            else:
                base = tokens[-1]
    return TypeInfo(base=base, pointer_depth=pointer_depth, reference=reference), name



def _strip_top_level_initializer(declarator: str) -> str:
    """Return a declarator without its top-level initializer."""
    parts = split_top_level(declarator, "=")
    return parts[0].strip() if parts else declarator.strip()


def _parse_simple_declarator(
    declarator: str,
    base_info: TypeInfo,
) -> tuple[Optional[TypeInfo], str]:
    """Parse ``*name``, ``name[10]`` and similar simple declarators."""
    declarator = _strip_top_level_initializer(declarator)
    if not declarator or "(" in declarator:
        # Function pointers and direct-initialisation need a real parser.
        return None, ""

    match = re.fullmatch(
        r"\s*(?P<prefix>(?:(?:\*|&)\s*(?:const\s+)?)*)"
        r"(?P<name>[A-Za-z_]\w*)"
        r"(?P<arrays>(?:\s*\[[^\]]*\])*)\s*",
        declarator,
    )
    if not match:
        return None, ""

    pointer_depth = base_info.pointer_depth + match.group("prefix").count("*")
    reference = base_info.reference or "&" in match.group("prefix")
    # A local array decays to a pointer in nearly every use relevant to the
    # stub inference. Preserve that ABI category rather than treating it as an
    # integer value.
    if match.group("arrays"):
        pointer_depth += 1
    return TypeInfo(base_info.base, pointer_depth, reference), match.group("name")


def _parse_declaration_statement(statement: str) -> dict[str, TypeInfo]:
    """Parse conservative, comma-separated C/C++ local declarations.

    Examples handled:
      ``int a, b``
      ``char *p, *q``
      ``register code_int code, oldcode, incode``
      ``struct item *head, value``

    Expressions are rejected by requiring every suffix to be a simple
    declarator. Missing a declaration is safer than inventing a type.
    """
    statement = statement.strip()
    if not statement or ":=" in statement:
        return {}

    first = re.match(r"([A-Za-z_]\w*)", statement)
    if not first or first.group(1) in CONTROL_WORDS | {
        "else", "goto", "break", "continue", "case", "default",
    }:
        return {}

    # Candidate split points are whitespace boundaries. The first split whose
    # suffix is a valid declarator list is the type/declarator boundary.
    boundaries = [m.end() for m in re.finditer(r"\s+", statement)]
    for boundary in boundaries:
        type_text = statement[:boundary].strip()
        declarator_text = statement[boundary:].strip()
        if not type_text or not declarator_text:
            continue
        if re.search(r"[=;{}()]", type_text):
            continue
        if not re.fullmatch(
            r"(?:(?:const|volatile|static|register|auto|extern|mutable|"
            r"unsigned|signed|long|short|struct|union|enum|class|typename)\s+)*"
            r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*"
            r"(?:\s+[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)*",
            type_text,
        ):
            continue

        base_info, dummy = parse_type_and_name(f"{type_text} __pv_decl")
        if not base_info or dummy != "__pv_decl":
            continue

        parsed: dict[str, TypeInfo] = {}
        parts = split_top_level(declarator_text)
        if not parts:
            continue
        for part in parts:
            info, name = _parse_simple_declarator(part, base_info)
            if not info or not name:
                parsed = {}
                break
            parsed[name] = info
        if parsed:
            return parsed
    return {}


def infer_local_variables(masked: str) -> dict[str, TypeInfo]:
    """Infer local variable types without requiring project headers."""
    variables: dict[str, TypeInfo] = {}
    body_start = masked.find("{")
    body = masked[body_start + 1:] if body_start >= 0 else masked

    statements = [piece.strip() for piece in re.split(r"[;{}]", body)]
    # Include the declaration portion of ``for (type var = ...; ...)``.
    statements.extend(
        match.group(1).strip()
        for match in re.finditer(r"\bfor\s*\(\s*([^;]+);", body)
    )

    for statement in statements:
        if not statement:
            continue
        for name, info in _parse_declaration_statement(statement).items():
            if info.base not in CONTROL_WORDS:
                variables.setdefault(name, info)
    return variables

def extract_member_chains(masked: str) -> list[tuple[str, list[tuple[str, str]], str]]:
    chains: list[tuple[str, list[tuple[str, str]], str]] = []
    token = re.compile(r"[A-Za-z_]\w*")
    for root_match in token.finditer(masked):
        root = root_match.group(0)
        if root in CONTROL_WORDS:
            continue
        prefix = masked[max(0, root_match.start() - 3):root_match.start()]
        # Do not start a second chain at a field that is already preceded by . or ->.
        if re.search(r"(?:\.|->)\s*$", prefix):
            continue
        pos = root_match.end()
        fields: list[tuple[str, str]] = []
        while True:
            ws = re.match(r"\s*", masked[pos:])
            pos += ws.end() if ws else 0
            if masked.startswith("->", pos):
                op = "->"
                pos += 2
            elif pos < len(masked) and masked[pos] == "." and not masked.startswith("...", pos):
                op = "."
                pos += 1
            else:
                break
            ws = re.match(r"\s*", masked[pos:])
            pos += ws.end() if ws else 0
            field_match = token.match(masked, pos)
            if not field_match:
                break
            fields.append((op, field_match.group(0)))
            pos = field_match.end()
        if fields:
            lookahead = masked[pos:pos + 16].lstrip()
            final_context = "function" if lookahead.startswith("(") else "array" if lookahead.startswith("[") else "scalar"
            chains.append((root, fields, final_context))
    # Remove suffix duplicates: scanning starts again at every field token.
    unique: list[tuple[str, list[tuple[str, str]], str]] = []
    seen: set[tuple[Any, ...]] = set()
    for root, fields, context in chains:
        key = (root, tuple(fields), context)
        if key not in seen:
            seen.add(key)
            unique.append((root, fields, context))
    return unique


def analyse_source(source: str) -> Analysis:
    masked = mask_comments_and_literals(source)
    signature = analyse_signature(source, masked)
    variables: dict[str, TypeInfo] = {}
    for param in split_top_level(signature.params_text):
        info, name = parse_type_and_name(param)
        if info and name:
            variables[name] = info
    variables.update({k: v for k, v in infer_local_variables(masked).items() if k not in variables})
    if signature.owner:
        variables["this"] = TypeInfo(base=signature.owner, pointer_depth=1)

    defined_types = set(re.findall(r"\b(?:struct|union|class|enum)\s+([A-Za-z_]\w*)\s*[{;]", masked))
    defined_types.update(re.findall(r"\btypedef\b[^;]*\b([A-Za-z_]\w*)\s*;", masked))
    defined_identifiers = set(variables)
    if signature.function_name:
        defined_identifiers.add(signature.function_name)

    member_chains = extract_member_chains(masked)
    first_ops: dict[str, str] = {}
    for root, fields, _ in member_chains:
        if fields:
            first_ops.setdefault(root, fields[0][0])

    return Analysis(
        source=source,
        masked=masked,
        signature=signature,
        variables=variables,
        root_first_operator=first_ops,
        member_chains=member_chains,
        defined_types=defined_types,
        defined_identifiers=defined_identifiers,
    )


def typeinfo_declaration(info: TypeInfo) -> str:
    suffix = " *" * info.pointer_depth
    if info.reference:
        suffix += " &"
    return f"{info.base}{suffix}".strip()


def member_expression_pattern(root: str, fields: Sequence[tuple[str, str]]) -> str:
    pattern = rf"\b{re.escape(root)}\b"
    for operator, field_name in fields:
        op_pattern = r"\s*->\s*" if operator == "->" else r"\s*\.\s*"
        pattern += op_pattern + re.escape(field_name)
    return pattern



def infer_array_field_type(
    analysis: Analysis,
    root: str,
    fields: Sequence[tuple[str, str]],
) -> Optional[FieldSpec]:
    """Infer an indexed field's element type from surrounding source usage."""
    expression = member_expression_pattern(root, fields)
    masked = analysis.masked

    def pointer_to(info: TypeInfo) -> str:
        return typeinfo_declaration(
            TypeInfo(
                base=info.base,
                pointer_depth=info.pointer_depth + 1,
                reference=False,
            )
        )

    # ``T *p = &obj->field[i]`` means field is T*.
    address_assignment = re.search(
        rf"\b(?P<type>(?:(?:const|volatile|unsigned|signed|long|short|"
        rf"struct|union|enum|class)\s+)*[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)"
        rf"\s*\*\s*[A-Za-z_]\w*\s*=\s*&\s*{expression}\s*\[",
        masked,
    )
    if address_assignment:
        return FieldSpec("typed", f"{address_assignment.group('type').strip()} *")

    # ``value = obj->field[i]`` and ``obj->field[i] = value``.
    for variable, info in analysis.variables.items():
        if re.search(
            rf"\b{re.escape(variable)}\b\s*=\s*{expression}\s*\[",
            masked,
        ):
            return FieldSpec("typed", pointer_to(info))
        if re.search(
            rf"{expression}\s*\[[^\]]+\]\s*=\s*\b{re.escape(variable)}\b",
            masked,
        ):
            return FieldSpec("typed", pointer_to(info))

    # Declaration assignment not captured by the local-variable pass.
    value_assignment = re.search(
        rf"\b(?P<type>(?:(?:const|volatile|unsigned|signed|long|short)\s+)*"
        rf"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)"
        rf"\s+[A-Za-z_]\w*\s*=\s*{expression}\s*\[",
        masked,
    )
    if value_assignment:
        return FieldSpec("typed", f"{value_assignment.group('type').strip()} *")

    # An explicit cast around the indexed value is strong local evidence.
    cast_use = re.search(
        rf"\(\s*(?P<type>(?:(?:const|volatile|unsigned|signed|long|short|"
        rf"struct|union|enum|class)\s+)*[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*"
        rf"(?:\s*\*)*)\s*\)\s*{expression}\s*\[",
        masked,
    )
    if cast_use:
        cast_type = re.sub(r"\s+", " ", cast_use.group("type")).strip()
        info, _ = parse_type_and_name(f"{cast_type} __pv_cast")
        if info:
            return FieldSpec("typed", pointer_to(info))

    # String APIs prove byte-addressable character storage.
    if re.search(
        rf"\b(?:strlen|strcmp|strncmp|strcpy|strncpy|strchr|strrchr|strstr)"
        rf"\s*\(\s*{expression}\b",
        masked,
    ):
        return FieldSpec("typed", "char *")

    # Memory APIs and character comparisons establish an i8 element width.
    if re.search(
        rf"\b(?:memcpy|memmove|memset|memcmp)\s*\([^;]*\b{expression}\b",
        masked,
    ):
        return FieldSpec("typed", "unsigned char *")
    if re.search(
        rf"{expression}\s*\[[^\]]+\]\s*(?:==|!=)\s*'[^']'",
        analysis.source,
    ):
        return FieldSpec("typed", "char *")
    if re.search(
        rf"{expression}\s*\[[^\]]+\]\s*&\s*0x(?:ff|ffu|fful)\b",
        masked,
        re.IGNORECASE,
    ):
        return FieldSpec("typed", "unsigned char *")

    # ``obj->field[i].member`` proves that field is an array/pointer of
    # synthetic element structs. This is much safer than rejecting the sample,
    # because the member chain that follows the index can be inferred in later
    # repair rounds from Clang's diagnostics.
    if re.search(rf"{expression}\s*\[[^\]]+\]\s*\.", masked):
        element_type = (
            f"__pv_arr_{safe_identifier(root)}_"
            f"{safe_identifier('_'.join(name for _op, name in fields))}"
        )
        return FieldSpec("typed", f"{element_type} *")

    # ``obj->field[i]->member`` means the indexed element is itself a pointer.
    if re.search(rf"{expression}\s*\[[^\]]+\]\s*->", masked):
        element_type = (
            f"__pv_arr_{safe_identifier(root)}_"
            f"{safe_identifier('_'.join(name for _op, name in fields))}"
        )
        return FieldSpec("typed", f"{element_type} **")

    # Numeric stores/comparisons prove an integer-like element.
    if re.search(
        rf"{expression}\s*\[[^\]]+\]\s*(?:=|==|!=|<=|>=|<|>|\+=|-=)\s*[-+]?\d+",
        masked,
    ):
        return FieldSpec("typed", "long long *")

    # Last-resort compileability fallback. Older preprocessors rejected these
    # as ambiguous and lost many otherwise useful functions. The current
    # compiler records the fallback in stub notes/status, and stores the exact
    # translation unit used to produce the LLVM.
    return FieldSpec("typed", "long long *")

def exact_variable_rhs_pattern(expression: str) -> str:
    """Pattern for ``field = var`` where ``var`` is exactly the RHS value.

    This intentionally refuses to match ``field = obj->other`` as variable
    ``obj``. That bug made fields such as ``cmap->ttop = cmap->tlen - 1`` look
    like they had type ``pdf_cmap *`` instead of integer.
    """
    return rf"{expression}\s*=\s*([A-Za-z_]\w*)\b(?!\s*(?:->|\.|\[))"


def exact_variable_lhs_pattern(expression: str) -> str:
    """Pattern for ``var = field`` where ``field`` is exactly the RHS value."""
    return rf"\b([A-Za-z_]\w*)\s*=\s*{expression}\b(?!\s*(?:->|\.|\[))"


def infer_terminal_field_spec(
    analysis: Analysis,
    root: str,
    fields: Sequence[tuple[str, str]],
    final_context: str,
) -> FieldSpec:
    expression = member_expression_pattern(root, fields)
    if final_context == "function":
        assignment = re.search(rf"\b([A-Za-z_]\w*)\s*=\s*{expression}\s*\(", analysis.masked)
        if assignment:
            target = analysis.variables.get(assignment.group(1))
            if target:
                return FieldSpec("function", typeinfo_declaration(target))
        method_name = fields[-1][1] if fields else ""
        if method_name in {"data", "c_str", "ptr", "get", "begin", "end"}:
            return FieldSpec("function", "char *")
        if method_name in {"size", "length", "count", "capacity"}:
            return FieldSpec("function", "size_t")
        if re.search(rf"\b(?:memcpy|memmove|memcmp|strlen|strcmp|strncmp)\s*\([^;]*{expression}\s*\(", analysis.masked):
            return FieldSpec("function", "void *")
        return FieldSpec("function", "int")
    if final_context == "array":
        inferred = infer_array_field_type(analysis, root, fields)
        if inferred is not None:
            return inferred
        return FieldSpec("unresolved_array")

    # Infer a field type from direct assignment in either direction. Prefer
    # declaration/assignment to a local variable over stores into the field,
    # because ``field = obj->other`` otherwise falsely captures ``obj``.
    lhs = re.search(exact_variable_lhs_pattern(expression), analysis.masked)
    if lhs:
        lhs_type = analysis.variables.get(lhs.group(1))
        if lhs_type:
            return FieldSpec("typed", typeinfo_declaration(lhs_type))
    rhs = re.search(exact_variable_rhs_pattern(expression), analysis.masked)
    if rhs:
        rhs_type = analysis.variables.get(rhs.group(1))
        if rhs_type:
            return FieldSpec("typed", typeinfo_declaration(rhs_type))

    # Integer assignments/arithmetic provide enough evidence for an int-like
    # field and are safer than the old 64-bit catch-all.
    if re.search(
        rf"{expression}\s*(?:\+\+|--|[+\-*/%&|^]?=\s*(?:[-+]?\d+|[A-Za-z_]\w*))",
        analysis.masked,
    ):
        return FieldSpec("typed", "int")
    if re.search(
        rf"{expression}\s*(?:==|!=|<=|>=|<|>)\s*[-+]?\d+",
        analysis.masked,
    ):
        return FieldSpec("typed", "int")
    return FieldSpec("scalar")


def ensure_struct(state: RepairState, type_name: str) -> dict[str, FieldSpec]:
    state.scalar_types.discard(type_name)
    state.opaque_types.discard(type_name)
    return state.structs.setdefault(type_name, {})


def merge_field(existing: Optional[FieldSpec], incoming: FieldSpec) -> FieldSpec:
    if existing is None:
        return incoming
    ranking = {"scalar": 0, "unresolved_array": 1, "array": 2, "typed": 3, "function": 4, "value": 5, "pointer": 6}
    return incoming if ranking.get(incoming.kind, 0) > ranking.get(existing.kind, 0) else existing



def infer_initializer_field(analysis: Analysis, expression: str, state: RepairState) -> FieldSpec:
    expr = expression.strip()
    if not expr:
        return FieldSpec("scalar")
    if re.search(r"\b(?:nullptr|NULL)\b", expr):
        return FieldSpec("typed", "void *")
    if re.search(r"\b(?:true|false)\b", expr) or "getAttrBool" in expr:
        return FieldSpec("typed", "bool")
    if re.match(r"^[uULR]*[\"']", expr):
        return FieldSpec("typed", "const char *")
    new_match = re.search(r"\bnew\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)", expr)
    if new_match:
        type_name = new_match.group(1)
        state.opaque_types.add(type_name)
        return FieldSpec("typed", f"{type_name} *")
    variable_match = re.fullmatch(r"&?\s*([A-Za-z_]\w*)", expr)
    if variable_match:
        info = analysis.variables.get(variable_match.group(1))
        if info:
            declaration = typeinfo_declaration(info)
            if expr.lstrip().startswith("&"):
                declaration += " *"
            return FieldSpec("typed", declaration)
    return FieldSpec("scalar")


def build_initial_state(analysis: Analysis, language: str) -> RepairState:
    state = RepairState()
    sig = analysis.signature

    # Register unknown parameter/local types.
    for var, info in analysis.variables.items():
        base = info.base
        if var == "this" or is_known_type(base) or base in analysis.defined_types or base == "__pv_function_pointer":
            continue
        first_op = analysis.root_first_operator.get(var, "")
        backing = scalar_underlying_type(base)
        if backing is not None:
            # Includes exact pointer aliases such as gpointer -> void *.
            state.scalar_types.add(base)
            state.note(f"scalar-type:{base}")
        elif _type_used_as_scalar(analysis, base):
            state.scalar_types.add(base)
            state.note(f"scalar-type-inferred:{base}")
        elif first_op == "->" and info.pointer_depth == 0:
            # ``Type value; value->field`` proves that Type is itself a pointer
            # typedef. This is stronger than guessing from naming conventions.
            object_type = f"__pv_obj_{safe_identifier(base)}"
            state.pointer_aliases[base] = object_type
            ensure_struct(state, object_type)
            state.note(f"pointer-alias:{base}")
        elif first_op in {"->", "."}:
            ensure_struct(state, base)
            state.note(f"struct:{base}")
        elif info.pointer_depth > 0:
            state.opaque_types.add(base)
            state.note(f"opaque:{base}")
        else:
            # By-value unknowns must be complete. A one-field struct is safer than
            # silently forcing every project type to integer.
            ensure_struct(state, base)
            state.note(f"value-type:{base}")
            state.mark_uncertain(f"value-type:{base}")

    # Build field layouts from actual member-access chains.
    for root, fields, final_context in analysis.member_chains:
        info = analysis.variables.get(root)
        if info:
            current = info.base
            if current in state.pointer_aliases:
                current = state.pointer_aliases[current]
        else:
            current = f"__pv_root_{safe_identifier(root)}"
            first_op = fields[0][0]
            state.variables.setdefault(root, f"{current} *" if first_op == "->" else current)
            state.note(f"root-variable:{root}")

        if current == "this" and sig.owner:
            current = sig.owner

        for index, (_operator, field_name) in enumerate(fields):
            is_last = index == len(fields) - 1
            if is_last:
                inferred = infer_terminal_field_spec(analysis, root, fields, final_context)
                kind, target = inferred.kind, inferred.target
                if kind == "scalar":
                    state.mark_uncertain(f"field-type:{current}.{field_name}")
                elif kind == "function" and target in {"", "int"}:
                    state.mark_uncertain(f"method-return:{current}.{field_name}")
                elif final_context == "array":
                    state.note(f"array-field:{current}.{field_name}:{target or kind}")
                    if target == "long long *":
                        state.note(f"array-field-fallback:{current}.{field_name}")
                    synthetic = re.match(r"(__pv_arr_[A-Za-z0-9_]+)\s*\*+\s*$", target or "")
                    if synthetic:
                        ensure_struct(state, synthetic.group(1))
            else:
                next_operator = fields[index + 1][0]
                kind = "pointer" if next_operator == "->" else "value"
                target = f"__pv_{safe_identifier(current)}_{safe_identifier(field_name)}"

            field_map = ensure_struct(state, current)
            field_map[field_name] = merge_field(field_map.get(field_name), FieldSpec(kind, target))
            if not is_last:
                ensure_struct(state, target)
                current = target

    return state


def render_qualified_block(type_name: str, body: str, keyword: str = "struct") -> str:
    return f"{keyword} {type_name} {{\n{body}\n}};"


def render_forward(type_name: str, language: str, keyword: str = "struct") -> str:
    return f"typedef struct {type_name} {type_name};"


def render_field(name: str, spec: FieldSpec, language: str) -> str:
    if spec.kind == "pointer":
        return f"    {spec.target} *{name};"
    if spec.kind == "value":
        return f"    {spec.target} {name};"
    if spec.kind == "function":
        return_type = spec.target or "int"
        return f"    {return_type} (*{name})();"
    if spec.kind == "array":
        raise ValueError(f"unresolved array field type for {name}")
    if spec.kind == "unresolved_array":
        raise ValueError(f"unresolved array field type for {name}")
    if spec.kind == "typed":
        return f"    {spec.target} {name};"
    return f"    int {name};"


def dependency_order(structs: Mapping[str, Mapping[str, FieldSpec]]) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            return
        visiting.add(name)
        for spec in structs.get(name, {}).values():
            if spec.kind == "value" and spec.target in structs:
                visit(spec.target)
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    for struct_name in structs:
        visit(struct_name)
    return ordered


def render_stubs(state: RepairState, language: str) -> str:
    lines: list[str] = ["/* ---- automatically generated PrimeVul stubs ---- */"]

    for macro in sorted(state.empty_macros):
        lines.extend([f"#ifndef {macro}", f"#define {macro}", "#endif"])
    for macro in sorted(state.semantic_macro_fallbacks):
        fallback = SEMANTIC_MACRO_FALLBACKS.get(macro)
        if fallback:
            lines.append(fallback)
    # Unknown constants remain low confidence, but give each a distinct,
    # deterministic value. Defining every constant as zero created duplicate
    # switch cases and silently collapsed control-flow alternatives.
    for ordinal, macro in enumerate(sorted(state.macro_constants), start=1):
        lines.extend([
            f"#ifndef {macro}",
            f"#define {macro} ({ordinal})",
            "#endif",
        ])

    full_types = set(state.structs)
    # A name must never be emitted as both a typedef scalar and a struct.
    # Clang reports this as redefinition of 'X' as different kind of symbol
    # (seen with Image/MagickBooleanType-like project names). Prefer the
    # complete type when field inference has already required one.
    scalar_type_names = sorted(
        t for t in state.scalar_types
        if "::" not in t and t not in full_types and t not in state.pointer_aliases
    )
    for type_name in scalar_type_names:
        backing = scalar_underlying_type(type_name) or "int"
        lines.append(f"typedef {backing} {type_name};")

    # Pointer typedefs need their backing object forward declarations first.
    for alias, object_type in sorted(state.pointer_aliases.items()):
        lines.append(render_forward(object_type, language))
        lines.append(f"typedef {object_type} *{alias};")

    for type_name in sorted(state.opaque_types - full_types):
        lines.append(render_forward(type_name, language))
    pointer_objects = set(state.pointer_aliases.values())
    for type_name in sorted(full_types - pointer_objects):
        lines.append(render_forward(type_name, language))

    # Define plain structs in value-dependency order.
    for type_name in dependency_order(state.structs):
        fields = state.structs[type_name]
        body = "\n".join(render_field(name, spec, language) for name, spec in sorted(fields.items()))
        if not body:
            body = "    int __pv_dummy;"
        lines.append(render_qualified_block(type_name, body, "struct"))

    for function_name, return_type in sorted(state.functions.items()):
        # An old-style declaration accepts the observed call arguments while
        # preserving the inferred return type.
        lines.append(f"extern {return_type} {function_name}();")

    for variable_name, variable_type in sorted(state.variables.items()):
        lines.append(f"extern {variable_type} {variable_name};")

    lines.append("/* ---- end generated stubs ---- */")
    return "\n".join(lines)


def find_unresolved_array_fields(state: RepairState) -> list[str]:
    unresolved: list[str] = []
    for type_name, fields in state.structs.items():
        for field_name, spec in fields.items():
            if spec.kind == "unresolved_array":
                unresolved.append(f"{type_name}.{field_name}")
    return sorted(unresolved)


def finalized_state_notes(state: RepairState) -> list[str]:
    """Drop uncertainty notes superseded by stronger later inference."""
    unresolved = set(state.uncertain)
    for item in list(unresolved):
        if not item.startswith("field-type:"):
            continue
        path = item.removeprefix("field-type:")
        if "." not in path:
            continue
        owner, field_name = path.rsplit(".", 1)
        spec: Optional[FieldSpec] = None
        if owner in state.structs:
            spec = state.structs[owner].get(field_name)
        if spec is not None and spec.kind not in {"scalar", "unresolved_array"}:
            unresolved.discard(item)

    return [
        note
        for note in state.notes
        if not note.startswith("uncertain:")
        or note.removeprefix("uncertain:") in unresolved
    ]


def find_unresolved_semantic_macros(source: str) -> list[str]:
    """Return project macros whose fake expansion would change control/data flow."""
    masked = mask_comments_and_literals(source)
    macro_calls = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", masked))
    unsafe: list[str] = []
    for macro in macro_calls:
        if macro in SAFE_FUNCTION_MACROS or macro in CONTROL_WORDS:
            continue
        upper = macro.upper()
        if macro in UNSAFE_SEMANTIC_MACROS:
            unsafe.append(macro)
            continue
        if macro == upper and any(pattern in upper for pattern in UNSAFE_MACRO_PATTERNS):
            unsafe.append(macro)
    return sorted(set(unsafe))


def function_is_called(masked: str, name: str) -> bool:
    return bool(re.search(rf"\b{re.escape(name)}\s*\(", masked))




def call_is_standalone_statement(masked: str, name: str) -> bool:
    """Return true only when ``name(...)`` is the complete statement."""
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*\(", re.MULTILINE)
    for match in pattern.finditer(masked):
        open_pos = masked.find("(", match.start())
        close_pos = find_matching(masked, open_pos)
        if close_pos < 0:
            continue
        line_end = masked.find("\n", close_pos)
        if line_end < 0:
            line_end = len(masked)
        if masked[close_pos + 1:line_end].strip() == ";":
            return True
    return False


def _signature_return_declaration(analysis: Analysis) -> str:
    """Return the target function's declared return type when it is explicit."""
    prefix = analysis.signature.return_prefix
    if not prefix:
        return ""
    prefix = re.sub(
        r"\b(?:static|extern|inline|__inline|__inline__|virtual|friend|"
        r"constexpr|consteval|constinit|register)\b",
        " ",
        prefix,
    )
    prefix = re.sub(r"__attribute__\s*\(\([^)]*\)\)", " ", prefix)
    prefix = re.sub(r"__declspec\s*\([^)]*\)", " ", prefix)
    prefix = re.sub(r"\s+", " ", prefix).strip()
    return prefix


def infer_function_return_type(analysis: Analysis, name: str, state: RepairState) -> str:
    """Infer an unknown call's return type from its use site."""
    escaped = re.escape(name)
    masked = analysis.masked

    # ``call(...)->member`` proves an object pointer result.
    if re.search(rf"\b{escaped}\s*\([^;{{}}]*\)\s*->", masked):
        result_type = f"__pv_result_{safe_identifier(name)}"
        ensure_struct(state, result_type)
        state.mark_uncertain(f"function-result-object:{name}")
        return f"{result_type} *"

    # Explicit casts are the strongest local evidence.
    cast_call = re.search(
        rf"\(\s*(?P<type>(?:(?:const|volatile|unsigned|signed|long|short|"
        rf"struct|union|enum|class)\s+)*[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*"
        rf"(?:\s*[*&]\s*)*)\)\s*{escaped}\s*\(",
        masked,
    )
    if cast_call:
        return re.sub(r"\s+", " ", cast_call.group("type")).strip()

    # ``*out = call()`` returns the pointee category.
    deref_assignment = re.search(
        rf"(?m)(?:^|[;{{}}])\s*\*\s*([A-Za-z_]\w*)\s*=\s*"
        rf"{escaped}\s*\(",
        masked,
    )
    if deref_assignment:
        target = analysis.variables.get(deref_assignment.group(1))
        if target and target.pointer_depth > 0:
            pointee = TypeInfo(
                base=target.base,
                pointer_depth=max(0, target.pointer_depth - 1),
                reference=False,
            )
            if (
                pointee.pointer_depth > 0
                and not is_known_type(pointee.base)
                and scalar_underlying_type(pointee.base) is None
            ):
                state.opaque_types.add(pointee.base)
            return typeinfo_declaration(pointee)

    # Assignment to a known variable.
    assignment = re.search(
        rf"\b([A-Za-z_]\w*)\s*=\s*(?:\([^;]*\)\s*)?{escaped}\s*\(",
        masked,
    )
    if assignment:
        target = analysis.variables.get(assignment.group(1))
        if target:
            if (
                target.pointer_depth > 0
                and not is_known_type(target.base)
                and scalar_underlying_type(target.base) is None
            ):
                state.opaque_types.add(target.base)
            return typeinfo_declaration(target)

    # Declaration assignment that the local parser may not have captured.
    declaration_assignment = re.search(
        rf"\b(?P<type>(?:(?:const|volatile|unsigned|signed|long|short|"
        rf"struct|union|enum|class)\s+)*[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*"
        rf"(?:\s*[*&]\s*)*)"
        rf"(?P<var>[A-Za-z_]\w*)\s*=\s*{escaped}\s*\(",
        masked,
    )
    if declaration_assignment:
        raw = re.sub(r"\s+", " ", declaration_assignment.group("type")).strip()
        if raw:
            return raw

    # Unary dereference proves a pointer result, for example
    # ``*sk_sleep(sk)`` in Linux networking code.
    if re.search(rf"\*\s*{escaped}\s*\(", masked):
        result_type = f"__pv_result_{safe_identifier(name)}"
        ensure_struct(state, result_type)
        state.mark_uncertain(f"function-pointee:{name}")
        return f"{result_type} *"

    # Directly returning the call inherits the target function's declared type.
    if re.search(rf"\breturn\s+{escaped}\s*\(", masked):
        target_return = _signature_return_declaration(analysis)
        if target_return:
            return target_return

    # Pointer-specific library contexts.
    if re.search(
        rf"\b(?:free|g_free|strlen|strcpy|strncpy|strcmp|strchr|strrchr)"
        rf"\s*\(\s*{escaped}\s*\(",
        masked,
    ):
        return "void *"
    if re.search(
        rf"\b{escaped}\s*\([^;]*\)\s*(?:==|!=)\s*(?:NULL|nullptr)\b",
        masked,
    ):
        return "void *"

    # Conditions consume an int-like value. Check this before the
    # standalone-call rule because a call can begin a continuation line inside
    # another function's argument list.
    if re.search(rf"\b(?:if|while)\s*\([^)]*\b{escaped}\s*\(", masked):
        return "int"

    # A standalone call has no consumed return value.
    if call_is_standalone_statement(masked, name):
        return "void"
    if re.search(rf"\b{escaped}\s*\([^;]*\)\s*(?:[=!<>+\-*/%&|]|\?)", masked):
        return "int"

    state.mark_uncertain(f"function-return:{name}")
    return "int"

def appears_as_modifier(analysis: Analysis, name: str) -> bool:
    sig_text = analysis.source[:analysis.signature.body_start] if analysis.signature.body_start >= 0 else analysis.source
    # Attribute/calling-convention macro immediately before a real type or after ')'.
    return bool(
        re.search(rf"\b{re.escape(name)}\b\s+(?:void|char|short|int|long|float|double|signed|unsigned|struct|class|enum)\b", sig_text)
        or re.search(rf"\)\s*\b{re.escape(name)}\b\s*$", sig_text.strip())
    )


def type_for_undeclared_variable(analysis: Analysis, name: str, state: RepairState) -> str:
    chain = next((item for item in analysis.member_chains if item[0] == name), None)
    if chain:
        root_type = f"__pv_root_{safe_identifier(name)}"
        ensure_struct(state, root_type)
        return f"{root_type} *" if chain[1][0][0] == "->" else root_type
    if re.search(rf"\b{re.escape(name)}\s*\[", analysis.masked):
        state.note(f"array-variable-fallback:{name}")
        return "long long *"
    return "long long"


def infer_missing_member_spec(
    analysis: Analysis,
    member_name: str,
) -> FieldSpec:
    """Infer a missing field from assignments involving that member."""
    escaped = re.escape(member_name)

    lhs = re.search(
        rf"\b([A-Za-z_]\w*)\s*=\s*[^;\n]*(?:->|\.)\s*{escaped}\b(?!\s*(?:->|\.|\[))",
        analysis.masked,
    )
    if lhs:
        target = analysis.variables.get(lhs.group(1))
        if target:
            return FieldSpec("typed", typeinfo_declaration(target))

    rhs = re.search(
        rf"(?:->|\.)\s*{escaped}\s*=\s*([A-Za-z_]\w*)\b(?!\s*(?:->|\.|\[))",
        analysis.masked,
    )
    if rhs:
        source_type = analysis.variables.get(rhs.group(1))
        if source_type:
            return FieldSpec("typed", typeinfo_declaration(source_type))

    if re.search(rf"(?:->|\.)\s*{escaped}\s*\[", analysis.masked):
        return FieldSpec("typed", "long long *")
    if re.search(rf"(?:->|\.)\s*{escaped}\s*\(", analysis.masked):
        return FieldSpec("function", "int")
    if re.search(rf"(?:->|\.)\s*{escaped}\s*->", analysis.masked):
        return FieldSpec("typed", "void *")
    return FieldSpec("scalar")


def normalize_diagnostic_type(value: str) -> str:
    value = re.sub(r"^(?:struct|class|union)\s+", "", value.strip())
    value = value.replace(" *", "").replace("&", "").strip()
    return value



def _type_used_as_scalar(
    analysis: Analysis,
    type_name: str,
) -> bool:
    """Infer enum/integer-like project types from local use sites."""
    variables = [
        name for name, info in analysis.variables.items()
        if info.base == type_name and info.pointer_depth == 0 and not info.reference
    ]
    if not variables:
        return False
    for variable in variables:
        escaped = re.escape(variable)
        if re.search(rf"\bswitch\s*\([^)]*\b{escaped}\b", analysis.masked):
            return True
        if re.search(rf"\b{escaped}\b\s*(?:==|!=|<=|>=|<|>|[+\-*/%&|^]=?)", analysis.masked):
            return True
        if re.search(rf"(?:==|!=|<=|>=|<|>|[+\-*/%&|^])\s*\b{escaped}\b", analysis.masked):
            return True
        if re.search(rf"\b(?:if|while)\s*\([^)]*\b{escaped}\b", analysis.masked):
            return True
    return False


def _type_used_as_pointer_alias(
    analysis: Analysis,
    type_name: str,
) -> bool:
    for variable, info in analysis.variables.items():
        if info.base != type_name or info.pointer_depth != 0:
            continue
        if analysis.root_first_operator.get(variable) == "->":
            return True
    return False


def _ensure_pointer_alias(
    state: RepairState,
    type_name: str,
) -> None:
    object_type = state.pointer_aliases.get(
        type_name,
        f"__pv_obj_{safe_identifier(type_name)}",
    )
    state.pointer_aliases[type_name] = object_type
    ensure_struct(state, object_type)
    state.note(f"pointer-alias:{type_name}")


def apply_diagnostic_repairs(stderr: str, analysis: Analysis, state: RepairState, language: str) -> bool:
    """Apply only repairs supported by local source evidence."""
    changed = False

    for type_name in DIAGNOSTIC_PATTERNS["unknown_type"].findall(stderr):
        type_name = normalize_diagnostic_type(type_name)
        if not re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", type_name):
            continue
        if type_name in analysis.defined_types or is_known_type(type_name):
            continue
        if appears_as_modifier(analysis, type_name):
            if type_name not in state.empty_macros:
                state.empty_macros.add(type_name)
                state.note(f"empty-macro:{type_name}")
                changed = True
        elif type_name in state.structs or type_name in state.pointer_aliases:
            continue
        elif scalar_underlying_type(type_name) is not None or _type_used_as_scalar(analysis, type_name):
            if type_name not in state.scalar_types:
                state.scalar_types.add(type_name)
                state.note(f"scalar-type:{type_name}")
                changed = True
        elif _type_used_as_pointer_alias(analysis, type_name):
            _ensure_pointer_alias(state, type_name)
            changed = True
        else:
            ensure_struct(state, type_name)
            state.note(f"unknown-type:{type_name}")
            state.mark_uncertain(f"value-type:{type_name}")
            changed = True

    undeclared_functions = set(DIAGNOSTIC_PATTERNS["undeclared_function"].findall(stderr))
    undeclared_names = set(DIAGNOSTIC_PATTERNS["undeclared"].findall(stderr))
    for name in sorted(undeclared_functions | undeclared_names):
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            continue
        if name in analysis.defined_identifiers or name in KNOWN_CALLS:
            continue
        if function_is_called(analysis.masked, name):
            if name not in state.functions:
                state.functions[name] = infer_function_return_type(analysis, name, state)
                state.note(f"function:{name}")
                changed = True
        elif name.isupper() or ("_" in name and name.upper() == name):
            if name not in state.macro_constants:
                state.macro_constants.add(name)
                state.note(f"constant:{name}")
                state.mark_uncertain(f"constant-value:{name}")
                changed = True
        else:
            if name not in state.variables:
                state.variables[name] = type_for_undeclared_variable(analysis, name, state)
                state.note(f"variable:{name}")
                changed = True

    for member_name, owner_type in DIAGNOSTIC_PATTERNS["no_member"].findall(stderr):
        owner_type = normalize_diagnostic_type(owner_type)
        # Diagnostics may name a pointer typedef rather than its backing object.
        owner_type = state.pointer_aliases.get(owner_type, owner_type)
        fields = ensure_struct(state, owner_type)
        if member_name not in fields:
            inferred_member = infer_missing_member_spec(analysis, member_name)
            fields[member_name] = inferred_member
            state.note(f"field:{owner_type}.{member_name}")
            if inferred_member.kind in {"scalar", "unresolved_array"}:
                state.mark_uncertain(f"field-type:{owner_type}.{member_name}")
            changed = True

    for type_name in DIAGNOSTIC_PATTERNS["incomplete"].findall(stderr):
        type_name = normalize_diagnostic_type(type_name)
        if not type_name:
            continue
        if scalar_underlying_type(type_name) is not None or _type_used_as_scalar(analysis, type_name):
            if type_name not in state.scalar_types:
                state.scalar_types.add(type_name)
                state.note(f"scalar-type:{type_name}")
                changed = True
        elif _type_used_as_pointer_alias(analysis, type_name):
            if type_name not in state.pointer_aliases:
                _ensure_pointer_alias(state, type_name)
                changed = True
        elif type_name not in state.structs:
            ensure_struct(state, type_name)
            state.note(f"complete-type:{type_name}")
            state.mark_uncertain(f"value-type:{type_name}")
            changed = True

    for _tag, type_name in DIAGNOSTIC_PATTERNS["must_tag"].findall(stderr):
        type_name = normalize_diagnostic_type(type_name)
        if language == "c" and type_name not in state.structs:
            ensure_struct(state, type_name)
            state.note(f"tag-type:{type_name}")
            changed = True

    return changed

def build_translation_unit(analysis: Analysis, state: RepairState, language: str) -> str:
    prelude = build_conditional_prelude(analysis.source, language)
    source_name = "primevul_input.c"
    compilable_source = analysis.source
    sig = analysis.signature

    if language == "c" and sig.function_name and not sig.return_prefix and not sig.owner:
        repair_type = infer_missing_return_type(analysis)
        if repair_type is None:
            raise ValueError("missing return type is ambiguous")
        compilable_source = re.sub(
            rf"\b{re.escape(sig.function_name)}\s*\(",
            f"{repair_type} {sig.function_name}(",
            compilable_source,
            count=1,
        )
        state.note(f"repair:return-type:{repair_type}")

    # Force the isolated target definition to be emitted without
    # ``-femit-all-decls``. The latter also emits hundreds of header functions.
    insertion = min(max(sig.declaration_start, 0), len(compilable_source))
    used_attribute = "__attribute__((used, noinline)) "
    compilable_source = (
        compilable_source[:insertion]
        + used_attribute
        + compilable_source[insertion:]
    )

    return (
        f"{prelude}\n{render_stubs(state, language)}\n"
        f"#line 1 \"{source_name}\"\n{compilable_source}\n"
    )


def _strip_outer_parentheses(expr: str) -> str:
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        close = find_matching(expr, 0)
        if close != len(expr) - 1:
            break
        expr = expr[1:-1].strip()
    return expr


def infer_return_expression_type(analysis: Analysis, expression: str) -> Optional[str]:
    """Infer a missing C return type from local return expressions only.

    This is deliberately conservative. It fixes high-confidence old-style C
    functions such as ``foo(...) { T *p; return p; }`` without inventing
    project-level semantics.
    """
    expr = _strip_outer_parentheses(expression)
    if not expr:
        return None

    cast = re.match(
        r"^\(\s*(?P<type>(?:(?:const|volatile|unsigned|signed|long|short|"
        r"struct|union|enum|class)\s+)*[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*"
        r"(?:\s*[*&]\s*)*)\)\s*.+$",
        expr,
    )
    if cast:
        return re.sub(r"\s+", " ", cast.group("type")).strip()

    address = re.fullmatch(r"&\s*([A-Za-z_]\w*)", expr)
    if address:
        info = analysis.variables.get(address.group(1))
        if info:
            return typeinfo_declaration(
                TypeInfo(
                    base=info.base,
                    pointer_depth=info.pointer_depth + 1,
                    reference=False,
                )
            )

    variable = re.fullmatch(r"([A-Za-z_]\w*)", expr)
    if variable:
        name = variable.group(1)
        info = analysis.variables.get(name)
        if info:
            return typeinfo_declaration(info)
        if name in {"NULL", "nullptr"}:
            return None
        if name in {"true", "false"}:
            return "bool"

    if re.fullmatch(r"[-+]?\d+[uUlL]*", expr):
        return "int"
    if re.fullmatch(r"(?:0x[0-9A-Fa-f]+|0[0-7]+)[uUlL]*", expr):
        return "int"
    if re.search(r"\bsizeof\s*\(", expr):
        return "size_t"

    # Pointer arithmetic still returns the pointer-like operand.
    pointer_arith = re.match(r"([A-Za-z_]\w*)\s*[+\-]\s*\d+", expr)
    if pointer_arith:
        info = analysis.variables.get(pointer_arith.group(1))
        if info and info.pointer_depth > 0:
            return typeinfo_declaration(info)

    # Conditional returns are safe only when both branches infer identically.
    if "?" in expr and ":" in expr:
        parts = split_top_level(expr, "?")
        if len(parts) == 2:
            branch_parts = split_top_level(parts[1], ":")
            if len(branch_parts) == 2:
                left = infer_return_expression_type(analysis, branch_parts[0])
                right = infer_return_expression_type(analysis, branch_parts[1])
                if left and left == right:
                    return left

    return None


def infer_missing_return_type(analysis: Analysis) -> Optional[str]:
    """Repair historical C implicit return types conservatively.

    Pre-C99 omitted return types defaulted to ``int``. For PrimeVul fragments,
    many omitted types are actually project pointer/value types. Inferring from
    returned local variables preserves ABI better than blindly forcing ``int``.
    """
    if analysis.signature.body_start < 0:
        return None
    body = analysis.masked[analysis.signature.body_start + 1:]
    returns = [
        match.group("expr").strip()
        for match in re.finditer(r"\breturn\b(?P<expr>[^;]*);", body)
    ]
    value_returns = [expr for expr in returns if expr]
    bare_returns = [expr for expr in returns if not expr]
    if value_returns and bare_returns:
        return None
    if not value_returns:
        return "void"

    inferred = [infer_return_expression_type(analysis, expr) for expr in value_returns]
    concrete = [item for item in inferred if item]
    if concrete and all(item == concrete[0] for item in concrete):
        return concrete[0]

    # Fall back to historical C only for clearly scalar-looking expressions.
    if all(
        re.fullmatch(r"[-+]?\d+[uUlL]*", expr)
        or re.search(r"(?:==|!=|<=|>=|<|>|\+|-|\*|/|%|&|\||\^)", expr)
        for expr in value_returns
    ):
        return "int"

    return None

def compact_error(stderr: str, limit: int = 240) -> str:
    lines = [line.strip() for line in stderr.splitlines() if "error:" in line]
    if lines:
        text = " | ".join(lines[:2])
    elif stderr.strip():
        text = stderr.strip().splitlines()[-1]
    else:
        text = "compiler failed"
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def _version_key(path: Path) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in re.split(r"[^\d]+", path.name):
        if piece:
            parts.append(int(piece))
    return tuple(parts)


def _existing_dirs(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        if path.is_dir():
            key = str(path).lower()
            if key not in seen:
                seen.add(key)
                result.append(path)
    return result


@functools.lru_cache(maxsize=1)
def discover_windows_system_include_dirs() -> tuple[str, ...]:
    """Find MSVC/Windows SDK headers for standalone LLVM-on-Windows installs."""
    if platform.system() != "Windows":
        return ()

    candidates: list[Path] = []

    include_env = os.environ.get("INCLUDE", "")
    for item in include_env.split(os.pathsep):
        if item:
            candidates.append(Path(item))

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))

    vs_roots = [
        program_files / "Microsoft Visual Studio" / "2022",
        program_files / "Microsoft Visual Studio" / "2019",
        program_files_x86 / "Microsoft Visual Studio" / "2019",
    ]
    for root in vs_roots:
        if not root.is_dir():
            continue
        for edition in root.iterdir():
            if not edition.is_dir():
                continue
            msvc_root = edition / "VC" / "Tools" / "MSVC"
            if msvc_root.is_dir():
                versions = sorted(
                    (path for path in msvc_root.iterdir() if path.is_dir()),
                    key=_version_key,
                    reverse=True,
                )
                if versions:
                    candidates.append(versions[0] / "include")

            candidates.append(edition / "SDK" / "ScopeCppSDK" / "vc15" / "VC" / "include")
            candidates.append(edition / "SDK" / "ScopeCppSDK" / "vc15" / "SDK" / "include" / "ucrt")

    windows_sdk_include = program_files_x86 / "Windows Kits" / "10" / "Include"
    if windows_sdk_include.is_dir():
        sdk_versions = sorted(
            (path for path in windows_sdk_include.iterdir() if path.is_dir()),
            key=_version_key,
            reverse=True,
        )
        if sdk_versions:
            latest_sdk = sdk_versions[0]
            candidates.extend([
                latest_sdk / "ucrt",
                latest_sdk / "shared",
                latest_sdk / "um",
                latest_sdk / "winrt",
            ])

    return tuple(str(path) for path in _existing_dirs(candidates))


def compiler_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    return env


def compiler_extra_flags(language: str) -> list[str]:
    flags = list(EXTRA_C_FLAGS)
    for include_dir in discover_windows_system_include_dirs():
        flags.extend(["-isystem", include_dir])
    return flags



# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------



def compiler_command(language: str, source_path: Path, ir_path: Path) -> list[str]:
    return [
        CLANG,
        "-x", "c",
        f"-std={C_STANDARD}",
        "-S", "-emit-llvm", "-O0", "-g0",
        "-fno-discard-value-names",
        "-fno-color-diagnostics",
        "-ferror-limit=0",
        "-fno-ident",

        # Errors that would alter the ABI or silently invent declarations.
        "-Werror=implicit-int",
        "-Werror=incompatible-pointer-types",
        "-Werror=int-conversion",
        "-Werror=implicit-function-declaration",

        # Real-world PrimeVul code often intentionally contains these defects
        # or uses GNU extensions. Let Clang emit faithful IR and record the
        # warning instead of rejecting the sample.
        "-Wno-error=return-type",
        "-Wno-error=pointer-sign",
        "-Wno-error=gnu-pointer-arith",
        "-Wno-error=zero-length-array",

        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-unused-function",
        "-Wno-strict-prototypes",
        "-Wno-gnu-zero-variadic-macro-arguments",
        "-Xclang", "-disable-O0-optnone",
        *compiler_extra_flags(language),
        str(source_path),
        "-o", str(ir_path),
    ]

def llvm_definition_records(llvm_ir: str) -> list[tuple[str, str, int, int]]:
    """Return ``(symbol, header, start_line, end_line)`` for function definitions."""
    lines = llvm_ir.splitlines()
    records: list[tuple[str, str, int, int]] = []
    index = 0
    pattern = re.compile(r'^define\b.*?@(?:"([^"]+)"|([A-Za-z0-9_.$?@-]+))\s*\(')
    while index < len(lines):
        line = lines[index]
        match = pattern.match(line)
        if not match:
            index += 1
            continue
        symbol = match.group(1) or match.group(2)
        start = index
        index += 1
        while index < len(lines) and lines[index].strip() != "}":
            index += 1
        end = min(index, len(lines) - 1)
        records.append((symbol, line, start, end))
        index = end + 1
    return records


def llvm_function_exists(llvm_ir: str, function_name: str) -> bool:
    return any(symbol == function_name for symbol, _header, _start, _end in llvm_definition_records(llvm_ir))


def target_c_function_exists(llvm_ir: str, function_name: str) -> bool:
    return llvm_function_exists(llvm_ir, function_name)


def llvm_return_type_for_c(expected_return_type: str) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", expected_return_type.strip())
    # LLVM 17 uses opaque pointers, so every C pointer return is ``ptr``.
    if "*" in normalized or normalized.endswith("]"):
        return "ptr"
    mapping = {
        "void": "void",
        "bool": "i1",
        "_Bool": "i1",
        "char": "i8",
        "signed char": "i8",
        "unsigned char": "i8",
        "short": "i16",
        "short int": "i16",
        "unsigned short": "i16",
        "unsigned short int": "i16",
        "int": "i32",
        "signed int": "i32",
        "unsigned int": "i32",
        "long": "i64",
        "long int": "i64",
        "unsigned long": "i64",
        "unsigned long int": "i64",
        "long long": "i64",
        "long long int": "i64",
        "unsigned long long": "i64",
        "unsigned long long int": "i64",
        "size_t": "i64",
        "ssize_t": "i64",
        "uintptr_t": "i64",
        "intptr_t": "i64",
        "uint64_t": "i64",
        "int64_t": "i64",
        "uint32_t": "i32",
        "int32_t": "i32",
        "uint16_t": "i16",
        "int16_t": "i16",
        "uint8_t": "i8",
        "int8_t": "i8",
    }
    return mapping.get(normalized)


def target_c_return_type_matches(llvm_ir: str, function_name: str, expected_return_type: str) -> bool:
    symbol_pattern = rf'(?:"{re.escape(function_name)}"|{re.escape(function_name)})'
    match = re.search(
        rf"(?m)^define\b[^\n]*\s+([A-Za-z_]\w*)\s+@{symbol_pattern}\s*\(",
        llvm_ir,
    )
    llvm_expected = llvm_return_type_for_c(expected_return_type)
    if llvm_expected is None:
        # Struct-by-value or unusual project scalar: do not pretend we can
        # prove the exact LLVM ABI here. Parameter ABI validation still runs.
        return True
    return bool(match and match.group(1) == llvm_expected)

def resolve_target_symbol(
    llvm_ir: str,
    signature: FunctionSignature,
    language: str,
    source_path: Path,
) -> str:
    return signature.function_name if llvm_function_exists(llvm_ir, signature.function_name) else ""


def discover_llvm_extract() -> str:
    direct = shutil.which(LLVM_EXTRACT)
    if direct:
        return direct
    compiler_path = shutil.which(CLANG)
    if compiler_path:
        compiler_dir = Path(compiler_path).resolve().parent
        names = ["llvm-extract.exe", "llvm-extract"] if platform.system() == "Windows" else ["llvm-extract"]
        for name in names:
            candidate = compiler_dir / name
            if candidate.is_file():
                return str(candidate)
    return ""


def definition_header_to_declaration(header: str) -> str:
    declaration = re.sub(r"^define\s+", "declare ", header.strip(), count=1)
    declaration = declaration.rsplit("{", 1)[0].rstrip()
    declaration = re.sub(
        r"\b(?:private|internal|available_externally|linkonce|linkonce_odr|"
        r"weak|weak_odr|common|appending|extern_weak)\b\s*",
        "",
        declaration,
    )
    declaration = re.sub(r"\s+comdat(?:\([^)]*\))?", "", declaration)
    declaration = re.sub(r"\s+section\s+\"[^\"]*\"", "", declaration)
    declaration = re.sub(r"\s+align\s+\d+", "", declaration)
    declaration = re.sub(r"\s+!dbg\s+!\d+", "", declaration)
    declaration = re.sub(r"\s+", " ", declaration).strip()
    return declaration


def extract_target_ir_text(llvm_ir: str, target_symbol: str) -> str:
    """Remove unrelated function bodies when llvm-extract is unavailable.

    Referenced helper definitions are retained as declarations. Type, global,
    attribute and metadata records are kept so the resulting module remains
    parseable by Clang.
    """
    lines = llvm_ir.splitlines()
    records = llvm_definition_records(llvm_ir)
    target_record = next((record for record in records if record[0] == target_symbol), None)
    if target_record is None:
        raise ValueError(f"target function not emitted: {target_symbol}")

    _symbol, _header, target_start, target_end = target_record
    target_text = "\n".join(lines[target_start:target_end + 1])
    referenced = {
        quoted or plain
        for quoted, plain in re.findall(
            r'@(?:"([^"]+)"|([A-Za-z0-9_.$?@-]+))',
            target_text,
        )
        if (quoted or plain) != target_symbol
    }

    starts = {start: (symbol, header, end) for symbol, header, start, end in records}
    output: list[str] = []
    index = 0
    while index < len(lines):
        record = starts.get(index)
        if record is None:
            output.append(lines[index])
            index += 1
            continue
        symbol, header, end = record
        if symbol == target_symbol:
            output.extend(lines[index:end + 1])
        elif symbol in referenced:
            output.append(definition_header_to_declaration(header))
        index = end + 1

    cleaned = "\n".join(output).strip() + "\n"
    if len(llvm_definition_records(cleaned)) != 1:
        raise ValueError("target-only extraction did not produce exactly one function")
    return cleaned


def extract_target_function(input_ir: Path, output_ir: Path, target_symbol: str) -> subprocess.CompletedProcess[str]:
    llvm_extract = discover_llvm_extract()
    if llvm_extract:
        return subprocess.run(
            [
                llvm_extract,
                f"--func={target_symbol}",
                str(input_ir),
                "-S",
                "-o",
                str(output_ir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            check=False,
        )

    try:
        llvm_ir = input_ir.read_text(encoding="utf-8", errors="replace")
        output_ir.write_text(
            extract_target_ir_text(llvm_ir, target_symbol),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=["python-target-extractor", target_symbol],
            returncode=0,
            stdout="",
            stderr="",
        )
    except (OSError, ValueError) as exc:
        return subprocess.CompletedProcess(
            args=["python-target-extractor", target_symbol],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )


def llvm_object_command(language: str, ir_path: Path, object_path: Path) -> list[str]:
    return [
        CLANG,
        "-c",
        "-fno-color-diagnostics",
        str(ir_path),
        "-o", str(object_path),
    ]



def detect_source_warning_codes(stderr: str) -> list[str]:
    """Return stable warning labels originating from the PrimeVul function."""
    codes: set[str] = set()
    for line in stderr.splitlines():
        if "warning:" not in line or "primevul_input." not in line:
            continue
        low = line.lower()
        if "does not return a value" in low or "-wreturn-type" in low:
            codes.add("return-type")
        elif "pointer-sign" in low or "pointers to integer types" in low:
            codes.add("pointer-sign")
        elif "pointer to void" in low and "arithmetic" in low:
            codes.add("gnu-pointer-arith")
        elif "zero size array" in low or "zero-length-array" in low:
            codes.add("zero-length-array")
        else:
            codes.add("compiler-warning")
    return sorted(codes)


def _llvm_definition_parameters(llvm_ir: str, target_symbol: str) -> list[str]:
    record = next(
        (item for item in llvm_definition_records(llvm_ir) if item[0] == target_symbol),
        None,
    )
    if record is None:
        return []
    header = record[1]
    marker_plain = f"@{target_symbol}("
    marker_quoted = f'@"{target_symbol}"('
    position = header.find(marker_plain)
    marker = marker_plain
    if position < 0:
        position = header.find(marker_quoted)
        marker = marker_quoted
    if position < 0:
        return []
    open_pos = position + len(marker) - 1
    close_pos = find_matching(header, open_pos)
    if close_pos < 0:
        return []
    params_text = header[open_pos + 1:close_pos].strip()
    if not params_text:
        return []
    return [part.strip() for part in split_top_level(params_text)]


def validate_c_parameter_abi(
    llvm_ir: str,
    target_symbol: str,
    analysis: Analysis,
    state: RepairState,
) -> tuple[bool, str]:
    """Verify pointer-vs-integer ABI categories for C parameters."""
    source_params = [
        part.strip()
        for part in split_top_level(analysis.signature.params_text)
        if part.strip() and part.strip() not in {"void", "..."}
    ]
    llvm_params = [
        part
        for part in _llvm_definition_parameters(llvm_ir, target_symbol)
        if part != "..."
    ]
    if len(source_params) != len(llvm_params):
        return False, (
            f"parameter count mismatch: source={len(source_params)} "
            f"llvm={len(llvm_params)}"
        )

    for index, (source_param, llvm_param) in enumerate(
        zip(source_params, llvm_params)
    ):
        info, name = parse_type_and_name(source_param)
        if info is None:
            continue
        backing = scalar_underlying_type(info.base) or ""
        expected_pointer = (
            info.pointer_depth > 0
            or info.reference
            or "[" in source_param
            or info.base in state.pointer_aliases
            or "*" in backing
        )
        actual_pointer = bool(
            re.match(r"^(?:ptr\b|[^,\s]+\*)", llvm_param)
        )

        # Only validate non-pointer scalar categories when the source type is
        # known. Unknown by-value aggregates can legitimately use ABI-specific
        # lowering and are left to Clang.
        known_scalar = (
            is_known_type(info.base)
            or scalar_underlying_type(info.base) is not None
        )
        if expected_pointer != actual_pointer and (
            expected_pointer or known_scalar
        ):
            return False, (
                f"parameter {index} ({name or '?'}) ABI mismatch: "
                f"source={source_param!r}, llvm={llvm_param!r}"
            )
    return True, ""


def compile_with_repairs(source: str, language: str) -> CompileAttempt:
    analysis = analyse_source(sanitize_for_compilation(source))
    if analysis.signature.body_start < 0:
        return CompileAttempt(False, language, stderr="no function body found", failure_status="failed_malformed_source")
    sig = analysis.signature
    repaired_return_type = ""
    if language == "c" and sig.function_name and not sig.return_prefix and not sig.owner:
        repaired_return_type = infer_missing_return_type(analysis) or ""
        if not repaired_return_type:
            return CompileAttempt(
                False,
                language,
                stderr="missing return type is ambiguous",
                failure_status="rejected_ambiguous_return_type",
            )
    state = build_initial_state(analysis, language)
    if ALLOW_SEMANTIC_MACRO_FALLBACKS:
        for macro in find_unresolved_semantic_macros(source):
            if macro in SEMANTIC_MACRO_FALLBACKS:
                state.semantic_macro_fallbacks.add(macro)
                state.note(f"semantic-macro-fallback:{macro}")
                state.mark_uncertain(f"semantic-macro:{macro}")
    last_stderr = ""
    last_round = 0
    last_translation_unit = ""

    with tempfile.TemporaryDirectory(prefix="primevul_ir_") as temp_dir:
        temp = Path(temp_dir)
        source_path = temp / "sample.c"
        ir_path = temp / "sample.ll"
        extracted_ir_path = temp / "target.ll"
        object_path = temp / ("sample.obj" if platform.system() == "Windows" else "sample.o")

        for round_index in range(MAX_REPAIR_ROUNDS + 1):
            last_round = round_index
            unresolved_fields = find_unresolved_array_fields(state)
            unresolved_variables = sorted(note.split(":", 1)[1] for note in state.notes if note.startswith("unresolved-array-variable:"))
            if unresolved_fields or unresolved_variables:
                details = ",".join(unresolved_fields + unresolved_variables)
                return CompileAttempt(
                    False,
                    language,
                    stderr=f"unresolved array element type: {details}",
                    rounds=round_index,
                    generated_stubs=state.notes,
                    failure_status="rejected_ambiguous_field_type",
                )
            try:
                translation_unit = build_translation_unit(analysis, state, language)
                last_translation_unit = translation_unit
                source_path.write_text(translation_unit, encoding="utf-8")
            except ValueError as exc:
                status = "rejected_ambiguous_return_type" if "return type" in str(exc) else "rejected_ambiguous_field_type"
                return CompileAttempt(False, language, compilable_source=last_translation_unit, stderr=str(exc), rounds=round_index, generated_stubs=state.notes, failure_status=status)
            try:
                completed = subprocess.run(
                    compiler_command(language, source_path, ir_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=TIMEOUT_SECONDS,
                    check=False,
                    cwd=temp,
                    env=compiler_environment(),
                )
            except subprocess.TimeoutExpired as exc:
                return CompileAttempt(False, language, compilable_source=translation_unit, stderr=f"compiler timeout after {TIMEOUT_SECONDS}s", rounds=round_index, generated_stubs=state.notes)
            except OSError as exc:
                return CompileAttempt(False, language, compilable_source=translation_unit, stderr=f"cannot execute compiler: {exc}", rounds=round_index, generated_stubs=state.notes)

            last_stderr = completed.stderr
            if completed.returncode == 0:
                for warning_code in detect_source_warning_codes(completed.stderr):
                    state.note(f"source-warning:{warning_code}")
            if completed.returncode == 0 and ir_path.exists():
                full_llvm_ir = ir_path.read_text(encoding="utf-8", errors="replace")
                target_symbol = resolve_target_symbol(
                    full_llvm_ir,
                    sig,
                    language,
                    source_path,
                )
                if not target_symbol:
                    return CompileAttempt(
                        False,
                        language,
                        compilable_source=translation_unit,
                        stderr="target function not emitted or could not be identified",
                        rounds=round_index,
                        generated_stubs=state.notes,
                        failure_status="rejected_target_not_emitted",
                    )

                if (
                    language == "c"
                    and repaired_return_type
                    and not target_c_return_type_matches(
                        full_llvm_ir,
                        sig.function_name,
                        repaired_return_type,
                    )
                ):
                    return CompileAttempt(
                        False,
                        language,
                        compilable_source=translation_unit,
                        stderr="repaired return type mismatch",
                        rounds=round_index,
                        generated_stubs=state.notes,
                        failure_status="rejected_ir_type_mismatch",
                    )

                if language == "c":
                    abi_ok, abi_error = validate_c_parameter_abi(
                        full_llvm_ir,
                        target_symbol,
                        analysis,
                        state,
                    )
                    if not abi_ok:
                        return CompileAttempt(
                            False,
                            language,
                            compilable_source=translation_unit,
                            stderr=abi_error,
                            rounds=round_index,
                            generated_stubs=state.notes,
                            failure_status="rejected_parameter_abi_mismatch",
                        )

                try:
                    extraction = extract_target_function(
                        ir_path,
                        extracted_ir_path,
                        target_symbol,
                    )
                except (subprocess.TimeoutExpired, OSError) as exc:
                    return CompileAttempt(
                        False,
                        language,
                        compilable_source=translation_unit,
                        stderr=f"target extraction failed: {exc}",
                        rounds=round_index,
                        generated_stubs=state.notes,
                        failure_status="rejected_target_extraction_failed",
                    )
                if extraction.returncode != 0 or not extracted_ir_path.exists():
                    return CompileAttempt(
                        False,
                        language,
                        compilable_source=translation_unit,
                        stderr=extraction.stderr or "target extraction failed",
                        rounds=round_index,
                        generated_stubs=state.notes,
                        failure_status="rejected_target_extraction_failed",
                    )

                ir_path_for_validation = extracted_ir_path
                target_ir = ir_path_for_validation.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                definition_count = len(llvm_definition_records(target_ir))
                if definition_count != MAX_TARGET_IR_FUNCTIONS:
                    return CompileAttempt(
                        False,
                        language,
                        compilable_source=translation_unit,
                        stderr=(
                            "target IR contains "
                            f"{definition_count} function definitions; expected "
                            f"{MAX_TARGET_IR_FUNCTIONS}"
                        ),
                        rounds=round_index,
                        generated_stubs=state.notes,
                        failure_status="rejected_non_target_ir_noise",
                    )

                if VALIDATE_LLVM_OBJECT:
                    try:
                        object_compile = subprocess.run(
                            llvm_object_command(
                                language,
                                ir_path_for_validation,
                                object_path,
                            ),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=TIMEOUT_SECONDS,
                            check=False,
                            cwd=temp,
                            env=compiler_environment(),
                        )
                    except subprocess.TimeoutExpired:
                        return CompileAttempt(
                            False,
                            language,
                            compilable_source=translation_unit,
                            stderr=(
                                "LLVM object compile timeout after "
                                f"{TIMEOUT_SECONDS}s"
                            ),
                            rounds=round_index,
                            generated_stubs=state.notes,
                        )
                    except OSError as exc:
                        return CompileAttempt(
                            False,
                            language,
                            compilable_source=translation_unit,
                            stderr=f"cannot execute compiler for LLVM object: {exc}",
                            rounds=round_index,
                            generated_stubs=state.notes,
                        )
                    if object_compile.returncode != 0 or not object_path.exists():
                        return CompileAttempt(
                        False,
                        language,
                        compilable_source=translation_unit,
                        stderr=compact_error(object_compile.stderr),
                            rounds=round_index,
                            generated_stubs=state.notes,
                            failure_status="rejected_invalid_extracted_ir",
                        )

                llvm_ir = target_ir
                if not KEEP_COMPILER_COMMENTS:
                    llvm_ir = re.sub(r"^; ModuleID = .*\n", "", llvm_ir)
                    llvm_ir = re.sub(r"^source_filename = .*\n", "", llvm_ir)
                return CompileAttempt(
                    True,
                    language,
                    llvm_ir=sanitize_llvm_metadata_paths(llvm_ir.strip()),
                    compilable_source=translation_unit,
                    target_symbol=target_symbol,
                    stderr=last_stderr,
                    rounds=round_index,
                    generated_stubs=finalized_state_notes(state),
                    repaired_return_type=repaired_return_type,
                )

            if round_index >= MAX_REPAIR_ROUNDS:
                break
            if not apply_diagnostic_repairs(last_stderr, analysis, state, language):
                break

    return CompileAttempt(
        False,
        language,
        compilable_source=last_translation_unit,
        stderr=compact_error(last_stderr),
        rounds=last_round,
        generated_stubs=state.notes,
    )


def make_sample_id(record: Mapping[str, Any], source: str) -> str:
    for key in ("sample_id", "idx", "id", "func_hash"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()[:20]


def sanitize_llvm_metadata_paths(llvm_ir: str) -> str:
    """Remove machine-specific temporary paths from LLVM debug metadata."""
    llvm_ir = re.sub(
        r'!DIFile\(filename: "[^"]*?([^"\\/]+)", directory: "[^"]*"\)',
        lambda m: f'!DIFile(filename: "{m.group(1)}", directory: ".")',
        llvm_ir,
    )
    return llvm_ir


def status_stub_count(status: str) -> int:
    match = re.search(r'(?:^|;)stubs=(\d+)(?:;|$)', status)
    return int(match.group(1)) if match else 0


def primevul_label(record: Mapping[str, Any]) -> int:
    """Return the PrimeVul binary label. Vulnerable samples are label 1."""
    try:
        return int(record.get("target", record.get("label", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def preprocess_record(
    record: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    """Convert one PrimeVul row to the requested Source/LLVM record format."""
    source = normalize_source(record.get("func") or record.get("source_code") or "")
    sample_id = make_sample_id(record, source)
    label = primevul_label(record)

    result: dict[str, Any] = {
        "sample_id": sample_id,
        # Keep the exact PrimeVul function in the output. Repairs are only
        # temporary scaffolding for LLVM generation.
        "source_code": source,
        "wrapped_source_code": "",
        "llvm_ir": "",
        "label": label,
        "language": "",
        "split": split,
        "ir_status": "",
        "compile_error": "",
    }

    if not source:
        result["ir_status"] = "failed_malformed_source"
        result["compile_error"] = "empty source function"
        return result
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        result["ir_status"] = "failed_source_too_large"
        result["compile_error"] = f"source exceeds {MAX_SOURCE_BYTES} bytes"
        return result
    candidates = language_candidates(record, source)
    if not candidates:
        extension = Path(str(record.get("file_name") or "")).suffix.lower()
        result["ir_status"] = (
            "rejected_cpp_file"
            if extension in DISCARD_SOURCE_EXTENSIONS
            else "rejected_non_c_file"
        )
        result["compile_error"] = f"unsupported file extension: {extension or '<missing>'}"
        return result
    unsafe_macros = find_unresolved_semantic_macros(source)
    unsupported_macros = [
        macro for macro in unsafe_macros
        if macro not in SEMANTIC_MACRO_FALLBACKS
    ]
    if unsafe_macros and (not ALLOW_SEMANTIC_MACRO_FALLBACKS or unsupported_macros):
        result["language"] = candidates[0]
        result["ir_status"] = (
            "rejected_unresolved_semantic_macro:"
            + ",".join(unsupported_macros or unsafe_macros)
        )
        result["compile_error"] = "unresolved semantic macro: " + ",".join(unsupported_macros or unsafe_macros)
        return result

    failures: list[str] = []
    for language in candidates:
        attempt = compile_with_repairs(source, language)
        if attempt.compilable_source:
            result["wrapped_source_code"] = attempt.compilable_source
        if attempt.success:
            result["language"] = language
            repairs = [
                note.removeprefix("repair:")
                for note in attempt.generated_stubs
                if note.startswith("repair:")
            ]
            uncertain = [
                note.removeprefix("uncertain:")
                for note in attempt.generated_stubs
                if note.startswith("uncertain:")
            ]
            source_warnings = [
                note.removeprefix("source-warning:")
                for note in attempt.generated_stubs
                if note.startswith("source-warning:")
            ]
            stub_notes = [
                note
                for note in attempt.generated_stubs
                if not note.startswith(("repair:", "uncertain:", "source-warning:"))
            ]
            stub_count = len(stub_notes)
            if stub_count > MAX_STUB_COUNT:
                result["llvm_ir"] = ""
                result["ir_status"] = (
                    f"rejected_excessive_synthetic_stubs;stubs={stub_count};"
                    f"limit={MAX_STUB_COUNT}"
                )
                result["compile_error"] = f"too many synthetic stubs: {stub_count} > {MAX_STUB_COUNT}"
                return result
            warning_suffix = (
                f";warnings={','.join(sorted(set(source_warnings)))}"
                if source_warnings
                else ""
            )

            if uncertain and not ALLOW_LOW_CONFIDENCE_STUBS:
                result["llvm_ir"] = ""
                result["ir_status"] = (
                    "rejected_low_confidence_stubs:"
                    + ",".join(sorted(set(uncertain))[:8])
                )
                result["compile_error"] = "low confidence synthetic stubs: " + ",".join(sorted(set(uncertain))[:8])
                return result

            result["llvm_ir"] = attempt.llvm_ir
            result["target_symbol"] = attempt.target_symbol
            result["source_view_code"] = source
            if (
                attempt.rounds == 0
                and not repairs
                and not uncertain
                and stub_count == 0
            ):
                result["ir_status"] = (
                    "success_clean_verified" + warning_suffix
                )
            elif repairs and stub_count == 0 and not uncertain:
                result["ir_status"] = (
                    f"success_repaired_verified;repair={','.join(repairs)};"
                    f"rounds={attempt.rounds};stubs=0{warning_suffix}"
                )
            elif uncertain:
                result["ir_status"] = (
                    f"success_stubbed_low_confidence;rounds={attempt.rounds};"
                    f"stubs={stub_count};uncertain={len(set(uncertain))}"
                    f"{warning_suffix}"
                )
            else:
                result["ir_status"] = (
                    f"success_stubbed_medium_confidence;rounds={attempt.rounds};"
                    f"stubs={stub_count}{warning_suffix}"
                )
            return result
        failures.append(f"{language}:{compact_error(attempt.stderr)}")
        if attempt.failure_status:
            result["language"] = language
            result["ir_status"] = attempt.failure_status
            result["compile_error"] = attempt.stderr
            return result

    result["language"] = candidates[0]
    result["ir_status"] = "failed_compile;" + " || ".join(failures)
    result["compile_error"] = " || ".join(failures)
    return result


# ---------------------------------------------------------------------------
# Single-function, local dataset, and run orchestration
# ---------------------------------------------------------------------------


def read_jsonl_records(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"non-object JSONL record at {path}:{line_number}")
            yield payload


def compile_c_function(
    source: str,
    *,
    file_name: str = "input.c",
    sample_id: str = "single_function",
    label: int = 0,
    split: str = "single",
) -> dict[str, Any]:
    """Compile one C function for scanner/UI use."""
    record = {
        "idx": sample_id,
        "file_name": file_name,
        "func": source,
        "target": label,
    }
    return preprocess_record(record, split)


def is_successful_llvm_record(record: Mapping[str, Any]) -> bool:
    """Return True only for records that contain emitted LLVM and a success status."""
    llvm_ir = str(record.get("llvm_ir") or "").strip()
    if not llvm_ir:
        return False
    status = str(record.get("ir_status") or "").strip()
    if not status:
        return True
    return (
        status.startswith("success")
        and bool(llvm_ir)
    )


def compact_compiled_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the training JSONL shape for a successfully compiled row."""
    return {
        "sample_id": str(record.get("sample_id", "")),
        "source_code": str(record.get("source_code", "")),
        "wrapped_source_code": str(record.get("wrapped_source_code", "")),
        "llvm_ir": str(record.get("llvm_ir", "")),
        "label": int(record.get("label", 0) or 0),
        "language": str(record.get("language", "")),
        "split": str(record.get("split", "")),
    }


def _preprocess_worker(
    payload: tuple[int, dict[str, Any], str],
) -> tuple[int, dict[str, Any]]:
    source_index, record, split = payload
    return source_index, preprocess_record(record, split)


def process_record_stream(
    indexed_records: Iterable[tuple[int, dict[str, Any]]],
    workers: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    payloads = (
        (
            source_index,
            record,
            str(record.get("split") or "processed"),
        )
        for source_index, record in indexed_records
    )
    if workers <= 1:
        for payload in payloads:
            yield _preprocess_worker(payload)
        return

    main_file = str(getattr(sys.modules.get("__main__"), "__file__", ""))
    if not main_file or main_file.endswith("<stdin>"):
        for payload in payloads:
            yield _preprocess_worker(payload)
        return

    import multiprocessing as mp
    context = mp.get_context("spawn" if os.name == "nt" else "fork")
    with context.Pool(processes=workers) as pool:
        yield from pool.imap(_preprocess_worker, payloads, chunksize=1)

def compile_processed_file(input_path: Path, output_path: Path, workers: int) -> dict[str, Any]:
    """Compile the processed PrimeVul JSONL and write accepted LLVM rows."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:  # type: ignore
            return iterable

    stats: dict[str, Any] = {
        "seen": 0,
        "written": 0,
        "success": 0,
        "failed": 0,
    }

    indexed_records = (
        (source_index, dict(record))
        for source_index, record in enumerate(read_jsonl_records(input_path), start=1)
    )
    processed_stream = process_record_stream(indexed_records, max(1, workers))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        progress = tqdm(
            processed_stream,
            desc="PrimeVul processed",
            unit="function",
        )
        try:
            for _source_index, processed in progress:
                stats["seen"] += 1
                success = is_successful_llvm_record(processed)
                stats["success" if success else "failed"] += 1
                if success:
                    output.write(json.dumps(compact_compiled_record(processed), ensure_ascii=False) + "\n")
                    stats["written"] += 1
        finally:
            close = getattr(progress, "close", None)
            if callable(close):
                close()
            close_stream = getattr(processed_stream, "close", None)
            if callable(close_stream):
                close_stream()

    return stats


def run() -> dict[str, Any]:
    if WORKERS < 1:
        raise SystemExit("WORKERS must be at least 1")
    if not INPUT_JSONL.exists():
        raise SystemExit(f"processed input not found: {INPUT_JSONL}")
    if shutil.which(CLANG) is None:
        raise SystemExit(f"C compiler not found: {CLANG}")

    stats = compile_processed_file(INPUT_JSONL, OUTPUT_JSONL, WORKERS)
    return {
        "input": str(INPUT_JSONL),
        "output": str(OUTPUT_JSONL),
        "preprocessing": stats,
    }


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
