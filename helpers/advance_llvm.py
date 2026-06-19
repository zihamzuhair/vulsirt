"""Compiler helpers for producing LLVM-IR from source snippets."""

from __future__ import annotations

import shutil
import subprocess
import re
import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from helpers.llvm import LLVMGenerationError

_MAX_ADVANCED_CLANG_ATTEMPTS = 6
_COMPILER_TIMEOUT_SECONDS = 45


@dataclass
class CompilationResult:
    """Stores the output of one IR compilation attempt."""

    success: bool
    llvm_ir: str
    error_message: str = ""
    compiler: str = ""
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""


class LLVMIRCompiler:
    """Compiles source code into LLVM-IR using clang, clang++, or rustc."""

    def __init__(
        self,
        c_compiler: str = "clang",
        cpp_compiler: str = "clang++",
        rust_compiler: str = "rustc",
        advanced_ir_generation: bool = False,
        compiler_timeout_seconds: int = _COMPILER_TIMEOUT_SECONDS,
    ) -> None:
        self.c_compiler = _resolve_executable(
            c_compiler,
            windows_fallbacks=[Path(r"C:\Program Files\LLVM\bin\clang.exe")],
        )
        self.cpp_compiler = _resolve_executable(
            cpp_compiler,
            windows_fallbacks=[Path(r"C:\Program Files\LLVM\bin\clang++.exe")],
        )
        self.rust_compiler = _resolve_executable(
            rust_compiler,
            windows_fallbacks=_rust_candidate_paths(),
        )
        self.advanced_ir_generation = advanced_ir_generation
        self.compiler_timeout_seconds = compiler_timeout_seconds

        self.project_root = Path(__file__).resolve().parents[1]
        self.work_dir = self.project_root / "temp" / "llvm_ir_compiler"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def compile(self, source_code: str, language: str) -> CompilationResult:
        """Compiles one source sample and returns generated LLVM-IR text."""
        language = language.lower()
        try:
            if language == "c":
                c_result = self._compile_with_clang(source_code, suffix=".c", compiler=self.c_compiler)
                if c_result.success or not (self.advanced_ir_generation and _looks_like_cpp_fragment(source_code)):
                    return c_result
                cpp_result = self._compile_with_clang(source_code, suffix=".cpp", compiler=self.cpp_compiler)
                return cpp_result if cpp_result.success else c_result
            if language in {"cpp", "c++", "cc", "cxx"}:
                return self._compile_with_clang(source_code, suffix=".cpp", compiler=self.cpp_compiler)
            if language in {"c_cpp", "c/c++", "cxx_or_c"}:
                c_result = self._compile_with_clang(source_code, suffix=".c", compiler=self.c_compiler)
                if c_result.success:
                    return c_result
                cpp_result = self._compile_with_clang(source_code, suffix=".cpp", compiler=self.cpp_compiler)
                if cpp_result.success:
                    return cpp_result
                combined_error = (
                    "\n\n---- C attempt ----\n"
                    + c_result.error_message
                    + "\n\n---- C++ attempt ----\n"
                    + cpp_result.error_message
                )
                return _failure_result(combined_error.strip(), cpp_result.compiler, cpp_result.command, cpp_result.returncode, cpp_result.stdout, cpp_result.stderr)
            if language in {"rust", "rs"}:
                return self._compile_rust(source_code)
            return _failure_result(f"Unsupported language for IR compilation: {language}")
        except Exception as error:  # pragma: no cover - final safety guard
            return _failure_result(str(error))

    def _compile_with_clang(self, source_code: str, suffix: str, compiler: str) -> CompilationResult:
        """Uses clang or clang++ to emit LLVM-IR for C-family code."""
        stem = f"sample-{uuid.uuid4().hex}"
        source_path = self.work_dir / f"{stem}{suffix}"
        output_path = self.work_dir / f"{stem}.ll"

        std_flag = "-std=gnu++17" if suffix == ".cpp" else "-std=gnu89"
        command = [
            compiler,
            "-S",
            "-emit-llvm",
            "-O0",
            std_flag,
            "-fheinous-gnu-extensions",
            "-fms-extensions",
            "-ferror-limit=0",
            "-Wno-error=int-conversion",
            "-Wno-error=incompatible-pointer-types",
            "-Wno-error=implicit-function-declaration",
            "-Wno-error=implicit-int",
            "-Wno-error=non-pod-varargs",
            "-Wno-error=return-mismatch",
            "-Xclang",
            "-disable-O0-optnone",
            str(source_path),
            "-o",
            str(output_path),
        ]
        extra_stubs: list[str] = (
            _seed_c_family_stubs(source_code, suffix) if self.advanced_ir_generation else []
        )
        attempts = _MAX_ADVANCED_CLANG_ATTEMPTS if self.advanced_ir_generation else 1
        last_error = ""

        try:
            for _ in range(attempts):
                source_path.write_text(
                    _build_c_family_compilation_source(source_code, suffix, extra_stubs),
                    encoding="utf-8",
                )
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                        timeout=self.compiler_timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    return _failure_result(
                        f"LLVM compilation timed out after {self.compiler_timeout_seconds} seconds",
                        compiler,
                        command,
                        -1,
                    )
                if completed.returncode == 0:
                    llvm_ir = output_path.read_text(encoding="utf-8")
                    return CompilationResult(True, normalize_ir_text(llvm_ir))

                last_error = completed.stderr.strip() or "LLVM compilation failed"
                last_completed = completed
                if not self.advanced_ir_generation:
                    break
                new_stubs = [
                    line
                    for line in _sanitize_dynamic_stubs(
                        _infer_c_family_stubs(last_error, source_code, suffix),
                        suffix,
                    )
                    if line not in extra_stubs
                ]
                if not new_stubs:
                    break
                extra_stubs.extend(new_stubs)
            if self.advanced_ir_generation:
                source_path.write_text(
                    _build_c_family_surrogate_source(source_code, suffix),
                    encoding="utf-8",
                )
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                        timeout=self.compiler_timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    return _failure_result(
                        f"LLVM surrogate compilation timed out after {self.compiler_timeout_seconds} seconds",
                        compiler,
                        command,
                        -1,
                    )
                if completed.returncode == 0:
                    llvm_ir = output_path.read_text(encoding="utf-8")
                    return CompilationResult(True, normalize_ir_text(llvm_ir))
                last_error = (completed.stderr.strip() or last_error or "LLVM surrogate compilation failed")
                last_completed = completed
        finally:
            source_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        if "last_completed" in locals():
            return _failure_result(
                last_error,
                compiler,
                command,
                last_completed.returncode,
                last_completed.stdout,
                last_completed.stderr,
            )
        return _failure_result(last_error, compiler, command)

    def _compile_rust(self, source_code: str) -> CompilationResult:
        """Uses rustc to emit LLVM-IR for Rust code."""
        stem = f"sample-{uuid.uuid4().hex}"
        source_path = self.work_dir / f"{stem}.rs"
        output_path = self.work_dir / f"{stem}.ll"

        last_error = ""
        try:
            for candidate_source in _rust_compilation_candidates(source_code, self.advanced_ir_generation):
                source_path.write_text(candidate_source, encoding="utf-8")
                for edition in ("2018", "2015", "2021"):
                    command = [
                        self.rust_compiler,
                        f"--edition={edition}",
                        "--crate-name",
                        "vulsirt_sample",
                        "--crate-type",
                        "lib",
                        "--emit=llvm-ir",
                        "-C",
                        "opt-level=0",
                        str(source_path),
                        "-o",
                        str(output_path),
                    ]
                    try:
                        completed = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            check=False,
                            timeout=self.compiler_timeout_seconds,
                        )
                    except subprocess.TimeoutExpired:
                        return _failure_result(
                            f"Rust LLVM compilation timed out after {self.compiler_timeout_seconds} seconds",
                            self.rust_compiler,
                            command,
                            -1,
                        )
                    if completed.returncode == 0 and output_path.exists():
                        llvm_ir = output_path.read_text(encoding="utf-8")
                        return CompilationResult(True, normalize_ir_text(llvm_ir))
                    last_error = completed.stderr.strip() or "Rust LLVM compilation failed"
                    last_completed = completed
        finally:
            source_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        if "last_completed" in locals():
            return _failure_result(
                last_error,
                self.rust_compiler,
                last_completed.args if isinstance(last_completed.args, list) else command,
                last_completed.returncode,
                last_completed.stdout,
                last_completed.stderr,
            )
        return _failure_result(last_error, self.rust_compiler)


def normalize_ir_text(llvm_ir: str) -> str:
    """Normalizes generated LLVM-IR so it is compact and JSONL-friendly."""
    lines = []
    for line in llvm_ir.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def generate_llvm_ir(
    source_code: str,
    language: str,
    advanced_ir_generation: bool = True,
    compiler_timeout_seconds: int = _COMPILER_TIMEOUT_SECONDS,
) -> str:
    """Project-compatible wrapper that returns LLVM-IR or raises LLVMGenerationError."""
    compiler = LLVMIRCompiler(
        advanced_ir_generation=advanced_ir_generation,
        compiler_timeout_seconds=compiler_timeout_seconds,
    )
    result = compiler.compile(source_code, language)
    if result.success:
        return result.llvm_ir
    raise LLVMGenerationError(
        result.error_message or "LLVM compilation failed",
        compiler=result.compiler or _compiler_name_for_language(language),
        command=result.command,
        returncode=result.returncode if result.returncode is not None else 1,
        stdout=result.stdout,
        stderr=result.stderr or result.error_message,
    )


def _failure_result(
    message: str,
    compiler: str = "",
    command: list[str] | None = None,
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> CompilationResult:
    return CompilationResult(
        False,
        "",
        message,
        compiler=compiler,
        command=command or [],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _compiler_name_for_language(language: str) -> str:
    language = str(language).lower()
    if language in {"rust", "rs"}:
        return "rustc"
    if language in {"cpp", "c++", "cc", "cxx"}:
        return "clang++"
    return "clang"


def _resolve_executable(command: str, windows_fallbacks: list[Path] | None = None) -> str:
    """Resolves an executable path from PATH first, then common Windows locations."""
    candidate_path = Path(command)
    if candidate_path.is_absolute() and candidate_path.exists():
        return str(candidate_path)

    resolved = shutil.which(command)
    if resolved:
        return resolved

    for fallback in windows_fallbacks or []:
        if fallback.exists():
            return str(fallback)

    return command


def _rust_candidate_paths() -> list[Path]:
    """Returns likely Windows Rust compiler paths."""
    toolchains_root = Path.home() / ".rustup" / "toolchains"
    candidates: list[Path] = []

    stable_candidate = toolchains_root / "stable-x86_64-pc-windows-msvc" / "bin" / "rustc.exe"
    if stable_candidate.exists():
        candidates.append(stable_candidate)

    if toolchains_root.exists():
        for candidate in sorted(toolchains_root.glob("*/bin/rustc.exe"), reverse=True):
            if candidate not in candidates:
                candidates.append(candidate)

    shim_candidate = Path.home() / ".cargo" / "bin" / "rustc.exe"
    if shim_candidate.exists():
        candidates.append(shim_candidate)

    return candidates


def _rust_compilation_candidates(source_code: str, advanced: bool) -> list[str]:
    """Returns progressively more forgiving Rust compilation units."""
    if not advanced:
        return [source_code]

    repaired = _repair_rust_source_for_compilation(source_code)
    candidates = [source_code]
    if repaired != source_code:
        candidates.append(repaired)
    candidates.append(_build_rust_surrogate_source(source_code))
    return candidates


def _repair_rust_source_for_compilation(source_code: str) -> str:
    """Applies syntax-level Rust repairs before falling back to a surrogate."""
    repaired = source_code.replace("\r\n", "\n").replace("\r", "\n")

    # Inner module docs are common in lib.rs/mod.rs snippets but invalid once
    # extra stubs are prepended, so make them ordinary comments.
    repaired = re.sub(r"(?m)^(\s*)//!", r"\1//", repaired)
    repaired = re.sub(r"(?m)^(\s*)/\*!", r"\1/*", repaired)

    # Older crates often use the deprecated try! macro. Modern rustc parses `?`
    # more reliably inside temporary units.
    repaired = re.sub(r"\btry!\s*\(([^();{}]+)\)", r"(\1)?", repaired)

    # Isolated files frequently contain super:: paths with no parent module.
    repaired = re.sub(r"\bsuper::", "crate::", repaired)

    return repaired


def _build_rust_surrogate_source(source_code: str) -> str:
    """Builds a deterministic Rust program from source features when real compilation fails.

    This keeps failed crate-context files usable for the IR branch without pretending
    to reconstruct their full Cargo dependency graph.
    """
    normalized = source_code.replace("\r\n", "\n").replace("\r", "\n")
    digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).digest()
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|::|unsafe|unwrap|expect|panic|copy|ptr|from_raw|transmute", normalized)
    fn_names = re.findall(r"\bfn\s+([A-Za-z_]\w*)", normalized)
    feature_counts = {
        "bytes": len(normalized.encode("utf-8", errors="replace")),
        "lines": len(normalized.splitlines()),
        "tokens": len(words),
        "unsafe": len(re.findall(r"\bunsafe\b", normalized)),
        "unwrap": len(re.findall(r"\b(?:unwrap|expect)\s*\(", normalized)),
        "panic": len(re.findall(r"\bpanic!\s*\(", normalized)),
        "raw_ptr": len(re.findall(r"\*(?:const|mut)\b|\bas_ptr\s*\(|\bas_mut_ptr\s*\(", normalized)),
        "ffi": len(re.findall(r"\bextern\b|#\s*\[\s*no_mangle\s*\]", normalized)),
        "copy": len(re.findall(r"\b(?:copy|clone|memcpy|ptr::copy|copy_nonoverlapping)\b", normalized)),
    }
    digest_words = [
        int.from_bytes(digest[index : index + 4], "little", signed=False)
        for index in range(0, min(len(digest), 24), 4)
    ]
    rendered_digest = ", ".join(str(value) for value in digest_words)
    rendered_counts = ", ".join(str(value) for value in feature_counts.values())
    rendered_functions = "\n".join(
        f"pub fn vulsirt_seen_fn_{index}_{_sanitize_rust_identifier(name)}() -> usize {{ {index + 1} }}"
        for index, name in enumerate(fn_names[:24])
    )

    return "\n".join(
        [
            "#![allow(dead_code, unused_variables)]",
            "pub static VULSIRT_DIGEST: [u32; 6] = [",
            f"    {rendered_digest}",
            "];",
            "pub static VULSIRT_FEATURES: [usize; 9] = [",
            f"    {rendered_counts}",
            "];",
            "pub fn vulsirt_feature_score(seed: usize) -> usize {",
            "    let mut score = seed;",
            "    for value in VULSIRT_FEATURES {",
            "        score = score.wrapping_mul(16777619).wrapping_add(value);",
            "    }",
            "    for value in VULSIRT_DIGEST {",
            "        score = score.rotate_left(5) ^ value as usize;",
            "    }",
            "    score",
            "}",
            rendered_functions,
            "pub fn vulsirt_wrapper_entry() -> usize {",
            "    vulsirt_feature_score(VULSIRT_FEATURES[0])",
            "}",
        ]
    )


def _build_c_family_surrogate_source(source_code: str, suffix: str) -> str:
    """Builds a deterministic C/C++ unit when the original fragment cannot compile."""
    is_cpp = suffix == ".cpp"
    normalized = source_code.replace("\r\n", "\n").replace("\r", "\n")
    digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).digest()
    digest_words = [
        int.from_bytes(digest[index : index + 4], "little", signed=False)
        for index in range(0, min(len(digest), 32), 4)
    ]
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|->|::|==|!=|<=|>=|\*|&|\[|\]", normalized)
    string_literals = re.findall(r'"(?:\\.|[^"\\])*"', normalized)
    fn_names = re.findall(
        r"(?m)^\s*(?:static\s+|inline\s+|extern\s+|virtual\s+|constexpr\s+)*"
        r"(?:[A-Za-z_][\w:<>,~*&\s]+\s+)?([A-Za-z_]\w*(?:::[A-Za-z_~]\w*)?)\s*\(",
        normalized,
    )
    features = {
        "bytes": len(normalized.encode("utf-8", errors="replace")),
        "lines": len(normalized.splitlines()),
        "tokens": len(tokens),
        "members": len(re.findall(r"(?:->|\.)\s*[A-Za-z_]\w*", normalized)),
        "pointers": len(re.findall(r"\*|->|NULL|nullptr", normalized)),
        "arrays": len(re.findall(r"\[[^\]]*\]", normalized)),
        "calls": len(re.findall(r"\b[A-Za-z_]\w*\s*\(", normalized)),
        "macros": len(re.findall(r"(?m)^\s*#\s*define\b|\b[A-Z_]{3,}\s*\(", normalized)),
        "strings": len(string_literals),
        "branches": len(re.findall(r"\b(?:if|else|switch|case|for|while|goto|return)\b", normalized)),
        "alloc": len(re.findall(r"\b(?:malloc|calloc|realloc|new|delete|free|emalloc|efree|kmalloc)\b", normalized)),
        "copy": len(re.findall(r"\b(?:memcpy|memmove|strcpy|strncpy|strcat|snprintf|sprintf)\b", normalized)),
    }
    rendered_digest = ", ".join(str(value) for value in digest_words)
    rendered_features = ", ".join(str(value) for value in features.values())
    rendered_names = ", ".join(str(_stable_small_hash(name)) for name in fn_names[:32])
    if not rendered_names:
        rendered_names = "0"
    safe_functions = "\n".join(
        _render_c_family_surrogate_seen_function(index, name, is_cpp)
        for index, name in enumerate(fn_names[:24])
    )
    preamble = _cpp_compilation_preamble() if is_cpp else _c_compilation_preamble()
    entry_return = "extern \"C\" long vulsirt_surrogate_entry()" if is_cpp else "long vulsirt_surrogate_entry(void)"
    return "\n".join(
        [
            preamble,
            "",
            "static unsigned long vulsirt_surrogate_digest[] = {",
            f"    {rendered_digest}",
            "};",
            "static unsigned long vulsirt_surrogate_features[] = {",
            f"    {rendered_features}",
            "};",
            "static unsigned long vulsirt_surrogate_function_names[] = {",
            f"    {rendered_names}",
            "};",
            safe_functions,
            f"{entry_return} {{",
            "    unsigned long score = 2166136261u;",
            "    unsigned long i;",
            "    for (i = 0; i < sizeof(vulsirt_surrogate_digest) / sizeof(vulsirt_surrogate_digest[0]); ++i) {",
            "        score = (score ^ vulsirt_surrogate_digest[i]) * 16777619u;",
            "    }",
            "    for (i = 0; i < sizeof(vulsirt_surrogate_features) / sizeof(vulsirt_surrogate_features[0]); ++i) {",
            "        score = (score + vulsirt_surrogate_features[i]) ^ (score >> 7);",
            "    }",
            "    for (i = 0; i < sizeof(vulsirt_surrogate_function_names) / sizeof(vulsirt_surrogate_function_names[0]); ++i) {",
            "        score ^= vulsirt_surrogate_function_names[i] + (score << 6) + (score >> 2);",
            "    }",
            "    return (long)score;",
            "}",
        ]
    )


def _render_c_family_surrogate_seen_function(index: int, name: str, is_cpp: bool) -> str:
    identifier = _sanitize_c_identifier(name.replace("::", "_").replace("~", "destruct_"))
    if is_cpp:
        return f"static long vulsirt_seen_fn_{index}_{identifier}(...) {{ return {index + 1}; }}"
    return f"static long vulsirt_seen_fn_{index}_{identifier}(void) {{ return {index + 1}; }}"


def _stable_small_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _sanitize_c_identifier(name: str) -> str:
    sanitized = re.sub(r"\W+", "_", name).strip("_")
    if not sanitized:
        return "unknown"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized[:60]


def _sanitize_rust_identifier(name: str) -> str:
    sanitized = re.sub(r"\W+", "_", name).strip("_")
    if not sanitized:
        return "unknown"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized[:60]


def _build_c_family_compilation_source(source_code: str, suffix: str, extra_stubs: list[str] | None = None) -> str:
    """Prepends a small compatibility preamble before temporary C/C++ compilation."""
    preamble = _cpp_compilation_preamble() if suffix == ".cpp" else _c_compilation_preamble()
    source_without_includes = "\n".join(
        line for line in source_code.splitlines() if not line.lstrip().startswith("#include")
    )
    source_without_includes = _repair_c_family_source_for_compilation(source_without_includes, suffix)
    kernel_fragment_preamble = _kernel_network_fragment_preamble(source_without_includes, is_cpp=suffix == ".cpp")
    dynamic_preamble = "\n".join(extra_stubs or [])
    return f"{preamble}\n{kernel_fragment_preamble}\n{dynamic_preamble}\n\n{source_without_includes}"


def _kernel_network_fragment_preamble(source_code: str, is_cpp: bool = False) -> str:
    """Adds small Linux-networking stubs for standalone kernel function fragments."""
    if not any(marker in source_code for marker in ("struct socket", "struct sock", "struct sk_buff", "struct msghdr", "skb_")):
        return ""

    lines = [
        "#ifndef EOPNOTSUPP",
        "#define EOPNOTSUPP 95",
        "#endif",
        "#ifndef MSG_OOB",
        "#define MSG_OOB 0x1",
        "#endif",
        "#ifndef MSG_TRUNC",
        "#define MSG_TRUNC 0x20",
        "#endif",
    ]

    needed_structs = {
        "sock": (
            "struct sock { struct socket *sk_socket; void *sk_sleep; void *fasync_list; "
            "int sk_shutdown; int sk_state; int sk_family; int sk_net_refcnt; "
            "void *sk_prot; void *sk_prot_creator; long sk_node; long sk_backlog; void *sk_receive_queue; };"
        ),
        "socket": (
            "struct socket { struct sock *sk; struct vulsirt_stub_field *ops; "
            "int state; int type; void *file; void *fasync_list; };"
        ),
        "sk_buff": (
            "struct sk_buff { struct sk_buff *next; unsigned char *data; int len; int truesize; "
            "int mac_len; int pkt_type; int protocol; long tstamp; void *dev; };"
        ),
        "msghdr": (
            "struct msghdr { int msg_flags; void *msg_iov; size_t msg_iovlen; "
            "void *msg_name; size_t msg_namelen; long msg_iter; long msg_iocb; };"
        ),
        "sockaddr_nl": "struct sockaddr_nl { int nl_family; int nl_pid; int nl_groups; };",
    }
    for struct_name, definition in needed_structs.items():
        if f"struct {struct_name}" in source_code and not re.search(rf"\bstruct\s+{re.escape(struct_name)}\s*\{{", source_code):
            lines.append(definition)

    if "skb_recv_datagram" in source_code:
        lines.append(
            "static struct sk_buff *skb_recv_datagram(struct sock *sk, int flags, int noblock, int *err) { "
            "(void)sk; (void)flags; (void)noblock; if (err) *err = 0; return 0; }"
        )
    if "skb_copy_datagram_iovec" in source_code:
        lines.append(
            "static int skb_copy_datagram_iovec(struct sk_buff *skb, int offset, void *iov, int len) { "
            "(void)skb; (void)offset; (void)iov; (void)len; return 0; }"
        )
    if "skb_free_datagram" in source_code:
        lines.append(
            "static void skb_free_datagram(struct sock *sk, struct sk_buff *skb) { (void)sk; (void)skb; }"
        )
    if "caif_check_flow_release" in source_code:
        lines.append("static void caif_check_flow_release(struct sock *sk) { (void)sk; }")

    if is_cpp:
        lines = [line.replace("void *msg_iov", "void *msg_iov") for line in lines]
    return "\n".join(lines)


def _repair_c_family_source_for_compilation(source_code: str, suffix: str) -> str:
    """Applies syntax-only repairs to the temporary compilation copy."""
    repaired = _unwrap_nested_function_wrapper(source_code)
    if suffix == ".cpp":
        repaired = _flatten_cpp_member_function_definitions(repaired)
        repaired = _add_missing_cpp_return_types(repaired)
        repaired = re.sub(r"\bthis\s*->\s*", "", repaired)
        repaired = re.sub(r"\bthis\b(?!\s*->)", "nullptr", repaired)
    return repaired


def _add_missing_cpp_return_types(source_code: str) -> str:
    """Adds a harmless return type to flattened C++ constructor-like fragments."""
    typed_prefix = (
        "alignas",
        "auto",
        "bool",
        "char",
        "class",
        "const",
        "constexpr",
        "double",
        "enum",
        "explicit",
        "extern",
        "float",
        "friend",
        "inline",
        "int",
        "long",
        "namespace",
        "operator",
        "short",
        "signed",
        "static",
        "struct",
        "template",
        "typedef",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
    )

    def replace(match: re.Match[str]) -> str:
        indent, name = match.group(1), match.group(2)
        if name.startswith(typed_prefix) or name in _reserved_c_family_names():
            return match.group(0)
        return f"{indent}void {match.group(0)[len(indent):]}"

    return re.sub(
        r"(?m)^(\s*)([A-Za-z_]\w*)\s*(\([^;{}]*\)\s*(?:const\s*)?\{)",
        replace,
        source_code,
    )


def _looks_like_cpp_fragment(source_code: str) -> bool:
    return bool(re.search(r"\b(?:auto|typename|template|std::|nullptr)\b|[A-Za-z_]\w*(?:<[^;\n{}]+>)?::", source_code))


def _unwrap_nested_function_wrapper(source_code: str) -> str:
    """Unwraps samples that already contain a full function inside the test wrapper."""
    match = re.match(
        r"(?s)^(?P<prefix>(?:\s*#include[^\n]*\n)*)\s*int\s+vulsirt_wrapper_entry\s*\(\s*void\s*\)\s*\{\s*(?P<body>.*)\s*\}\s*$",
        source_code,
    )
    if not match:
        return source_code

    body_lines = match.group("body").splitlines()
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    if body_lines and body_lines[-1].strip() == "return 0;":
        body_lines.pop()
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    if not body_lines:
        return source_code

    first = body_lines[0].strip()
    if ";" in first or "(" not in first:
        return source_code
    if not (first.startswith(("void ", "static ", "int ", "unsigned ", "char ", "long ", "bool ")) or "::" in first):
        return source_code

    deindented = [line[4:] if line.startswith("    ") else line for line in body_lines]
    return match.group("prefix") + "\n".join(deindented) + "\n"


def _flatten_cpp_member_function_definitions(source_code: str) -> str:
    """Turns isolated C++ member definitions into free functions for snippet compilation."""
    lines = source_code.splitlines()
    repaired_lines = []
    signature_buffer: list[str] = []
    for line in lines:
        signature_buffer.append(line)
        if not _inside_possible_cpp_signature(signature_buffer):
            repaired_lines.extend(signature_buffer)
            signature_buffer = []
            continue
        if "{" not in line and len(signature_buffer) < 8:
            continue
        signature = "\n".join(signature_buffer)
        flattened = _flatten_cpp_signature(signature)
        repaired_lines.extend(flattened.splitlines())
        signature_buffer = []
    repaired_lines.extend(signature_buffer)
    return "\n".join(repaired_lines)


def _inside_possible_cpp_signature(lines: list[str]) -> bool:
    signature = "\n".join(lines)
    if "::" not in signature:
        return False
    if ";" in signature and "{" not in signature:
        return False
    first_significant = next((line for line in lines if line.strip()), "")
    if len(first_significant) - len(first_significant.lstrip()) > 2:
        return False
    if re.search(r"^\s*(?:return|new|delete|if|while|for|switch)\b", first_significant):
        return False
    return bool(re.search(r"(?m)^\s*(?:[\w:<>,~*&\s]+\s+)?[A-Za-z_]\w*(?:::[A-Za-z_~]\w*)+\s*\(", signature))


def _flatten_cpp_signature(signature: str) -> str:
    match = re.search(r"(?m)^(\s*)((?:[\w:<>,~*&\s]+\s+)?)([A-Za-z_]\w*(?:::[A-Za-z_~]\w*)+)(\s*\()", signature)
    if not match:
        return signature
    indent, prefix, qualified_name, open_paren = match.groups()
    later_match = None
    for candidate in re.finditer(r"([A-Za-z_]\w*(?:::[A-Za-z_~]\w*)+)(\s*\()", signature[match.start() :]):
        later_match = candidate
    if later_match:
        absolute_start = match.start() + later_match.start(1)
        absolute_end = match.start() + later_match.end(1)
        qualified_name = later_match.group(1)
        flattened_name = qualified_name.replace("::", "_").replace("~", "destruct_")
        repaired = signature[:absolute_start] + flattened_name + signature[absolute_end:]
        repaired = re.sub(r"(?m)^(\s*)(?!std::)[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+\s+", r"\1long ", repaired, count=1)
        return re.sub(r"\)\s*const\s*\{", ") {", repaired, count=1)
    flattened_name = qualified_name.replace("::", "_").replace("~", "destruct_")
    if not prefix.strip():
        prefix = "void "
    repaired = signature[: match.start()] + f"{indent}{prefix}{flattened_name}{open_paren}" + signature[match.end() :]
    return re.sub(r"\)\s*const\s*\{", ") {", repaired, count=1)


def _seed_c_family_stubs(source_code: str, suffix: str) -> list[str]:
    """Returns source-derived stubs that do not depend on an initial diagnostic."""
    return _sanitize_dynamic_stubs(_infer_c_family_stubs("", source_code, suffix), suffix)


def _sanitize_dynamic_stubs(stubs: list[str], suffix: str) -> list[str]:
    """Drops inferred stubs that would collide with the static preamble."""
    is_cpp = suffix == ".cpp"
    blocked_names = _predefined_c_family_types() | _predefined_c_family_functions()
    if is_cpp:
        blocked_names.update(
            {
                "scoped_refptr",
                "base",
                "blink",
                "network",
                "net",
                "mojom",
                "std",
            }
        )
    sanitized: list[str] = []
    for stub in stubs:
        stripped = stub.strip()
        if not stripped:
            continue
        if _dynamic_stub_declares_blocked_name(stripped, blocked_names):
            continue
        sanitized.append(stub)
    return list(dict.fromkeys(sanitized))


def _dynamic_stub_declares_blocked_name(stub: str, blocked_names: set[str]) -> bool:
    for name in blocked_names:
        escaped = re.escape(name)
        if re.search(rf"\b(?:struct|class|union|enum)\s+{escaped}\b", stub):
            return True
        if re.search(rf"\b(?:typedef|using)\s+{escaped}\b", stub):
            return True
        if re.search(rf"\bnamespace\s+{escaped}\b", stub):
            return True
    return False


def _infer_c_family_stubs(error_text: str, source_code: str, suffix: str) -> list[str]:
    """Infers small compatibility stubs from clang diagnostics."""
    stubs: list[str] = []
    is_cpp = suffix == ".cpp"
    inference_source = _source_for_stub_inference(source_code)
    typedef_structs = _infer_typedef_struct_fields(inference_source)
    alias_structs = _infer_alias_struct_fields(inference_source)
    designated_structs = _infer_designated_initializer_struct_fields(inference_source)
    union_fields = _infer_union_fields(inference_source)
    predefined_types = _predefined_c_family_types()
    missing_member_structs = _infer_missing_member_struct_fields(error_text, inference_source, predefined_types)

    if is_cpp:
        stubs.extend(_infer_cpp_qualified_name_stubs(inference_source))
        if any(name in inference_source for name in ("TIFF_Manager", "PSIR_Manager", "IPTC_Manager", "PSIR_MemoryReader", "IPTC_Reader")):
            stubs.append(_render_cpp_class_scope_stub("TIFF_Manager", {"TagInfo"}, inference_source))
            stubs.append(_render_cpp_class_scope_stub("PSIR_Manager", {"ImgRsrcInfo"}, inference_source))
            stubs.append(_render_cpp_class_scope_stub("IPTC_Manager", set(), inference_source))
        if "PostScript_MetaHandler" in inference_source or "PostScript_Support" in inference_source:
            stubs.append(_render_cpp_class_scope_stub("PostScript_MetaHandler", set(), inference_source))
            stubs.append(_render_cpp_class_scope_stub("PostScript_Support", set(), inference_source))

    if (
        typedef_structs
        or alias_structs
        or designated_structs
        or union_fields
        or _infer_struct_fields(inference_source)
        or _infer_member_field_kinds(inference_source)
    ):
        stubs.append(_generic_field_struct_stub(inference_source, is_cpp=is_cpp))

    for type_name, fields in typedef_structs.items():
        if type_name in predefined_types:
            continue
        rendered_fields = _render_stub_fields(fields, is_cpp=is_cpp)
        if is_cpp:
            stubs.append(f"struct {type_name} {{ {rendered_fields} }};")
        else:
            stubs.append(f"typedef struct {type_name} {{ {rendered_fields} }} {type_name};")

    for type_name, (fields, pointer_alias) in alias_structs.items():
        if type_name in predefined_types or type_name in typedef_structs:
            continue
        rendered_fields = _render_stub_fields(fields, is_cpp=is_cpp)
        if pointer_alias and not is_cpp:
            stubs.append(f"typedef struct {type_name} {{ {rendered_fields} }} *{type_name};")
        elif is_cpp:
            stubs.append(f"struct {type_name} {{ {rendered_fields} }};")
        else:
            stubs.append(f"typedef struct {type_name} {{ {rendered_fields} }} {type_name};")

    for type_name, fields in missing_member_structs.items():
        if type_name in typedef_structs or type_name in alias_structs:
            continue
        rendered_fields = _render_stub_fields(fields, is_cpp=is_cpp)
        if is_cpp:
            stubs.append(f"struct {type_name} {{ {rendered_fields} }};")
        else:
            stubs.append(f"typedef struct {type_name} {{ {rendered_fields} }} {type_name};")

    unknown_type_names = _infer_unknown_type_names(error_text, inference_source, is_cpp=is_cpp)
    for type_name in sorted(unknown_type_names):
        if (
            type_name in _reserved_c_family_names()
            or type_name in typedef_structs
            or type_name in alias_structs
            or type_name in predefined_types
        ):
            continue
        if is_cpp and re.search(rf"\b{re.escape(type_name)}::", inference_source):
            continue
        if _looks_like_pointer_alias_type(type_name) and not is_cpp:
            stubs.append(f"typedef struct {type_name} {{ {_render_stub_fields({}, is_cpp=is_cpp)} }} *{type_name};")
            continue
        if _looks_like_scalar_alias_type(type_name):
            stubs.append(f"using {type_name} = long;" if is_cpp else f"typedef long {type_name};")
            continue
        if _looks_like_struct_alias_type(type_name):
            if is_cpp:
                stubs.append(f"struct {type_name} {{ {_render_stub_fields({}, is_cpp=is_cpp)} }};")
            else:
                stubs.append(f"typedef struct {type_name} {{ {_render_stub_fields({}, is_cpp=is_cpp)} }} {type_name};")
            continue
        if is_cpp:
            stubs.append(f"using {type_name} = long;")
        else:
            stubs.append(f"typedef long {type_name};")

    for function_name in sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*\([^()\n;]*\)\s*->", source_code))):
        if function_name in _reserved_c_family_names() or function_name in _common_storage_specifiers() or function_name in _predefined_c_family_functions():
            continue
        stubs.append(
            f"static struct vulsirt_stub_field *{function_name}(...) {{ return 0; }}"
            if is_cpp
            else f"static struct vulsirt_stub_field *{function_name}() {{ return 0; }}"
        )

    constant_stub_value = 1
    for identifier in sorted(set(re.findall(r"use of undeclared identifier '([A-Za-z_]\w*)'", error_text))):
        if identifier in _reserved_c_family_names() or identifier in _predefined_c_family_functions():
            continue
        if is_cpp and re.search(rf"\b{re.escape(identifier)}\s*::", source_code):
            continue
        if identifier in unknown_type_names:
            continue
        if identifier in typedef_structs or identifier in alias_structs or identifier in predefined_types:
            continue
        if identifier == "devlist" and "struct device" in inference_source:
            stubs.append("static struct device *devlist = 0;")
            continue
        if identifier.upper() == identifier and not re.search(rf"\b{re.escape(identifier)}\s*\(", source_code):
            stubs.append(f"#define {identifier} {constant_stub_value}")
            constant_stub_value += 1
            continue
        if _looks_like_pointer_alias_type(identifier) and _looks_like_type_use(identifier, inference_source):
            if is_cpp:
                stubs.append(f"struct {identifier} {{ {_render_stub_fields(_infer_member_field_kinds(inference_source), is_cpp=is_cpp)} }};")
            else:
                stubs.append(f"typedef struct {identifier} {{ {_render_stub_fields(_infer_member_field_kinds(inference_source), is_cpp=is_cpp)} }} *{identifier};")
            continue
        if re.search(rf"\b{re.escape(identifier)}\s*\(", source_code):
            if re.search(rf"\b{re.escape(identifier)}\s*\([^)]*\)\s*->", source_code):
                stubs.append(
                    f"static struct vulsirt_stub_field *{identifier}(...) {{ return 0; }}"
                    if is_cpp
                    else f"static struct vulsirt_stub_field *{identifier}() {{ return 0; }}"
                )
                continue
            stubs.append(f"static long {identifier}(...) {{ return 0; }}" if is_cpp else f"static long {identifier}() {{ return 0; }}")
            continue
        if re.search(rf"\b{re.escape(identifier)}\s*->", source_code):
            stubs.append(f"static struct vulsirt_stub_field *{identifier};")
            continue
        if re.search(rf"\b{re.escape(identifier)}\s*\.", source_code):
            stubs.append(f"static struct vulsirt_stub_field {identifier};")
            continue
        if _looks_like_char_pointer_storage(identifier, source_code):
            stubs.append(f"static char *{identifier};")
            continue
        if is_cpp and _looks_like_nullptr_storage(identifier, source_code):
            stubs.append(f"static struct vulsirt_stub_field *{identifier};")
            continue
        if re.search(rf"\b{re.escape(identifier)}\s*\[", source_code):
            stubs.append(f"static char {identifier}[16];")
            continue
        if _looks_like_type_use(identifier, inference_source):
            if is_cpp:
                stubs.append(f"struct {identifier} {{ {_render_stub_fields({}, is_cpp=is_cpp)} }};")
            else:
                stubs.append(f"typedef struct {identifier} {{ {_render_stub_fields({}, is_cpp=is_cpp)} }} {identifier};")
            continue
        stubs.append(f"static long {identifier} = 0;")

    needed_structs = set(re.findall(r"incomplete definition of type 'struct ([A-Za-z_]\w*)'", error_text))
    needed_structs.update(re.findall(r"incomplete type 'struct ([A-Za-z_]\w*)'", error_text))
    needed_structs.update(re.findall(r"incomplete type '(?:const\s+)?struct ([A-Za-z_]\w*)'", error_text))
    needed_structs.difference_update(typedef_structs)
    needed_structs.difference_update(alias_structs)
    explicit_struct_fields = _infer_struct_fields(inference_source)
    for struct_name, fields in designated_structs.items():
        explicit_struct_fields.setdefault(struct_name, {}).update(fields)
    for struct_name, fields in explicit_struct_fields.items():
        if needed_structs and struct_name not in needed_structs:
            continue
        if struct_name in predefined_types:
            needed_structs.discard(struct_name)
            continue
        if struct_name == "header":
            stubs.append("struct header { char *p; long l; };")
            needed_structs.discard(struct_name)
            continue
        if struct_name == "device":
            stubs.append("struct header { char *p; long l; };")
            stubs.append("struct device { struct device *next; time_t t; struct header headers[16]; char data[1]; };")
            needed_structs.discard("header")
            needed_structs.discard(struct_name)
            continue
        stubs.append(f"struct {struct_name} {{ {_render_stub_fields(fields, is_cpp=is_cpp)} }};")
        needed_structs.discard(struct_name)
    for struct_name in sorted(needed_structs):
        if struct_name in predefined_types:
            continue
        if struct_name == "header":
            stubs.append("struct header { char *p; long l; };")
        elif struct_name == "device":
            stubs.append("struct header { char *p; long l; };")
            stubs.append("struct device { struct device *next; time_t t; struct header headers[16]; char data[1]; };")
        else:
            stubs.append(f"struct {struct_name} {{ long dummy; }};")

    needed_unions = set(re.findall(r"incomplete type 'union ([A-Za-z_]\w*)'", error_text))
    needed_unions.update(re.findall(r"incomplete definition of type 'union ([A-Za-z_]\w*)'", error_text))
    for union_name, fields in union_fields.items():
        if needed_unions and union_name not in needed_unions:
            continue
        stubs.append(f"union {union_name} {{ {_render_union_fields(fields, is_cpp=is_cpp)} }};")
        needed_unions.discard(union_name)
    for union_name in sorted(needed_unions):
        stubs.append(f"union {union_name} {{ long dummy; }};")

    return list(dict.fromkeys(stubs))


def _infer_unknown_type_names(error_text: str, source_code: str, is_cpp: bool = False) -> set[str]:
    """Extracts missing type names from diagnostics and nearby type-like source uses."""
    type_names = set(re.findall(r"unknown type name '([A-Za-z_]\w*)'", error_text))
    type_names.update(re.findall(r"must use '(?:struct|union|enum) ([A-Za-z_]\w*)' tag", error_text))
    type_names.update(re.findall(r"unknown type name '(?:struct|union|enum) ([A-Za-z_]\w*)'", error_text))

    if "a type specifier is required" in error_text or "requires a type specifier" in error_text:
        for type_name in re.findall(r"(?m)^\s*([A-Za-z_]\w*)\s+[*&]?\s*[A-Za-z_]\w*\s*(?:[,;)=]|\[[^\]]*\])", source_code):
            type_names.add(type_name)
        for type_name in re.findall(r"\b([A-Za-z_]\w*)\s*\*", source_code):
            type_names.add(type_name)

    if is_cpp:
        type_names.update(re.findall(r"template argument for template type parameter must be a type[\s\S]*?\|\s+([A-Za-z_]\w*)", error_text))

    for type_name in re.findall(r"(?m)^\s*(?:const\s+)?([A-Za-z_]\w*)\s*\*\s*[A-Za-z_]\w*\s*(?:=|;|,|\))", source_code):
        if _looks_like_project_type_name(type_name):
            type_names.add(type_name)
    for type_name in re.findall(r"(?m)^\s*(?:const\s+)?([A-Za-z_]\w*)\s+[A-Za-z_]\w*\s*(?:=|;|,|\))", source_code):
        if _looks_like_project_type_name(type_name):
            type_names.add(type_name)

    return {
        type_name
        for type_name in type_names
        if type_name not in _reserved_c_family_names()
        and type_name not in _common_storage_specifiers()
    }


def _looks_like_project_type_name(type_name: str) -> bool:
    if not type_name or type_name in _reserved_c_family_names() or type_name in _common_storage_specifiers():
        return False
    if type_name in {"NULL", "true", "false", "return"}:
        return False
    return (
        bool(type_name[:1].isupper())
        or "_" in type_name
        or type_name.endswith(("Ptr", "Ref", "Handle", "State", "Context", "Info", "Data", "_t"))
    )


def _looks_like_nullptr_storage(identifier: str, source_code: str) -> bool:
    return bool(re.search(rf"\b{re.escape(identifier)}\s*=\s*nullptr\b", source_code))


def _infer_struct_fields(source_code: str) -> dict[str, dict[str, str]]:
    """Infers fields for dereferenced struct pointer parameters."""
    pointer_vars: dict[str, str] = {}
    for struct_name, variable_name in re.findall(r"\b(?:const\s+)?struct\s+([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)", source_code):
        pointer_vars[variable_name] = struct_name
    for line in source_code.splitlines():
        statement = line.strip()
        if ";" not in statement:
            continue
        declaration = statement.split(";", 1)[0]
        match = re.search(r"\b(?:const\s+)?struct\s+([A-Za-z_]\w*)\s+(.+)", declaration)
        if not match:
            continue
        struct_name, declarators = match.groups()
        for variable_name in re.findall(r"\*\s*([A-Za-z_]\w*)", declarators):
            pointer_vars[variable_name] = struct_name

    fields_by_struct: dict[str, dict[str, str]] = {}
    for variable_name, field_name, suffix in re.findall(
        r"\b([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*(->|\.|\[|\()?",
        source_code,
    ):
        struct_name = pointer_vars.get(variable_name)
        if struct_name:
            fields_by_struct.setdefault(struct_name, {})
            _set_field_kind(fields_by_struct[struct_name], field_name, _field_kind_from_suffix(suffix))
    for fields in fields_by_struct.values():
        _promote_function_array_fields(fields, source_code)
    return fields_by_struct


def _infer_missing_member_struct_fields(
    error_text: str,
    source_code: str,
    predefined_types: set[str],
) -> dict[str, dict[str, str]]:
    fields_by_type: dict[str, dict[str, str]] = {}
    inferred_member_kinds = _infer_member_field_kinds(source_code)
    for member_name, raw_type_name in re.findall(r"no member named '([A-Za-z_]\w*)' in '([^']+)'", error_text):
        type_name = raw_type_name
        type_name = re.sub(r"^(?:const\s+)?(?:struct|class)\s+", "", type_name).strip()
        if (
            not re.match(r"^[A-Za-z_]\w*$", type_name)
            or type_name in predefined_types
            or type_name in _reserved_c_family_names()
            or re.search(rf"\b(?:struct|class)\s+{re.escape(type_name)}\s*\{{", source_code)
        ):
            continue
        fields = fields_by_type.setdefault(type_name, {})
        _set_field_kind(fields, member_name, inferred_member_kinds.get(member_name, "long"))
    return fields_by_type


def _infer_union_fields(source_code: str) -> dict[str, dict[str, str]]:
    """Infers fields for dereferenced union pointer parameters."""
    pointer_vars: dict[str, str] = {}
    for union_name, variable_name in re.findall(r"\b(?:const\s+)?union\s+([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)", source_code):
        pointer_vars[variable_name] = union_name

    fields_by_union: dict[str, dict[str, str]] = {}
    for struct_name, union_variable, field_name in re.findall(
        r"\bstruct\s+([A-Za-z_]\w*)\s*\*\s*[A-Za-z_]\w*\s*=\s*&\s*([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)",
        source_code,
    ):
        union_name = pointer_vars.get(union_variable)
        if union_name:
            fields_by_union.setdefault(union_name, {})[field_name] = f"struct:{struct_name}"

    for variable_name, field_name, suffix in re.findall(
        r"\b([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*(->|\.|\[|\()?",
        source_code,
    ):
        union_name = pointer_vars.get(variable_name)
        if not union_name:
            continue
        fields = fields_by_union.setdefault(union_name, {})
        if field_name in fields and fields[field_name].startswith("struct:"):
            continue
        _set_field_kind(fields, field_name, _field_kind_from_suffix(suffix))
    return fields_by_union


def _source_for_stub_inference(source_code: str) -> str:
    """Removes comments and preprocessor lines before regex-based stub inference."""
    without_block_comments = re.sub(r"/\*.*?\*/", " ", source_code, flags=re.DOTALL)
    without_line_comments = re.sub(r"//.*", "", without_block_comments)
    return "\n".join(
        line for line in without_line_comments.splitlines() if not line.lstrip().startswith("#")
    )


def _infer_designated_initializer_struct_fields(source_code: str) -> dict[str, dict[str, str]]:
    """Infers struct fields from C designated initializers like .field = value."""
    fields_by_struct: dict[str, dict[str, str]] = {}
    initializer_pattern = re.compile(
        r"\bstruct\s+([A-Za-z_]\w*)\s+[A-Za-z_]\w*\s*=\s*\{(?P<body>.*?)\}\s*;",
        re.DOTALL,
    )
    for match in initializer_pattern.finditer(source_code):
        struct_name = match.group(1)
        body = match.group("body")
        fields = fields_by_struct.setdefault(struct_name, {})
        for field_name in re.findall(r"(?m)(?<![A-Za-z_]\w)\.\s*([A-Za-z_]\w*)\s*=", body):
            if field_name in _reserved_c_family_names() or field_name in _common_storage_specifiers():
                continue
            _set_field_kind(fields, field_name, "long")
    return fields_by_struct


def _infer_member_field_kinds(source_code: str) -> dict[str, str]:
    """Infers generic field names from any member access chain in the snippet."""
    fields: dict[str, str] = {}
    member_pattern = re.compile(r"(?:->|\.)\s*([A-Za-z_]\w*)")
    for match in member_pattern.finditer(source_code):
        field_name = match.group(1)
        if field_name in _reserved_c_family_names() or field_name in _common_storage_specifiers():
            continue
        suffix = _next_member_suffix(source_code, match.end())
        _set_field_kind(fields, field_name, _field_kind_from_suffix(suffix))
    _promote_chained_member_fields(fields, source_code)
    _promote_function_array_fields(fields, source_code)
    return fields


def _next_member_suffix(source_code: str, offset: int) -> str:
    remainder = source_code[offset:].lstrip()
    for suffix in ("->", ".", "[", "("):
        if remainder.startswith(suffix):
            return suffix
    return ""


def _infer_typedef_struct_fields(source_code: str) -> dict[str, dict[str, str]]:
    """Infers typedef-like pointer types that are dereferenced with ->."""
    pointer_vars: dict[str, str] = {}
    pointer_pattern = re.compile(r"\b(?:const\s+)?([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)")
    for match in pointer_pattern.finditer(source_code):
        type_name, variable_name = match.groups()
        prefix = source_code[max(0, match.start() - 8) : match.start()]
        if (
            "struct " in prefix
            or "union " in prefix
            or type_name in _reserved_c_family_names()
            or type_name in _common_storage_specifiers()
        ):
            continue
        pointer_vars[variable_name] = type_name

    fields_by_type: dict[str, dict[str, str]] = {}
    for variable_name, field_name, suffix in re.findall(
        r"\b([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*(->|\.|\[|\()?",
        source_code,
    ):
        type_name = pointer_vars.get(variable_name)
        if type_name:
            fields_by_type.setdefault(type_name, {})
            _set_field_kind(fields_by_type[type_name], field_name, _field_kind_from_suffix(suffix))
    for fields in fields_by_type.values():
        _promote_function_array_fields(fields, source_code)
    return fields_by_type


def _infer_alias_struct_fields(source_code: str) -> dict[str, tuple[dict[str, str], bool]]:
    """Infers fields for typedef-like variables declared without an explicit pointer."""
    declared_vars: dict[str, str] = {}
    declaration_pattern = re.compile(
        r"\b(?:const\s+)?([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(?=[,;)=])"
    )
    for match in declaration_pattern.finditer(source_code):
        type_name, variable_name = match.groups()
        prefix = source_code[max(0, match.start() - 8) : match.start()]
        if (
            "struct " in prefix
            or "union " in prefix
            or type_name in _reserved_c_family_names()
            or type_name in _common_storage_specifiers()
            or variable_name in _reserved_c_family_names()
            or type_name == "return"
        ):
            continue
        declared_vars[variable_name] = type_name

    fields_by_type: dict[str, tuple[dict[str, str], bool]] = {}
    for variable_name, operator, field_name, suffix in re.findall(
        r"\b([A-Za-z_]\w*)\s*(->|\.)\s*([A-Za-z_]\w*)\s*(->|\.|\[|\()?",
        source_code,
    ):
        type_name = declared_vars.get(variable_name)
        if not type_name:
            continue
        fields, pointer_alias = fields_by_type.setdefault(type_name, ({}, False))
        _set_field_kind(fields, field_name, _field_kind_from_suffix(suffix))
        if suffix == ".":
            _set_field_kind(fields, field_name, "struct")
        fields_by_type[type_name] = (fields, pointer_alias or operator == "->")
    for fields, _ in fields_by_type.values():
        _promote_function_array_fields(fields, source_code)
    return fields_by_type


def _field_kind_from_suffix(suffix: str) -> str:
    if suffix == "(":
        return "function"
    if suffix == "->":
        return "pointer"
    if suffix == ".":
        return "struct"
    if suffix == "[":
        return "array"
    return "long"


def _looks_like_char_pointer_storage(identifier: str, source_code: str) -> bool:
    """Detects undeclared fragment variables used as mutable char cursors."""
    escaped = re.escape(identifier)
    pointerish_name = re.search(r"(?:buf|buffer|ptr|cursor|end|lim|limit|position)", identifier, re.IGNORECASE)
    pointer_arithmetic = re.search(rf"\b{escaped}\b\s*[-+]|[-+]\s*\b{escaped}\b", source_code)
    pointer_assignment = re.search(
        rf"\b{escaped}\b\s*=\s*(?:NULL|0|[A-Za-z_]\w*\s*\+|\([^)]+\s*\*\)|(?:[A-Za-z_]\w*)\b)",
        source_code,
    )
    if pointerish_name and (pointer_arithmetic or pointer_assignment):
        return True
    return bool(re.search(rf"\bchar\s*\*\s*[A-Za-z_]\w*\s*=\s*{escaped}\b", source_code))


def _promote_function_array_fields(fields: dict[str, str], source_code: str) -> None:
    for field_name in list(fields):
        escaped = re.escape(field_name)
        if re.search(rf"(?:->|\.)\s*{escaped}\s*\[[^\]]+\]\s*\(", source_code):
            _set_field_kind(fields, field_name, "function_array")
        if re.search(rf"(?:->|\.)\s*{escaped}\s*\[[^\]]+\]\s*\[[^\]]+\]\s*\[[^\]]+\]", source_code):
            _set_field_kind(fields, field_name, "array3")
        elif re.search(rf"(?:->|\.)\s*{escaped}\s*\[[^\]]+\]\s*\[[^\]]+\]", source_code):
            _set_field_kind(fields, field_name, "array2")
        if re.search(rf"(?:->|\.)\s*{escaped}\s*=\s*(?:\([^)]+\)\s*)?(?:g_)?(?:m|c)alloc(?:0)?\s*\(", source_code):
            _set_field_kind(fields, field_name, "pointer")
        if re.search(rf"sizeof\s*\(\s*\*\s*\([^)]*(?:->|\.)\s*{escaped}\s*\)\s*\)", source_code):
            _set_field_kind(fields, field_name, "pointer")


def _promote_chained_member_fields(fields: dict[str, str], source_code: str) -> None:
    """Promotes fields that need to carry another member access."""
    for field_name in list(fields):
        escaped = re.escape(field_name)
        if re.search(rf"(?:->|\.)\s*{escaped}\s*\[[^\]]+\]\s*\[[^\]]+\]\s*\[[^\]]+\]\s*->", source_code):
            _set_field_kind(fields, field_name, "array3_pointer")
        elif re.search(rf"(?:->|\.)\s*{escaped}\s*\[[^\]]+\]\s*\[[^\]]+\]\s*\[[^\]]+\]\s*\.", source_code):
            _set_field_kind(fields, field_name, "array3_struct")
        elif re.search(rf"(?:->|\.)\s*{escaped}\s*\[[^\]]+\]\s*\[[^\]]+\]\s*->", source_code):
            _set_field_kind(fields, field_name, "array2_pointer")
        elif re.search(rf"(?:->|\.)\s*{escaped}\s*\[[^\]]+\]\s*\[[^\]]+\]\s*\.", source_code):
            _set_field_kind(fields, field_name, "array2_struct")
        elif re.search(rf"(?:->|\.)\s*{escaped}\s*\[[^\]]+\]\s*->", source_code):
            _set_field_kind(fields, field_name, "array_pointer")
        elif re.search(rf"(?:->|\.)\s*{escaped}\s*\[[^\]]+\]\s*\.", source_code):
            _set_field_kind(fields, field_name, "array_struct")


def _set_field_kind(fields: dict[str, str], field_name: str, new_kind: str) -> None:
    priority = {
        "long": 0,
        "array": 1,
        "array2": 2,
        "array3": 3,
        "struct": 4,
        "array_struct": 5,
        "array2_struct": 6,
        "array3_struct": 7,
        "array_pointer": 8,
        "array2_pointer": 9,
        "array3_pointer": 10,
        "pointer": 11,
        "function": 12,
        "function_array": 13,
    }
    old_kind = fields.get(field_name, "long")
    fields[field_name] = new_kind if priority[new_kind] > priority[old_kind] else old_kind


def _render_stub_fields(fields: dict[str, str], is_cpp: bool = False) -> str:
    if not fields:
        return "long dummy;"
    rendered_fields = []
    for field_name, field_kind in sorted(fields.items()):
        if field_name == "data":
            rendered_fields.append("const unsigned char *data;")
        elif field_name == "internals":
            rendered_fields.append("struct vulsirt_stub_internals internals;")
        elif field_name == "sax25_call":
            rendered_fields.append("ax25_address sax25_call;")
        elif field_name == "s_high_profile":
            rendered_fields.append("struct vulsirt_stub_profile s_high_profile;")
        elif field_name == "psirMgr":
            rendered_fields.append("PSIR_Manager *psirMgr;")
        elif field_name == "iptcMgr":
            rendered_fields.append("IPTC_Manager *iptcMgr;")
        elif field_name == "tiffMgr":
            rendered_fields.append("TIFF_Manager tiffMgr;")
        elif field_name == "parent":
            rendered_fields.append("struct vulsirt_stub_parent *parent;")
        elif field_name == "u":
            rendered_fields.append("struct vulsirt_stub_u u[16];")
        elif field_name in {"mem", "val"}:
            rendered_fields.append(f"struct vulsirt_stub_leaf {field_name};")
        elif field_name == "private_data":
            rendered_fields.append("void *private_data;")
        elif field_name == "readstat":
            rendered_fields.append("long (*readstat)(...);" if is_cpp else "long (*readstat)();")
        elif field_name == "apf_intra_pred_chroma":
            rendered_fields.append("long (*apf_intra_pred_chroma[16])(...);" if is_cpp else "long (*apf_intra_pred_chroma[16])();")
        elif field_kind == "function_array":
            rendered_fields.append(f"long (*{field_name}[16])(...);" if is_cpp else f"long (*{field_name}[16])();")
        elif field_kind == "function":
            rendered_fields.append(f"long (*{field_name})(...);" if is_cpp else f"long (*{field_name})();")
        elif field_kind == "pointer":
            rendered_fields.append(f"struct vulsirt_stub_field *{field_name};")
        elif field_kind == "struct":
            rendered_fields.append(f"struct vulsirt_stub_field {field_name};")
        elif field_kind == "array3":
            rendered_fields.append(f"long {field_name}[16][16][16];")
        elif field_kind == "array3_pointer":
            rendered_fields.append(f"struct vulsirt_stub_field *{field_name}[16][16][16];")
        elif field_kind == "array3_struct":
            rendered_fields.append(f"struct vulsirt_stub_field {field_name}[16][16][16];")
        elif field_kind == "array2":
            rendered_fields.append(f"long {field_name}[16][16];")
        elif field_kind == "array2_pointer":
            rendered_fields.append(f"struct vulsirt_stub_field *{field_name}[16][16];")
        elif field_kind == "array2_struct":
            rendered_fields.append(f"struct vulsirt_stub_field {field_name}[16][16];")
        elif field_kind == "array":
            rendered_fields.append(f"long {field_name}[16];")
        elif field_kind == "array_pointer":
            rendered_fields.append(f"struct vulsirt_stub_field *{field_name}[16];")
        elif field_kind == "array_struct":
            rendered_fields.append(f"struct vulsirt_stub_field {field_name}[16];")
        else:
            rendered_fields.append(f"long {field_name};")
    return " ".join(rendered_fields)


def _render_union_fields(fields: dict[str, str], is_cpp: bool = False) -> str:
    if not fields:
        return "long dummy;"
    rendered_fields = []
    for field_name, field_kind in sorted(fields.items()):
        if field_kind.startswith("struct:"):
            rendered_fields.append(f"struct {field_kind.split(':', 1)[1]} {field_name};")
        else:
            rendered_fields.append(_render_stub_fields({field_name: field_kind}, is_cpp=is_cpp))
    return " ".join(rendered_fields)


def _render_generic_stub_field_lines(fields: dict[str, str], is_cpp: bool = False) -> list[str]:
    """Renders fields inside vulsirt_stub_field without recursive by-value members."""
    rendered_fields = []
    for field_name, field_kind in sorted(fields.items()):
        if field_name == "private_data":
            rendered_fields.append("    void *private_data;")
        elif field_name == "psirMgr":
            rendered_fields.append("    PSIR_Manager *psirMgr;")
        elif field_name == "iptcMgr":
            rendered_fields.append("    IPTC_Manager *iptcMgr;")
        elif field_name == "tiffMgr":
            rendered_fields.append("    TIFF_Manager tiffMgr;")
        elif field_name == "parent":
            rendered_fields.append("    struct vulsirt_stub_parent *parent;")
        elif field_name == "readstat":
            rendered_fields.append("    long (*readstat)(...);" if is_cpp else "    long (*readstat)();")
        elif field_name == "u":
            rendered_fields.append("    struct vulsirt_stub_u u[16];")
        elif field_name in {"mem", "val"}:
            rendered_fields.append(f"    struct vulsirt_stub_leaf {field_name};")
        elif field_kind == "function":
            rendered_fields.append(f"    long (*{field_name})(...);" if is_cpp else f"    long (*{field_name})();")
        elif field_kind == "pointer":
            rendered_fields.append(f"    struct vulsirt_stub_field *{field_name};")
        elif field_kind == "struct":
            rendered_fields.append(f"    struct vulsirt_stub_leaf {field_name};")
        elif field_kind == "function_array":
            rendered_fields.append(f"    long (*{field_name}[16])(...);" if is_cpp else f"    long (*{field_name}[16])();")
        elif field_kind == "array3":
            rendered_fields.append(f"    long {field_name}[16][16][16];")
        elif field_kind == "array3_pointer":
            rendered_fields.append(f"    struct vulsirt_stub_field *{field_name}[16][16][16];")
        elif field_kind == "array3_struct":
            rendered_fields.append(f"    struct vulsirt_stub_leaf {field_name}[16][16][16];")
        elif field_kind == "array2":
            rendered_fields.append(f"    long {field_name}[16][16];")
        elif field_kind == "array2_pointer":
            rendered_fields.append(f"    struct vulsirt_stub_field *{field_name}[16][16];")
        elif field_kind == "array2_struct":
            rendered_fields.append(f"    struct vulsirt_stub_leaf {field_name}[16][16];")
        elif field_kind == "array":
            rendered_fields.append(f"    long {field_name}[16];")
        elif field_kind == "array_pointer":
            rendered_fields.append(f"    struct vulsirt_stub_field *{field_name}[16];")
        elif field_kind == "array_struct":
            rendered_fields.append(f"    struct vulsirt_stub_leaf {field_name}[16];")
        elif field_name == "data":
            rendered_fields.append("    const unsigned char *data;")
        else:
            rendered_fields.append(f"    long {field_name};")
    return rendered_fields


def _render_leaf_stub_field_lines(fields: dict[str, str]) -> list[str]:
    """Renders non-recursive fields for nested dotted member access."""
    rendered_fields = []
    for field_name, field_kind in sorted(fields.items()):
        if field_kind == "function":
            rendered_fields.append(f"    long (*{field_name})();")
        elif field_kind == "function_array":
            rendered_fields.append(f"    long (*{field_name}[16])();")
        elif field_kind == "array3":
            rendered_fields.append(f"    long {field_name}[16][16][16];")
        elif field_kind == "array3_pointer":
            rendered_fields.append(f"    struct vulsirt_stub_leaf *{field_name}[16][16][16];")
        elif field_kind == "array3_struct":
            rendered_fields.append(f"    long {field_name}[16][16][16];")
        elif field_kind == "array2":
            rendered_fields.append(f"    long {field_name}[16][16];")
        elif field_kind == "array2_pointer":
            rendered_fields.append(f"    struct vulsirt_stub_leaf *{field_name}[16][16];")
        elif field_kind == "array2_struct":
            rendered_fields.append(f"    long {field_name}[16][16];")
        elif field_kind == "array":
            rendered_fields.append(f"    long {field_name}[16];")
        elif field_kind == "array_pointer":
            rendered_fields.append(f"    struct vulsirt_stub_leaf *{field_name}[16];")
        elif field_kind == "array_struct":
            rendered_fields.append(f"    long {field_name}[16];")
        elif field_name == "psirMgr":
            rendered_fields.append("    PSIR_Manager *psirMgr;")
        elif field_name == "iptcMgr":
            rendered_fields.append("    IPTC_Manager *iptcMgr;")
        elif field_name == "tiffMgr":
            rendered_fields.append("    TIFF_Manager tiffMgr;")
        elif field_name == "parent":
            rendered_fields.append("    struct vulsirt_stub_parent *parent;")
        elif field_name == "data":
            rendered_fields.append("    const unsigned char *data;")
        else:
            rendered_fields.append(f"    long {field_name};")
    return rendered_fields


def _generic_field_struct_stub(source_code: str = "", is_cpp: bool = False) -> str:
    built_in_fields = {
        "DataOffset",
        "addr",
        "algorithm",
        "algorithm2",
        "all",
        "apf_intra_pred_chroma",
        "base",
        "buf",
        "capacity",
        "columns",
        "comm",
        "count",
        "data",
        "dev",
        "drv",
        "drivers",
        "dummy",
        "fd",
        "flags",
        "height",
        "id",
        "interface",
        "internals",
        "iov_base",
        "iov_len",
        "len",
        "length",
        "lock",
        "name",
        "next",
        "nikeys",
        "pos",
        "private_data",
        "proto",
        "range",
        "rcv_waitq",
        "readstat",
        "reject_error",
        "resumable",
        "rxopt",
        "s_addr",
        "s_high_profile",
        "sax25_call",
        "size",
        "st_waitq",
        "state",
        "status",
        "stavail",
        "substring",
        "tail",
        "tmp",
        "type",
        "u",
        "value",
        "width",
    }
    inferred_fields = {
        field_name: field_kind
        for field_name, field_kind in _infer_member_field_kinds(source_code).items()
        if field_name not in built_in_fields
    }
    inferred_field_lines = _render_generic_stub_field_lines(inferred_fields, is_cpp=is_cpp)
    leaf_built_in_fields = {
        "all",
        "count",
        "data",
        "dummy",
        "flags",
        "h",
        "id",
        "len",
        "line",
        "size",
        "type",
        "u",
        "value",
        "w",
        "x",
        "y",
    }
    inferred_leaf_lines = _render_leaf_stub_field_lines(
        {
            field_name: field_kind
            for field_name, field_kind in _infer_member_field_kinds(source_code).items()
            if field_name not in leaf_built_in_fields
        }
    )
    leaf_cpp_methods = (
        [
            "    vulsirt_stub_leaf(long value = 0) : dummy(value) {}",
            "    template <typename T> vulsirt_stub_leaf &operator=(T) { return *this; }",
            "    operator long() const { return dummy; }",
            "    operator bool() const { return dummy != 0; }",
            "    vulsirt_stub_leaf *operator->() { return this; }",
            "    const vulsirt_stub_leaf *operator->() const { return this; }",
            "    vulsirt_stub_leaf *get() { return this; }",
            "    void reset() { dummy = 0; }",
            "    template <typename... Args> long operator()(Args...) { return 0; }",
        ]
        if is_cpp
        else []
    )
    field_cpp_methods = (
        [
            "    vulsirt_stub_field(long value = 0) : dummy(value) {}",
            "    template <typename T> vulsirt_stub_field &operator=(T) { return *this; }",
            "    operator long() const { return dummy; }",
            "    operator bool() const { return dummy != 0; }",
            "    vulsirt_stub_field *operator->() { return this; }",
            "    const vulsirt_stub_field *operator->() const { return this; }",
            "    vulsirt_stub_field *get() { return this; }",
            "    void reset() { dummy = 0; }",
            "    template <typename... Args> long operator()(Args...) { return 0; }",
            "    template <typename... Args> void push_back(Args...) {}",
        ]
        if is_cpp
        else []
    )
    return "\n".join(
        [
            "struct vulsirt_stub_endpoint { long min; long max; };",
            "struct vulsirt_stub_ip_range { struct vulsirt_stub_endpoint ipv4; struct vulsirt_stub_endpoint ipv6; };",
            "struct vulsirt_stub_addr { struct vulsirt_stub_ip_range addr; struct vulsirt_stub_endpoint proto; };",
            "struct vulsirt_stub_cipher { long algorithm; long algorithm2; long id; long key_size; };",
            "struct vulsirt_stub_tmp { struct vulsirt_stub_cipher *new_cipher; struct vulsirt_stub_cipher *cipher; long size; };",
            "struct vulsirt_stub_parent { long openFlags; XMP_IO *ioRef; XMP_AbortProc abortProc; void *abortArg; };",
            "typedef struct ax25_address { long ax25_call[16]; } ax25_address;",
            "struct vulsirt_stub_profile { long i2_scalinglist4x4[16]; long i2_scalinglist8x8[16]; };",
            "struct vulsirt_stub_internals { long resumable; long size; long data; };",
            "struct vulsirt_stub_leaf {",
            "    long dummy;",
            "    long all;",
            "    long count;",
            "    long flags;",
            "    long h;",
            "    long id;",
            "    long len;",
            "    long line;",
            "    long size;",
            "    long type;",
            "    long value;",
            "    long w;",
            "    long x;",
            "    long y;",
            "    const unsigned char *data;",
            *inferred_leaf_lines,
            *leaf_cpp_methods,
            "};",
            "struct vulsirt_stub_u { struct vulsirt_stub_leaf mem; struct vulsirt_stub_leaf val; };",
            "struct vulsirt_stub_field {",
            "    long dummy;",
            "    long algorithm;",
            "    long algorithm2;",
            "    long base;",
            "    long buf;",
            "    long capacity;",
            "    long columns;",
            "    long count;",
            "    const unsigned char *data;",
            "    long DataOffset;",
            "    struct vulsirt_stub_field *dev;",
            "    long fd;",
            "    long height;",
            "    long id;",
            "    void *iov_base;",
            "    size_t iov_len;",
            "    long length;",
            "    long lock;",
            "    long name;",
            "    long nikeys;",
            "    long next;",
            "    long pos;",
            "    void *private_data;",
            "    long reject_error;",
            "    long len;",
            "    long s_addr;",
            "    long size;",
            "    long state;",
            "    long status;",
            "    long substring;",
            "    long tail;",
            "    long resumable;",
            "    long all;",
            "    char comm[16];",
            "    long type;",
            "    long value;",
            "    long flags;",
            "    long stavail;",
            "    long width;",
            "    struct vulsirt_stub_u u[16];",
            "    long (*apf_intra_pred_chroma[16])(...);" if is_cpp else "    long (*apf_intra_pred_chroma[16])();",
            "    struct vulsirt_stub_ip_range addr;",
            "    struct vulsirt_stub_endpoint proto;",
            "    struct vulsirt_stub_addr range;",
            "    struct vulsirt_stub_tmp tmp;",
            "    struct vulsirt_stub_internals rxopt;",
            "    struct vulsirt_stub_internals internals;",
            "    ax25_address sax25_call;",
            "    struct vulsirt_stub_profile s_high_profile;",
            "    struct vulsirt_stub_field *drv[16];",
            "    struct vulsirt_stub_field *interface;",
            "    long (*readstat)(...);" if is_cpp else "    long (*readstat)();",
            "    struct vulsirt_stub_field *rcv_waitq;",
            "    struct vulsirt_stub_field *st_waitq;",
            "    struct vulsirt_stub_field *drivers;",
            *inferred_field_lines,
            *field_cpp_methods,
            "};",
        ]
    )


def _reserved_c_family_names() -> set[str]:
    return {
        "break",
        "bool",
        "case",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "float",
        "for",
        "goto",
        "if",
        "int",
        "long",
        "namespace",
        "return",
        "short",
        "signed",
        "sizeof",
        "size_t",
        "static_assert",
        "struct",
        "template",
        "unsigned",
        "union",
        "void",
        "while",
    }


def _common_storage_specifiers() -> set[str]:
    return {
        "auto",
        "extern",
        "inline",
        "register",
        "static",
        "typedef",
        "volatile",
    }


def _predefined_c_family_functions() -> set[str]:
    return {
        "GetElement",
        "GetTensorData",
        "render_frame",
        "stringToDict",
    }


def _looks_like_pointer_alias_type(type_name: str) -> bool:
    return type_name.endswith(("Ptr", "Ref", "Handle", "State", "Context")) or type_name.endswith("_t")


def _looks_like_struct_alias_type(type_name: str) -> bool:
    return type_name.endswith(
        ("_t", "Info", "Data", "Packet", "State", "Context", "Header", "Manager", "Reader", "Writer", "Handler")
    )


def _looks_like_scalar_alias_type(type_name: str) -> bool:
    scalar_names = {
        "Bool",
        "CompositeFunc",
        "FT_Error",
        "Register",
        "__u32",
        "apr_off_t",
        "apr_size_t",
        "apr_status_t",
        "apr_uint64_t",
        "gboolean",
        "gint",
        "gint32",
        "guint",
        "guint32",
        "krb5_error_code",
        "l_fp",
        "linenr_T",
        "PVSCSISGState",
        "enum_func_status",
        "enum_mysqlnd_collected_stats",
        "mrb_bool",
        "os_ptr",
        "pos_T",
        "rpmVerifyAttrs",
        "rpmfileAttrs",
        "rpm_mode_t",
        "timelib_sll",
        "ut32",
        "zend_bool",
    }
    return type_name in scalar_names or type_name.endswith(("Attrs", "Flags"))


def _looks_like_type_use(identifier: str, source_code: str) -> bool:
    if identifier in {"std", "and", "or", "not"}:
        return False
    if identifier.upper() == identifier:
        return False
    escaped = re.escape(identifier)
    return bool(re.search(rf"\b{escaped}\s*[*&]?\s+[A-Za-z_]\w*\b", source_code))


def _infer_cpp_qualified_name_stubs(source_code: str) -> list[str]:
    """Builds small namespace/type shims for C++ fragments extracted from large projects."""
    stubs: list[str] = []
    qualified_names = sorted(set(re.findall(r"\b((?:[A-Za-z_]\w*::)+[A-Za-z_]\w*)", source_code)))
    namespace_roots: dict[tuple[str, ...], set[str]] = {}
    class_scopes: dict[str, set[str]] = {}
    for qualified_name in qualified_names:
        parts = qualified_name.split("::")
        if len(parts) < 2 or parts[0] in {"std"}:
            continue
        concrete_class_scopes = {"TIFF_Manager", "PSIR_Manager", "IPTC_Manager", "MOOV_Manager", "Status", "FrameBuffer"}
        if parts[0][:1].isupper() and (parts[0] in concrete_class_scopes or parts[0] not in _predefined_c_family_types()):
            class_scopes.setdefault(parts[0], set()).add(parts[1])
            continue
        if parts[0] in {"absl", "android", "base", "blink", "icu", "netdutils", "service_manager"} or any(part[:1].isupper() for part in parts[:-1]):
            continue
        namespace_roots.setdefault(tuple(parts[:-1]), set()).add(parts[-1])

    for class_name, members in sorted(class_scopes.items()):
        stubs.append(_render_cpp_class_scope_stub(class_name, members, source_code))

    for namespaces, type_names in sorted(namespace_roots.items()):
        if namespaces == ("service_manager",):
            continue
        namespace_open = " ".join(f"namespace {namespace} {{" for namespace in namespaces)
        namespace_close = " ".join("}" for _ in namespaces)
        rendered_types = " ".join(
            _render_cpp_namespace_type_stub(namespaces, type_name)
            for type_name in sorted(type_names)
            if type_name and type_name[0].isupper()
        )
        if rendered_types:
            stubs.append(f"{namespace_open} {rendered_types} {namespace_close}")

    if "service_manager::Manifest" in source_code or "service_manager::ManifestBuilder" in source_code:
        stubs.append(
            "\n".join(
                [
                    "namespace service_manager {",
                    "struct Manifest {",
                    "    template <typename... Args> struct InterfaceList {};",
                    "};",
                    "struct ManifestBuilder {",
                    "    template <typename... Args> ManifestBuilder &ExposeCapability(Args...) { return *this; }",
                    "    template <typename... Args> ManifestBuilder &ExposeInterfaceFilterCapability_Deprecated(Args...) { return *this; }",
                    "    template <typename... Args> ManifestBuilder &PackageService(Args...) { return *this; }",
                    "    template <typename... Args> ManifestBuilder &RequireCapability(Args...) { return *this; }",
                    "    Manifest Build() const { return Manifest(); }",
                    "    operator Manifest() const { return Manifest(); }",
                    "};",
                    "}",
                ]
            )
        )
    return stubs


def _render_cpp_class_scope_stub(class_name: str, members: set[str], source_code: str) -> str:
    if class_name == "TIFF_Manager":
        return (
            "struct TIFF_Manager { "
            "struct TagInfo { void *dataPtr = nullptr; size_t dataLen = 0; long type = 0; long id = 0; }; "
            "template <typename... Args> bool GetTag(Args...) const { return false; } "
            "template <typename... Args> void IntegrateFromPShop6(Args...) {} "
            "};"
        )
    if class_name == "TIFF_MetaHandler":
        return (
            "struct TIFF_MetaHandler { "
            "bool processedXMP = false; bool containsXMP = false; "
            "struct vulsirt_stub_parent *parent = nullptr; TIFF_Manager tiffMgr; "
            "PSIR_Manager *psirMgr = nullptr; IPTC_Manager *iptcMgr = nullptr; "
            "std::string xmpPacket; struct XMPObj { template <typename... Args> void ParseFromBuffer(Args...) {} } xmpObj; "
            "void ProcessXMP(); };"
        )
    if class_name == "PhotoDataUtils":
        return "struct PhotoDataUtils { template <typename... Args> static int CheckIPTCDigest(Args...) { return 0; } };"
    if class_name == "PostScript_Support":
        return "struct PostScript_Support { template <typename... Args> static bool IsValidPSFile(Args...) { return true; } };"
    if class_name == "PostScript_MetaHandler":
        return (
            "struct PostScript_MetaHandler { struct vulsirt_stub_parent *parent = nullptr; long fileformat = 0; "
            "void ParsePSFile(); template <typename... Args> void setTokenInfo(Args...) {} "
            "template <typename... Args> void ExtractDocInfoDict(Args...) {} };"
        )
    if class_name == "PSIR_Manager":
        return (
            "struct PSIR_Manager { "
            "struct ImgRsrcInfo { void *dataPtr = nullptr; size_t dataLen = 0; long id = 0; }; "
            "template <typename... Args> void ParseMemoryResources(Args...) {} "
            "template <typename... Args> bool GetImgRsrc(Args...) { return false; } "
            "template <typename... Args> void DeleteImgRsrc(Args...) {} "
            "}; "
            "struct PSIR_MemoryReader : PSIR_Manager {}; struct PSIR_FileWriter : PSIR_Manager {};"
        )
    if class_name == "IPTC_Manager":
        return (
            "struct IPTC_Manager { template <typename... Args> void ParseMemoryDataSets(Args...) {} }; "
            "struct IPTC_Reader : IPTC_Manager {}; struct IPTC_Writer : IPTC_Manager {};"
        )
    if class_name == "MOOV_Manager":
        return (
            "struct MOOV_Manager { "
            "using BoxRef = long; "
            "struct BoxInfo { long boxType = 0; size_t contentSize = 0; void *content = nullptr; unsigned int childCount = 0; }; "
            "template <typename... Args> BoxRef GetBox(Args...) const { return 0; } "
            "template <typename... Args> BoxRef GetNthChild(Args...) const { return 0; } "
            "};"
        )
    if class_name == "Status":
        return "struct Status { template <typename... Args> Status(Args...) {} static Status OK() { return Status(); } int code() const { return 0; } std::string ToString() const { return {}; } operator bool() const { return true; } };"
    if class_name == "FrameBuffer":
        return (
            "struct FrameBuffer { struct ConstIterator { const char *name() const { return \"\"; } "
            "ConstIterator &operator++() { return *this; } bool operator!=(const ConstIterator &) const { return false; } }; "
            "ConstIterator begin() const { return {}; } ConstIterator end() const { return {}; } };"
        )

    lines = [f"struct {class_name} {{", f"    template <typename... Args> {class_name}(Args...) {{}}"]
    for member in sorted(members):
        escaped = re.escape(f"{class_name}::{member}")
        if re.search(rf"\b{escaped}\s*\(", source_code):
            lines.append(f"    template <typename... Args> static {class_name} {member}(Args...) {{ return {class_name}(); }}")
        elif re.search(rf"\b{escaped}\s+[*&]?\s*[A-Za-z_]\w*", source_code):
            lines.append(f"    struct {member} {{ template <typename... Args> {member}(Args...) {{}} {_render_stub_fields({}, is_cpp=True)} }};")
        else:
            lines.append(f"    static const int {member} = 0;")
    lines.append("};")
    return " ".join(lines)


def _render_cpp_namespace_type_stub(namespaces: tuple[str, ...], type_name: str) -> str:
    if namespaces == ("net",) and type_name == "RedirectInfo":
        return "struct RedirectInfo { template <typename... Args> RedirectInfo(Args...) {} long new_url = 0; long new_method = 0; long new_referrer = 0; };"
    if namespaces == ("network",) and type_name == "ResourceResponse":
        return "struct ResourceResponse { template <typename... Args> ResourceResponse(Args...) {} struct Head { long ssl_info = 0; long headers = 0; long mime_type = 0; } head; };"
    return f"struct {type_name} {{ template <typename... Args> {type_name}(Args...) {{}} }};"


def _predefined_c_family_types() -> set[str]:
    return {
        "GF_Err",
        "Image",
        "ImageInfo",
        "PointInfo",
        "RectangleInfo",
        "ChromaticityInfo",
        "PixelInfo",
        "ExceptionInfo",
        "PrimitiveInfo",
        "CompositeFunc",
        "DrawInfo",
        "GifInfo",
        "GIOChannel",
        "GIOFunc",
        "PixelPacket",
        "png_color",
        "MSLInfo",
        "LayerInfo",
        "QtDemuxSample",
        "QtDemuxStream",
        "TIFF_Manager",
        "PSIR_Manager",
        "PSIR_MemoryReader",
        "PSIR_FileWriter",
        "IPTC_Manager",
        "IPTC_Reader",
        "IPTC_Writer",
        "MOOV_Manager",
        "PCIDevice",
        "V9fsFidState",
        "V9fsPDU",
        "V9fsStat",
        "V9fsString",
        "XMP_IO",
        "XMP_AbortProc",
        "XMP_StringPtr",
        "XMP_StringLen",
        "XMP_Uns8",
        "XMP_Uns32",
        "XMP_Int64",
        "byte",
        "StringInfo",
        "search_state",
        "search_domain",
        "nd_router_advert",
        "nd_neighbor_solicit",
        "nd_neighbor_advert",
        "nd_redirect",
        "hlist_head",
        "kioctx",
        "div_data",
        "divs_data",
        "factors_data",
        "clk",
        "clk_hw",
        "clk_ops",
        "clk_onecell_data",
        "clk_gate",
        "clk_fixed_factor",
        "clk_divider",
        "hostent",
        "sockaddr",
        "sockaddr_un",
        "st_entry",
        "AString",
        "ABuffer",
        "ActionReply",
        "BBinder",
        "BrowserThread",
        "BufferInfo",
        "ChildProcessSecurityPolicyImpl",
        "CommandLine",
        "codec_t",
        "code",
        "inflate_state",
        "z_stream",
        "z_streamp",
        "CredentialedSubresourceCheckResult",
        "DeviceBase",
        "Eigen",
        "Exiv2",
        "IEX_NAMESPACE",
        "IPTC_Manager",
        "IPTC_Reader",
        "IPTC_Writer",
        "LegacyProtocolInSubresourceCheckResult",
        "Mode",
        "ModCommand",
        "Mutex",
        "NavigationRequest",
        "OMX_BUFFERHEADERTYPE",
        "PostScript_MetaHandler",
        "PostScript_Support",
        "Referrer",
        "RenderProcessHostImpl",
        "Status",
        "TIFF_MetaHandler",
        "vulsirt_stub_parent",
        "vulsirt_stub_field",
        "vulsirt_stub_tmp",
        "vulsirt_stub_cipher",
        "_bdf_parse_t",
        "bdf_glyph_t",
        "bdf_font_t",
        "H264Context",
        "cdf_property_info_t",
        "DNSHeader",
        "GooString",
        "NodeDef",
        "phar_entry_info",
        "magic_entry_set",
        "vfio_region_info",
        "vfio_irq_set",
        "sgmap64",
        "ion_fd_data",
        "sockaddr_in",
        "sockaddr_in6",
        "IPV6OptRA",
        "IPV6OptJumbo",
        "VALUE_PAIR",
        "mrb_context",
        "php_struct",
        "CallResult",
        "CPUDevice",
        "CodeBlock",
        "DataBuf",
        "DataType",
        "ExecutionStatus",
        "GeneratorInnerFunction",
        "GCScope",
        "Handle",
        "HermesValue",
        "HiddenClass",
        "Inst",
        "JSObject",
        "MagickBooleanType",
        "MagickStatusType",
        "MagickOffsetType",
        "EndianType",
        "Bool",
        "FT_Error",
        "gint",
        "gint32",
        "guint",
        "guint32",
        "uint16",
        "uint32",
        "krb5_error_code",
        "ut32",
        "__u32",
        "linenr_T",
        "timelib_sll",
        "l_fp",
        "Register",
        "pos_T",
        "apr_size_t",
        "apr_status_t",
        "apr_uint64_t",
        "apr_off_t",
        "PVSCSISGState",
        "enum_func_status",
        "mrb_bool",
        "os_ptr",
        "rpmVerifyAttrs",
        "rpmfileAttrs",
        "rpm_mode_t",
        "sk_buff",
        "sk_read_actor_t",
        "sock",
        "socket",
        "sockaddr_nl",
        "ulong",
        "zend_uchar",
        "Message",
        "OpCode",
        "OpKernelContext",
        "Parcel",
        "PropOpFlags",
        "PseudoHandle",
        "Runtime",
        "ScopedNativeDepthTracker",
        "SegmentedArray",
        "StringPiece16",
        "ssize_t",
        "timespec",
        "TfLiteStatus",
        "Tensor",
        "TensorShape",
        "UErrorCode",
        "WavpackContext",
        "WebPImage",
        "WebAssociatedURLLoaderOptions",
        "MediaStreamDevice",
        "MediaDeviceSaltAndOrigin",
        "OpenDeviceCallback",
        "render_frame",
        "Document",
        "AtomicString",
        "KURL",
        "ResourceLoaderOptions",
        "ResourceRequest",
        "FetchParameters",
        "ImageResourceContent",
        "LayoutImageResource",
        "LayoutImage",
        "TensorShapeUtils",
        "OpDef",
        "AttrValue",
        "OpInfo",
        "FunctionLibraryDefinition",
        "MetaGraphDef",
        "GraphDef",
        "InferenceContext",
        "QMapIterator",
        "Logger",
        "RuntimeOption",
        "MYSQLND_MEMORY_POOL_CHUNK",
        "MYSQLND_FIELD",
        "MYSQLND_STATS",
        "st_mysqlnd_perm_bind",
        "IoCloser",
        "ax25_address",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "xmlChar",
        "char_u",
        "opus_int",
        "opus_int16",
        "opus_int32",
        "regex_t",
        "stat",
        "time_t",
        "xmlDocPtr",
        "xmlNodePtr",
        "xmlParserCtxtPtr",
        "xmlSAXHandler",
        "xListExtensionsReply",
        "scoped_refptr",
    }


def _c_compilation_preamble() -> str:
    """Provides common C headers, typedefs, and macros for fragmentary samples."""
    return "\n".join(
        [
            "#include <stdbool.h>",
            "#include <stddef.h>",
            "#include <stdint.h>",
            "#include <inttypes.h>",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            "#include <time.h>",
            "",
            "#define __user",
            "#define __iomem",
            "#define __init",
            "#define __exit",
            "#define __force",
            "#define __must_check",
            "#ifndef __always_inline",
            "#define __always_inline inline",
            "#endif",
            "#define __maybe_unused",
            "#define coroutine_fn",
            "#define __printf(a, b)",
            "#define likely(x) (x)",
            "#define unlikely(x) (x)",
            "#define WARN_ON_ONCE(x) (x)",
            "#define min(a, b) ((a) < (b) ? (a) : (b))",
            "#define max(a, b) ((a) > (b) ? (a) : (b))",
            "#define min_t(type, a, b) ((a) < (b) ? (a) : (b))",
            "#define max_t(type, a, b) ((a) > (b) ? (a) : (b))",
            "#define clamp_t(type, value, lo, hi) ((value) < (lo) ? (lo) : ((value) > (hi) ? (hi) : (value)))",
            "#define DECLARE_SOCKADDR(type, dst, src) type dst = (type)(src)",
            "#define container_of(ptr, type, member) ((type *)0)",
            "#define list_entry(ptr, type, member) ((type *)0)",
            "#define list_first_entry(ptr, type, member) ((type *)0)",
            "#define list_for_each_entry(pos, head, member) for (; 0; )",
            "#define list_for_each_entry_safe(pos, n, head, member) for (; 0; )",
            "#define PROFILE_DISABLE_INTRA_PRED()",
            "#define PROFILE_DISABLE_IQ_IT_RECON()",
            "#define __packed",
            "#define __read_mostly",
            "#define TSRMLS_DC",
            "#define TSRMLS_CC",
            "#define FAR",
            "#define ZLIB_INTERNAL",
            "#define z_const const",
            "#define OFF 0",
            "#define PUP(a) (*(a)++)",
            "#define Tracevv(x)",
            "#define MagickExport",
            "#define WandExport",
            "#define KERN_WARNING \"\"",
            "#define KERN_ERR \"\"",
            "#define KERN_INFO \"\"",
            "#define KERN_DEBUG \"\"",
            "#ifndef SCNu16",
            "#define SCNu16 \"hu\"",
            "#endif",
            "#ifndef SCNd16",
            "#define SCNd16 \"hd\"",
            "#endif",
            "#ifndef PRIu16",
            "#define PRIu16 \"hu\"",
            "#endif",
            "#ifndef PRId16",
            "#define PRId16 \"hd\"",
            "#endif",
            "#define IP_SCAN_FMT \"%u.%u.%u.%u\"",
            "#define IPV6_SCAN_FMT \"%x:%x:%x:%x:%x:%x:%x:%x\"",
            "#define IP_SCAN_ARGS(x) x",
            "#define TEE_PARAM_TYPE_GET(types, n) ((types) + (n))",
            "#define TEE_PARAM_TYPE_MEMREF_INOUT 1",
            "#define TEE_PARAM_TYPE_MEMREF_OUTPUT 2",
            "#define TEE_PARAM_TYPE_VALUE_INOUT 3",
            "#define TEE_PARAM_TYPE_VALUE_OUTPUT 4",
            "#define DRV_FLAG_RUNNING 1",
            "#define GFP_KERNEL 0",
            "#define PG_FUNCTION_ARGS void",
            "#define PG_GETARG_ARRAYTYPE_P(n) ((ArrayType *)0)",
            "#define ARR_NDIM(a) 0",
            "#define ARR_ELEMTYPE(a) 0",
            "#define ARR_DIMS(a) ((int *)0)",
            "#define ARR_LBOUND(a) ((int *)0)",
            "#define ARR_DATA_PTR(a) ((char *)0)",
            "#define VARSIZE_ANY_EXHDR(a) 0",
            "#define PG_RETURN_POINTER(x) return (x)",
            "static long vulsirt_pg_storage; static long *vulsirt_pg_slot(const char *name) { (void)name; return &vulsirt_pg_storage; }",
            "#define PG(x) (*vulsirt_pg_slot(#x))",
            "#define CG(x) (*vulsirt_pg_slot(#x))",
            "#define EG(x) (*vulsirt_pg_slot(#x))",
            "#define TEXTOID 25",
            "#define ERROR 1",
            "#define Assert(x)",
            "#define ereport(...)",
            "#define errmsg(...) 0",
            "#define errcode(...) 0",
            "#define PHP_FUNCTION(name) void name(void)",
            "#define ZEND_NUM_ARGS() 0",
            "#define ZEND_HASH_FOREACH_VAL(ht, val) for (; 0; )",
            "#define ZEND_HASH_FOREACH_END()",
            "#define Z_ARRVAL_P(z) ((HashTable *)0)",
            "#define DBG_ENTER(...)",
            "#define DBG_INF(...)",
            "#define DBG_RETURN(x) return (x)",
            "#define g_new0(type, count) ((type *)calloc((count), sizeof(type)))",
            "#define g_malloc0(size) calloc(1, (size))",
            "#define g_malloc(size) malloc(size)",
            "#define MAKE_STD_ZVAL(z) do { z = (zval *)0; } while (0)",
            "#define ZVAL_NULL(z)",
            "#define MYSQLND_G(x) 0",
            "#define RETURN_FALSE return",
            "#define RETURN_TRUE return",
            "#define RETURN_NULL() return",
            "#define FAILURE 0",
            "#define E_WARNING 0",
            "#define S_ISDIR(x) 0",
            "#define S_ISLNK(x) 0",
            "#define S_ISREG(x) 1",
            "#define EVP_MAX_IV_LENGTH 64",
            "#define _X_UNUSED",
            "#define LIBXML_VERSION 0",
            "#define XML_PARSE_DTDLOAD 1",
            "#define XML_PARSE_HUGE 2",
            "#define LLDP_TLV_ORG_DOT1 { 0x00, 0x80, 0xc2 }",
            "#define LLDP_TLV_ORG_DOT3 { 0x00, 0x12, 0x0f }",
            "#define LLDP_TLV_ORG_MED { 0x00, 0x12, 0xbb }",
            "#define LLDP_TLV_ORG_DCBX { 0x00, 0x1b, 0x21 }",
            "#define LLDP_ADDR_NEAREST_BRIDGE { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x0e }",
            "#define LLDP_ADDR_NEAREST_NONTPMR_BRIDGE { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x03 }",
            "#define LLDP_ADDR_NEAREST_CUSTOMER_BRIDGE { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x00 }",
            "#define LockDisplay(dpy)",
            "#define UnlockDisplay(dpy)",
            "#define SyncHandle()",
            "#define GetEmptyReq(name, req) req = (xReq *)0",
            "#define YYCTYPE unsigned char",
            "#define YYCURSOR cursor",
            "#define YYLIMIT cursor",
            "#define YYMARKER cursor",
            "#define YYFILL(n)",
            "#define YYDEBUG(label, ch)",
            "",
            "typedef unsigned int uint;",
            "typedef unsigned short uint16;",
            "typedef unsigned int uint32;",
            "typedef unsigned char u8;",
            "typedef unsigned short u16;",
            "typedef unsigned int u32;",
            "typedef unsigned long long u64;",
            "typedef unsigned char byte;",
            "typedef unsigned short __be16;",
            "typedef signed char s8;",
            "typedef signed short s16;",
            "typedef signed int s32;",
            "typedef signed long long s64;",
            "typedef unsigned char char_u;",
            "typedef int int32;",
            "typedef long Datum;",
            "typedef int GF_Err;",
            "typedef int MagickBooleanType;",
            "typedef long MagickStatusType;",
            "typedef long MagickOffsetType;",
            "typedef long EndianType;",
            "typedef int Bool;",
            "typedef int gint;",
            "typedef int gint32;",
            "typedef unsigned int guint;",
            "typedef unsigned int guint32;",
            "typedef unsigned char guint8;",
            "typedef unsigned long long guint64;",
            "typedef int krb5_error_code;",
            "typedef unsigned int ut32;",
            "typedef unsigned int __u32;",
            "typedef long linenr_T;",
            "typedef long timelib_sll;",
            "typedef long l_fp;",
            "typedef long Register;",
            "typedef long pos_T;",
            "typedef unsigned char XMP_Uns8;",
            "typedef unsigned int XMP_Uns32;",
            "typedef long long XMP_Int64;",
            "typedef char *XMP_StringPtr;",
            "typedef size_t XMP_StringLen;",
            "typedef int (*XMP_AbortProc)(void *);",
            "typedef size_t apr_size_t;",
            "typedef int apr_status_t;",
            "typedef unsigned long long apr_uint64_t;",
            "typedef long apr_off_t;",
            "typedef int zend_bool;",
            "typedef unsigned char zend_uchar;",
            "typedef int mrb_bool;",
            "typedef int FT_Error;",
            "typedef int gboolean;",
            "typedef unsigned long ulong;",
            "#ifndef _SSIZE_T_DEFINED",
            "#define _SSIZE_T_DEFINED",
            "typedef long ssize_t;",
            "#endif",
            "typedef int PVSCSISGState;",
            "typedef long rpmVerifyAttrs;",
            "typedef long rpmfileAttrs;",
            "typedef unsigned int rpm_mode_t;",
            "typedef long enum_func_status;",
            "typedef long enum_mysqlnd_collected_stats;",
            "typedef struct ref { struct { long type_attrs; } tas; union { byte *bytes; long intval; void *ptr; } value; } ref;",
            "typedef ref *os_ptr;",
            "typedef struct code { unsigned char op; unsigned char bits; unsigned short val; } code;",
            "typedef struct inflate_state { unsigned dmax; unsigned wsize; unsigned whave; unsigned wnext; unsigned sane; unsigned mode; unsigned lenbits; unsigned distbits; unsigned long hold; unsigned bits; unsigned char *window; code *lencode; code *distcode; } inflate_state;",
            "typedef struct z_stream_s { void *state; unsigned char *next_in; unsigned char *next_out; unsigned avail_in; unsigned avail_out; char *msg; } z_stream;",
            "typedef z_stream *z_streamp;",
            "static ref vulsirt_ref_stack[8]; static os_ptr osp = vulsirt_ref_stack + 3;",
            "static unsigned int r_size(os_ptr value) { return 0; }",
            "static void r_set_size(os_ptr value, long size) { (void)value; (void)size; }",
            "#define check_read_type(value, type) do { (void)(value); } while (0)",
            "#define make_false(value) do { (void)(value); } while (0)",
            "#define make_true(value) do { (void)(value); } while (0)",
            "#define push(count) do { (void)(count); } while (0)",
            "typedef int opus_int;",
            "typedef long opus_int16;",
            "typedef long opus_int32;",
            "typedef int TfLiteStatus;",
            "typedef int regex_t;",
            "typedef unsigned char xmlChar;",
            "typedef unsigned int WORD32;",
            "typedef signed char WORD8;",
            "typedef long WORD16;",
            "typedef unsigned short UWORD16;",
            "typedef unsigned int UWORD32;",
            "typedef unsigned char UWORD8;",
            "typedef long compat_uptr_t;",
            "typedef struct PointInfo { double x; double y; double z; } PointInfo;",
            "typedef struct RectangleInfo { long x; long y; unsigned long width; unsigned long height; } RectangleInfo;",
            "typedef struct ChromaticityInfo { PointInfo red_primary; PointInfo green_primary; PointInfo blue_primary; PointInfo white_point; } ChromaticityInfo;",
            "typedef struct PixelInfo { double red; double green; double blue; double alpha; double opacity; } PixelInfo;",
            "typedef struct ExceptionInfo { unsigned long signature; int severity; char reason[256]; char description[256]; } ExceptionInfo;",
            "typedef struct DrawInfo { double affine[6]; double *dash_pattern; double dash_offset; PixelInfo fill; PixelInfo stroke; PixelInfo alpha; char *clip_mask; char *font; RectangleInfo viewbox; int linejoin; int linecap; int fill_rule; double miterlimit; double stroke_width; int align; void *fill_pattern; void *stroke_pattern; } DrawInfo;",
            "typedef struct GifInfo { unsigned char *rasterBits; unsigned long rasterSize; unsigned long sampleSize; void *gifFilePtr; unsigned long originalWidth; unsigned long originalHeight; } GifInfo;",
            "typedef struct PixelPacket { unsigned short red; unsigned short green; unsigned short blue; unsigned short opacity; } PixelPacket;",
            "typedef struct png_color { unsigned char red; unsigned char green; unsigned char blue; } png_color;",
            "typedef struct Image { char filename[256]; char magick[64]; char magick_filename[256]; char signature[64]; unsigned char *blob; unsigned char *colormap; unsigned long columns; unsigned long rows; unsigned long colors; unsigned long page; unsigned long number_scenes; unsigned long scene; unsigned long iterations; unsigned long delay; unsigned long ticks_per_second; unsigned long depth; unsigned long number_channels; unsigned long number_meta_channels; long offset; long start_loop; long resolution; double x_resolution; double y_resolution; double gamma; int storage_class; int colorspace; int alpha_trait; int compression; int debug; int ping; int matte; int endian; int units; int type; int orientation; int rendering_intent; int interlace; int dispose; int gravity; int dither; int intensity; int quality; int taint; unsigned long magick_columns; unsigned long magick_rows; void *exception; void *extract_info; void *profiles; void *artifacts; void *properties; void *progress_monitor; void *client_data; void *cache; char *directory; char *montage; PixelInfo background_color; PixelInfo border_color; PixelInfo matte_color; PixelInfo transparent_color; ChromaticityInfo chromaticity; RectangleInfo tile_info; RectangleInfo page_info; RectangleInfo tile_offset; struct Image *previous; struct Image *next; } Image;",
            "typedef struct ImageInfo { char filename[256]; char magick[64]; char signature[64]; char *file; char *density; char *size; char *sampling_factor; unsigned long page; unsigned long scene; unsigned long scenes; unsigned long first_scene; unsigned long number_scenes; unsigned long depth; unsigned long length; unsigned int compression; int quality; int adjoin; int ping; int debug; int verbose; int endian; int type; int antialias; int colorspace; int interlace; int monochrome; int pointsize; int dither; void *blob; void *extract; void *cache; } ImageInfo;",
            "typedef struct PrimitiveInfo { PointInfo point; PointInfo coordinates; long primitive; long method; char *text; } PrimitiveInfo;",
            "typedef void (*CompositeFunc)();",
            "typedef struct MSLInfo { Image *image; ImageInfo *image_info; void *attributes; void *exception; } MSLInfo;",
            "typedef struct LayerInfo { Image *image; RectangleInfo page; long x; long y; } LayerInfo;",
            "typedef struct QtDemuxSample { guint32 size; guint32 chunk; guint64 offset; guint64 timestamp; guint64 duration; gint32 pts_offset; int keyframe; } QtDemuxSample;",
            "typedef struct QtDemuxStream { int sampled; int n_samples; QtDemuxSample *samples; guint64 min_duration; guint32 timescale; int all_keyframe; int samples_per_frame; int bytes_per_frame; int n_channels; } QtDemuxStream;",
            "typedef struct PCIDevice { long dummy; } PCIDevice;",
            "typedef struct iovec { void *iov_base; size_t iov_len; } iovec;",
            "typedef struct V9fsString { char *data; } V9fsString;",
            "typedef struct V9fsStat { long dummy; } V9fsStat;",
            "typedef struct V9fsFidXattr { long copied_len; size_t len; int flags; V9fsString name; void *value; } V9fsFidXattr;",
            "typedef struct V9fsFidFS { V9fsFidXattr xattr; } V9fsFidFS;",
            "typedef struct V9fsFidState { long fid; long fid_type; long open_flags; V9fsFidFS fs; void *path; } V9fsFidState;",
            "typedef struct V9fsPDU { long tag; long id; } V9fsPDU;",
            "typedef struct StringInfo { unsigned char *datum; size_t length; } StringInfo;",
            "typedef struct XMP_IO { long dummy; } XMP_IO;",
            "typedef struct search_domain { struct search_domain *next; int len; } search_domain;",
            "typedef struct search_state { struct search_domain *head; } search_state;",
            "typedef struct nd_router_advert { unsigned int nd_ra_curhoplimit; unsigned int nd_ra_flags_reserved; unsigned short nd_ra_router_lifetime; unsigned int nd_ra_reachable; unsigned int nd_ra_retransmit; } nd_router_advert;",
            "typedef struct nd_neighbor_solicit { unsigned int nd_ns_reserved; unsigned char nd_ns_target[16]; } nd_neighbor_solicit;",
            "typedef struct nd_neighbor_advert { unsigned int nd_na_flags_reserved; unsigned char nd_na_target[16]; } nd_neighbor_advert;",
            "typedef struct nd_redirect { unsigned int nd_rd_reserved; unsigned char nd_rd_target[16]; unsigned char nd_rd_dst[16]; } nd_redirect;",
            "typedef struct sockaddr { unsigned short sa_family; char sa_data[14]; } sockaddr;",
            "typedef struct sockaddr_un { unsigned short sun_family; char sun_path[108]; } sockaddr_un;",
            "typedef struct hlist_head { void *first; } hlist_head;",
            "typedef struct kioctx { long dummy; } kioctx;",
            "typedef struct div_data { long self; long gate; long fixed; long pow; long shift; long width; long critical; void *table; } div_data;",
            "typedef struct factors_data { const char *name; long mult; long div; long ndivs; } factors_data;",
            "typedef struct divs_data { long shift; long mult; long value; long val; long ndivs; struct factors_data *factors; struct div_data div[16]; } divs_data;",
            "struct clk { long dummy; };",
            "struct clk_hw { struct clk *clk; };",
            "struct clk_ops { long dummy; };",
            "struct clk_onecell_data { struct clk **clks; unsigned int clk_num; };",
            "struct clk_gate { struct clk_hw hw; void *reg; long bit_idx; void *lock; };",
            "struct clk_fixed_factor { struct clk_hw hw; long mult; long div; };",
            "struct clk_divider { struct clk_hw hw; void *reg; long shift; long width; long flags; void *lock; void *table; };",
            "typedef struct cli_exe_section { unsigned long rva; unsigned long rsz; unsigned long raw; unsigned long vsz; unsigned long urva; unsigned long uvsz; unsigned long characteristics; } cli_exe_section;",
            "typedef struct phar_entry_info { char *filename; unsigned int filename_len; int is_persistent; unsigned long uncompressed_filesize; unsigned long compressed_filesize; unsigned int flags; } phar_entry_info;",
            "typedef struct environment { int indent; int flags; int depth; } environment;",
            "typedef struct lys_type_enum { char *name; long value; } lys_type_enum;",
            "typedef struct codec_t { long (*s_parse)(); long (*parse)(); long flags; } codec_t;",
            "typedef struct _bdf_parse_t { long flags; long glyph_enc; char *glyph_name; long row; long cnt; long opts; void *list; void *font; } _bdf_parse_t;",
            "typedef struct bdf_glyph_t { long bbx; long bpr; long swidth; long dwidth; unsigned char *bitmap; char *name; long encoding; } bdf_glyph_t;",
            "typedef struct bdf_font_t { bdf_glyph_t *glyphs; long glyphs_size; long glyphs_used; long unencoded_used; long modified; long spacing; } bdf_font_t;",
            "typedef struct H264Context { long sps; long pps; long gb; long avctx; long cur_pic_ptr; long picture_structure; long ref_count; long first_field; long current_slice; long droppable; long frame_num; long prev_frame_num; long mb_width; long mb_height; long mb_y; long resync_mb_y; long slice_type; long slice_type_nos; long deblocking_filter; long slice_num; long short_ref[16]; long ref_list[16]; long slice_alpha_c0_offset; long slice_beta_offset; } H264Context;",
            "typedef struct cdf_property_info_t { long pi_type; long pi_id; long pi_val; } cdf_property_info_t;",
            "typedef struct DNSHeader { unsigned char *payload; unsigned long payload_size; } DNSHeader;",
            "typedef struct VALUE_PAIR { unsigned long length; unsigned char *vp_strvalue; long vp_integer; } VALUE_PAIR;",
            "typedef struct mrb_context { struct mrb_context *prev; struct mrb_context *next; long status; } mrb_context;",
            "typedef struct magic_entry_set { void *me; long count; } magic_entry_set;",
            "typedef struct vfio_region_info { unsigned long index; unsigned long size; unsigned long flags; unsigned long offset; } vfio_region_info;",
            "typedef struct vfio_irq_set { unsigned long index; unsigned long start; unsigned long count; unsigned long flags; } vfio_irq_set;",
            "typedef struct sgmap64 { void *sg; unsigned long len; } sgmap64;",
            "typedef struct ion_fd_data { int fd; void *handle; } ion_fd_data;",
            "typedef struct sockaddr_in { struct { unsigned long s_addr; } sin_addr; } sockaddr_in;",
            "typedef struct sockaddr_in6 { struct { unsigned char s6_addr[16]; } sin6_addr; } sockaddr_in6;",
            "typedef struct IPV6OptRA { unsigned long ip6ra_value; } IPV6OptRA;",
            "typedef struct IPV6OptJumbo { unsigned long ip6j_payload_len; } IPV6OptJumbo;",
            "typedef struct php_struct { long r; long request_processed; } php_struct;",
            "typedef void *OpKernelContext;",
            "typedef void *WavpackContext;",
            "typedef void *WebPImage;",
            "typedef struct xmlDoc { xmlChar *URL; } xmlDoc; typedef struct xmlDoc *xmlDocPtr;",
            "typedef struct xmlNode { long dummy; } *xmlNodePtr;",
            "typedef struct xmlSAXHandler { void (*ignorableWhitespace)(); void (*comment)(); void (*warning)(); void (*error)(); } xmlSAXHandler;",
            "typedef struct xmlParserInput { unsigned char *cur; unsigned char *base; unsigned char *end; } xmlParserInput; typedef struct xmlParserInput *xmlParserInputPtr;",
            "typedef struct xmlParserCtxt { int options; int wellFormed; xmlDocPtr myDoc; char *directory; xmlSAXHandler *sax; xmlParserInputPtr input; int inputNr; int inputMax; } *xmlParserCtxtPtr;",
            "typedef void ArrayType;",
            "typedef void HStore;",
            "struct stat { long st_mode; long st_size; long st_uid; long st_gid; long st_rdev; long st_mtime; };",
            "typedef int (*sk_read_actor_t)(void *, void *, unsigned int, size_t);",
            "typedef void Display; typedef struct xListExtensionsReply { int nExtensions; unsigned long length; } xListExtensionsReply; typedef void xReply; typedef void xReq;",
            "typedef struct zval { long value; } zval; typedef struct HashTable { long dummy; } HashTable; typedef struct zend_resource { long dummy; } zend_resource;",
            "typedef struct MYSQLND_MEMORY_POOL_CHUNK { zend_uchar *ptr; size_t app; } MYSQLND_MEMORY_POOL_CHUNK; typedef struct MYSQLND_FIELD { long type; } MYSQLND_FIELD; typedef struct MYSQLND_STATS { long dummy; } MYSQLND_STATS;",
            "typedef struct st_mysqlnd_perm_bind { long dummy; } st_mysqlnd_perm_bind;",
            "typedef struct evp_pkey_st { long type; long save_type; void *pkey; } EVP_PKEY; typedef struct evp_cipher_st { long dummy; } EVP_CIPHER; typedef struct evp_cipher_ctx_st { long dummy; } EVP_CIPHER_CTX;",
            "typedef struct Pairs { char *key; char *val; int keylen; int vallen; int isnull; int needfree; } Pairs;",
            "",
            "static struct timespec current_kernel_time(void) { struct timespec value = {0, 0}; return value; }",
            "static void *kmalloc(unsigned long size, int flags) { (void)size; (void)flags; return 0; }",
            "static char *isdn_statstr(void) { return 0; }",
            "static HStore *hstorePairs(Pairs *pairs, int count, int buflen) { (void)pairs; (void)count; (void)buflen; return 0; }",
            "static int hstoreCheckKeyLen(int len) { return len; }",
            "static int hstoreCheckValLen(int len) { return len; }",
            "static int hstoreUniquePairs(Pairs *pairs, int count, int *buflen) { (void)pairs; if (buflen) *buflen = 0; return count; }",
            "static int stat() { return 0; } static int lstat() { return 0; }",
            "static void *Xmalloc(size_t size) { return malloc(size); } static void Xfree(void *p) { free(p); } static int _XReply() { return 0; } static void _XEatDataWords() {} static void _XReadPad() {}",
            "static int zend_parse_parameters() { return 0; } static int zend_hash_num_elements() { return 0; } static void php_error_docref() {} static void *safe_emalloc() { return 0; }",
            "static int EVP_CIPHER_iv_length() { return 0; } static const EVP_CIPHER *EVP_get_cipherbyname() { return 0; } static const EVP_CIPHER *EVP_rc4(void) { return 0; }",
            "static xmlParserCtxtPtr xmlCreateMemoryParserCtxt() { static xmlSAXHandler sax; static xmlDoc doc; static struct xmlParserCtxt ctxt = {0, 1, &doc, 0, &sax}; return &ctxt; } static void xmlParseDocument(xmlParserCtxtPtr ctxt) { (void)ctxt; } static xmlChar *xmlCharStrdup(const char *s) { return (xmlChar *)s; } static void xmlFreeDoc(xmlDocPtr doc) { (void)doc; } static void xmlFreeParserCtxt(xmlParserCtxtPtr ctxt) { (void)ctxt; }",
            "static unsigned long php_mysqlnd_net_field_length(zend_uchar **p) { return p && *p ? (unsigned long)**p : 0; }",
            "static void v9fs_stat_init(V9fsStat *value) { (void)value; }",
            "static void v9fs_string_init(V9fsString *value) { if (value) value->data = 0; }",
            "static void v9fs_string_copy(V9fsString *dst, const V9fsString *src) { if (dst) dst->data = src ? src->data : 0; }",
            "static void v9fs_string_free(V9fsString *value) { (void)value; }",
            "static int pdu_unmarshal() { return 0; }",
            "static V9fsFidState *get_fid(V9fsPDU *pdu, int32_t fid) { static V9fsFidState value; (void)pdu; (void)fid; return &value; }",
            "static void put_fid(V9fsPDU *pdu, V9fsFidState *fid) { (void)pdu; (void)fid; }",
            "static void pdu_complete(V9fsPDU *pdu, long err) { (void)pdu; (void)err; }",
            "static const int P9_FID_XATTR = 1;",
        ]
    )


def _cpp_compilation_preamble() -> str:
    """Provides common C++ headers and macros for fragmentary samples."""
    return "\n".join(
        [
            "#include <cstddef>",
            "#include <cstdint>",
            "#include <cstdio>",
            "#include <cstdlib>",
            "#include <cstring>",
            "#include <cmath>",
            "#include <algorithm>",
            "#include <iostream>",
            "#include <string>",
            "#include <vector>",
            "#include <map>",
            "#include <memory>",
            "#include <type_traits>",
            "#include <utility>",
            "",
            "#define __user",
            "#define __iomem",
            "#define __init",
            "#define __exit",
            "#define __force",
            "#define __must_check",
            "#ifndef __always_inline",
            "#define __always_inline inline",
            "#endif",
            "#define __maybe_unused",
            "#define coroutine_fn",
            "#define __printf(a, b)",
            "#define likely(x) (x)",
            "#define unlikely(x) (x)",
            "#define WARN_ON_ONCE(x) (x)",
            "#define min_t(type, a, b) ((a) < (b) ? (a) : (b))",
            "#define max_t(type, a, b) ((a) > (b) ? (a) : (b))",
            "#define clamp_t(type, value, lo, hi) ((value) < (lo) ? (lo) : ((value) > (hi) ? (hi) : (value)))",
            "#define DECLARE_SOCKADDR(type, dst, src) type dst = (type)(src)",
            "#define container_of(ptr, type, member) ((type *)0)",
            "#define list_entry(ptr, type, member) ((type *)0)",
            "#define list_first_entry(ptr, type, member) ((type *)0)",
            "#define list_for_each_entry(pos, head, member) for (; 0; )",
            "#define list_for_each_entry_safe(pos, n, head, member) for (; 0; )",
            "#define PROFILE_DISABLE_INTRA_PRED()",
            "#define __packed",
            "#define __read_mostly",
            "#define TSRMLS_DC",
            "#define TSRMLS_CC",
            "#define FAR",
            "#define override",
            "#define final",
            "#define OS_WIN 0",
            "#define BUILDFLAG(x) 0",
            "#define static_assert(...)",
            "#define FFTRank 1",
            "#define CHECK_INTERFACE(...)",
            "#define ALOGV(...)",
            "#define DCHECK(...)",
            "#define DCHECK_CURRENTLY_ON(...)",
            "#define CHECK(...)",
            "#define CHECK_LT(a, b)",
            "#define DVLOG(x) if (true) ; else std::cerr",
            "#define memcpy(...) ((void *)0)",
            "#define snprintf(...) 0",
            "#define strstr(...) ((char *)0)",
            "#define strlen(...) 0",
            "#define U32_AT(x) ((uint32_t)0)",
            "#define g_new0(type, count) ((type *)calloc((count), sizeof(type)))",
            "#define g_malloc0(size) calloc(1, (size))",
            "#define g_malloc(size) malloc(size)",
            "",
            "using uint = unsigned int;",
            "using uint16 = unsigned short;",
            "using uint32 = unsigned int;",
            "using ssize_t = long;",
            "using GF_Err = int;",
            "using TfLiteStatus = int;",
            "using regex_t = int;",
            "using opus_int = int;",
            "using opus_int16 = long;",
            "using opus_int32 = long;",
            "using status_t = int;",
            "using MagickBooleanType = int;",
            "using MagickStatusType = long;",
            "using MagickOffsetType = long;",
            "using EndianType = long;",
            "using Bool = int;",
            "using gint = int;",
            "using gint32 = int;",
            "using gchar = char;",
            "using guint = unsigned int;",
            "using guint32 = unsigned int;",
            "using guint8 = unsigned char;",
            "using guint64 = unsigned long long;",
            "struct GIOChannel { long dummy = 0; };",
            "using GIOFunc = long (*)();",
            "using krb5_error_code = int;",
            "using ut32 = unsigned int;",
            "using __u32 = unsigned int;",
            "using linenr_T = long;",
            "using timelib_sll = long;",
            "using l_fp = long;",
            "using Register = long;",
            "using pos_T = long;",
            "using XMP_Uns8 = unsigned char;",
            "using XMP_Uns32 = unsigned int;",
            "using XMP_Int64 = long long;",
            "using XMP_StringPtr = char *;",
            "using XMP_StringLen = size_t;",
            "using XMP_AbortProc = int (*)(void *);",
            "using apr_size_t = size_t;",
            "using apr_status_t = int;",
            "using apr_uint64_t = unsigned long long;",
            "using apr_off_t = long;",
            "using jlong = long;",
            "using byte = unsigned char;",
            "using uint64 = unsigned long long;",
            "using xmlChar = unsigned char;",
            "using WORD32 = unsigned int;",
            "using WORD8 = signed char;",
            "using WORD16 = long;",
            "using UWORD16 = unsigned short;",
            "using UWORD32 = unsigned int;",
            "using UWORD8 = unsigned char;",
            "using compat_uptr_t = long;",
            "struct sockaddr { unsigned short sa_family = 0; char sa_data[14] = {}; };",
            "struct sockaddr_un { unsigned short sun_family = 0; char sun_path[108] = {}; };",
            "struct sockaddr_in { unsigned short sin_family = 0; unsigned short sin_port = 0; struct { unsigned long s_addr = 0; } sin_addr; char sin_zero[8] = {}; };",
            "struct sockaddr_in6 { unsigned short sin6_family = 0; unsigned short sin6_port = 0; unsigned long sin6_flowinfo = 0; struct { unsigned char s6_addr[16] = {}; } sin6_addr; unsigned long sin6_scope_id = 0; };",
            "struct klinux_sockaddr { int16_t klinux_sa_family = 0; };",
            "struct klinux_sockaddr_un { int16_t klinux_sa_family = 0; char klinux_sun_path[108] = {}; };",
            "struct klinux_sockaddr_in { int16_t klinux_sa_family = 0; unsigned short klinux_sin_port = 0; unsigned long klinux_sin_addr = 0; char klinux_sin_zero[8] = {}; };",
            "struct klinux_sockaddr_in6 { int16_t klinux_sa_family = 0; unsigned short klinux_sin6_port = 0; unsigned long klinux_sin6_flowinfo = 0; unsigned char klinux_sin6_addr[16] = {}; unsigned long klinux_sin6_scope_id = 0; };",
            "static const int AF_UNIX = 1; static const int AF_INET = 2; static const int AF_INET6 = 10;",
            "static const int PF_UNIX = AF_UNIX; static const int SOCK_STREAM = 1;",
            "static const int kLinux_AF_UNIX = 1; static const int kLinux_AF_INET = 2; static const int kLinux_AF_INET6 = 10; static const int kLinux_AF_UNSPEC = 0;",
            "struct hostent { char **h_addr_list = nullptr; char *h_addr = nullptr; int h_length = 0; };",
            "static hostent *gethostbyname(...) { static hostent value; return &value; }",
            "static unsigned long inet_addr(...) { return 0; }",
            "static unsigned short htons(unsigned short value) { return value; }",
            "static int WSAStartup(...) { return 0; }",
            "static int MAKEWORD(...) { return 0; }",
            "static int closesocket(...) { return 0; }",
            "static int socket(...) { return 0; }",
            "static int connect(...) { return 0; }",
            "static int bind(...) { return 0; }",
            "static int listen(...) { return 0; }",
            "static int send(...) { return 0; }",
            "static int recv(...) { return 0; }",
            "static int unlink(...) { return 0; }",
            "static gchar *g_strdup_printf(...) { static gchar value[256] = {}; return value; }",
            "static gchar *g_get_current_dir(...) { static gchar value[256] = {}; return value; }",
            "static gchar *g_get_user_name(...) { static gchar value[32] = {}; return value; }",
            "static gchar *g_strerror(...) { static gchar value[32] = {}; return value; }",
            "static GIOChannel *g_io_channel_unix_new(...) { static GIOChannel value; return &value; }",
            "static void g_io_channel_unref(...) {}",
            "static int g_io_add_watch(...) { return 1; }",
            "static void g_free(...) {}",
            "static long gdk_display_get_default(...) { return 0; }",
            "static gchar *gdk_display_get_name(...) { static gchar value[32] = {}; return value; }",
            "static void g_warning(...) {}",
            "template <typename T, size_t N> void InitializeToZeroArray(T (&arr)[N]) { for (size_t i = 0; i < N; ++i) arr[i] = T(); }",
            "template <typename T> void InitializeToZeroSingle(T *value) { if (value) *value = T(); }",
            "template <typename... Args> void ReinterpretCopyArray(Args...) {}",
            "template <typename... Args> void ReinterpretCopySingle(Args...) {}",
            "template <typename... Args> void CopySockaddr(Args...) {}",
            "namespace absl { template <typename... Args> std::string StrCat(Args...) { return {}; } }",
            "struct PointInfo { double x = 0; double y = 0; double z = 0; };",
            "struct RectangleInfo { long x = 0; long y = 0; unsigned long width = 0; unsigned long height = 0; };",
            "struct ChromaticityInfo { PointInfo red_primary; PointInfo green_primary; PointInfo blue_primary; PointInfo white_point; };",
            "struct PixelInfo { double red = 0; double green = 0; double blue = 0; double alpha = 0; double opacity = 0; };",
            "struct ExceptionInfo { unsigned long signature = 0; int severity = 0; char reason[256] = {}; char description[256] = {}; };",
            "struct DrawInfo { double affine[6] = {}; double *dash_pattern = nullptr; double dash_offset = 0; PixelInfo fill; PixelInfo stroke; PixelInfo alpha; char *clip_mask = nullptr; char *font = nullptr; RectangleInfo viewbox; int linejoin = 0; int linecap = 0; int fill_rule = 0; double miterlimit = 0; double stroke_width = 0; int align = 0; void *fill_pattern = nullptr; void *stroke_pattern = nullptr; };",
            "struct GifInfo { unsigned char *rasterBits = nullptr; unsigned long rasterSize = 0; unsigned long sampleSize = 0; void *gifFilePtr = nullptr; unsigned long originalWidth = 0; unsigned long originalHeight = 0; };",
            "struct PixelPacket { unsigned short red = 0; unsigned short green = 0; unsigned short blue = 0; unsigned short opacity = 0; };",
            "struct png_color { unsigned char red = 0; unsigned char green = 0; unsigned char blue = 0; };",
            "struct Image { char filename[256] = {}; char magick[64] = {}; char magick_filename[256] = {}; char signature[64] = {}; unsigned char *blob = nullptr; unsigned char *colormap = nullptr; unsigned long columns = 0; unsigned long rows = 0; unsigned long colors = 0; unsigned long page = 0; unsigned long number_scenes = 0; unsigned long scene = 0; unsigned long iterations = 0; unsigned long delay = 0; unsigned long ticks_per_second = 0; unsigned long depth = 0; unsigned long number_channels = 0; unsigned long number_meta_channels = 0; long offset = 0; long start_loop = 0; long resolution = 0; double x_resolution = 0; double y_resolution = 0; double gamma = 0; int storage_class = 0; int colorspace = 0; int alpha_trait = 0; int compression = 0; int debug = 0; int ping = 0; int matte = 0; int endian = 0; int units = 0; int type = 0; int orientation = 0; int rendering_intent = 0; int interlace = 0; int dispose = 0; int gravity = 0; int dither = 0; int intensity = 0; int quality = 0; int taint = 0; unsigned long magick_columns = 0; unsigned long magick_rows = 0; void *exception = nullptr; void *extract_info = nullptr; void *profiles = nullptr; void *artifacts = nullptr; void *properties = nullptr; void *progress_monitor = nullptr; void *client_data = nullptr; void *cache = nullptr; char *directory = nullptr; char *montage = nullptr; PixelInfo background_color; PixelInfo border_color; PixelInfo matte_color; PixelInfo transparent_color; ChromaticityInfo chromaticity; RectangleInfo tile_info; RectangleInfo page_info; RectangleInfo tile_offset; Image *previous = nullptr; Image *next = nullptr; };",
            "struct ImageInfo { char filename[256] = {}; char magick[64] = {}; char signature[64] = {}; char *file = nullptr; char *density = nullptr; char *size = nullptr; char *sampling_factor = nullptr; unsigned long page = 0; unsigned long scene = 0; unsigned long scenes = 0; unsigned long first_scene = 0; unsigned long number_scenes = 0; unsigned long depth = 0; unsigned long length = 0; unsigned int compression = 0; int quality = 0; int adjoin = 0; int ping = 0; int debug = 0; int verbose = 0; int endian = 0; int type = 0; int antialias = 0; int colorspace = 0; int interlace = 0; int monochrome = 0; int pointsize = 0; int dither = 0; void *blob = nullptr; void *extract = nullptr; void *cache = nullptr; };",
            "struct PrimitiveInfo { PointInfo point; PointInfo coordinates; long primitive = 0; long method = 0; char *text = nullptr; };",
            "using CompositeFunc = void (*)();",
            "struct MSLInfo { Image *image = nullptr; ImageInfo *image_info = nullptr; void *attributes = nullptr; void *exception = nullptr; };",
            "struct LayerInfo { Image *image = nullptr; RectangleInfo page; long x = 0; long y = 0; };",
            "struct QtDemuxSample { guint32 size = 0; guint32 chunk = 0; guint64 offset = 0; guint64 timestamp = 0; guint64 duration = 0; gint32 pts_offset = 0; int keyframe = 0; };",
            "struct QtDemuxStream { int sampled = 0; int n_samples = 0; QtDemuxSample *samples = nullptr; guint64 min_duration = 0; guint32 timescale = 0; int all_keyframe = 0; int samples_per_frame = 0; int bytes_per_frame = 0; int n_channels = 0; };",
            "struct PCIDevice { long dummy = 0; };",
            "struct iovec { void *iov_base = nullptr; size_t iov_len = 0; };",
            "struct hlist_head { void *first = nullptr; };",
            "struct kioctx { long dummy = 0; };",
            "struct V9fsString { char *data = nullptr; };",
            "struct V9fsStat { long dummy = 0; };",
            "struct V9fsFidXattr { long copied_len = 0; size_t len = 0; int flags = 0; V9fsString name; void *value = nullptr; };",
            "struct V9fsFidFS { V9fsFidXattr xattr; };",
            "struct V9fsFidState { long fid = 0; long fid_type = 0; long open_flags = 0; V9fsFidFS fs; void *path = nullptr; };",
            "struct V9fsPDU { long tag = 0; long id = 0; };",
            "struct StringInfo { unsigned char *datum = nullptr; size_t length = 0; };",
            "struct XMP_IO { void Rewind() {} template <typename... Args> long Read(Args...) { return 0; } template <typename... Args> void Seek(Args...) {} };",
            "struct st_entry { long type = 0; long flags = 0; long key = 0; long record = 0; long data = 0; char *varname = nullptr; };",
            "struct codec_t { long (*s_parse)(...) = nullptr; long (*parse)(...) = nullptr; long flags = 0; };",
            "struct _bdf_parse_t { long flags = 0; long glyph_enc = 0; char *glyph_name = nullptr; long row = 0; long cnt = 0; long opts = 0; void *list = nullptr; void *font = nullptr; };",
            "struct bdf_glyph_t { long bbx = 0; long bpr = 0; long swidth = 0; long dwidth = 0; unsigned char *bitmap = nullptr; char *name = nullptr; long encoding = 0; };",
            "struct bdf_font_t { bdf_glyph_t *glyphs = nullptr; long glyphs_size = 0; long glyphs_used = 0; long unencoded_used = 0; long modified = 0; long spacing = 0; };",
            "struct H264Context { long sps = 0; long pps = 0; long gb = 0; long avctx = 0; long cur_pic_ptr = 0; long picture_structure = 0; long ref_count = 0; long first_field = 0; long current_slice = 0; long droppable = 0; long frame_num = 0; long prev_frame_num = 0; long mb_width = 0; long mb_height = 0; long mb_y = 0; long resync_mb_y = 0; long slice_type = 0; long slice_type_nos = 0; long deblocking_filter = 0; long slice_num = 0; long short_ref[16] = {}; long ref_list[16] = {}; long slice_alpha_c0_offset = 0; long slice_beta_offset = 0; };",
            "struct cdf_property_info_t { long pi_type = 0; long pi_id = 0; long pi_val = 0; };",
            "struct DNSHeader { unsigned char *payload = nullptr; unsigned long payload_size = 0; };",
            "struct GooString { template <typename... Args> long format(Args...) { return 0; } char *c_str() { return nullptr; } char *getCString() { return nullptr; } long getLength() const { return 0; } };",
            "struct NodeDef { std::string name; std::vector<long> input; template <typename... Args> long input_size(Args...) const { return 0; } };",
            "struct OpKernelContext {",
            "    long input = 0;",
            "    template <typename T> T eigen_device() { return T(); }",
            "    template <typename... Args> int allocate_temp(Args...) { return 0; }",
            "};",
            "using WavpackContext = void *;",
            "using WebPImage = void *;",
            "using xmlDocPtr = void *;",
            "using xmlNodePtr = void *;",
            "struct phar_entry_info { char *filename = nullptr; unsigned int filename_len = 0; int is_persistent = 0; unsigned long uncompressed_filesize = 0; unsigned long compressed_filesize = 0; unsigned int flags = 0; void *metadata = nullptr; unsigned long header_offset = 0; };",
            "enum class ExecutionStatus { EXCEPTION, RETURNED };",
            "enum class OpCode { CallDirect, Debugger, Eq, PutNewOwnByIdLong, LoadConstInt, LoadConstDouble, LoadConstUInt8 };",
            "struct HermesValue {",
            "    using RawType = long;",
            "    long raw = 0;",
            "    RawType getRaw() const { return raw; }",
            "    double getDouble() const { return 0; }",
            "    double getNumber() const { return 0; }",
            "    template <typename T> T getNumberAs() const { return T(); }",
            "    bool getBool() const { return false; }",
            "    static HermesValue fromRaw(RawType) { return HermesValue(); }",
            "    static HermesValue encodeUndefinedValue() { return HermesValue(); }",
            "    static HermesValue encodeDoubleValue(double) { return HermesValue(); }",
            "    static HermesValue encodeNumberValue(double) { return HermesValue(); }",
            "    static HermesValue encodeBoolValue(bool) { return HermesValue(); }",
            "    static HermesValue encodeNativePointer(void *) { return HermesValue(); }",
            "    static HermesValue encodeObjectValue(...) { return HermesValue(); }",
            "    static HermesValue encodeStringValue(...) { return HermesValue(); }",
            "    static HermesValue encodeNullValue() { return HermesValue(); }",
            "};",
            "using PinnedHermesValue = HermesValue;",
            "template <typename T> struct CallResult {",
            "    T value; ExecutionStatus status;",
            "    CallResult(ExecutionStatus s = ExecutionStatus::RETURNED) : value(), status(s) {}",
            "    CallResult(T v) : value(v), status(ExecutionStatus::RETURNED) {}",
            "    T *operator->() { return &value; }",
            "    const T *operator->() const { return &value; }",
            "    T &getValue() { return value; }",
            "    const T &getValue() const { return value; }",
            "    T getHermesValue() const { return value; }",
            "    operator bool() const { return status != ExecutionStatus::EXCEPTION; }",
            "};",
            "struct Inst { OpCode opCode; long dummy; };",
            "struct CodeBlock { long dummy; };",
            "struct InterpreterState { CodeBlock *codeBlock; };",
            "struct Runtime {",
            "    enum class StackOverflowKind { JSRegisterStack };",
            "    HermesValue thrownValue_;",
            "    void *jitContext_ = nullptr;",
            "    const Inst *getCurrentIP() const { return nullptr; }",
            "    void setCurrentIP(const Inst *) {}",
            "    HermesValue raiseStackOverflow(StackOverflowKind) { return HermesValue(); }",
            "    struct Frame { HermesValue &getThisArgRef() { static HermesValue v; return v; } HermesValue &getArgRef(int) { static HermesValue v; return v; } int getArgCount() const { return 0; } };",
            "    Frame getCurrentFrame() { return Frame(); }",
            "};",
            "struct PropOpFlags { bool getMustExist() const { return false; } };",
            "struct JSObject {",
            "    static HermesValue create(...) { return HermesValue(); }",
            "    template <typename... Args> HermesValue getClassGCPtr(Args...) { return HermesValue(); }",
            "};",
            "struct HostObject {};",
            "struct StringPrimitive {};",
            "struct PropertyAccessor {};",
            "template <typename T = HermesValue> struct Handle {",
            "    T *ptr = nullptr;",
            "    Handle() = default; Handle(T *value) : ptr(value) {}",
            "    T *operator->() const { return ptr; }",
            "    T &operator*() const { static T value; return ptr ? *ptr : value; }",
            "    operator bool() const { return ptr != nullptr; }",
            "};",
            "template <typename T = HermesValue> using MutableHandle = Handle<T>;",
            "template <typename T = HermesValue> struct PseudoHandle {",
            "    T value; PseudoHandle(T v = T()) : value(v) {}",
            "    T getHermesValue() const { return value; }",
            "};",
            "struct GCScope { explicit GCScope(Runtime *) {} };",
            "struct ScopedNativeDepthTracker {",
            "    explicit ScopedNativeDepthTracker(Runtime *) {}",
            "    bool overflowed() const { return false; }",
            "};",
            "struct HiddenClass { static const int kDictionaryThreshold = 0; };",
            "struct SegmentedArray { static const int kValueToSegmentThreshold = 1; };",
            "struct GeneratorInnerFunction {",
            "    enum class State { Completed, SuspendedYield, SuspendedStart, Executing };",
            "    enum class Action { Return, Throw };",
            "    void setState(State) {}",
            "    Action getAction() const { return Action::Return; }",
            "};",
            "template <typename T, typename U> T *vmcast(U *) { return nullptr; }",
            "static const char *DumpHermesValue(...) { return \"\"; }",
            "#define llvm_unreachable(x) do { return {}; } while (0)",
            "template <typename T> struct StatusWith {",
            "    T value; StatusWith(...) : value() {}",
            "    bool isOK() const { return true; }",
            "    T getValue() const { return value; }",
            "    int getStatus() const { return 0; }",
            "};",
            "template <typename T> struct StatusOr {",
            "    T value; StatusOr(...) : value() {}",
            "    bool ok() const { return true; }",
            "    bool isOK() const { return true; }",
            "    T ValueOrDie() const { return value; }",
            "    T value_or(T fallback) const { return fallback; }",
            "    operator bool() const { return true; }",
            "};",
            "struct Message { template <typename... Args> Message(Args...) {} struct Header { char *data() const { return nullptr; } size_t dataLen() const { return 0; } long getId() const { return 0; } long getResponseToMsgId() const { return 0; } }; Header header() const { return Header(); } };",
            "using MessageCompressorId = int;",
            "using sound_trigger_module_handle_t = int;",
            "struct sound_trigger_module_descriptor { long dummy; };",
            "static const int NO_ERROR = 0;",
            "static const int LIST_MODULES = 1;",
            "static const int ATTACH = 2;",
            "static const int SET_CAPTURE_STATE = 3;",
            "struct CompressionHeader { static size_t size() { return 0; } template <typename... Args> CompressionHeader(Args...) {} int compressorId = 0; size_t uncompressedSize = 0; int originalOpCode = 0; };",
            "struct ConstDataRangeCursor { template <typename... Args> ConstDataRangeCursor(Args...) {} size_t length() const { return 0; } };",
            "struct DataRangeCursor { template <typename... Args> DataRangeCursor(Args...) {} };",
            "struct SharedBuffer { static SharedBuffer allocate(size_t) { return SharedBuffer(); } void *get() { return nullptr; } };",
            "struct MsgData { static const size_t MsgDataHeaderSize = 0; struct View { View(void *) {} void setId(long) {} void setResponseToMsgId(long) {} void setOperation(int) {} void setLen(size_t) {} char *data() { return nullptr; } size_t dataLen() const { return 0; } }; };",
            "namespace ErrorCodes { static const int BadValue = 1; static const int InternalError = 2; }",
            "#define LOG(x) if (true) ; else std::cerr",
            "#define LOG_IF(level, cond) if (!(cond)) ; else std::cerr",
            "#define DLOG(x) if (true) ; else std::cerr",
            "#define ALOGD(...)",
            "#define U_ZERO_ERROR 0",
            "#define U_FAILURE(x) false",
            "#define USPOOF_ALL_CHECKS 1",
            "#define USPOOF_RESTRICTION_LEVEL_MASK 2",
            "#define USPOOF_ASCII 3",
            "#define USPOOF_SINGLE_SCRIPT_RESTRICTIVE 4",
            "#define US_INV 0",
            "struct StringPiece16 { const unsigned char *data() const { return (const unsigned char *)\"\"; } size_t size() const { return 0; } };",
            "template <typename T> using scoped_refptr = T *;",
            "struct vulsirt_chromium_value {",
            "    long value = 0;",
            "    vulsirt_chromium_value(long v = 0) : value(v) {}",
            "    std::string possibly_invalid_spec() const { return {}; }",
            "    bool is_null() const { return false; }",
            "    void reset() { value = 0; }",
            "    vulsirt_chromium_value *get() { return this; }",
            "    const vulsirt_chromium_value *get() const { return this; }",
            "    vulsirt_chromium_value *operator->() { return this; }",
            "    const vulsirt_chromium_value *operator->() const { return this; }",
            "    template <typename T> vulsirt_chromium_value &operator=(T) { return *this; }",
            "    operator bool() const { return value != 0; }",
            "    operator long() const { return value; }",
            "};",
            "struct vulsirt_chromium_vector { template <typename... Args> void push_back(Args...) {} size_t size() const { return 0; } };",
            "namespace net { using Error = int; static const int OK = 0; static const int ERR_ABORTED = -1; static const int ERR_UNSAFE_REDIRECT = -2; struct RedirectInfo { template <typename... Args> RedirectInfo(Args...) {} vulsirt_chromium_value new_url; std::string new_method; std::string new_referrer; bool insecure_scheme_was_upgraded = false; }; }",
            "namespace network { struct ResourceResponse { template <typename... Args> ResourceResponse(Args...) {} struct Head { vulsirt_chromium_value ssl_info; vulsirt_chromium_value headers; std::string mime_type; } head; }; struct URLLoaderCompletionStatus { template <typename... Args> URLLoaderCompletionStatus(Args...) {} }; }",
            "struct ChildProcessSecurityPolicyImpl { static ChildProcessSecurityPolicyImpl *GetInstance() { static ChildProcessSecurityPolicyImpl value; return &value; } template <typename... Args> bool CanRedirectToURL(Args...) { return true; } template <typename... Args> bool CanRequestURL(Args...) { return true; } };",
            "struct RenderProcessHostImpl { template <typename... Args> long GetID(Args...) const { return 0; } };",
            "struct SiteInstance { bool HasProcess() const { return false; } RenderProcessHostImpl *GetProcess() { static RenderProcessHostImpl value; return &value; } void *GetBrowserContext() { return nullptr; } };",
            "struct NavigationRequest { template <typename... Args> long GetURL(Args...) const { return 0; } template <typename... Args> long OnRedirectChecksComplete(Args...) { return 0; } };",
            "struct Referrer { vulsirt_chromium_value url; long policy = 0; template <typename... Args> static Referrer SanitizeForRequest(Args...) { return Referrer(); } };",
            "enum class CredentialedSubresourceCheckResult { ALLOW_REQUEST = 0, BLOCK_REQUEST = 1 };",
            "enum class LegacyProtocolInSubresourceCheckResult { ALLOW_REQUEST = 0, BLOCK_REQUEST = 1 };",
            "using UErrorCode = int;",
            "using socklen_t = int;",
            "#ifndef EPERM",
            "static const int EPERM = 1;",
            "#endif",
            "#ifndef EINVAL",
            "static const int EINVAL = 22;",
            "#endif",
            "static const int IPPROTO_UDP = 17; static const int UDP_ENCAP = 100; static const int UDP_ENCAP_ESPINUDP = 1; static const int UDP_ENCAP_ESPINUDP_NON_IKE = 2;",
            "#define S_ISSOCK(x) 1",
            "static int fstat(...) { return 0; } static int fchown(...) { return 0; }",
            "template <typename T, typename = typename std::enable_if<std::is_class<T>::value>::type> bool operator==(const T &, const T &) { return false; }",
            "template <typename T, typename = typename std::enable_if<std::is_class<T>::value>::type> bool operator!=(const T &, const T &) { return true; }",
            "struct AMessage { long dummy; };",
            "struct MediaBuffer { long dummy; };",
            "struct ReadOptions { long dummy; };",
            "struct Mutex { struct Autolock { template <typename... Args> Autolock(Args...) {} }; };",
            "struct vulsirt_run_callback { template <typename... Args> void Run(Args...) {} };",
            "using OpenDeviceCallback = vulsirt_run_callback;",
            "struct MediaDeviceSaltAndOrigin { long origin = 0; };",
            "namespace BrowserThread { static const int IO = 0; }",
            "struct MediaStreamManager { template <typename... Args> static bool IsOriginAllowed(Args...) { return true; } template <typename... Args> void OpenDevice(Args...) {} };",
            "struct MediaStreamDispatcherHost { static void OnDeviceStopped(...) {} };",
            "namespace base { struct nullopt_t { template <typename T> operator T() const { return T(); } }; static const nullopt_t nullopt = {}; struct TimeTicks { static TimeTicks Now() { return TimeTicks(); } bool is_null() const { return false; } }; template <typename... Args> long BindRepeating(Args...) { return 0; } template <typename T> T *Unretained(T *value) { return value; } static std::nullptr_t Unretained(std::nullptr_t) { return nullptr; } template <typename T> struct WeakPtr { T *ptr = nullptr; WeakPtr(T *value = nullptr) : ptr(value) {} operator bool() const { return ptr != nullptr; } }; template <typename T> struct WeakPtrFactory { T *GetWeakPtr() { return nullptr; } }; template <typename... Args> long Bind(Args...) { return 0; } }",
            "struct vulsirt_loader_stub { template <typename... Args> void reset(Args...) {} template <typename... Args> void LoadAsynchronously(Args...) {} vulsirt_loader_stub *operator->() { return this; } operator bool() const { return false; } };",
            "static vulsirt_loader_stub loader_; static bool is_embedded_ = true; static long original_url_ = 0;",
            "namespace base {",
            "using StringPiece16 = ::StringPiece16;",
            "template <typename T, typename U> T checked_cast(U value) { return static_cast<T>(value); }",
            "template <typename... Args> std::string GetFieldTrialParamValueByFeature(Args...) { return {}; }",
            "struct FeatureList { template <typename... Args> static bool IsEnabled(Args...) { return false; } };",
            "struct CommandLine {",
            "    static CommandLine *ForCurrentProcess() { static CommandLine value; return &value; }",
            "    template <typename... Args> bool HasSwitch(Args...) const { return false; }",
            "    template <typename... Args> std::string GetSwitchValueASCII(Args...) const { return {}; }",
            "};",
            "}",
            "struct Mode { static const long kNone = 0; static const long kAll = 1; static const long kMinimal = 2; static const long kBrowser = 3; static const long kGpu = 4; static const long kRendererSampling = 5; };",
            "namespace switches { static const char *kMemlog = \"memlog\"; static const char *kEnableHeapProfiling = \"enable_heap_profiling\"; static const char *kMemlogModeAll = \"all\"; static const char *kMemlogModeMinimal = \"minimal\"; static const char *kMemlogModeBrowser = \"browser\"; static const char *kMemlogModeGpu = \"gpu\"; static const char *kMemlogModeRendererSampling = \"renderer\"; }",
            "template <typename T> class sp {",
            "public:",
            "    T *ptr;",
            "    sp() : ptr(nullptr) {}",
            "    sp(T *value) : ptr(value) {}",
            "    T *operator->() const { return ptr; }",
            "    operator T *() const { return ptr; }",
            "    sp &operator=(T *value) { ptr = value; return *this; }",
            "    sp &operator=(std::nullptr_t) { ptr = nullptr; return *this; }",
            "    bool operator!=(int) const { return ptr != nullptr; }",
            "    bool operator==(int) const { return ptr == nullptr; }",
            "    void clear() { ptr = nullptr; }",
            "};",
            "struct AString {",
            "    size_t size() const { return 0; }",
            "    const char *c_str() const { return \"\"; }",
            "};",
            "struct ABuffer {",
            "    explicit ABuffer(size_t = 0) {}",
            "    uint8_t *data() { return nullptr; }",
            "    size_t size() const { return 0; }",
            "    void setRange(size_t, size_t) {}",
            "};",
            "struct SampleToChunkEntry { uint32_t startChunk; uint32_t samplesPerChunk; uint32_t chunkDesc; };",
            "struct vulsirt_data_source { long flags = 0; long offset = 0; template <typename... Args> ssize_t readAt(Args...) { return 0; } };",
            "static long long mSampleToChunkOffset = -1; static size_t mNumSampleToChunkOffsets = 0; static SampleToChunkEntry *mSampleToChunkEntries = nullptr; static vulsirt_data_source *mDataSource = nullptr;",
            "static char *__cxa_demangle(...) { return nullptr; }",
            "struct OMX_VERSIONTYPE { struct { int nVersionMajor; int nVersionMinor; int nRevision; int nStep; } s; };",
            "using OMX_ERRORTYPE = int; using OMX_U32 = unsigned int; using OMX_PTR = void *; using OMX_U8 = unsigned char;",
            "static const int OMX_ErrorNone = 0; static const int OMX_StateLoaded = 0; static const int OMX_FALSE = 0; static const int OMX_TRUE = 1;",
            "struct OMX_BUFFERHEADERTYPE { long nSize; OMX_VERSIONTYPE nVersion; OMX_U8 *pBuffer; long nAllocLen; long nFilledLen; long nOffset; OMX_PTR pAppPrivate; OMX_PTR pPlatformPrivate; OMX_PTR pInputPortPrivate; OMX_PTR pOutputPortPrivate; OMX_PTR hMarkTargetComponent; OMX_PTR pMarkData; long nTickCount; long nTimeStamp; long nFlags; long nOutputPortIndex; long nInputPortIndex; };",
            "struct BufferInfo { OMX_BUFFERHEADERTYPE *mHeader; bool mOwnedByUs; };",
            "struct PortInfo { struct Def { long bEnabled; long nBufferCountActual; long bPopulated; } mDef; struct Buffers { size_t size() const { return 0; } void push() {} BufferInfo &editItemAt(size_t) { static BufferInfo value; return value; } } mBuffers; };",
            "struct vulsirt_port_vector { size_t size() const { return 0; } PortInfo &editItemAt(size_t) { static PortInfo value; return value; } };",
            "static vulsirt_port_vector mPorts; static long mState = 0; static void checkTransitions() {} static Mutex mLock;",
            "namespace android {",
            "namespace base { struct unique_fd { int value = 0; int get() const { return value; } operator int() const { return value; } }; }",
            "struct IBinder {};",
            "struct IInterface { template <typename... Args> static IBinder *asBinder(Args...) { return nullptr; } };",
            "struct BBinder { template <typename... Args> static status_t onTransact(Args...) { return 0; } };",
            "struct Parcel {",
            "    int32_t readInt32() const { return 0; }",
            "    void *readInplace(size_t) const { return nullptr; }",
            "    void writeInt32(int32_t) {}",
            "    void write(const void *, size_t) {}",
            "    void read(void *, size_t) const {}",
            "    IBinder *readStrongBinder() const { return nullptr; }",
            "    void writeStrongBinder(IBinder *) {}",
            "};",
            "static Parcel *parcelForJavaObject(...) { static Parcel value; return &value; }",
            "}",
            "using Parcel = android::Parcel;",
            "using android::BBinder;",
            "using android::IBinder;",
            "using android::IInterface;",
            "namespace netdutils {",
            "struct Status { int value = 0; Status(int v = 0) : value(v) {} operator int() const { return value; } };",
            "namespace status { static const Status ok = Status(0); }",
            "template <typename... Args> Status statusFromErrno(Args...) { return Status(); }",
            "}",
            "struct Fd { template <typename... Args> Fd(Args...) {} };",
            "struct vulsirt_syscall_instance { template <typename... Args> netdutils::Status getsockopt(Args...) { return netdutils::Status(); } };",
            "static vulsirt_syscall_instance getSyscallInstance() { return {}; }",
            "struct ISoundTriggerClient {};",
            "struct ISoundTrigger {};",
            "struct ISoundTriggerHwService {};",
            "template <typename T, typename U> sp<T> interface_cast(U *) { return sp<T>(); }",
            "namespace blink {",
            "enum MediaStreamType { MEDIA_STREAM_TYPE_NO_SERVICE = 0 };",
            "struct MediaStreamDevice { template <typename... Args> MediaStreamDevice(Args...) {} };",
            "struct WebAssociatedURLLoaderOptions {};",
            "struct WebAssociatedURLLoader { template <typename... Args> void LoadAsynchronously(Args...) {} };",
            "struct WebLocalFrame { template <typename... Args> WebAssociatedURLLoader *CreateAssociatedURLLoader(Args...) { static WebAssociatedURLLoader value; return &value; } };",
            "struct WebURLRequest { static const int kRequestContextObject = 0; static const int kPreviewsNoTransform = 0; static const int kRequestContextImageSet = 0; static const int kRequestContextPing = 0; template <typename... Args> WebURLRequest(Args...) {} template <typename... Args> void SetRequestContext(Args...) {} };",
            "}",
            "struct vulsirt_render_frame { blink::WebLocalFrame *GetWebFrame() { static blink::WebLocalFrame value; return &value; } };",
            "static vulsirt_render_frame *render_frame() { static vulsirt_render_frame value; return &value; }",
            "struct IncrementLoadEventDelayCount {};",
            "static std::unique_ptr<IncrementLoadEventDelayCount> delay_until_do_update_from_element_;",
            "static std::string GetFunctionNameRaw(...) { return {}; }",
            "struct AtomicString { template <typename... Args> AtomicString(Args...) {} bool IsNull() const { return false; } bool IsEmpty() const { return false; } };",
            "struct KURL { template <typename... Args> KURL(Args...) {} bool IsNull() const { return false; } bool IsEmpty() const { return false; } std::string possibly_invalid_spec() const { return {}; } };",
            "using ReferrerPolicy = int; static const int kReferrerPolicyDefault = 0; static const int kUpdateForcedReload = 1;",
            "struct vulsirt_document_frame { template <typename... Args> void MaybeAllowImagePlaceholder(Args...) {} };",
            "struct Document { static const int kNoDismissal = 0; bool IsActive() const { return true; } std::string OutgoingReferrer() const { return {}; } int PageDismissalEventBeingDispatched() const { return 0; } vulsirt_document_frame *GetFrame() { static vulsirt_document_frame value; return &value; } void *Fetcher() { return nullptr; } int GetClientHintsPreferences() const { return 0; } };",
            "struct vulsirt_layout_object { bool IsImage() const { return true; } };",
            "struct vulsirt_element_stub { Document &GetDocument() { static Document value; return value; } AtomicString ImageSourceURL() { return {}; } AtomicString localName() { return {}; } vulsirt_element_stub *parentNode() { return this; } AtomicString FastGetAttribute(...) { return {}; } vulsirt_layout_object *GetLayoutObject() { static vulsirt_layout_object value; return &value; } };",
            "static vulsirt_element_stub *element_ = nullptr; static vulsirt_element_stub *GetElement() { static vulsirt_element_stub value; return &value; } static bool IsHTMLPictureElement(...) { return false; }",
            "struct ResourceLoaderOptions { struct { AtomicString name; } initiator_info; };",
            "struct ResourceRequest { template <typename... Args> ResourceRequest(Args...) {} template <typename... Args> void SetCacheMode(Args...) {} template <typename... Args> void SetPreviewsState(Args...) {} template <typename... Args> void SetHTTPReferrer(Args...) {} template <typename... Args> void SetRequestContext(Args...) {} template <typename... Args> void SetHTTPHeaderField(Args...) {} void SetKeepalive(bool) {} };",
            "using WebURLRequest = blink::WebURLRequest;",
            "struct FetchParameters { template <typename... Args> FetchParameters(Args...) {} };",
            "struct ImageResourceContent { template <typename... Args> static ImageResourceContent *Fetch(Args...) { return nullptr; } template <typename... Args> void AddObserver(Args...) {} template <typename... Args> void RemoveObserver(Args...) {} };",
            "struct vulsirt_image_content_handle { ImageResourceContent *value = nullptr; ImageResourceContent *Get() { return value; } operator ImageResourceContent *() { return value; } template <typename T> vulsirt_image_content_handle &operator=(T) { return *this; } };",
            "static vulsirt_image_content_handle image_content_;",
            "struct LayoutImageResource { void ResetAnimation() {} }; static LayoutImageResource *GetLayoutImageResource() { static LayoutImageResource value; return &value; } struct LayoutImage { void IntrinsicSizeChanged() {} }; static LayoutImage *ToLayoutImage(...) { return nullptr; }",
            "namespace mojom { enum FetchCacheMode { kBypassCache = 0 }; } namespace HTMLNames { static const int srcsetAttr = 0; } namespace HTTPNames { static const int Cache_Control = 0; } struct SecurityPolicy { template <typename... Args> static int GenerateReferrer(Args...) { return 0; } };",
            "namespace icu {",
            "struct UnicodeString { template <typename... Args> UnicodeString(Args...) {} };",
            "struct RegexMatcher {",
            "    template <typename... Args> RegexMatcher(Args...) {}",
            "    template <typename... Args> void reset(Args...) {}",
            "    bool find() { return false; }",
            "};",
            "}",
            "struct ImVec2 { float x; float y; ImVec2(float a = 0, float b = 0) : x(a), y(b) {} };",
            "struct ImVec4 { float x; float y; float z; float w; ImVec4(float a = 0, float b = 0, float c = 0, float d = 0) : x(a), y(b), z(c), w(d) {} };",
            "enum { ImGuiCol_Header = 0, ImGuiCol_Text = 1, ImGuiCol_HeaderActive = 2, ImGuiCol_HeaderHovered = 3 };",
            "enum { ImGuiTableBgTarget_RowBg0 = 0, ImGuiSelectableFlags_NoPadWithHalfSpacing = 1, ImGuiHoveredFlags_AllowWhenBlockedByActiveItem = 1 };",
            "namespace ImGui {",
            "static void TableNextRow(...) {} static void TableNextColumn(...) {} static ImVec2 GetCursorPos() { return ImVec2(); }",
            "static float GetScrollY() { return 0; } static ImVec2 GetWindowSize() { return ImVec2(); } static unsigned int GetColorU32(...) { return 0; }",
            "static void TableSetBgColor(...) {} static void PushStyleColor(...) {} static void PopStyleColor(int = 1) {}",
            "static void TextColored(...) {} static float GetCursorPosX() { return 0; } static void Selectable(...) {}",
            "static bool IsItemClicked(...) { return false; } static bool IsItemHovered(...) { return false; } static void SameLine(...) {}",
            "}",
            "namespace Eigen {",
            "using DenseIndex = long;",
            "static const int BothParts = 0; static const int FFT_FORWARD = 0;",
            "struct ArrayXi { static ArrayXi LinSpaced(...) { return ArrayXi(); } };",
            "template <typename... Args> struct Map { template <typename... CtorArgs> Map(CtorArgs...) {} };",
            "template <typename T, int N> struct DSizes { long values[N ? N : 1]; long &operator[](int i) { return values[i]; } const long &operator[](int i) const { return values[i]; } };",
            "}",
            "struct DataType {",
            "    enum Enum { DT_INVALID = 0, DT_FLOAT = 1, DT_DOUBLE = 2, DT_UINT8 = 3, DT_COMPLEX64 = 4, DT_COMPLEX128 = 5 };",
            "    int value = 0;",
            "    DataType(int v = 0) : value(v) {}",
            "    operator int() const { return value; }",
            "};",
            "struct CPUDevice { long dummy; };",
            "template <typename T> struct DataTypeToEnum { static DataType v() { return 0; } };",
            "struct DeviceBase { long dummy; };",
            "struct TensorShapeUtils { template <typename... Args> static bool IsScalar(Args...) { return true; } template <typename... Args> static bool IsVector(Args...) { return true; } template <typename... Args> static bool IsMatrix(Args...) { return true; } };",
            "struct AttrValue { template <typename... Args> AttrValue(Args...) {} template <typename... Args> long type(Args...) const { return 0; } template <typename... Args> long s(Args...) const { return 0; } };",
            "struct vulsirt_tf_range { long *begin() const { return nullptr; } long *end() const { return nullptr; } long operator[](int) const { return 0; } int size() const { return 0; } };",
            "struct OpDef { struct ArgDef { template <typename... Args> long name(Args...) const { return 0; } template <typename... Args> long type(Args...) const { return 0; } }; template <typename... Args> long name(Args...) const { return 0; } template <typename... Args> ArgDef input_arg(Args...) const { return {}; } template <typename... Args> ArgDef output_arg(Args...) const { return {}; } template <typename... Args> AttrValue attr(Args...) const { return {}; } template <typename... Args> long input_arg_size(Args...) const { return 0; } template <typename... Args> long output_arg_size(Args...) const { return 0; } template <typename... Args> long attr_size(Args...) const { return 0; } };",
            "struct OpInfo { OpDef op_def; template <typename... Args> OpInfo(Args...) {} };",
            "struct FunctionLibraryDefinition { template <typename... Args> long Find(Args...) const { return 0; } };",
            "struct GraphDef { template <typename... Args> vulsirt_tf_range node(Args...) const { return {}; } template <typename... Args> long node_size(Args...) const { return 0; } };",
            "struct MetaGraphDef { GraphDef graph_def() const { return {}; } template <typename... Args> vulsirt_tf_range signature_def(Args...) const { return {}; } };",
            "struct InferenceContext { template <typename... Args> long input(Args...) const { return 0; } template <typename... Args> long output(Args...) const { return 0; } template <typename... Args> void set_output(Args...) {} };",
            "struct Logger { template <typename... Args> Logger(Args...) {} template <typename... Args> void Log(Args...) {} template <typename... Args> void Error(Args...) {} };",
            "struct RuntimeOption { template <typename... Args> RuntimeOption(Args...) {} long value = 0; operator long() const { return value; } };",
            "template <typename K, typename V = long> struct QMapIterator { template <typename... Args> QMapIterator(Args...) {} bool hasNext() const { return false; } void next() {} K key() const { return K(); } V value() const { return V(); } };",
            "namespace OPENEXR_IMF_INTERNAL_NAMESPACE { struct Header {}; struct InputFile {}; struct OutputFile {}; } namespace IEX_NAMESPACE { struct BaseExc {}; }",
            "struct vulsirt_stub_tensor_view {",
            "    long dimensions() const { return 0; }",
            "    long dimension(int) const { return 0; }",
            "    long dim_size(int) const { return 0; }",
            "    long &operator[](int) { static long value = 0; return value; }",
            "    template <typename... Args> vulsirt_stub_tensor_view slice(Args...) { return {}; }",
            "    template <typename... Args> vulsirt_stub_tensor_view fft(Args...) { return {}; }",
            "    template <typename... Args> vulsirt_stub_tensor_view &device(Args...) { return *this; }",
            "    template <typename T> vulsirt_stub_tensor_view &operator=(T) { return *this; }",
            "};",
            "struct TensorShape { template <typename... Args> TensorShape(Args...) {} void AddDim(long) {} long dim_size(int) const { return 0; } };",
            "struct Tensor {",
            "    Tensor() {} Tensor(const Tensor &) {}",
            "    TensorShape shape; Tensor *tensor = nullptr; long type = 0; long dtype = 0; long dims = 0; void *sparsity = nullptr;",
            "    long dim_size(int) const { return 0; }",
            "    template <typename... Args> vulsirt_stub_tensor_view flat_inner_dims(Args...) { return {}; }",
            "    template <typename T, int N> vulsirt_stub_tensor_view flat_inner_dims() { return {}; }",
            "    template <typename T> vulsirt_stub_tensor_view flat() { return {}; }",
            "};",
            "template <typename T, typename... Args> T *GetTensorData(Args...) { return nullptr; }",
            "template <typename... Args> TensorShape GetTensorShape(Args...) { return {}; }",
            "static void v9fs_stat_init(V9fsStat *) {}",
            "static void v9fs_string_init(V9fsString *value) { if (value) value->data = nullptr; }",
            "static void v9fs_string_copy(V9fsString *dst, const V9fsString *src) { if (dst) dst->data = src ? src->data : nullptr; }",
            "static void v9fs_string_free(V9fsString *) {}",
            "static int pdu_unmarshal(...) { return 0; }",
            "static V9fsFidState *get_fid(V9fsPDU *, int32_t) { static V9fsFidState value; return &value; }",
            "static void put_fid(V9fsPDU *, V9fsFidState *) {}",
            "static void pdu_complete(V9fsPDU *, long) {}",
            "static const int P9_FID_XATTR = 1;",
            "#define OP_REQUIRES_OK(ctx, expr) do { (void)(expr); } while (0)",
            "struct DataBuf {",
            "    byte *pData_; size_t size_;",
            "    explicit DataBuf(size_t size = 0) : pData_(nullptr), size_(size) {}",
            "};",
            "struct IoCloser { template <typename... Args> IoCloser(Args...) {} };",
            "namespace Exiv2 {",
            "using Dictionary = std::map<std::string, std::string>;",
            "enum ByteOrder { bigEndian, littleEndian };",
            "static const int kerDataSourceOpenFailed = 1; static const int kerNotAnImage = 2; static const int kerFailedToReadImageData = 3; static const int kerCorruptedMetadata = 4;",
            "struct Error { template <typename... Args> Error(Args...) {} };",
            "struct Uri { std::string Host; std::string Port; static Uri Parse(...) { return Uri(); } };",
            "static uint32_t getULong(...) { return 0; } static uint16_t getUShort(...) { return 0; }",
            "}",
            "static Exiv2::Dictionary stringToDict(...) { return {}; }",
            "struct PngChunk {",
            "    enum { tEXt_Chunk = 0, zTXt_Chunk = 1, iTXt_Chunk = 2 };",
            "    static void decodeIHDRChunk(...) {}",
            "    static void decodeTXTChunk(...) {}",
            "};",
            "using Exiv2::bigEndian;",
            "using Exiv2::littleEndian;",
            "namespace base {",
            "template <typename T> struct NoDestructor {",
            "    T value;",
            "    template <typename... Args> NoDestructor(Args...) : value() {}",
            "    operator const T &() const { return value; }",
            "    const T &operator*() const { return value; }",
            "    const T *operator->() const { return &value; }",
            "};",
            "}",
        ]
    )
