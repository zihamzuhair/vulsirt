import subprocess
import tempfile
from pathlib import Path


class LLVMGenerationError(RuntimeError):
    def __init__(self, message, compiler, command, returncode, stdout, stderr):
        super().__init__(message)
        self.compiler = compiler
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def can_generate_ir(language):
    return str(language).lower() in {"c", "cpp", "c++", "cc", "cxx", "rust", "rs"}


def clang_language(language):
    language = str(language).lower()
    if language in {"cpp", "c++", "cc", "cxx"}:
        return "c++", ".cpp"
    return "c", ".c"


def generate_llvm_ir(source_code, language):
    if str(language).lower() in {"rust", "rs"}:
        return generate_rust_llvm_ir(source_code)

    clang_lang, suffix = clang_language(language)
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / f"sample{suffix}"
        ir_path = Path(temp_dir) / "sample.ll"
        source_path.write_text(source_code, encoding="utf-8")
        command = [
            "clang",
            "-S",
            "-emit-llvm",
            "-O0",
            "-x",
            clang_lang,
            str(source_path),
            "-o",
            str(ir_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise LLVMGenerationError(
                result.stderr.strip() or "clang failed",
                compiler="clang",
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return ir_path.read_text(encoding="utf-8")


def generate_rust_llvm_ir(source_code):
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / "sample.rs"
        ir_path = Path(temp_dir) / "sample.ll"
        wrapped_source = source_code
        if "fn main" not in wrapped_source and "#![crate_type" not in wrapped_source:
            wrapped_source = "#![allow(dead_code)]\n" + wrapped_source
        source_path.write_text(wrapped_source, encoding="utf-8")
        command = [
            "rustc",
            "--crate-type",
            "lib",
            "--emit",
            "llvm-ir",
            str(source_path),
            "-o",
            str(ir_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise LLVMGenerationError(
                result.stderr.strip() or "rustc failed",
                compiler="rustc",
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return ir_path.read_text(encoding="utf-8")


def llvm_error_category(error, stage):
    text_parts = [stage, type(error).__name__, str(error)]
    if isinstance(error, LLVMGenerationError):
        text_parts.extend([error.compiler, error.stdout, error.stderr])
    text = "\n".join(part for part in text_parts if part).lower()

    if stage == "unsupported_language":
        return "unsupported_language"
    if isinstance(error, FileNotFoundError) or "no such file or directory" in text and "clang" in text:
        return "compiler_not_found"
    if "fatal error:" in text and "file not found" in text:
        return "missing_include"
    if "can't find crate" in text or "cannot find crate" in text or "unresolved import" in text:
        return "missing_dependency"
    if "failed to resolve" in text or "unresolved module" in text:
        return "missing_dependency"
    if "use of undeclared identifier" in text or "undeclared" in text:
        return "unresolved_symbol"
    if "permission denied" in text or "access is denied" in text:
        return "permission_error"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "expected" in text or "syntax error" in text or "parse error" in text:
        return "syntax_error"
    if "error:" in text:
        return "compiler_error"
    return "unknown"
