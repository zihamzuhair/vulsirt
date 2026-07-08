#!/usr/bin/env python3
"""Raw Rust code-to-LLVM compiler pipeline.

This script mirrors the compact JSONL output produced by
``helpers/primevul_code_compiler.py`` while compiling Rust rows with ``rustc``. It reads
raw RustSec/OSV source records, emits LLVM IR for records that compile, and
writes only successful rows in the same training shape.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional


PREPROCESSOR_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Static run configuration
# ---------------------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Edit these variables for each run. Then execute:
#   python compiler.py --dataset rust
INPUT_JSONL = DATA_DIR / "raw" / "rust" / "rustsec_osv_dataset.jsonl"
OUTPUT_JSONL = DATA_DIR / "processed" / "rustsec_osv_dataset_with_llvm.jsonl"
WORKERS = 4

RUSTC = "rustc"
RUST_EDITIONS: tuple[str, ...] = ("2021", "2018", "2015")
TIMEOUT_SECONDS = 30
MAX_SOURCE_BYTES = 1_000_000
VALIDATE_LLVM_OBJECT = True
KEEP_COMPILER_COMMENTS = False
REQUIRE_LLVM_FUNCTION_DEFINITION = True
EXTRA_RUSTC_FLAGS: tuple[str, ...] = ()


def normalize_source(value: Any) -> str:
    if value is None:
        return ""
    source = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return source.strip("\ufeff")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def make_sample_id(record: Mapping[str, Any], source: str) -> str:
    for key in ("sample_id", "idx", "id", "commit_id", "project_commit"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)

    file_name = str(record.get("file_name") or record.get("path") or "rust_source")
    return f"{file_name}:{stable_hash(source)}"


def rust_label(record: Mapping[str, Any]) -> int:
    try:
        return int(record.get("target", record.get("label", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def compact_error(stderr: str, *, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", stderr or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def sanitize_llvm_metadata_paths(llvm_ir: str) -> str:
    escaped_temp = re.escape(tempfile.gettempdir().replace("\\", "/"))
    llvm_ir = re.sub(escaped_temp + r"[^\"'\s)]*", "<temp>", llvm_ir.replace("\\", "/"))
    return llvm_ir


def has_llvm_function_definition(llvm_ir: str) -> bool:
    return bool(re.search(r"^define\b", llvm_ir or "", flags=re.MULTILINE))


def crate_name_for(sample_id: str) -> str:
    name = re.sub(r"\W+", "_", sample_id).strip("_").lower()
    if not name or name[0].isdigit():
        name = f"rust_sample_{stable_hash(sample_id)}"
    return name[:64]


def rustc_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("RUST_BACKTRACE", "0")
    return env


def rustc_command(
    source_path: Path,
    llvm_path: Path,
    *,
    crate_name: str,
    edition: str,
) -> list[str]:
    return [
        RUSTC,
        str(source_path),
        "--crate-type=lib",
        "--crate-name",
        crate_name,
        "--edition",
        edition,
        "--emit=llvm-ir",
        "-C",
        "opt-level=0",
        "-C",
        "debuginfo=0",
        "-C",
        "embed-bitcode=no",
        "-C",
        "metadata=codex",
        "-o",
        str(llvm_path),
        *EXTRA_RUSTC_FLAGS,
    ]


def rustc_object_command(source_path: Path, object_path: Path, *, crate_name: str, edition: str) -> list[str]:
    return [
        RUSTC,
        str(source_path),
        "--crate-type=lib",
        "--crate-name",
        crate_name,
        "--edition",
        edition,
        "--emit=obj",
        "-C",
        "opt-level=0",
        "-C",
        "debuginfo=0",
        "-o",
        str(object_path),
        *EXTRA_RUSTC_FLAGS,
    ]


def compile_with_rustc(source: str, sample_id: str) -> dict[str, Any]:
    crate_name = crate_name_for(sample_id)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="rust_ir_") as temp_dir:
        temp_path = Path(temp_dir)
        source_path = temp_path / "input.rs"
        source_path.write_text(source, encoding="utf-8")

        for edition in RUST_EDITIONS:
            llvm_path = temp_path / f"output_{edition}.ll"
            try:
                result = subprocess.run(
                    rustc_command(
                        source_path,
                        llvm_path,
                        crate_name=crate_name,
                        edition=edition,
                    ),
                    cwd=temp_path,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=TIMEOUT_SECONDS,
                    env=rustc_environment(),
                )
            except subprocess.TimeoutExpired:
                failures.append(f"rust{edition}:compiler timeout after {TIMEOUT_SECONDS}s")
                continue
            except OSError as exc:
                return {
                    "success": False,
                    "edition": edition,
                    "llvm_ir": "",
                    "stderr": f"cannot execute rust compiler: {exc}",
                    "ir_status": "failed_compiler_unavailable",
                }

            if result.returncode != 0 or not llvm_path.exists():
                failures.append(f"rust{edition}:{compact_error(result.stderr)}")
                continue

            if VALIDATE_LLVM_OBJECT:
                object_path = temp_path / f"output_{edition}.o"
                try:
                    object_result = subprocess.run(
                        rustc_object_command(
                            source_path,
                            object_path,
                            crate_name=crate_name,
                            edition=edition,
                        ),
                        cwd=temp_path,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=TIMEOUT_SECONDS,
                        env=rustc_environment(),
                    )
                except subprocess.TimeoutExpired:
                    failures.append(f"rust{edition}:object compile timeout after {TIMEOUT_SECONDS}s")
                    continue
                except OSError as exc:
                    return {
                        "success": False,
                        "edition": edition,
                        "llvm_ir": "",
                        "stderr": f"cannot execute rust compiler for object validation: {exc}",
                        "ir_status": "failed_compiler_unavailable",
                    }
                if object_result.returncode != 0 or not object_path.exists():
                    failures.append(f"rust{edition}:object validation failed:{compact_error(object_result.stderr)}")
                    continue

            llvm_ir = llvm_path.read_text(encoding="utf-8", errors="replace").strip()
            if not KEEP_COMPILER_COMMENTS:
                llvm_ir = re.sub(r"^; ModuleID = .*\n", "", llvm_ir)
                llvm_ir = re.sub(r"^source_filename = .*\n", "", llvm_ir)

            return {
                "success": True,
                "edition": edition,
                "llvm_ir": sanitize_llvm_metadata_paths(llvm_ir),
                "stderr": result.stderr,
                "ir_status": f"success_clean_verified;edition={edition}",
            }

    return {
        "success": False,
        "edition": RUST_EDITIONS[0],
        "llvm_ir": "",
        "stderr": " || ".join(failures),
        "ir_status": "failed_compile;" + " || ".join(failures),
    }


def preprocess_record(record: Mapping[str, Any], split: str) -> dict[str, Any]:
    source = normalize_source(record.get("source_code") or record.get("func") or "")
    sample_id = make_sample_id(record, source)
    label = rust_label(record)

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "source_code": source,
        "wrapped_source_code": "",
        "llvm_ir": "",
        "label": label,
        "language": "rust",
        "split": split,
        "ir_status": "",
        "compile_error": "",
    }

    if not source:
        result["ir_status"] = "failed_malformed_source"
        result["compile_error"] = "empty source"
        return result
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        result["ir_status"] = "failed_source_too_large"
        result["compile_error"] = f"source exceeds {MAX_SOURCE_BYTES} bytes"
        return result

    attempt = compile_with_rustc(source, sample_id)
    result["ir_status"] = str(attempt["ir_status"])
    result["compile_error"] = "" if attempt["success"] else str(attempt["stderr"])
    if attempt["success"]:
        result["llvm_ir"] = str(attempt["llvm_ir"])
    return result


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


def compile_rust_source(
    source: str,
    *,
    sample_id: str = "single_rust_source",
    label: int = 0,
    split: str = "single",
) -> dict[str, Any]:
    record = {
        "sample_id": sample_id,
        "source_code": source,
        "label": label,
        "language": "rust",
    }
    return preprocess_record(record, split)


def is_successful_llvm_record(record: Mapping[str, Any]) -> bool:
    llvm_ir = str(record.get("llvm_ir") or "").strip()
    if not llvm_ir:
        return False
    if REQUIRE_LLVM_FUNCTION_DEFINITION and not has_llvm_function_definition(llvm_ir):
        return False
    status = str(record.get("ir_status") or "").strip()
    if not status:
        return True
    return status.startswith("success") and bool(llvm_ir)


def compact_compiled_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(record.get("sample_id", "")),
        "source_code": str(record.get("source_code", "")),
        "wrapped_source_code": str(record.get("wrapped_source_code", "")),
        "llvm_ir": str(record.get("llvm_ir", "")),
        "label": int(record.get("label", 0) or 0),
        "language": str(record.get("language", "")),
        "split": str(record.get("split", "")),
    }


def _preprocess_worker(payload: tuple[int, dict[str, Any], str]) -> tuple[int, dict[str, Any]]:
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
        progress = tqdm(processed_stream, desc="Rust processed", unit="module")
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
        raise SystemExit(f"Rust input not found: {INPUT_JSONL}")
    if shutil.which(RUSTC) is None:
        raise SystemExit(f"Rust compiler not found: {RUSTC}")

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
